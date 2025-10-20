# train_denoiser_cli.py
import argparse, os, datetime, torch
from text8_beta_diffusion.config import (
    DataConfig, DiffusionConfig, MLPConfig,
    InfoProviderDenoiserConfig, InfoProviderRuntimeConfig
)
from text8_beta_diffusion.setup_torch import set_seed, enable_fast_kernels, add_torchversion_safe_globals
from text8_beta_diffusion.utils.precision import PrecisionPolicy, to_model_dtype
from text8_beta_diffusion.data import make_dataloaders
from text8_beta_diffusion.models.transformer_compat import DenoiserCompat
from text8_beta_diffusion.models.transformer_llama2_like import DenoiserLlama2Like

from text8_beta_diffusion.models.scheduler_mlp import SchedulerMLP
from text8_beta_diffusion.info.denoiser_info import DenoiserInfoProvider
from text8_beta_diffusion.training.train_denoiser import train_diffusion
from text8_beta_diffusion.models.ema import ModelEMA
# AJOUTS:
from text8_beta_diffusion.config import ValidationConfig
from text8_beta_diffusion.training.sampling import sample_md4
from text8_beta_diffusion.data import decode_ids


def _flex_load_state_dict(model: torch.nn.Module, path: str):
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return missing, unexpected

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bf16-only", action="store_true")
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--logdir", type=str, default="runs", help="Dossier de sortie pour le CSV")
    args = p.parse_args()

    set_seed(1234); enable_fast_kernels(); add_torchversion_safe_globals()

    data_cfg = DataConfig()
    diff_cfg = DiffusionConfig()
    mlp_cfg  = MLPConfig()
    ip_den_cfg = InfoProviderDenoiserConfig()
    ip_run_cfg = InfoProviderRuntimeConfig()

    val_cfg = ValidationConfig()

    if args.steps is not None:
        diff_cfg.steps = args.steps

    precision = PrecisionPolicy(bf16_only=args.bf16_only, use_autocast=True, upcast_softmax_to_fp32=True)
    train_loader, val_loader, vocab_size, mask_id = make_dataloaders(data_cfg, diff_cfg.batch_size_denoiser)

    # -----------------------------------------------------------
    # 1) Denoiser A ENTRAINER (cible)
    # -----------------------------------------------------------
    tcfg = diff_cfg.transformer
    if tcfg.arch == "compat":
        denoiser = DenoiserCompat(
            vocab_size=vocab_size, d_model=tcfg.d_model, n_heads=tcfg.n_heads, n_layers=tcfg.n_layers,
            ff_mult=tcfg.ff_mult, dropout=tcfg.dropout, mask_id=mask_id, max_seq_len=tcfg.max_seq_len,
        )
    elif tcfg.arch == "llama2":
        denoiser = DenoiserLlama2Like(
            vocab_size=vocab_size, mask_id=mask_id, d_model=tcfg.d_model, n_heads=tcfg.n_heads, n_layers=tcfg.n_layers,
            n_kv_heads=tcfg.n_kv_heads, mlp_type=tcfg.mlp_type, multiple_of=tcfg.multiple_of, ff_mult=tcfg.ff_mult,
            dropout=tcfg.dropout, norm_eps=tcfg.norm_eps, w_init_scale=tcfg.w_init_scale,
            depth_scaled_init=tcfg.depth_scaled_init, cond_type=tcfg.cond_type, embed_input=tcfg.embed_input,
            rope_theta=tcfg.rope_theta, max_seq_len=tcfg.max_seq_len, weight_tying=tcfg.weight_tying,
            causal=tcfg.causal, time_scale=tcfg.time_scale,
        )
    else:
        raise ValueError(f"Unknown transformer arch: {tcfg.arch}")

    to_model_dtype(denoiser, precision)
    denoiser.classifier.prepare_rope(device=next(denoiser.parameters()).device, dtype=torch.float32)

    denoiser = torch.compile(denoiser, mode="reduce-overhead", dynamic=False, fullgraph=True)

    mlp_model = None
    info_provider = None

    if(not diff_cfg.train_objective == "md4"):

        # -----------------------------------------------------------
        # 2) Scheduler MLP (pré-entraîné) — utilisé pour le sampling/poids
        # -----------------------------------------------------------
        mlp_model = SchedulerMLP(
            seq_len=data_cfg.seq_len, hidden=mlp_cfg.hidden, depth=mlp_cfg.depth, dropout=mlp_cfg.dropout,
            alpha_min=mlp_cfg.alpha_min, alpha_max=mlp_cfg.alpha_max,
            beta_min=mlp_cfg.beta_min, beta_max=mlp_cfg.beta_max
        )
        to_model_dtype(mlp_model, precision)

        # Charger ses poids depuis le chemin configuré
        if mlp_cfg.load_path_for_info:
            if not os.path.isfile(mlp_cfg.load_path_for_info):
                if mlp_cfg.required_for_info:
                    raise FileNotFoundError(
                        f"[Config] MLPConfig.load_path_for_info='{mlp_cfg.load_path_for_info}' introuvable. "
                        f"Entraîne d'abord le scheduler (train_scheduler_cli.py) ou mets required_for_info=False."
                    )
                else:
                    print(f"[WARN] MLP pré-entraîné introuvable ({mlp_cfg.load_path_for_info}); on continue non-préentraîné.")
            else:
                missing, unexpected = _flex_load_state_dict(mlp_model, mlp_cfg.load_path_for_info)
                print(f"[MLP] Chargé depuis {mlp_cfg.load_path_for_info}. Missing={len(missing)} Unexpected={len(unexpected)}")

        # Geler le MLP (pas d'entraînement ici)
        mlp_model.eval()
        for p in mlp_model.parameters():
            p.requires_grad = False

        # -----------------------------------------------------------
        # 3) InfoProvider avec SON denoiser pré-entraîné (distinct)
        # -----------------------------------------------------------
        if ip_den_cfg.weights_path and not os.path.isfile(ip_den_cfg.weights_path):
            if ip_den_cfg.require_weights:
                raise FileNotFoundError(
                    f"[Config] InfoProviderDenoiserConfig.weights_path='{ip_den_cfg.weights_path}' introuvable. "
                    f"Fournis un denoiser pré-entraîné pour l'InfoProvider ou mets require_weights=False."
                )
            else:
                print(f"[WARN] Denoiser pré-entraîné pour InfoProvider introuvable ({ip_den_cfg.weights_path}).")

        info_provider = DenoiserInfoProvider(
            denoiser=None,                               # IMPORTANT: on n’utilise PAS le denoiser à entraîner
            weights_path=ip_den_cfg.weights_path,        # -> on reconstruit/charge depuis ce checkpoint
            vocab_size=vocab_size,
            d_model=ip_den_cfg.d_model,
            n_heads=ip_den_cfg.n_heads,
            n_layers=ip_den_cfg.n_layers,
            ff_mult=ip_den_cfg.ff_mult,
            dropout=ip_den_cfg.dropout,
            mask_id=mask_id,
            device=next(denoiser.parameters()).device,
            max_forward_batch=ip_run_cfg.max_forward_batch,
            K=ip_run_cfg.K,
            ensure_coverage=ip_run_cfg.ensure_coverage,
            normalize_sum_to_one=ip_run_cfg.normalize_sum_to_one,
            agg_mode=ip_run_cfg.agg_mode,
            ht_exclude_forced=ip_run_cfg.ht_exclude_forced,
            time_from_pi=ip_run_cfg.time_from_pi,
            mix_weight=ip_run_cfg.mix_weight,
            use_exp_perplexity=ip_run_cfg.use_exp_perplexity,
            precision=precision,
            max_seq_len=ip_den_cfg.max_seq_len,
            verbose_load=True
        )
        # Le modèle interne de l'InfoProvider est en eval() et gelé par construction.
        info_provider.model.eval()
        for p in info_provider.model.parameters():
            p.requires_grad = False
        info_provider.model = torch.compile(
            info_provider.model,
            mode="reduce-overhead",
            dynamic=True,        # micro-batches de tailles variables
            fullgraph=True
        )

        # Le modèle interne de l'InfoProvider est en eval() et gelé par construction.
        info_provider.model.eval()
        for p in info_provider.model.parameters():
            p.requires_grad = False

        # AJOUT: Étape de sampling initiale MD4 avec le denoiser de l'InfoProvider
        if val_cfg.enable_initial_md4_from_infoprovider:
            try:
                print("==== Initial MD4 sampling with InfoProvider's denoiser ====")
                # S'assurer que RoPE est prêt (normalement déjà fait dans DenoiserInfoProvider.__init__)
                try:
                    info_provider.model.classifier.prepare_rope(
                        device=next(info_provider.model.parameters()).device,
                        dtype=torch.float32
                    )
                except AttributeError:
                    pass

                final_tokens, _ = sample_md4(
                    model=info_provider.model,
                    n_tokens=data_cfg.seq_len,
                    steps=val_cfg.sampling_steps,
                    grid=val_cfg.sampling_grid,
                    schedule=val_cfg.sampling_schedule,
                    eps=1e-4,
                    temperature=val_cfg.sampling_temperature,
                    snapshots_every=val_cfg.sampling_snapshots_every,
                    n_samples=val_cfg.sampling_n_samples,
                    prefix=val_cfg.sampling_prefix,
                    top_p=val_cfg.sampling_top_p,
                )
                # Affiche le premier échantillon
                print(decode_ids(final_tokens[0]))
                print("==== End of initial MD4 sampling ====")
            except Exception as e:
                print(f"[WARN] Initial MD4 sampling failed: {e}")

    # -----------------------------------------------------------
    # 4) Dossier logs + lancement training
    # -----------------------------------------------------------
    os.makedirs(args.logdir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = os.path.join(args.logdir, f"denoiser_metrics_{ts}.csv")

    ema = None if (not diff_cfg.ema) else ModelEMA(denoiser, decay=diff_cfg.ema_decay)

    print("START TRAINING")

    train_diffusion(
        denoiser=denoiser, mlp_model=mlp_model, info_provider=info_provider,
        train_loader=train_loader, val_loader=val_loader, diff_cfg=diff_cfg,
        precision_policy=precision, weighted_loss=diff_cfg.weighted_loss,
        sample_every=diff_cfg.sample_every, top_p=args.top_p, ema=ema,
        data_cfg=data_cfg, mlp_cfg=mlp_cfg, sampling_cfg=None, log_csv_path=csv_path
    )

if __name__ == "__main__":
    main()
