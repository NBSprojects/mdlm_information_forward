import argparse, os, datetime, torch
from text8_beta_diffusion.config import DataConfig, DiffusionConfig, MLPConfig
from text8_beta_diffusion.setup_torch import set_seed, enable_fast_kernels, add_torchversion_safe_globals
from text8_beta_diffusion.utils.precision import PrecisionPolicy, to_model_dtype
from text8_beta_diffusion.data import make_dataloaders
from text8_beta_diffusion.models.transformer_compat import DenoiserCompat
from text8_beta_diffusion.models.scheduler_mlp import SchedulerMLP
from text8_beta_diffusion.info.denoiser_info import DenoiserInfoProvider
from text8_beta_diffusion.training.train_denoiser import train_diffusion
from text8_beta_diffusion.models.ema import ModelEMA

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bf16-only", action="store_true")
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--logdir", type=str, default="runs", help="Dossier de sortie pour le CSV")
    args = p.parse_args()
    set_seed(1234); enable_fast_kernels(); add_torchversion_safe_globals()
    data_cfg = DataConfig(); diff_cfg = DiffusionConfig(); mlp_cfg  = MLPConfig()

    

    if args.steps is not None:
        diff_cfg.steps = args.steps
    precision = PrecisionPolicy(bf16_only=args.bf16_only, use_autocast=True, upcast_softmax_to_fp32=True)
    train_loader, val_loader, vocab_size, mask_id = make_dataloaders(data_cfg, diff_cfg.batch_size_denoiser)
    denoiser = DenoiserCompat(vocab_size=vocab_size, d_model=diff_cfg.d_model,
                              n_heads=diff_cfg.n_heads, n_layers=diff_cfg.n_layers,
                              ff_mult=diff_cfg.ff_mult, dropout=diff_cfg.dropout, max_seq_len=data_cfg.seq_len)
    to_model_dtype(denoiser, precision)
    denoiser.classifier.prepare_rope(device=next(denoiser.parameters()).device, dtype=torch.float32)

    denoiser = torch.compile(
        denoiser,
        mode="reduce-overhead",   # bon défaut pour un entraînement itératif
        dynamic=False,            # formes stables = meilleur graphe + moins de recompiles
        fullgraph=True            # essaye d'abord True; si ça "break" => enlève-le
    )
    # MLP
    mlp_model = SchedulerMLP(seq_len=data_cfg.seq_len, hidden=mlp_cfg.hidden, depth=mlp_cfg.depth, dropout=mlp_cfg.dropout,
                             alpha_min=mlp_cfg.alpha_min, alpha_max=mlp_cfg.alpha_max,
                             beta_min=mlp_cfg.beta_min, beta_max=mlp_cfg.beta_max)
    to_model_dtype(mlp_model, precision)
    info_provider = DenoiserInfoProvider(
        denoiser=denoiser, mask_id=mask_id, device=next(denoiser.parameters()).device, K=4,
        ensure_coverage=False, normalize_sum_to_one=True, agg_mode="hajek",
        mix_weight=0.5, use_exp_perplexity=False, max_forward_batch=None,
        precision=precision, max_seq_len=data_cfg.seq_len
    )

    os.makedirs(args.logdir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = os.path.join(args.logdir, f"denoiser_metrics_{ts}.csv")


    ema = None if (not diff_cfg.ema) else ModelEMA(denoiser, decay=diff_cfg.ema_decay)
    train_diffusion(
        denoiser=denoiser, mlp_model=mlp_model, info_provider=info_provider,
        train_loader=train_loader, val_loader=val_loader, diff_cfg=diff_cfg,
        precision_policy=precision, weighted_loss=diff_cfg.weighted_loss,
        sample_every=diff_cfg.sample_every, top_p=args.top_p, ema=ema,
        data_cfg=data_cfg, mlp_cfg=mlp_cfg, sampling_cfg=None,
        log_csv_path=csv_path
    )

if __name__ == "__main__":
    main()
