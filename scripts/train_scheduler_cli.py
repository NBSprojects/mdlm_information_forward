# train_scheduler_cli.py
import argparse, os, torch
from text8_beta_diffusion.config import DataConfig, MLPConfig, InfoProviderDenoiserConfig, InfoProviderRuntimeConfig
from text8_beta_diffusion.setup_torch import set_seed, enable_fast_kernels, add_torchversion_safe_globals
from text8_beta_diffusion.utils.precision import PrecisionPolicy, to_model_dtype
from text8_beta_diffusion.data import make_dataloaders
from text8_beta_diffusion.models.scheduler_mlp import SchedulerMLP
from text8_beta_diffusion.training.train_scheduler import train_scheduler_mlp
from text8_beta_diffusion.info.denoiser_info import DenoiserInfoProvider
from text8_beta_diffusion.diagnostics.eval_forward import integrated_mse_vs_baseline

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bf16-only", action="store_true")
    p.add_argument("--steps", type=int, default=None)
    args = p.parse_args()
    set_seed(1234); enable_fast_kernels(); add_torchversion_safe_globals()

    data_cfg = DataConfig(); mlp_cfg = MLPConfig()
    ip_den_cfg = InfoProviderDenoiserConfig()
    ip_run_cfg = InfoProviderRuntimeConfig()
    if args.steps is not None:
        mlp_cfg.steps = args.steps

    precision = PrecisionPolicy(bf16_only=args.bf16_only, use_autocast=True, upcast_softmax_to_fp32=True)
    train_loader, val_loader, vocab_size, mask_id = make_dataloaders(data_cfg, mlp_cfg.batch_size_MLP)

    model = SchedulerMLP(
        seq_len=data_cfg.seq_len, hidden=mlp_cfg.hidden, depth=mlp_cfg.depth, dropout=mlp_cfg.dropout,
        alpha_min=mlp_cfg.alpha_min, alpha_max=mlp_cfg.alpha_max,
        beta_min=mlp_cfg.beta_min, beta_max=mlp_cfg.beta_max
    )
    to_model_dtype(model, precision)

    device = next(model.parameters()).device

    # InfoProvider (reconstruit depuis checkpoint)
    info_provider = DenoiserInfoProvider(
        denoiser=None,
        weights_path=ip_den_cfg.weights_path,
        vocab_size=vocab_size,
        d_model=ip_den_cfg.d_model,
        n_heads=ip_den_cfg.n_heads,
        n_layers=ip_den_cfg.n_layers,
        ff_mult=ip_den_cfg.ff_mult,
        dropout=ip_den_cfg.dropout,
        mask_id=mask_id,
        device=device,
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
        verbose_load=True,
    )

    # PRE-EVAL
    if getattr(mlp_cfg, "mse_eval_enable", True):
        m_model, m_base, rel = integrated_mse_vs_baseline(
            mlp_model=model,
            info_provider=info_provider,
            val_loader=val_loader,
            batches=mlp_cfg.mse_eval_batches,
            t_samples=mlp_cfg.mse_eval_t_samples,
        )
        print(f"[MLP][PRE] integrated_mse={m_model:.7f}  baseline={m_base:.7f}  rel_gain={(rel):.5f}")


    train_scheduler_mlp(
        model, steps=mlp_cfg.steps, lr=mlp_cfg.lr, wd=mlp_cfg.wd,
        batch_size=mlp_cfg.batch_size_MLP, seq_len=data_cfg.seq_len,
        t_samples_per_step=mlp_cfg.t_samples_per_step, device=next(model.parameters()).device
    )

    # POST-EVAL
    if getattr(mlp_cfg, "mse_eval_enable", True):
        m_model, m_base, rel = integrated_mse_vs_baseline(
            mlp_model=model,
            info_provider=info_provider,
            val_loader=val_loader,
            batches=mlp_cfg.mse_eval_batches,
            t_samples=mlp_cfg.mse_eval_t_samples,
        )
        print(f"[MLP][POST] integrated_mse={m_model:.7f}  baseline={m_base:.7f}  rel_gain={(rel):.5f}")

    # === Sauvegarde ===
    if mlp_cfg.save_path:
        os.makedirs(os.path.dirname(mlp_cfg.save_path), exist_ok=True)
        torch.save(model.state_dict(), mlp_cfg.save_path)
        print(f"[MLP] Weights saved to: {mlp_cfg.save_path}")

if __name__ == "__main__":
    main()
