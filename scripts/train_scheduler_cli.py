import argparse
from text8_beta_diffusion.config import DataConfig, MLPConfig
from text8_beta_diffusion.setup_torch import set_seed, enable_fast_kernels, add_torchversion_safe_globals
from text8_beta_diffusion.utils.precision import PrecisionPolicy, to_model_dtype
from text8_beta_diffusion.data import make_dataloaders
from text8_beta_diffusion.models.scheduler_mlp import SchedulerMLP
from text8_beta_diffusion.training.train_scheduler import train_scheduler_mlp

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bf16-only", action="store_true")
    p.add_argument("--steps", type=int, default=None)
    args = p.parse_args()
    set_seed(1234); enable_fast_kernels(); add_torchversion_safe_globals()
    data_cfg = DataConfig(); mlp_cfg = MLPConfig()
    if args.steps is not None:
        mlp_cfg.steps = args.steps
    precision = PrecisionPolicy(bf16_only=args.bf16_only, use_autocast=True, upcast_softmax_to_fp32=True)
    train_loader, _, _, _ = make_dataloaders(data_cfg, mlp_cfg.batch_size_MLP)
    model = SchedulerMLP(seq_len=data_cfg.seq_len, hidden=mlp_cfg.hidden, depth=mlp_cfg.depth, dropout=mlp_cfg.dropout,
                         alpha_min=mlp_cfg.alpha_min, alpha_max=mlp_cfg.alpha_max,
                         beta_min=mlp_cfg.beta_min, beta_max=mlp_cfg.beta_max)
    to_model_dtype(model, precision)
    train_scheduler_mlp(model, steps=mlp_cfg.steps, lr=mlp_cfg.lr, wd=mlp_cfg.wd,
                        batch_size=mlp_cfg.batch_size_MLP, seq_len=data_cfg.seq_len,
                        t_samples_per_step=mlp_cfg.t_samples_per_step, device=next(model.parameters()).device)

if __name__ == "__main__":
    main()
