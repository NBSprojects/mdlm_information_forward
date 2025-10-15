import argparse
import torch
from text8_beta_diffusion.config import DataConfig, DiffusionConfig, MLPConfig
from text8_beta_diffusion.setup_torch import set_seed, enable_fast_kernels, add_torchversion_safe_globals
from text8_beta_diffusion.data import make_dataloaders, decode_ids
from text8_beta_diffusion.models.transformer_compat import DenoiserCompat
from text8_beta_diffusion.models.scheduler_mlp import SchedulerMLP
from text8_beta_diffusion.training.sampling import sample_md4, sample_md4_beta_gw

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--grid", type=str, default="cosine")
    p.add_argument("--temperature", type=float, default=1.0)
    args = p.parse_args()
    set_seed(1234); enable_fast_kernels(); add_torchversion_safe_globals()
    data_cfg = DataConfig(); diff_cfg = DiffusionConfig(); mlp_cfg = MLPConfig()
    _, _, vocab_size, mask_id = make_dataloaders(data_cfg, diff_cfg.batch_size_denoiser)
    denoiser = DenoiserCompat(vocab_size=vocab_size, d_model=diff_cfg.d_model,
                              n_heads=diff_cfg.n_heads, n_layers=diff_cfg.n_layers,
                              ff_mult=diff_cfg.ff_mult, dropout=diff_cfg.dropout, max_seq_len=data_cfg.seq_len)
    denoiser.classifier.prepare_rope(device=next(denoiser.parameters()).device, dtype=torch.float32)
    mlp_model = SchedulerMLP(seq_len=data_cfg.seq_len, hidden=mlp_cfg.hidden, depth=mlp_cfg.depth, dropout=mlp_cfg.dropout,
                             alpha_min=mlp_cfg.alpha_min, alpha_max=mlp_cfg.alpha_max,
                             beta_min=mlp_cfg.beta_min, beta_max=mlp_cfg.beta_max)
    final_tokens, snaps = sample_md4(
        model=denoiser, n_tokens=data_cfg.seq_len, steps=args.steps,
        grid=args.grid, schedule="linear", temperature=args.temperature,
        snapshots_every=100, n_samples=1, prefix=None, top_p=args.top_p
    )
    print("==== Final sample (baseline) ====")
    print(decode_ids(final_tokens[0]))
    final_tokens, snaps = sample_md4_beta_gw(
        denoiser=denoiser, mlp_model=mlp_model, n_tokens=data_cfg.seq_len, steps=args.steps,
        grid=args.grid, temperature=args.temperature, snapshots_every=100, n_samples=1,
        prefix=None, top_p=args.top_p
    )
    print("==== Final sample (Beta GW) ====")
    print(decode_ids(final_tokens[0]))

if __name__ == "__main__":
    main()
