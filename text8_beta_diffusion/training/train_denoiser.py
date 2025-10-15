from __future__ import annotations
import torch, torch.nn as nn
from ..data import mask_id, decode_ids
from .masking import sample_xt_and_weights
from .losses import masked_ce_losses



def train_diffusion(
    denoiser, mlp_model, info_provider, train_loader, val_loader, diff_cfg,
    precision_policy=None, weighted_loss: bool = True, sample_every: int = 0,
    sample_n: int = 8, sample_steps: int = 200, sample_grid: str = "cosine",
    sample_temperature: float = 1.0, sample_prefix=None, top_p: float = 1.0,
    ema=None, ckpt_path: str | None = None
):
    denoiser.train(); mlp_model.eval()
    opt = torch.optim.AdamW(denoiser.parameters(), lr=diff_cfg.lr, weight_decay=diff_cfg.wd, fused=True)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: min((it+1)/max(diff_cfg.warmup,1), 1.0))
    it = iter(train_loader)
    device = next(denoiser.parameters()).device
    for step in range(1, diff_cfg.steps+1):
        try: x0, _ = next(it)
        except StopIteration: it = iter(train_loader); x0, _ = next(it)
        x0 = x0.to(device, non_blocking=True)
        info = info_provider.compute_info(x0)
        xt, t, W, p = sample_xt_and_weights(x0, info, mlp_model)
        logits = denoiser(xt, t)
        losses = masked_ce_losses(logits, x0, xt, W, mask_id=mask_id)
        loss = losses["loss_weighted"] if weighted_loss else losses["loss_unweighted"]
        opt.zero_grad(set_to_none=True); loss.backward()
        try:
            nn.utils.clip_grad_norm_(denoiser.parameters(), diff_cfg.grad_clip, foreach=True)
        except TypeError:
            nn.utils.clip_grad_norm_(denoiser.parameters(), diff_cfg.grad_clip)
        opt.step(); sched.step()
        if ema is not None and step >= diff_cfg.ema_start:
            ema.update(denoiser)
        if step % diff_cfg.log_interval == 0:
            H_seq = info.sum(dim=1)
            target_info = (1.0 - t) * H_seq
            mask_beta = (xt == mask_id).float()
            info_remain_beta = (info * (1.0 - mask_beta)).sum(dim=1)
            mse_beta = ((target_info - info_remain_beta) ** 2).mean().item()
            print(
              f"[Den] step {step}/{diff_cfg.steps} "
              f"loss(w)={float(losses['loss_weighted']):.4f} "
              f"loss(uw)={float(losses['loss_unweighted']):.4f} "
              f"lr={sched.get_last_lr()[0]:.2e} "
              f"mse_beta={mse_beta:.6f} masked_frac={float(p.mean()):.3f} n_masked={losses['n_masked']}"
            )
        if step % diff_cfg.eval_interval == 0:
            eval_net = ema.ema if ema is not None else denoiser
            eval_net.eval()
            with torch.no_grad():
                x0_eval, _ = next(iter(val_loader))
                x0_eval = x0_eval.to(device, non_blocking=True)
                info_eval = info_provider.compute_info(x0_eval)
                xt_eval, t_eval, W_eval, _ = sample_xt_and_weights(x0_eval, info_eval, mlp_model)
                logits_eval = eval_net(xt_eval, t_eval)
                val_losses = masked_ce_losses(logits_eval, x0_eval, xt_eval, W_eval, mask_id=mask_id)
                print(f"         [Eval] loss(w)={float(val_losses['loss_weighted']):.4f} "
                      f"loss(uw)={float(val_losses['loss_unweighted']):.4f}")
            denoiser.train()
        if sample_every and (step % sample_every == 0):
            from .sampling import sample_md4_beta_gw
            eval_net = ema.ema if ema is not None else denoiser
            eval_net.eval()
            with torch.no_grad():
                gen, _ = sample_md4_beta_gw(
                    denoiser=eval_net, mlp_model=mlp_model, n_tokens=x0.size(1),
                    steps=sample_steps, grid=sample_grid, temperature=sample_temperature,
                    snapshots_every=None, n_samples=sample_n, prefix=sample_prefix, top_p=top_p
                )
                print(f"\n[Sampling @ {step}]")
                for i in range(min(sample_n, gen.size(0))):
                    print(">>>", i, ":", decode_ids(gen[i].cpu()))
            denoiser.train()
        if ckpt_path and step % (10*diff_cfg.eval_interval) == 0:
            torch.save({"denoiser": denoiser.state_dict(), "step": step}, ckpt_path)
