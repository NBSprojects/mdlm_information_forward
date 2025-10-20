# --- MODIFS PRINCIPALES DANS train_denoiser.py ---

from __future__ import annotations
import time, math, torch, torch.nn as nn
import torch.nn.functional as F
import os
from ..data import mask_id, decode_ids
from .masking import sample_xt_and_weights          # <-- EXISTANT (mode beta_gw)
from .losses import masked_ce_losses
from ..utils.timing import CudaTimers
from ..utils.logging import log, CSVLogger


# >>> AJOUTS :
from .md4_objective import (
    sample_times, forward_sample, dgamma_times_alpha, logsnr, alpha,
    masked_ce_per_sample as md4_masked_ce_per_sample
)


def train_diffusion(
    denoiser, mlp_model, info_provider, train_loader, val_loader, diff_cfg,
    precision_policy=None, weighted_loss: bool = True, sample_every: int = 0,
    sample_n: int = 8, sample_steps: int = 200, sample_grid: str = "cosine",
    sample_temperature: float = 1.0, sample_prefix=None, top_p: float = 1.0,
    ema=None, ckpt_path: str | None = None,
    data_cfg=None, mlp_cfg=None, sampling_cfg=None,
    log_csv_path: str = "runs/denoiser_train_metrics.csv"
):
    denoiser.train();  # mlp_model peut être None en mode md4
    opt = torch.optim.AdamW(denoiser.parameters(), lr=diff_cfg.lr, weight_decay=diff_cfg.wd, fused=True)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: min((it+1)/max(diff_cfg.warmup,1), 1.0))
    it = iter(train_loader)
    device = next(denoiser.parameters()).device

    timers = CudaTimers(enabled=torch.cuda.is_available())
    csv = CSVLogger(log_csv_path, data_cfg, diff_cfg, mlp_cfg, sampling_cfg)

    for step in range(1, diff_cfg.steps+1):
        try:
            x0, _ = next(it)
        except StopIteration:
            it = iter(train_loader); x0, _ = next(it)

        timers.start("step_total")

        timers.start("h2d")
        x0 = x0.to(device, non_blocking=True)
        timers.end("h2d")

        if diff_cfg.train_objective == "beta_gw":
            # === EXISTANT (inchangé) ===
            with timers.phase("info_compute"):
                info = info_provider.compute_info(x0)
            with timers.phase("sample_xt"):
                xt, t, W, p = sample_xt_and_weights(x0, info, mlp_model)
            with timers.phase("forward"):
                logits = denoiser(xt, t)
            with timers.phase("loss"):
                losses = masked_ce_losses(logits, x0, xt, W, mask_id=mask_id)
                loss = losses["loss_weighted"] if weighted_loss else losses["loss_unweighted"]
            loss_unweighted_common = losses["loss_unweighted"].item()
            masked_frac = float(p.mean().item())
            n_masked    = int(losses["n_masked"])
            sum_weights = float(losses["sum_weights"])
        # ---------------------------------------------------
        else:
            # === NOUVEAU : OBJECTIF MD4 (CODE 1) ===
            with timers.phase("sample_xt"):
                B = x0.size(0)
                # 1) échantillonner t (antithétique optionnel)
                t_raw = sample_times(B, device, diff_cfg.md4_antithetic_time_sampling)  # [B] ~ U(0,1)
                if diff_cfg.md4_cont_time:
                    t_eff = t_raw
                else:
                    # t ∈ {1/T, 2/T, ..., 1}
                    T = float(diff_cfg.md4_timesteps)
                    t_eff = (torch.floor(t_raw * T) + 1.0) / T

                # 2) bruitage x_t
                xt = forward_sample(x0, t_eff, schedule=diff_cfg.md4_noise_schedule,
                                    eps=diff_cfg.md4_eps, mask_id=mask_id)

            with timers.phase("forward"):
                logits = denoiser(xt, t_eff)

            with timers.phase("loss"):
                # 3) Somme des log-probas vraies masquées (négatives), comme Google
                B, L, V = logits.shape
                logp = F.log_softmax(logits, dim=-1)                  # [B, L, V]
                logp_true = logp.gather(-1, x0.unsqueeze(-1)).squeeze(-1)  # [B, L] (<= 0)
                mask_f = (xt == mask_id).float()                      # [B, L]
                masked_neg_cross_ent = (mask_f * logp_true).sum(dim=1)     # [B]

                if diff_cfg.md4_cont_time:
                    w = dgamma_times_alpha(
                        t_eff, schedule=diff_cfg.md4_noise_schedule, eps=diff_cfg.md4_eps
                    )  # [B] (<= 0)
                else:
                    T = float(diff_cfg.md4_timesteps)
                    s_eff = t_eff - (1.0 / T)
                    g_t = logsnr(t_eff, diff_cfg.md4_noise_schedule, diff_cfg.md4_eps)
                    g_s = logsnr(s_eff, diff_cfg.md4_noise_schedule, diff_cfg.md4_eps)
                    a_s = alpha(s_eff, diff_cfg.md4_noise_schedule, diff_cfg.md4_eps)
                    w = T * torch.expm1(g_t - g_s) * a_s               # [B] (<= 0)

                loss = (w * masked_neg_cross_ent).mean()               # produit >= 0

                n_mask = mask_f.sum(dim=1)                             # [B]
                ce_mask_mean = (-(mask_f * logp_true).sum(dim=1) / n_mask.clamp_min(1)).mean()
                loss_unweighted_common = float(ce_mask_mean.item())

                loss = (w * masked_neg_cross_ent).mean()

            # stats pour logs
            masked_frac = float((xt == mask_id).float().mean().item())
            n_masked    = int(n_mask.sum().item())    # somme sur le batch
            sum_weights = float(w.sum().item())
            # === FIN OBJECTIF MD4 ===
        # ----------------- FIN BRANCHE D'OBJECTIF -----------------

        opt.zero_grad(set_to_none=True)
        with timers.phase("backward"):
            loss.backward()

        try:
            grad_total_norm = nn.utils.clip_grad_norm_(denoiser.parameters(), diff_cfg.grad_clip, foreach=True)
        except TypeError:
            grad_total_norm = nn.utils.clip_grad_norm_(denoiser.parameters(), diff_cfg.grad_clip)

        with timers.phase("optim_step"):
            opt.step(); sched.step()

        if ema is not None and step >= getattr(diff_cfg, "ema_start", 0):
            with timers.phase("ema_update"):
                ema.update(denoiser)

        timers.end("step_total")

        times_ms = timers.elapsed_all_ms()
        row = {
            "step": step,
            "loss_weighted": float(loss.item()),           # pour uniformiser le CSV
            "loss_unweighted": float(loss_unweighted_common),               # N/A en md4
            "lr": float(sched.get_last_lr()[0]),
            "grad_norm": float(grad_total_norm if isinstance(grad_total_norm, (int, float)) else grad_total_norm.item()),
            "time.step_total_ms": float(times_ms.get("step_total", float("nan"))),
            "time.h2d_ms": float(times_ms.get("h2d", float("nan"))),
            "time.info_compute_ms": float(times_ms.get("info_compute", float("nan")) if diff_cfg.train_objective=="beta_gw" else float("nan")),
            "time.sample_xt_ms": float(times_ms.get("sample_xt", float("nan"))),
            "time.forward_ms": float(times_ms.get("forward", float("nan"))),
            "time.loss_ms": float(times_ms.get("loss", float("nan"))),
            "time.backward_ms": float(times_ms.get("backward", float("nan"))),
            "time.optim_step_ms": float(times_ms.get("optim_step", float("nan"))),
            "time.ema_update_ms": float(times_ms.get("ema_update", float("nan"))),
            "masked_frac": masked_frac,
            "n_masked": n_masked,
            "sum_mask_weights": sum_weights,
        }

        # --- LOG console + verbose batch (restent inchangés côté beta_gw).
        # On peut rendre verbose_batch inactif en md4 ou l’adapter ; ici je le laisse inactif en md4 pour rester simple.
        if step % diff_cfg.log_interval == 0:
            info_str = (f" info={row['time.info_compute_ms']:.2f}" if diff_cfg.train_objective=="beta_gw" else "")
            log(f"[Den] step {step}/{diff_cfg.steps} "
                f"loss={row['loss_weighted']:.4f} loss_u={row['loss_unweighted']:.4f} "
                f"lr={row['lr']:.2e} grad_norm={row['grad_norm']:.3f} "
                f"masked_frac={row['masked_frac']:.3f} n_masked={row['n_masked']} | "
                f"t_step={row['time.step_total_ms']:.2f}ms "
                f"[h2d={row['time.h2d_ms']:.2f}{info_str} samp={row['time.sample_xt_ms']:.2f} "
                f"fwd={row['time.forward_ms']:.2f} loss={row['time.loss_ms']:.2f} "
                f"bwd={row['time.backward_ms']:.2f} opt={row['time.optim_step_ms']:.2f}]"
            )

        # --- Eval périodique (je conserve votre logique ; en md4 on évalue la même perte md4)
        if step % diff_cfg.eval_interval == 0:
            eval_net = ema.ema if ema is not None else denoiser
            eval_net.eval()
            with torch.no_grad():
                x0_eval, _ = next(iter(val_loader))
                x0_eval = x0_eval.to(device, non_blocking=True)
                if diff_cfg.train_objective == "beta_gw":
                    info_eval = info_provider.compute_info(x0_eval)
                    xt_eval, t_eval, W_eval, _ = sample_xt_and_weights(x0_eval, info_eval, mlp_model)
                    logits_eval = eval_net(xt_eval, t_eval)
                    val_losses = masked_ce_losses(logits_eval, x0_eval, xt_eval, W_eval, mask_id=mask_id)
                    row["val_loss_weighted"]   = float(val_losses["loss_weighted"].item())
                    row["val_loss_unweighted"] = float(val_losses["loss_unweighted"].item())
                else:
                    # md4
                    B_eval = x0_eval.size(0)
                    t_eval = sample_times(B_eval, device, diff_cfg.md4_antithetic_time_sampling)
                    if not diff_cfg.md4_cont_time:
                        T = float(diff_cfg.md4_timesteps)
                        t_eval = (torch.floor(t_eval * T) + 1.0) / T
                    xt_eval = forward_sample(x0_eval, t_eval, diff_cfg.md4_noise_schedule, diff_cfg.md4_eps, mask_id)
                    logits_eval = eval_net(xt_eval, t_eval)
                    ce_batch, _ = md4_masked_ce_per_sample(
                        logits_eval, x0_eval, xt_eval, mask_id=mask_id,
                        normalize_by_masked=diff_cfg.md4_normalize_by_masked_tokens
                    )
                    if diff_cfg.md4_cont_time:
                        w_eval = dgamma_times_alpha(t_eval, diff_cfg.md4_noise_schedule, diff_cfg.md4_eps)
                    else:
                        T = float(diff_cfg.md4_timesteps)
                        s_eval = t_eval - (1.0 / T)
                        g_t = logsnr(t_eval, diff_cfg.md4_noise_schedule, diff_cfg.md4_eps)
                        g_s = logsnr(s_eval, diff_cfg.md4_noise_schedule, diff_cfg.md4_eps)
                        a_s = alpha(s_eval, diff_cfg.md4_noise_schedule, diff_cfg.md4_eps)
                        w_eval = T * torch.expm1(g_t - g_s) * a_s
                    row["val_loss_weighted"]   = float((w_eval * ce_batch).mean().item())
                    row["val_loss_unweighted"] = float("nan")
                log(f"         [Eval] loss={row['val_loss_weighted']:.4f}")
            denoiser.train()

        csv.log_row(row)

        # --- Sampling "demo" pendant l'entraînement (je garde votre beta_gw).
        if sample_every and (step % sample_every == 0):
            from .sampling import sample_md4_beta_gw, sample_md4
            eval_net = ema.ema if ema is not None else denoiser
            eval_net.eval()
            with torch.no_grad():
                if diff_cfg.train_objective == "beta_gw":
                    gen, _ = sample_md4_beta_gw(
                        denoiser=eval_net, mlp_model=mlp_model, n_tokens=x0.size(1),
                        steps=sample_steps, grid=sample_grid, temperature=sample_temperature,
                        snapshots_every=None, n_samples=sample_n, prefix=sample_prefix, top_p=top_p
                    )
                else:
                    gen, _ = sample_md4(
                        model=eval_net, n_tokens=x0.size(1),
                        steps=sample_steps, grid=sample_grid, schedule=sampling_cfg.schedule if sampling_cfg else "linear",
                        eps=1e-4, temperature=sample_temperature,
                        snapshots_every=None, n_samples=sample_n, prefix=sample_prefix, top_p=top_p
                    )
                print(f"\n[Sampling @ {step}]")
                for i in range(min(sample_n, gen.size(0))):
                    print(">>>", i, ":", decode_ids(gen[i].cpu()))
            denoiser.train()

        save_path_base = ckpt_path or getattr(diff_cfg, "ckpt_path", None)
        save_every = getattr(diff_cfg, "ckpt_every", 0)
        overwrite = getattr(diff_cfg, "ckpt_overwrite", True)

        if save_path_base and save_every > 0 and (step % save_every == 0):
            root, ext = os.path.splitext(save_path_base)
            ext = ext if ext else ".pt"
            path = save_path_base if overwrite else f"{root}_step{step}{ext}"
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            torch.save({"denoiser": denoiser.state_dict(), "step": step}, path)

    csv.flush()
    log(f"[CSV] écrit: {log_csv_path}")
