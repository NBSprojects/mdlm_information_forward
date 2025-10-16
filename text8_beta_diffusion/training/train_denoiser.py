from __future__ import annotations
import time, math, torch, torch.nn as nn
from ..data import mask_id, decode_ids
from .masking import sample_xt_and_weights
from .losses import masked_ce_losses
from ..utils.timing import CudaTimers
from ..utils.logging import log, CSVLogger



def train_diffusion(
    denoiser, mlp_model, info_provider, train_loader, val_loader, diff_cfg,
    precision_policy=None, weighted_loss: bool = True, sample_every: int = 0,
    sample_n: int = 8, sample_steps: int = 200, sample_grid: str = "cosine",
    sample_temperature: float = 1.0, sample_prefix=None, top_p: float = 1.0,
    ema=None, ckpt_path: str | None = None,
    # NOUVEAU :
    data_cfg=None, mlp_cfg=None, sampling_cfg=None,
    log_csv_path: str = "runs/denoiser_train_metrics.csv"
):
    denoiser.train(); mlp_model.eval()
    opt = torch.optim.AdamW(denoiser.parameters(), lr=diff_cfg.lr, weight_decay=diff_cfg.wd, fused=True)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: min((it+1)/max(diff_cfg.warmup,1), 1.0))
    it = iter(train_loader)
    device = next(denoiser.parameters()).device

    # NOUVEAU: timers + logger CSV
    timers = CudaTimers(enabled=torch.cuda.is_available())
    csv = CSVLogger(log_csv_path, data_cfg, diff_cfg, mlp_cfg, sampling_cfg)

    for step in range(1, diff_cfg.steps+1):
        try:
            x0, _ = next(it)
        except StopIteration:
            it = iter(train_loader); x0, _ = next(it)

        # --- Chronométrage GPU par phases ---
        timers.start("step_total")

        timers.start("h2d")
        x0 = x0.to(device, non_blocking=True)
        timers.end("h2d")

        with timers.phase("info_compute"):
            info = info_provider.compute_info(x0)

        with timers.phase("sample_xt"):
            xt, t, W, p = sample_xt_and_weights(x0, info, mlp_model)

        with timers.phase("forward"):
            logits = denoiser(xt, t)

        with timers.phase("loss"):
            losses = masked_ce_losses(logits, x0, xt, W, mask_id=mask_id)
            loss = losses["loss_weighted"] if weighted_loss else losses["loss_unweighted"]

        opt.zero_grad(set_to_none=True)

        with timers.phase("backward"):
            loss.backward()

        # grad norm AVANT clipping (valeur retournée = norm totale avant clip)
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

        # Récupérer tous les timings (ms)
        times_ms = timers.elapsed_all_ms()
        # lignes du CSV (val.* sera renseigné plus bas si on évalue à ce step)
        row = {
            "step": step,
            "loss_weighted": float(losses["loss_weighted"]),
            "loss_unweighted": float(losses["loss_unweighted"]),
            "lr": float(sched.get_last_lr()[0]),
            "grad_norm": float(grad_total_norm if isinstance(grad_total_norm, (int, float)) else grad_total_norm.item()),
            # temps clés
            "time.step_total_ms": float(times_ms.get("step_total", float("nan"))),
            "time.h2d_ms": float(times_ms.get("h2d", float("nan"))),
            "time.info_compute_ms": float(times_ms.get("info_compute", float("nan"))),
            "time.sample_xt_ms": float(times_ms.get("sample_xt", float("nan"))),
            "time.forward_ms": float(times_ms.get("forward", float("nan"))),
            "time.loss_ms": float(times_ms.get("loss", float("nan"))),
            "time.backward_ms": float(times_ms.get("backward", float("nan"))),
            "time.optim_step_ms": float(times_ms.get("optim_step", float("nan"))),
            "time.ema_update_ms": float(times_ms.get("ema_update", float("nan"))),
            # info supplémentaire utile déjà calculée dans le code existant
            "masked_frac": float(p.mean().item()) if isinstance(p, torch.Tensor) else float("nan"),
            "n_masked": int(losses["n_masked"]),
            "sum_mask_weights": float(losses["sum_weights"]),
        }

        # Logging console périodique (enrichi)
        if step % diff_cfg.log_interval == 0:
            H_seq = info.sum(dim=1)                 # inchangé
            target_info = (1.0 - t) * H_seq
            mask_beta = (xt == mask_id).float()
            info_remain_beta = (info * (1.0 - mask_beta)).sum(dim=1)
            mse_beta = ((target_info - info_remain_beta) ** 2).mean().item()
            log(
              f"[Den] step {step}/{diff_cfg.steps} "
              f"loss(w)={row['loss_weighted']:.4f} loss(uw)={row['loss_unweighted']:.4f} "
              f"lr={row['lr']:.2e} grad_norm={row['grad_norm']:.3f} "
              f"mse_beta={mse_beta:.6f} masked_frac={row['masked_frac']:.3f} n_masked={row['n_masked']} | "
              f"t_step={row['time.step_total_ms']:.2f}ms "
              f"[h2d={row['time.h2d_ms']:.2f} info={row['time.info_compute_ms']:.2f} "
              f"samp={row['time.sample_xt_ms']:.2f} fwd={row['time.forward_ms']:.2f} "
              f"loss={row['time.loss_ms']:.2f} bwd={row['time.backward_ms']:.2f} "
              f"opt={row['time.optim_step_ms']:.2f} ema={row['time.ema_update_ms']:.2f}]"
            )

            # ====== NOUVEAU: verbose_batch ======
            if getattr(diff_cfg, "verbose_batch", False):
                # Fenêtre de 20 tokens, p dans [0, 256-20]; si L < 256 on borne pour ne pas dépasser
                B, L = x0.size(0), x0.size(1)
                width = 20
                base_len = 256
                L_eff = min(base_len, L)
                # borne supérieure inclusive (si L_eff == width => 0)
                max_p = max(L_eff - width, 0)
                p_pos = int(torch.randint(low=0, high=max_p + 1, size=(1,), device=x0.device).item())
                p_end = p_pos + width
                Bv = min(10, B)

                # Paramètres MLP sur le batch courant (pas de gradient)
                with torch.no_grad():
                    mlp_out = mlp_model(info)  # sorties typiques: (a, b) de shape [B, L]
                    # Supporte tuple/list, dict({'a','b'}), ou Tensor [..., 2]
                    if isinstance(mlp_out, (tuple, list)) and len(mlp_out) >= 2:
                        a_full, b_full = mlp_out[0], mlp_out[1]
                    elif isinstance(mlp_out, dict):
                        a_full = mlp_out.get("a", mlp_out.get("alpha", None))
                        b_full = mlp_out.get("b", mlp_out.get("beta", None))
                        if a_full is None or b_full is None:
                            # prend les deux premières clés si noms inattendus
                            _vals = list(mlp_out.values())
                            a_full = _vals[0]; b_full = _vals[1]
                    else:
                        # Tensor unique avec dernière dim=2
                        if mlp_out.dim() >= 3 and mlp_out.size(-1) >= 2:
                            a_full = mlp_out[..., 0]
                            b_full = mlp_out[..., 1]
                        else:
                            a_full = mlp_out
                            b_full = torch.zeros_like(mlp_out)

                # Slices [first 10, p:p+20]
                toks_slice = x0[:Bv, p_pos:p_end]      # [Bv, <=20]
                info_slice = info[:Bv, p_pos:p_end]    # [Bv, <=20]
                a_slice = a_full[:Bv, p_pos:p_end]
                b_slice = b_full[:Bv, p_pos:p_end]

                # Affichage via log(...), pas de print
                log(f"[VerboseBatch] step={step} p={p_pos} window=[{p_pos}:{p_end}] (first {Bv}/{B})")
                for i in range(Bv):
                    ids = toks_slice[i].detach().cpu().tolist()
                    a_vals = [round(float(v), 3) for v in a_slice[i].detach().cpu().flatten().tolist()]
                    b_vals = [round(float(v), 3) for v in b_slice[i].detach().cpu().flatten().tolist()]
                    info_vals = [round(float(v), 3) for v in info_slice[i].detach().cpu().flatten().tolist()]

                    log(f"  -- seq[{i}] tokens[{p_pos}:{p_end}]: {ids}")
                    log(f"     mlp.a: {a_vals}")
                    log(f"     mlp.b: {b_vals}")
                    log(f"     info : {info_vals}")
                # ====== FIN verbose_batch ======


        # Évaluation périodique (inchangée, on ajoute seulement l'écriture dans 'row')
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
                row["val_loss_weighted"] = float(val_losses["loss_weighted"])
                row["val_loss_unweighted"] = float(val_losses["loss_unweighted"])
                log(f"         [Eval] loss(w)={row['val_loss_weighted']:.4f} loss(uw)={row['val_loss_unweighted']:.4f}")
            denoiser.train()

        # Enregistrement de la ligne CSV pour CE step
        csv.log_row(row)

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


    # === FIN D'ENTRAÎNEMENT: écrire le CSV ===
    csv.flush()
    log(f"[CSV] écrit: {log_csv_path}")

