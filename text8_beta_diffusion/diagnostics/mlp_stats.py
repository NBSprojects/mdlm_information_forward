import torch
from typing import Tuple

@torch.no_grad()
def report_mlp_param_stats(
    mlp_model,
    info_provider,
    loader,
    batches: int = 1,
    threshold: float = 19.9,
    positions_per_seq: int = 10,
    max_seqs_to_show: int = 10
):
    """Statistiques globales sur (a,b) et échantillons par séquence (proche du notebook)."""
    device = next(mlp_model.parameters()).device
    shown = 0
    for _ in range(batches):
        try:
            x0, _ = next(iter(loader))
        except StopIteration:
            break
        x0 = x0.to(info_provider.device, non_blocking=True)
        I = info_provider.compute_info(x0)  # (B,L)
        B, L = I.shape
        prev_training = mlp_model.training
        mlp_model.eval()
        a, b = mlp_model(I.to(device))
        if prev_training:
            mlp_model.train()

        # --- Stats globales
        a_min, a_max, a_mean = float(a.min().item()), float(a.max().item()), float(a.mean().item())
        b_min, b_max, b_mean = float(b.min().item()), float(b.max().item()), float(b.mean().item())
        ab_min = torch.minimum(a, b)
        ab_max = torch.maximum(a, b)
        mean_ab_min = float(ab_min.mean().item())
        mean_ab_max = float(ab_max.mean().item())
        var_ab_min  = float(ab_min.float().var(unbiased=False).item())
        var_ab_max  = float(ab_max.float().var(unbiased=False).item())
        count_a_high = int((a > threshold).sum().item())
        count_b_high = int((b > threshold).sum().item())
        count_anb_high = int(((a > threshold) & (b > threshold)).sum().item())
        total = int(B * L)

        print("\n=== Stats globales (sur le batch) ===")
        print(f"a: min={a_min:.6f} | max={a_max:.6f} | mean={a_mean:.6f}")
        print(f"b: min={b_min:.6f} | max={b_max:.6f} | mean={b_mean:.6f}")
        print(f"min(a,b): mean={mean_ab_min:.6f} | var={var_ab_min:.6f}")
        print(f"max(a,b): mean={mean_ab_max:.6f} | var={var_ab_max:.6f}")
        print(f"a>{threshold}: {count_a_high}/{total}  | b>{threshold}: {count_b_high}/{total}  | both>{threshold}: {count_anb_high}/{total}")

        # --- Détails par séquence
        use_B = min(max_seqs_to_show, B)
        print(f"\n=== Détails par séquence (montrées: {use_B} / {B}) ===")
        for i in range(use_B):
            ai, bi = a[i], b[i]
            ai_min, ai_max, ai_mean = float(ai.min().item()), float(ai.max().item()), float(ai.mean().item())
            bi_min, bi_max, bi_mean = float(bi.min().item()), float(bi.max().item()), float(bi.mean().item())
            ab_min_i = torch.minimum(ai, bi).mean().item()
            ab_max_i = torch.maximum(ai, bi).mean().item()
            count_a_high_i = int((ai > threshold).sum().item())
            count_b_high_i = int((bi > threshold).sum().item())
            count_anb_high_i = int(((ai > threshold) & (bi > threshold)).sum().item())
            print(f"\n[seq {i}] a(min={ai_min:.6f}, max={ai_max:.6f}, mean={ai_mean:.6f}) | b(min={bi_min:.6f}, max={bi_max:.6f}, mean={bi_mean:.6f})")
            print(f"      min(a,b): mean={ab_min_i:.6f} | max(a,b): mean={ab_max_i:.6f}")
            print(f"      a>{threshold}: {count_a_high_i}/{L} | b>{threshold}: {count_b_high_i}/{L} | both>{threshold}: {count_anb_high_i}/{L}")
            # échantillon de positions
            k = min(positions_per_seq, L)
            idxs = torch.randperm(L)[:k].tolist()
            rows = []
            for j in idxs:
                rows.append(f"(pos {j:3d}) a={float(ai[j].item()):.6f}  b={float(bi[j].item()):.6f}")
            print("  éléments aléatoires:")
            print("  " + " | ".join(rows))

        shown += 1
        if shown >= batches:
            break
