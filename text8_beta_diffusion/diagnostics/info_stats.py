import math, torch
import numpy as np
from typing import Tuple
from ..data import decode_ids
import ot  # POT

def kl_to_uniform(info_batch: torch.Tensor) -> float:
    """KL(info || unif) moyen sur le batch (par séquence). info est (B,L), >=0.
       On renormalise par séquence pour robustesse."""
    eps = 1e-12
    B, L = info_batch.shape
    p = info_batch / info_batch.sum(dim=1, keepdim=True).clamp_min(eps)
    log_u = -math.log(L)
    kl = (p * (p.clamp_min(eps).log() - log_u)).sum(dim=1).mean().item()
    return float(kl)

def kl_mean_dist_to_uniform(info_batch: torch.Tensor) -> float:
    """KL( mean(info) || unif ), où mean porte sur le batch (B)."""
    eps = 1e-12
    if info_batch.dim() != 2:
        raise ValueError("info_batch doit être (B,L)")
    p = info_batch / info_batch.sum(dim=1, keepdim=True).clamp_min(eps)
    mean_p = p.mean(dim=0)  # (L,)
    log_u = -math.log(p.size(1))
    return float((mean_p * (mean_p.clamp_min(eps).log() - log_u)).sum().item())

def ot_distance_to_uniform_simplex(info_batch: torch.Tensor) -> float:
    """Distance de Wasserstein-2 (OT) entre le batch info et un batch Dir(1) de même taille."""
    eps = 1e-9
    if info_batch.dim() != 2:
        raise ValueError("info_batch doit être (B,L)")
    if (info_batch < -eps).any():
        raise ValueError("info_batch doit être >= 0")
    p = info_batch / info_batch.sum(dim=1, keepdim=True).clamp_min(eps)
    B, L = p.shape
    exp_dist = torch.distributions.exponential.Exponential(rate=1.0)
    q = exp_dist.sample(sample_shape=(B, L))
    q = q / q.sum(dim=1, keepdim=True)
    P = p.detach().cpu().numpy()
    Q = q.detach().cpu().numpy()
    wa = np.full(B, 1.0 / B); wb = np.full(B, 1.0 / B)
    M = ot.dist(P, Q, metric='sqeuclidean')
    cost = ot.emd2(wa, wb, M)
    return float(np.sqrt(cost))

@torch.no_grad()
def report_info_stats(
    x0: torch.Tensor, info: torch.Tensor,
    show_sequences: int = 10, window_len: int = 20,
    compute_kl: bool = True, compute_ot: bool = False,
    entropy_in_bits: bool = True
):
    """Affiche un aperçu des séquences (token slice + vecteur info) + agrégats batch."""
    eps = 1e-12
    B, L = info.shape
    use_B = min(show_sequences, B)

    # --- Aperçus par séquence
    print(f"\n[Info stats] Batch B={B}, L={L} | Aperçu {use_B} séquences")
    for i in range(use_B):
        idx_drawn = torch.randint(0, L, ()).item()
        start = min(idx_drawn, max(0, L - window_len))
        end = min(start + window_len, L)
        tokens_slice = x0[i, start:end]
        infos_slice  = info[i, start:end]
        text_slice   = decode_ids(tokens_slice)
        p = info[i] / (info[i].sum() + eps)
        H_nats = float((-(p.clamp_min(eps) * p.clamp_min(eps).log())).sum().item())
        H_bits = H_nats / math.log(2.0)
        print(f"\n[seq {i}] window=[{start}:{end})")
        print("texte[20]:", text_slice)
        print("info[20] :", " ".join(f"{v:.5f}" for v in infos_slice.tolist()))
        if entropy_in_bits:
            print(f"Entropy H(info) = {H_bits:.4f} bits ({H_nats:.4f} nats)")
        else:
            print(f"Entropy H(info) = {H_nats:.4f} nats")

    # --- Agrégats batch
    p_norm = info / (info.sum(dim=1, keepdim=True) + eps)
    mean_across_seqs = p_norm.mean(dim=0)       # (L,)
    var_across_seqs  = p_norm.var(dim=0, unbiased=False)
    global_mean   = float(mean_across_seqs.mean().item())
    mean_var_pos  = float(var_across_seqs.mean().item())
    per_seq_var   = p_norm.var(dim=1, unbiased=False)
    mean_var_intra = float(per_seq_var.mean().item())
    print("\n=== Agrégats batch (distribution d'information par token) ===")
    print(f"Global mean p[pos] : {global_mean:.6f} (théorique ~ {1.0/L:.6f})")
    print(f"Variance moyenne par position : {mean_var_pos:.6e}")
    print(f"Variance intra-séquence moyenne : {mean_var_intra:.6e}")

    if compute_kl:
        kl = kl_to_uniform(info)
        kl_mean = kl_mean_dist_to_uniform(info)
        print(f"KL(info || uniform) (moyenne séquence)     : {kl:.6f} nats")
        print(f"KL(mean(info) || uniform) (sur le batch) : {kl_mean:.6f} nats")
    if compute_ot:
        ot_w2 = ot_distance_to_uniform_simplex(info)
        print(f"OT W2(info vs Dir(1)) : {ot_w2:.6f}")
