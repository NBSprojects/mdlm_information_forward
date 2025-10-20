from __future__ import annotations
import torch, torch.nn.functional as F



def masked_ce_losses(
    logits: torch.Tensor, x0: torch.Tensor, xt: torch.Tensor, W: torch.Tensor | None, mask_id: int
) -> dict:
    B, L, V = logits.shape
    mask_pos = (xt == mask_id)
    n_mask = int(mask_pos.sum().item())
    if n_mask == 0:
        ce_full = F.cross_entropy(logits.reshape(-1, V), x0.reshape(-1), reduction="mean")
        return dict(loss_unweighted=ce_full, loss_weighted=ce_full, n_masked=0, sum_weights=torch.tensor(0.))
    logits_m  = logits[mask_pos]
    targets_m = x0[mask_pos]
    ce = F.cross_entropy(logits_m, targets_m, reduction="none")
    loss_unweighted = ce.mean()
    if W is None:
        loss_weighted = loss_unweighted
        sum_w = torch.tensor(float(n_mask), device=logits.device)
    else:
        w = W[mask_pos]
        sum_w = w.sum().clamp_min(1.0)
        loss_weighted = (w * ce).sum() / sum_w
    return dict(loss_unweighted=loss_unweighted, loss_weighted=loss_weighted, n_masked=n_mask, sum_weights=sum_w)


def masked_ce_per_sample(
    logits: torch.Tensor, x0: torch.Tensor, xt: torch.Tensor, mask_id: int, normalize_by_masked: bool
) -> dict:
    """
    Version parallèle à celle du module md4_objective, exposée via losses.py pour usage centralisé.
    """
    B, L, V = logits.shape
    ce = F.cross_entropy(logits.reshape(-1, V), x0.reshape(-1), reduction="none").view(B, L)
    masked = (xt == mask_id)
    ce_sum = (ce * masked).sum(dim=1)              # [B]
    n_masked = masked.sum(dim=1)                   # [B]
    if normalize_by_masked:
        denom = n_masked.clamp_min(1)
        ce_batch = ce_sum / denom
    else:
        ce_batch = ce_sum
    return dict(per_sample_loss=ce_batch, n_masked=n_masked)