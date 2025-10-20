# text8_beta_diffusion/training/md4_objective.py
from __future__ import annotations
import math
import torch
import torch.nn.functional as F

def _parse_poly(schedule: str) -> float | None:
    if schedule.startswith("poly"):
        try:
            return float(schedule.replace("poly", ""))
        except Exception:
            return None
    return None

# --------- alpha(t), d alpha(t), dgamma*alpha(t) (CODE 1) ---------

def _alpha_base(t: torch.Tensor, schedule: str) -> torch.Tensor:
    """Alpha sans eps (0..1), broadcast sur t."""
    poly = _parse_poly(schedule)
    if schedule == "linear":
        return 1.0 - t
    elif schedule == "cosine":
        return 1.0 - torch.cos(math.pi/2.0 * (1.0 - t))
    elif poly is not None:
        return 1.0 - (t ** poly)
    else:
        raise ValueError(f"Unknown schedule '{schedule}'")

def alpha(t: torch.Tensor, schedule: str, eps: float) -> torch.Tensor:
    """Alpha avec eps, comme CODE 1: (1-2eps)*alpha_base + eps"""
    return (1.0 - 2*eps) * _alpha_base(t, schedule) + eps

def _dalpha_base(t: torch.Tensor, schedule: str) -> torch.Tensor:
    poly = _parse_poly(schedule)
    if schedule == "linear":
        return -torch.ones_like(t)
    elif schedule == "cosine":
        return -(math.pi/2.0) * torch.sin(math.pi/2.0 * (1.0 - t))
    elif poly is not None:
        return -poly * (t ** (poly - 1.0))
    else:
        raise ValueError(f"Unknown schedule '{schedule}'")

def dalpha(t: torch.Tensor, schedule: str, eps: float) -> torch.Tensor:
    """(1-2eps) * d alpha_base/dt"""
    return (1.0 - 2*eps) * _dalpha_base(t, schedule)

def dgamma_times_alpha(t: torch.Tensor, schedule: str, eps: float) -> torch.Tensor:
    """d gamma/dt * alpha(t) où gamma(t)=log(alpha/(1-alpha)).
       Dans CODE 1 c’est: dalpha / (1 - alpha)."""
    a = alpha(t, schedule, eps)
    da = dalpha(t, schedule, eps)
    return da / (1.0 - a + 1e-12)

def logsnr(t: torch.Tensor, schedule: str, eps: float) -> torch.Tensor:
    """gamma(t)=log(alpha/(1-alpha))."""
    a = alpha(t, schedule, eps)
    return torch.log(a / (1.0 - a + 1e-12))

# --------- bruitage avant (forward_sample) ---------

def forward_sample(x0: torch.Tensor, t: torch.Tensor, schedule: str, eps: float, mask_id: int) -> torch.Tensor:
    """
    CODE 1 : pour chaque position, on "garde" x0 avec proba a(t) et on met <mask> sinon (i.i.d.).
    x0: [B, L] (long)
    t : [B] ou scalaire dans [0,1]
    """
    B, L = x0.shape
    if t.ndim == 0:
        t = t.expand(B)
    a = alpha(t, schedule, eps)             # [B]
    p_keep = a.view(B, 1).expand(B, L)      # broadcast
    keep = torch.bernoulli(p_keep).to(dtype=torch.bool, device=x0.device)
    xt = torch.where(keep, x0, torch.full_like(x0, mask_id))
    return xt

# --------- temps antithétique (optionnel) ---------

def sample_times(B: int, device: torch.device, antithetic: bool) -> torch.Tensor:
    if not antithetic:
        return torch.rand(B, device=device)
    # CODE 1 : t0 ~ U(0,1), puis vecteur "antithétique"
    t0 = torch.rand((), device=device)
    arange = torch.arange(B, device=device, dtype=torch.float32) / float(B)
    return torch.fmod(t0 + arange, 1.0)

# --------- CE (par échantillon) sur positions masquées ---------

def masked_ce_per_sample(logits: torch.Tensor, x0: torch.Tensor, xt: torch.Tensor,
                         mask_id: int, normalize_by_masked: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Renvoie:
      ce_batch: [B] somme (ou moyenne) des CE sur positions masquées, par échantillon
      n_masked: [B] nb de positions masquées par échantillon
    """
    B, L, V = logits.shape
    ce = F.cross_entropy(
        logits.reshape(-1, V), x0.reshape(-1),
        reduction="none"
    ).view(B, L)
    masked = (xt == mask_id)
    ce_sum = (ce * masked).sum(dim=1)          # [B]
    n_masked = masked.sum(dim=1)               # [B]
    if normalize_by_masked:
        denom = n_masked.clamp_min(1)
        ce_batch = ce_sum / denom
    else:
        ce_batch = ce_sum
    return ce_batch, n_masked
