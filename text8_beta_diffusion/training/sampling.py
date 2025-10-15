import math, torch, torch.nn.functional as F
from typing import Optional
from ..data import mask_id, vocab_size, stoi, decode_ids
from .masking import _ibeta_reg

def alpha_linear(t, eps: float = 1e-4):
    return (1.0 - 2*eps) * (1.0 - t) + eps

def alpha_cosine(t, eps: float = 1e-4):
    return (1.0 - 2*eps) * (1.0 - torch.cos(math.pi/2 * (1.0 - t))) + eps

def dgamma_times_alpha(t, schedule: str = "linear", eps: float = 1e-4):
    if schedule == "linear":
        a = alpha_linear(t, eps)
    elif schedule == "cosine":
        a = alpha_cosine(t, eps)
    else:
        raise ValueError("schedule must be 'linear' or 'cosine'")
    dummy = torch.zeros_like(t)
    return dummy, a

def _map_time_grid(u: float, grid: str = "cosine"):
    if grid == "cosine":
        return math.cos(math.pi/2 * (1.0 - u))
    elif grid == "linear":
        return u
    else:
        raise ValueError("grid must be 'cosine' or 'linear'")

def _encode_prefix(prefix: str):
    ids = [stoi.get(ch, stoi[" "]) for ch in prefix if ch in stoi]
    return torch.tensor(ids, dtype=torch.long)

def _nucleus_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return probs
    sorted_p, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cum = torch.cumsum(sorted_p, dim=-1)
    keep = cum <= top_p
    keep[..., 0] = True
    filtered = torch.zeros_like(probs).scatter_(-1, sorted_idx, torch.where(keep, sorted_p, torch.zeros_like(sorted_p)))
    z = filtered.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return filtered / z

@torch.no_grad()
def sample_md4(
    model, n_tokens: int = 256, steps: int = 1000, grid: str = "cosine",
    schedule: str = "linear", eps: float = 1e-4, temperature: float = 1.0,
    snapshots_every: Optional[int] = 100, n_samples: int = 1, prefix: Optional[str] = None,
    top_p: float = 1.0
):
    device = next(model.parameters()).device
    B = int(n_samples)
    zt = torch.full((B, n_tokens), mask_id, dtype=torch.long, device=device)
    fixed = torch.zeros_like(zt, dtype=torch.bool)
    if prefix:
        pref_ids = _encode_prefix(prefix).to(device)
        Lp = min(pref_ids.numel(), n_tokens)
        if Lp > 0:
            zt[:, :Lp] = pref_ids[:Lp]
            fixed[:, :Lp] = True
    snapshots = []
    for i in range(steps, 0, -1):
        t = i / steps; s = (i - 1) / steps
        t_mapped = _map_time_grid(t, grid=grid)
        s_mapped = _map_time_grid(s, grid=grid)
        t_tensor = torch.full((B,), t_mapped, device=device)
        s_tensor = torch.full((B,), s_mapped, device=device)
        _, a_t = dgamma_times_alpha(t_tensor, schedule=schedule, eps=eps)
        _, a_s = dgamma_times_alpha(s_tensor, schedule=schedule, eps=eps)
        logits = model(zt, t_tensor)
        if temperature != 1.0:
            logits = logits / float(temperature)
        pvocab = F.softmax(logits, dim=-1)
        pvocab = _nucleus_filter(pvocab, top_p=top_p)
        u = (a_s - a_t) / (1.0 - a_t + 1e-12)
        u = u.view(B, 1).expand(B, n_tokens)
        probs_mask = (1.0 - u).unsqueeze(-1)
        probs_vocab_scaled = u.unsqueeze(-1) * pvocab
        probs_ext = torch.cat([probs_vocab_scaled, probs_mask], dim=-1)
        ext = torch.distributions.Categorical(probs=probs_ext.reshape(-1, vocab_size + 1)).sample().view(B, n_tokens)
        is_mask = (zt == mask_id) & (~fixed)
        zt = torch.where(is_mask, ext, zt)
        zt = torch.where(zt > (vocab_size - 1), torch.full_like(zt, mask_id), zt)
        if (snapshots_every is not None) and (i % snapshots_every == 0 or i == 1):
            snapshots.append(zt.detach().clone().cpu())
    return zt.detach().cpu(), snapshots

@torch.no_grad()
def sample_md4_beta_gw(
    denoiser, mlp_model, n_tokens: int = 256, steps: int = 1000, grid: str = "cosine",
    temperature: float = 1.0, snapshots_every: Optional[int] = 100, n_samples: int = 1,
    prefix: Optional[str] = None, dirichlet_alpha: float = 1.0, top_p: float = 1.0
):
    device = next(denoiser.parameters()).device
    B = int(n_samples); L = int(n_tokens)
    zt = torch.full((B, L), mask_id, dtype=torch.long, device=device)
    fixed = torch.zeros_like(zt, dtype=torch.bool)
    if prefix:
        pref_ids = _encode_prefix(prefix).to(device)
        Lp = min(pref_ids.numel(), L)
        if Lp > 0:
            zt[:, :Lp] = pref_ids[:Lp]
            fixed[:, :Lp] = True
    conc = torch.full((L,), float(dirichlet_alpha), device=device, dtype=torch.float32)
    w = torch.distributions.Dirichlet(conc).sample((B,))
    mlp_model.eval()
    a, b = mlp_model(w)
    snapshots = []
    for i in range(steps, 0, -1):
        t = i / steps; s = (i - 1) / steps
        t_mapped = torch.tensor(_map_time_grid(t, grid=grid), device=device, dtype=torch.float32)
        s_mapped = torch.tensor(_map_time_grid(s, grid=grid), device=device, dtype=torch.float32)
        It = _ibeta_reg(a, b, t_mapped.view(1, 1).expand(B, L)).to(torch.float32)
        Is = _ibeta_reg(a, b, s_mapped.view(1, 1).expand(B, L)).to(torch.float32)
        u = ((It - Is) / It.clamp_min(1e-6)).clamp(0.0, 1.0)
        logits = denoiser(zt, t_mapped.expand(B))
        if temperature != 1.0:
            logits = logits / float(temperature)
        pvocab = F.softmax(logits, dim=-1)
        pvocab = _nucleus_filter(pvocab, top_p=top_p)
        probs_mask  = (1.0 - u).unsqueeze(-1)
        probs_vocab = u.unsqueeze(-1) * pvocab
        probs_ext   = torch.cat([probs_vocab, probs_mask], dim=-1)
        is_mask = (zt == mask_id) & (~fixed)
        if is_mask.any():
            ext = torch.distributions.Categorical(probs=probs_ext[is_mask]).sample()
            new_vals = torch.where(ext > (vocab_size - 1), torch.full_like(ext, mask_id), ext)
            zt[is_mask] = new_vals
        if snapshots_every and (i % snapshots_every == 0 or i == 1):
            snapshots.append(zt.detach().clone().cpu())
    return zt.detach().cpu(), snapshots
