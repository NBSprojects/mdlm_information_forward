import torch
from typing import Tuple
from ..data import mask_id

def _betacf(a, b, x, max_iter=200, tol=1e-12):
    tiny = torch.finfo(a.dtype).tiny
    one  = torch.ones_like(x)
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = one.clone()
    d = one - qab * x / qap
    d = torch.where(d.abs() < tiny, torch.full_like(d, tiny), d)
    d = 1.0 / d
    h = d.clone()
    for i in range(1, max_iter + 1):
        ii  = torch.tensor(float(i), dtype=a.dtype, device=a.device)
        ii2 = 2.0 * ii
        aa = (ii * (b - ii) * x) / ((qam + ii2) * (a + ii2))
        d = one + aa * d
        d = torch.where(d.abs() < tiny, torch.full_like(d, tiny), d)
        c = one + aa / c
        c = torch.where(c.abs() < tiny, torch.full_like(c, tiny), c)
        d = 1.0 / d
        h = h * d * c
        aa = -((a + ii) * (qab + ii) * x) / ((a + ii2) * (qap + ii2))
        d = one + aa * d
        d = torch.where(d.abs() < tiny, torch.full_like(d, tiny), d)
        c = one + aa / c
        c = torch.where(c.abs() < tiny, torch.full_like(c, tiny), c)
        d = 1.0 / d
        delta = d * c
        h = h * delta
    return h

def _ibeta_reg(a, b, x, tol=1e-12, max_iter=200):
    a = a.to(torch.float64); b = b.to(torch.float64); x = x.to(torch.float64)
    eps = torch.finfo(x.dtype).eps
    x = x.clamp(eps, 1.0 - eps)
    flip = x > (a + 1.0) / (a + b + 2.0)
    xx = torch.where(flip, 1.0 - x, x)
    aa = torch.where(flip, b, a)
    bb = torch.where(flip, a, b)
    lnB   = torch.lgamma(aa) + torch.lgamma(bb) - torch.lgamma(aa + bb)
    front = torch.exp(aa * torch.log(xx) + bb * torch.log1p(-xx) - lnB) / aa
    cf = _betacf(aa, bb, xx, max_iter=max_iter, tol=tol)
    I  = front * cf
    I  = torch.where(flip, 1.0 - I, I)
    return I

def beta_pdf_cdf(t: torch.Tensor, a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6):
    t64 = t.clamp(eps, 1.0 - eps).to(torch.float64)
    a64 = a.to(torch.float64); b64 = b.to(torch.float64)
    logB    = torch.lgamma(a64) + torch.lgamma(b64) - torch.lgamma(a64 + b64)
    log_pdf = (a64 - 1.0) * torch.log(t64) + (b64 - 1.0) * torch.log1p(-t64) - logB
    pdf     = torch.exp(log_pdf).to(t.dtype)
    cdf64 = _ibeta_reg(a64, b64, t64)
    cdf   = cdf64.to(t.dtype)
    cdf = cdf.clamp(0.0, 1.0)
    return pdf, cdf

def sample_xt_and_weights(x0: torch.Tensor, info: torch.Tensor, mlp_model, eps: float = 1e-6):
    B, L = x0.shape
    mlp_model.eval()
    with torch.no_grad():
        a, b = mlp_model(info)
    t = torch.rand(B, device=x0.device)
    T = t.unsqueeze(-1).expand(-1, L)
    pdf, cdf = beta_pdf_cdf(T, a, b)
    p = cdf.clamp(1e-8, 1-1e-8)
    M = torch.bernoulli(p)
    xt = torch.where(M>0.5, torch.full_like(x0, mask_id), x0)
    W = (pdf / p.clamp_min(1e-8)).detach()
    return xt, t, W, p
