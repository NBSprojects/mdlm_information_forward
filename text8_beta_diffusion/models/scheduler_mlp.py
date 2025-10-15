import numpy as np, torch, torch.nn as nn
from typing import Tuple, Optional

class SchedulerMLP(nn.Module):
    def __init__(self, seq_len: int, hidden: int = 512, depth: int = 3, dropout: float = 0.1,
                 alpha_min: float = 0.1, alpha_max: float = 20.0, beta_min: float = 0.1, beta_max: float = 20.0):
        super().__init__()
        self.seq_len = seq_len
        self.alpha_min = alpha_min; self.alpha_max = alpha_max
        self.beta_min  = beta_min;  self.beta_max  = beta_max
        d_in = seq_len; d = hidden
        layers = [nn.Linear(d_in, d), nn.BatchNorm1d(d), nn.ReLU(inplace=True)]
        for _ in range(depth - 1):
            layers += [nn.Linear(d, d), nn.BatchNorm1d(d), nn.ReLU(inplace=True)]
        layers += [nn.Linear(d, 2 * seq_len)]
        self.net = nn.Sequential(*layers)
        self.softplus = nn.Softplus()
        self.eps = 1e-6
    def forward(self, info_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(info_seq)
        B, L = info_seq.shape
        out = out.view(B, 2, L)
        a_raw, b_raw = out[:, 0, :], out[:, 1, :]
        a = self.softplus(a_raw) + self.eps
        b = self.softplus(b_raw) + self.eps
        a = a.clamp(min=self.alpha_min, max=self.alpha_max)
        b = b.clamp(min=self.beta_min,  max=self.beta_max)
        return a, b

def leggauss_torch(K: int, device: torch.device, dtype: torch.dtype):
    xs_np, ws_np = np.polynomial.legendre.leggauss(K)
    xs = torch.from_numpy((xs_np + 1.0) / 2.0).to(device=device, dtype=dtype)
    ws = torch.from_numpy(ws_np / 2.0).to(device=device, dtype=dtype)
    ws = ws / ws.sum()
    return xs, ws

def uniform_t_grid(K: int, device: torch.device, dtype: torch.dtype):
    if K <= 1:
        t = torch.tensor([0.5], device=device, dtype=dtype)
    else:
        t = torch.linspace(0.05, 0.95, K, device=device, dtype=dtype)
    w = torch.full((K,), 1.0 / max(K, 1), device=device, dtype=dtype)
    return t, w

def betainc_quadrature_torch(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor,
                             quad_nodes: torch.Tensor, quad_weights: torch.Tensor) -> torch.Tensor:
    while t.dim() < a.dim():
        t = t.unsqueeze(-1)
    t = t.expand_as(a).clamp(1e-7, 1 - 1e-7)
    x = quad_nodes.view(*([1] * a.dim()), -1)
    w = quad_weights.view(*([1] * a.dim()), -1)
    u = t.unsqueeze(-1) * x
    log_pdf = (a.unsqueeze(-1) - 1.0) * torch.log(u.clamp(1e-9)) \
            + (b.unsqueeze(-1) - 1.0) * torch.log((1.0 - u).clamp(1e-9))
    log_B   = torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
    pdf     = torch.exp(log_pdf - log_B.unsqueeze(-1))
    cdf = (pdf * w).sum(dim=-1) * t
    return cdf.clamp(0.0, 1.0)

def mse_integrated_beta(I: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                        t_nodes: torch.Tensor, t_weights: torch.Tensor,
                        inner_nodes: torch.Tensor, inner_weights: torch.Tensor) -> torch.Tensor:
    B, L = I.shape
    K = t_nodes.numel()
    t = t_nodes.view(1, K, 1).expand(B, K, L)
    a3 = a.unsqueeze(1).expand_as(t)
    b3 = b.unsqueeze(1).expand_as(t)
    I3 = I.unsqueeze(1).expand_as(t)
    p = betainc_quadrature_torch(a3, b3, t, inner_nodes, inner_weights)
    Eh   = (I3 * (1.0 - p)).sum(dim=2, keepdim=True)
    Varh = (I3.pow(2) * p * (1.0 - p)).sum(dim=2, keepdim=True)
    H = I.sum(dim=1, keepdim=True).unsqueeze(1)
    target = (1.0 - t_nodes.view(1, K, 1)) * H
    mse_k = Varh + (Eh - target).pow(2)
    loss_per_sample = (mse_k.squeeze(-1) * t_weights.view(1, K)).sum(dim=1)
    return loss_per_sample.mean()

def baseline_mse_integrated(I: torch.Tensor, t_nodes: torch.Tensor, t_weights: torch.Tensor, alpha_nodes: Optional[torch.Tensor] = None) -> torch.Tensor:
    device = I.device; dtype = I.dtype
    B, L = I.shape; K = int(t_nodes.numel())
    H  = I.sum(dim=1, keepdim=True)
    H2 = (I * I).sum(dim=1, keepdim=True)
    t = t_nodes.view(1, K, 1).to(device=device, dtype=dtype)
    w = t_weights.view(1, K).to(device=device, dtype=dtype)
    if alpha_nodes is None:
        alpha_nodes = (1.0 - t_nodes)
    alpha_nodes = alpha_nodes.view(1, K, 1).to(device=device, dtype=dtype)
    var_k = H2.unsqueeze(1) * (t * (1.0 - t))
    Eh_k     = H.unsqueeze(1) * (1.0 - t)
    target_k = H.unsqueeze(1) * alpha_nodes
    bias2_k  = (Eh_k - target_k).pow(2)
    mse_per_sample = ((var_k + bias2_k).squeeze(-1) * w).sum(dim=1)
    return mse_per_sample.mean()
