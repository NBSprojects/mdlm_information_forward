import math, torch
import matplotlib.pyplot as plt
from ..training.masking import beta_pdf_cdf

@torch.no_grad()
def test_infoprovider_entropy_curve(info_provider, data_loader, num_points: int = 11, use_bits: bool = True, clamp_t_eps: float = 1e-6):
    try:
        x0, _ = next(iter(data_loader))
    except StopIteration:
        data_iter = iter(data_loader)
        x0, _ = next(data_iter)
    x0 = x0.to(info_provider.device, non_blocking=True)
    B, L = x0.shape
    old_time_from_pi = info_provider.time_from_pi
    old_t_info_level = info_provider.t_info_level
    info_provider.time_from_pi = False
    ts = torch.linspace(0.0, 1.0, steps=num_points)
    print(f"[InfoProvider curve] Batch B={B}, L={L} | points={num_points}\n{'t':>6}  {'mean_H(nats)':>18}  {'mean_H(bits)':>18}")
    for t in ts:
        t_val = float(max(clamp_t_eps, min(1.0 - clamp_t_eps, float(t.item()))))
        info_provider.t_info_level = t_val
        info = info_provider.compute_info(x0)
        eps = 1e-12
        p = info / info.sum(dim=1, keepdim=True).clamp_min(eps)
        H_nats = (-(p.clamp_min(eps) * p.clamp_min(eps).log()).sum(dim=1))
        H_mean_nats = H_nats.mean().item()
        if use_bits:
            H_mean_bits = (H_nats / math.log(2.0)).mean().item()
            print(f"{t.item():6.2f}  {H_mean_nats:18.12f}  {H_mean_bits:18.12f}")
        else:
            print(f"{t.item():6.2f}  {H_mean_nats:18.12f}")
    info_provider.time_from_pi = old_time_from_pi
    info_provider.t_info_level = old_t_info_level

@torch.no_grad()
def mse_curve_vs_baseline(mlp_model, info_provider, val_loader, batches: int = 5, T: int = 25, show=True):
    grid = torch.linspace(0, 1, T, device=next(mlp_model.parameters()).device)
    model_vals = torch.zeros(T, device=grid.device)
    base_vals  = torch.zeros(T, device=grid.device)
    it = iter(val_loader)
    for _ in range(batches):
        try:
            x0, _ = next(it)
        except StopIteration:
            break
        x0 = x0.to(grid.device)
        I = info_provider.compute_info(x0).to(grid.device)
        H = I.sum(dim=1, keepdim=True).clamp_min(1e-6)
        a, b = mlp_model(I)
        B, L = I.shape
        for j, t in enumerate(grid):
            Tmat = t.repeat(B, L)
            pdf, cdf = beta_pdf_cdf(Tmat, a, b)
            p = cdf.clamp(1e-7, 1-1e-7)
            M = torch.bernoulli(p)
            h = (I * (1 - M)).sum(dim=1, keepdim=True)
            info_frac = (h / H).squeeze(-1)
            target = (1 - t).expand_as(info_frac)
            model_vals[j] += (info_frac - target).pow(2).mean()
            M_b = torch.bernoulli(Tmat)
            h_b = (I * (1 - M_b)).sum(dim=1, keepdim=True)
            info_frac_b = (h_b / H).squeeze(-1)
            base_vals[j] += (info_frac_b - target).pow(2).mean()
    model_vals /= max(batches,1); base_vals  /= max(batches,1)
    if show:
        plt.figure()
        plt.plot(grid.detach().cpu().numpy(), model_vals.detach().cpu().numpy(), label="MLP Beta masking")
        plt.plot(grid.detach().cpu().numpy(), base_vals.detach().cpu().numpy(), label="Baseline p=t")
        plt.title("MSE_t curve: (info_fraction - (1 - t))^2")
        plt.xlabel("t"); plt.ylabel("MSE"); plt.legend(); plt.show()
    return grid.detach().cpu().numpy(), model_vals.detach().cpu().numpy(), base_vals.detach().cpu().numpy()
