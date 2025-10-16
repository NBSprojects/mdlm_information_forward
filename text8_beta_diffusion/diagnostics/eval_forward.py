import numpy as np, torch
from ..models.scheduler_mlp import leggauss_torch, mse_integrated_beta, baseline_mse_integrated

@torch.no_grad()
def integrated_mse_vs_baseline(
    mlp_model,
    info_provider,
    val_loader,
    batches: int = 10,
    t_samples: int = 64
) -> tuple[float, float, float]:
    """
    Calcule la MSE intégrée sur t∈[0,1] du forward Beta (via MLP) vs baseline p(t)=t.
    Retourne: (m_model, m_base, relative_gain) avec relative_gain = (m_base - m_model) / max(m_base, eps)
    """
    device = next(mlp_model.parameters()).device
    K = int(max(2, t_samples))
    T_NODES, T_WEIGHTS = leggauss_torch(K, device=device, dtype=torch.float32)
    INNER_NODES, INNER_WEIGHTS = leggauss_torch(16, device=device, dtype=torch.float32)

    mse_model, mse_base, n = 0.0, 0.0, 0
    it = iter(val_loader)
    for _ in range(batches):
        try:
            x0, _ = next(it)
        except StopIteration:
            break
        x0 = x0.to(info_provider.device, non_blocking=True)
        I = info_provider.compute_info(x0).to(device)   # (B,L)
        a, b = mlp_model(I)
        loss_m = mse_integrated_beta(I, a, b, T_NODES, T_WEIGHTS, INNER_NODES, INNER_WEIGHTS).item()
        loss_b = baseline_mse_integrated(I, T_NODES, T_WEIGHTS).item()
        mse_model += loss_m; mse_base += loss_b; n += 1

    if n == 0:
        return 0.0, 0.0, 0.0
    mse_model /= n; mse_base /= n
    rel = (mse_base - mse_model) / max(mse_base, 1e-8)
    return float(mse_model), float(mse_base), float(rel)
