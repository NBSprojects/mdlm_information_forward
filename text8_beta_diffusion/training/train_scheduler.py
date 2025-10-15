import time, torch
from ..models.scheduler_mlp import SchedulerMLP, leggauss_torch, mse_integrated_beta, baseline_mse_integrated

def train_scheduler_mlp(model: SchedulerMLP, steps: int, lr: float, wd: float,
                        batch_size: int, seq_len: int, t_samples_per_step: int = 32,
                        log_every: int = 50, device=None):
    model.train()
    device = device or next(model.parameters()).device
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    T_NODES, T_WEIGHTS = leggauss_torch(max(1, t_samples_per_step), device=device, dtype=torch.float32)
    INNER_NODES, INNER_WEIGHTS = leggauss_torch(16, device=device, dtype=torch.float32)
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        with torch.no_grad():
            conc = torch.ones(seq_len, device=device, dtype=torch.float32)
            I = torch.distributions.Dirichlet(conc).sample((batch_size,))
        a, b = model(I)
        loss = mse_integrated_beta(I, a, b, T_NODES, T_WEIGHTS, INNER_NODES, INNER_WEIGHTS)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % log_every == 0:
            base = baseline_mse_integrated(I, T_NODES, T_WEIGHTS).detach()
            dt = time.perf_counter() - t0
            print(f"[MLP] step {step}/{steps}  loss={loss.item():.7f}  base={base.item():.7f}  ({dt:.1f}s)")
            t0 = time.perf_counter()
