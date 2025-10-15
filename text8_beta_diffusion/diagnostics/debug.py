import torch, torch.nn.functional as F
from ..data import decode_ids

@torch.no_grad()
def debug_show_masked_variants(info_provider, batch_ids, seq_idx: int = 0, positions=(50, 60, 70),
                               n_show: int = 10, window: int = 20, seed: int = 123, topk: int = 5):
    model = info_provider.model.eval()
    device = info_provider.device
    x0 = batch_ids.to(device)
    B, L = x0.shape
    N = info_provider.N
    pi = info_provider.pi.to(device)

    def _vdc_forward_once():
        g = torch.Generator(device=device); g.manual_seed(seed)
        U = torch.rand((B, N, L), generator=g, device=device)
        M = U < pi.view(1, N, 1)
        if getattr(info_provider, 'ensure_coverage', True):
            ever = M.any(dim=1)
            b_idx, l_idx = torch.where(~ever)
            if b_idx.numel() > 0:
                n_idx = torch.randint(low=0, high=N, size=(b_idx.numel(),), generator=g, device=device)
                M[b_idx, n_idx, l_idx] = True
        x_rep = x0.unsqueeze(1).expand(B, N, L).clone()
        x_rep[M] = info_provider.mask_id
        xt = x_rep.reshape(B * N, L)
        if getattr(info_provider, 'time_from_pi', True):
            t_vec = pi.repeat(B)
        else:
            t_vec = torch.full((B * N,), float(info_provider.t_info_level), device=device)
        return M, xt, t_vec

    M, xt, t_vec = _vdc_forward_once()
    print(f"\n[SEQ {seq_idx}] L={L}")
    print("ORIGINAL:"); print(decode_ids(x0[seq_idx]))
    base = seq_idx * N
    for pos in positions:
        if not (0 <= pos < L): continue
        print("\n" + "="*90)
        print(f"[pos={pos}]")
        masked_i = torch.nonzero(M[seq_idx,:,pos]).squeeze(-1).tolist()[:n_show]
        print(f"  Masqué dans {len(masked_i)}/{N} répliques.")
        for i in masked_i:
            bn = base + i
            seq_i = xt[bn].unsqueeze(0)
            t_i   = t_vec[bn].view(1)
            logits = model(seq_i, t_i)
            lp = F.log_softmax(logits, dim=-1)[0, pos]
            p  = lp.exp()
            vals, idxs = torch.topk(p, k=topk)
            topk_tokens = [(int(ix.item()), float(v.item())) for v, ix in zip(vals, idxs)]
            s = max(0, pos - window); e = min(seq_i.numel(), pos + window)
            masked_str = decode_ids(seq_i[0, s:e])
            print(f"  (i={i:3d}) t={float(t_i.item()):.3f}  top-{topk} @pos: {topk_tokens}")
            print("  window:", masked_str)
