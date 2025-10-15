import math, torch, torch.nn.functional as F
from typing import Optional
from .base import BaseInfoProvider
from ..models.transformer_compat import build_denoiser_for_info_from_checkpoint
from ..utils.precision import PrecisionPolicy, autocast_ctx, maybe_upcast_for_stability

class DenoiserInfoProvider(BaseInfoProvider):
    def __init__(
        self, denoiser: Optional[torch.nn.Module] = None, weights_path: Optional[str] = None,
        vocab_size: Optional[int] = None, d_model: Optional[int] = None, n_heads: Optional[int] = None,
        n_layers: Optional[int] = None, ff_mult: float = 4.0, dropout: float = 0.0,
        mask_id: int = 0, t_info_level: float = 0.5, mix_weight: float = 0.0, use_exp_perplexity: bool = False,
        device: torch.device = torch.device("cpu"), max_forward_batch: Optional[int] = None, verbose_load: bool = True,
        K: int = 6, ensure_coverage: bool = True, normalize_sum_to_one: bool = True,
        agg_mode: str = "hajek", ht_exclude_forced: bool = False, time_from_pi: bool = True,
        precision: PrecisionPolicy = PrecisionPolicy(), max_seq_len: int = 256
    ):
        super().__init__()
        self.mask_id = int(mask_id)
        self.t_info_level = float(t_info_level)
        self.device = device
        self.max_forward_batch = max_forward_batch
        self.normalize_sum_to_one = bool(normalize_sum_to_one)
        self.mix_weight = float(mix_weight)
        self.use_exp_perplexity = bool(use_exp_perplexity)
        self.agg_mode = str(agg_mode).lower()
        assert self.agg_mode in ("naive", "hajek", "ht"), "agg_mode must be 'naive', 'hajek' or 'ht'"
        self.ht_exclude_forced = bool(ht_exclude_forced)
        self.time_from_pi = bool(time_from_pi)
        self.precision = precision

        if denoiser is not None:
            self.model = denoiser.to(self.device).eval()
        else:
            assert weights_path is not None, "Provide 'weights_path' or a prebuilt 'denoiser'."
            assert all(x is not None for x in [vocab_size, d_model, n_heads, n_layers]), \
                "To rebuild the model, provide vocab_size, d_model, n_heads, n_layers."
            self.model, missing, unexpected = build_denoiser_for_info_from_checkpoint(
                weights_path=weights_path, device=self.device, vocab_size=int(vocab_size),
                mask_id=int(mask_id), d_model=int(d_model), n_heads=int(n_heads), n_layers=int(n_layers),
                ff_mult=float(ff_mult), dropout=float(dropout), max_seq_len=int(max_seq_len), trust_checkpoint=True,
            )
            if verbose_load:
                if missing:    print("[InfoProvider] Missing keys:", missing)
                if unexpected: print("[InfoProvider] Unexpected keys:", unexpected)
            self.model.eval()

        if K < 2:
            raise ValueError("K must be ≥ 2 so that N = 2^K - 2 ≥ 2.")
        self.K = int(K)
        self.N = (1 << self.K) - 2
        self.ensure_coverage = bool(ensure_coverage)
        self.pi = self._van_der_corput_seq(self.N, base=2, device=self.device)

    @staticmethod
    def _van_der_corput_seq(N: int, base: int = 2, device=None, dtype=torch.float32) -> torch.Tensor:
        def vdc_single(n: int, b: int) -> float:
            v = 0.0; denom = 1.0
            while n > 0:
                n, r = divmod(n, b)
                denom *= b; v += r / denom
            return v
        vals = [vdc_single(i, base) for i in range(1, N + 1)]
        t = torch.tensor(vals, dtype=dtype, device=device)
        eps = 1e-6
        return t.clamp(eps, 1.0 - eps)

    @torch.no_grad()
    def compute_info(self, batch_ids: torch.Tensor) -> torch.Tensor:
        x0 = batch_ids.to(self.device, non_blocking=True)
        B, L = x0.shape
        N = self.N
        pi = self.pi
        x_rep = x0.unsqueeze(1).expand(B, N, L).clone()
        U = torch.rand((B, N, L), device=self.device, dtype=torch.float32)
        M = U < pi.view(1, N, 1)
        forced_mask = torch.zeros_like(M)
        if self.ensure_coverage:
            ever = M.any(dim=1)
            b_idx, l_idx = torch.where(~ever)
            if b_idx.numel() > 0:
                n_idx = torch.randint(low=0, high=N, size=(b_idx.numel(),), device=self.device)
                M[b_idx, n_idx, l_idx] = True
                forced_mask[b_idx, n_idx, l_idx] = True
        x_rep[M] = self.mask_id
        BN = B * N
        xt = x_rep.view(BN, L)
        if self.time_from_pi:
            t_vec = pi.repeat(B).to(self.device, dtype=torch.float32)
        else:
            t_vec = torch.full((BN,), float(self.t_info_level), device=self.device, dtype=torch.float32)
        b_idx_full = (torch.arange(BN, device=self.device) // N)
        w_flat = (1.0 / pi).repeat(B).view(BN, 1)
        eps = 1e-8
        if self.agg_mode == "naive":
            sum_H = torch.zeros((B, L), device=self.device, dtype=torch.float32)
            cnt   = torch.zeros((B, L), device=self.device, dtype=torch.float32)
        elif self.agg_mode == "hajek":
            sum_wH = torch.zeros((B, L), device=self.device, dtype=torch.float32)
            sum_w  = torch.zeros((B, L), device=self.device, dtype=torch.float32)
        elif self.agg_mode == "ht":
            tot_HT = torch.zeros((B, L), device=self.device, dtype=torch.float32)
        mb = self.max_forward_batch or BN
        M_flat = M.view(BN, L)
        forced_mask_flat = forced_mask.view(BN, L)
        for start in range(0, BN, mb):
            end = min(start + mb, BN)
            with autocast_ctx(self.precision):
                logits = self.model(xt[start:end], t_vec[start:end])
            log_probs = torch.nn.functional.log_softmax(
                maybe_upcast_for_stability(logits, self.precision), dim=-1
            )
            probs = log_probs.exp()
            H = -(probs * log_probs).sum(dim=-1)
            targets = x0.index_select(0, b_idx_full[start:end])
            logp_true = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            S = torch.exp(-logp_true) if self.use_exp_perplexity else -logp_true
            lam = self.mix_weight
            measure = (1.0 - lam) * H + lam * S
            mask_chunk = M_flat[start:end]
            if self.ht_exclude_forced:
                mask_chunk = mask_chunk & (~forced_mask_flat[start:end])
            mask_f = mask_chunk.to(measure.dtype)
            if self.agg_mode == "naive":
                sum_H.index_add_(0, b_idx_full[start:end], measure * mask_f)
                cnt.index_add_(0,  b_idx_full[start:end], mask_f)
            elif self.agg_mode == "hajek":
                w_chunk = w_flat[start:end]
                sum_wH.index_add_(0, b_idx_full[start:end], measure * mask_f * w_chunk)
                sum_w.index_add_(0,  b_idx_full[start:end], mask_f * w_chunk)
            elif self.agg_mode == "ht":
                w_chunk = w_flat[start:end]
                tot_HT.index_add_(0, b_idx_full[start:end], measure * mask_f * w_chunk)
        if self.agg_mode == "naive":
            out = sum_H / cnt.clamp_min(eps)
        elif self.agg_mode == "hajek":
            out = sum_wH / sum_w.clamp_min(eps)
        elif self.agg_mode == "ht":
            out = tot_HT / float(N)
        if self.normalize_sum_to_one:
            out = out / out.sum(dim=1, keepdim=True).clamp_min(eps)
        return out.to(torch.float32)
