from __future__ import annotations
import math, torch, torch.nn as nn, torch.nn.functional as F
from typing import Optional
from ..utils.checkpoints import load_state_dict_from_checkpoint
from .base_denoiser import BaseDenoiser
from text8_beta_diffusion.utils.logging import get_model_summary


class CondEmbeddingCompat(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(embedding_dim * 4, embedding_dim),
        )
    def forward(self, t: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        half = self.embedding_dim // 2
        inv = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) *
                        (-math.log(10_000) / max(1, (half - 1))))
        sinus = t.float().unsqueeze(-1) * inv.unsqueeze(0)
        temb = torch.cat([torch.sin(sinus), torch.cos(sinus)], dim=-1)
        if self.embedding_dim % 2 == 1:
            temb = F.pad(temb, (0, 1), value=0.0)
        h = temb if cond is None else torch.cat([temb, cond], dim=-1)
        return self.mlp(h)

def precompute_freqs_cis_compat(dim: int, seq_len: int, theta: float = 10_000.0, device=None, dtype=None):
    base = torch.arange(0, dim, 2, device=device, dtype=torch.float32)[:(dim // 2)]
    freqs = 1.0 / (theta ** (base / dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    cos = freqs.cos().to(dtype=dtype); sin = freqs.sin().to(dtype=dtype)
    return cos, sin

def apply_rotary_emb_compat(xq, xk, freqs_cos, freqs_sin):
    B, T, H, Dh = xq.shape
    assert Dh % 2 == 0
    xq_ = xq.float().reshape(B, T, H, Dh//2, 2)
    xk_ = xk.float().reshape(B, T, H, Dh//2, 2)
    fc = freqs_cos[:T].unsqueeze(0).unsqueeze(2)
    fs = freqs_sin[:T].unsqueeze(0).unsqueeze(2)
    xq_r, xq_i = xq_[..., 0], xq_[..., 1]
    xk_r, xk_i = xk_[..., 0], xk_[..., 1]
    xq_out_r = xq_r * fc - xq_i * fs
    xq_out_i = xq_r * fs + xq_i * fc
    xk_out_r = xk_r * fc - xk_i * fs
    xk_out_i = xk_r * fs + xk_i * fc
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).reshape(B, T, H, Dh)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).reshape(B, T, H, Dh)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class MultiheadSelfAttentionCompat(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0, causal: bool = False):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim; self.n_heads = n_heads; self.head_dim = dim // n_heads
        self.causal = causal
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout
    def forward(self, x, freqs_cos, freqs_sin, train: bool = False):
        B, T, C = x.shape
        H, Dh = self.n_heads, self.head_dim
        q = self.q_proj(x).view(B, T, H, Dh)
        k = self.k_proj(x).view(B, T, H, Dh)
        v = self.v_proj(x).view(B, T, H, Dh)
        q, k = apply_rotary_emb_compat(q, k, freqs_cos, freqs_sin)
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
        attn = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if train else 0.0, is_causal=self.causal
        )
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(attn)

class FeedForwardCompat(nn.Module):
    def __init__(self, dim: int, mlp_mult: float, dropout: float = 0.0):
        super().__init__()
        hidden = int(2 * (4 * dim) / 3)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim,  bias=False)
        self.w3 = nn.Linear(dim, hidden,  bias=False)
        self.dropout = float(dropout)
    def forward(self, x, train: bool = False):
        y = self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))
        if self.dropout and train:
            y = torch.nn.functional.dropout(y, p=self.dropout, training=True)
        return y

class TransformerBlockCompat(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_mult: float, dropout: float, cond_type: str = "adaln"):
        super().__init__()
        self.cond_type = cond_type
        self.attn = MultiheadSelfAttentionCompat(dim, n_heads, dropout=dropout, causal=False)
        self.ffn  = FeedForwardCompat(dim, mlp_mult, dropout=dropout)
        self.norm_att = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm_mlp = nn.LayerNorm(dim, elementwise_affine=False)
        if cond_type in ("adaln", "adaln_zero"):
            self.ada_mlp = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6*dim, bias=True))
            if cond_type == "adaln_zero":
                nn.init.zeros_(self.ada_mlp[1].weight); nn.init.zeros_(self.ada_mlp[1].bias)
        else:
            raise NotImplementedError("Only AdaLN variant is used.")
    def forward(self, x, freqs_cos, freqs_sin, cond_vec: Optional[torch.Tensor] = None, train: bool = False):
        if self.cond_type in ("adaln", "adaln_zero"):
            assert cond_vec is not None
            shift_att, scale_att, gate_att, shift_mlp, scale_mlp, gate_mlp = self.ada_mlp(cond_vec).chunk(6, dim=-1)
            h = x + gate_att * self.attn(self.norm_att(x) * (1.0 + scale_att) + shift_att, freqs_cos, freqs_sin, train=train)
            out = h + gate_mlp * self.ffn(self.norm_mlp(h) * (1.0 + scale_mlp) + shift_mlp, train=train)
        else:
            h = x + self.attn(self.norm_att(x), freqs_cos, freqs_sin, train=train)
            out = h + self.ffn(self.norm_mlp(h), train=train)
        return out

class TransformerClassifierCompat(nn.Module):
    def __init__(self, vocab_size: int, mask_id: int, dim: int, n_layers: int, n_heads: int, mlp_mult: float, dropout: float, max_seq_len: int):
        super().__init__()
        self.vocab_size = vocab_size; self.mask_id = mask_id; self.dim = dim
        self.embed = nn.Embedding(vocab_size + 1, dim)
        self.time_cond = CondEmbeddingCompat(dim)
        self.blocks = nn.ModuleList([
            TransformerBlockCompat(dim, n_heads, mlp_mult, dropout, cond_type="adaln") for _ in range(n_layers)
        ])
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False)
        self.to_logits = nn.Linear(dim, vocab_size, bias=False)
        self.max_seq_len = int(max_seq_len)
        self.head_dim = dim // n_heads
        cos_f32, sin_f32 = precompute_freqs_cis_compat(self.head_dim, self.max_seq_len, theta=10_000.0, device='cpu', dtype=torch.float32)
        self.register_buffer("rope_cos_f32", cos_f32, persistent=False)
        self.register_buffer("rope_sin_f32", sin_f32, persistent=False)

    def prepare_rope(self, device=None, dtype=torch.float32):
        if device is None:
            device = next(self.parameters()).device
        self.rope_cos_f32 = self.rope_cos_f32.to(device=device, dtype=dtype)
        self.rope_sin_f32 = self.rope_sin_f32.to(device=device, dtype=dtype)

    def forward(self, z: torch.Tensor, t: torch.Tensor | None, train: bool = False) -> torch.Tensor:
        B, T = z.shape
        h = self.embed(z)
        cond_vec = None
        if t is not None:
            cond_vec = self.time_cond(t.view(-1) * 1000.0).unsqueeze(1)
        freqs_cos = self.rope_cos_f32[:T]; freqs_sin = self.rope_sin_f32[:T]
        for blk in self.blocks:
            h = blk(h, freqs_cos, freqs_sin, cond_vec=cond_vec, train=train)
        h = self.norm_out(h)
        logits = self.to_logits(h)
        return logits



class DenoiserCompat(BaseDenoiser):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int,
                 ff_mult: float = 4.0, dropout: float = 0.0, mask_id: int | None = None, max_seq_len: int = 256):
        super().__init__()
        if mask_id is None:
            mask_id = vocab_size
        self.classifier = TransformerClassifierCompat(
            vocab_size=vocab_size, mask_id=mask_id, dim=d_model, n_layers=n_layers,
            n_heads=n_heads, mlp_mult=ff_mult, dropout=dropout, max_seq_len=max_seq_len
        )
        get_model_summary(self.classifier)
        
    def forward(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.classifier(xt, t, train=self.training)
    def load_state_dict(self, state_dict, strict: bool = True):
        if any(k.startswith("classifier.") for k in state_dict.keys()):
            sd_cls = {k[len("classifier."):]: v for k, v in state_dict.items() if k.startswith("classifier.")}
        else:
            sd_cls = state_dict
        return self.classifier.load_state_dict(sd_cls, strict=strict)

    def rope_target(self):
        return self.classifier

def build_denoiser_for_info_from_checkpoint(
    weights_path: str, device: torch.device, vocab_size: int, mask_id: int,
    d_model: int, n_heads: int, n_layers: int, ff_mult: float, dropout: float = 0.0,
    max_seq_len: int = 256, trust_checkpoint: bool = True,
):
    sd = load_state_dict_from_checkpoint(weights_path, device=device, trust_checkpoint=trust_checkpoint)
    model = DenoiserCompat(vocab_size=vocab_size, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                           ff_mult=ff_mult, dropout=dropout, mask_id=mask_id, max_seq_len=max_seq_len).to(device)
    missing_unexp = model.load_state_dict(sd, strict=True)
    try:
        missing, unexpected = missing_unexp.missing_keys, missing_unexp.unexpected_keys
    except Exception:
        missing = getattr(missing_unexp, "missing_keys", [])
        unexpected = getattr(missing_unexp, "unexpected_keys", [])
    return model, missing, unexpected
