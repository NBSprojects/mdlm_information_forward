# text8_beta_diffusion/models/transformer_llama2_like.py

from __future__ import annotations
import math, torch, torch.nn as nn, torch.nn.functional as F
from typing import Optional, Literal
from .base_denoiser import BaseDenoiser

# ---------- utils ----------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = float(eps)
        self.scale = nn.Parameter(torch.ones(dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x * self.scale

def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10_000.0, device=None, dtype=None):
    base = torch.arange(0, dim, 2, device=device, dtype=torch.float32)[:(dim // 2)]
    freqs = 1.0 / (theta ** (base / dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return freqs.cos().to(dtype=dtype), freqs.sin().to(dtype=dtype)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
    # x*: [B, T, H, Dh]
    B, T, H, Dh = xq.shape
    assert Dh % 2 == 0
    xq_ = xq.float().reshape(B, T, H, Dh//2, 2)
    xk_ = xk.float().reshape(B, T, H, Dh//2, 2)
    fc = freqs_cos[:T].unsqueeze(0).unsqueeze(2)  # [1,T,1,Dh/2]
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

def repeat_kv(x: torch.Tensor, n_rep: int):
    # x: [B, T, H_kv, Dh] -> [B, T, H_kv * n_rep, Dh]
    if n_rep == 1:
        return x
    B, T, H, Dh = x.shape
    x = x.unsqueeze(3).expand(B, T, H, n_rep, Dh)
    return x.reshape(B, T, H * n_rep, Dh)

class Dropout1d(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)
    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        if self.p > 0.0 and training:
            # same mask over time dimension
            B, T, C = x.shape
            mask = torch.empty(B, 1, C, device=x.device, dtype=x.dtype).bernoulli_(1 - self.p)
            x = x * mask / (1 - self.p)
        return x

# ---------- condition temporelle ----------

def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    # timesteps: [B] (float)
    half = embedding_dim // 2
    inv = torch.exp(torch.arange(half, device=timesteps.device, dtype=torch.float32) *
                    (-math.log(10_000) / max(1, (half - 1))))
    sinus = timesteps.float().unsqueeze(-1) * inv.unsqueeze(0)
    emb = torch.cat([torch.sin(sinus), torch.cos(sinus)], dim=-1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1), value=0.0)
    return emb

class CondEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.proj0 = nn.Linear(embedding_dim, embedding_dim * 4)
        self.proj1 = nn.Linear(embedding_dim * 4, embedding_dim)
    def forward(self, t: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        temb = get_timestep_embedding(t, self.embedding_dim)
        h = temb if cond is None else torch.cat([temb, cond], dim=-1)
        h = F.silu(self.proj0(h))
        return self.proj1(h)

# ---------- composants Transformer ----------

_ACT = {
    "swiglu": F.silu,
    "geglu":  F.gelu,
    "glu":    torch.sigmoid,
}

class FeedForward(nn.Module):
    def __init__(self, dim: int, multiple_of: int, dropout: float, hidden_dim: Optional[int] = None,
                 w_init_scale: float = 1.0, mlp_type: Literal["swiglu","geglu","glu"] = "swiglu"):
        super().__init__()
        self.dim = dim
        self.dropout = float(dropout)
        self.act = _ACT[mlp_type]
        if hidden_dim is None:
            h = 4 * dim
            h = int(2 * h / 3)
            h = multiple_of * ((h + multiple_of - 1) // multiple_of)
        else:
            h = int(hidden_dim)
        self.w1 = nn.Linear(dim, h, bias=False)
        self.w2 = nn.Linear(h, dim,  bias=False)
        self.w3 = nn.Linear(dim, h,  bias=False)
        self.resid_drop = Dropout1d(dropout)
        # On pourrait appliquer une init spéciale si besoin (w_init_scale / depth_scaled_init)
    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        y = self.w2(self.act(self.w1(x)) * self.w3(x))
        return self.resid_drop(y, training=training) if self.dropout > 0 else y

class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: Optional[int] = None,
                 dropout: float = 0.0, causal: bool = False, qkv_bias: bool = False):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self._n_kv = n_heads if n_kv_heads is None else int(n_kv_heads)
        assert n_heads % self._n_kv == 0
        self.n_rep = n_heads // self._n_kv
        self.causal = bool(causal)
        self.q = nn.Linear(dim, n_heads * self.head_dim, bias=qkv_bias)
        self.k = nn.Linear(dim, self._n_kv * self.head_dim, bias=qkv_bias)
        self.v = nn.Linear(dim, self._n_kv * self.head_dim, bias=qkv_bias)
        self.o = nn.Linear(dim, dim, bias=False)
        self.attn_drop = nn.Dropout(dropout) if dropout > 0 else None
        self.resid_drop = Dropout1d(dropout)
    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor, training: bool = False) -> torch.Tensor:
        B, T, C = x.shape
        H, Dh = self.n_heads, self.head_dim
        q = self.q(x).view(B, T, H, Dh)
        k = self.k(x).view(B, T, self._n_kv, Dh)
        v = self.v(x).view(B, T, self._n_kv, Dh)
        q, k = apply_rotary_emb(q, k, freqs_cos, freqs_sin)
        k = repeat_kv(k, self.n_rep)  # [B,T,H,Dh]
        v = repeat_kv(v, self.n_rep)
        q = q.transpose(1, 2)  # [B,H,T,Dh]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(Dh)  # [B,H,T,T]
        if self.causal:
            mask = torch.full((1, 1, T, T), float("-inf"), device=x.device, dtype=att.dtype)
            att = att + torch.triu(mask, diagonal=1)
        att = F.softmax(att, dim=-1)
        if self.attn_drop is not None and training:
            att = self.attn_drop(att)
        out = att @ v  # [B,H,T,Dh]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.o(out)
        return self.resid_drop(out, training=training)

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: Optional[int], dropout: float,
                 mlp_multiple_of: int, hidden_dim: Optional[int], w_init_scale: float,
                 mlp_type: str, cond_type: Literal["adaln","adaln_zero"], causal: bool, norm_eps: float,
                 depth_scaled_init: bool, n_layers: int):
        super().__init__()
        self.cond_type = cond_type
        self.attn = Attention(dim, n_heads, n_kv_heads=n_kv_heads, dropout=dropout, causal=causal, qkv_bias=False)
        self.ffn  = FeedForward(dim, multiple_of=mlp_multiple_of, dropout=dropout,
                                hidden_dim=hidden_dim, w_init_scale=w_init_scale, mlp_type=mlp_type)
        # normalisations: si cond -> LN sans affine ; sinon RMSNorm (comme la version JAX)
        if cond_type in ("adaln", "adaln_zero"):
            self.norm_att = nn.LayerNorm(dim, elementwise_affine=False, eps=norm_eps)
            self.norm_mlp = nn.LayerNorm(dim, elementwise_affine=False, eps=norm_eps)
            self.ada_mlp = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6*dim, bias=True))
            if cond_type == "adaln_zero":
                nn.init.zeros_(self.ada_mlp[1].weight); nn.init.zeros_(self.ada_mlp[1].bias)
        else:
            self.norm_att = RMSNorm(dim, eps=norm_eps)
            self.norm_mlp = RMSNorm(dim, eps=norm_eps)
            self.ada_mlp = None
    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor,
                cond_vec: Optional[torch.Tensor], training: bool) -> torch.Tensor:
        if self.ada_mlp is not None:
            assert cond_vec is not None
            shift_att, scale_att, gate_att, shift_mlp, scale_mlp, gate_mlp = self.ada_mlp(cond_vec).chunk(6, dim=-1)
            h = x + gate_att * self.attn(self.norm_att(x) * (1.0 + scale_att) + shift_att, freqs_cos, freqs_sin, training)
            out = h + gate_mlp * self.ffn(self.norm_mlp(h) * (1.0 + scale_mlp) + shift_mlp, training)
        else:
            h = x + self.attn(self.norm_att(x), freqs_cos, freqs_sin, training)
            out = h + self.ffn(self.norm_mlp(h), training)
        return out

class TransformerLlama2Like(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 mask_id: int,
                 dim: int, n_layers: int, n_heads: int, n_kv_heads: Optional[int],
                 mlp_type: str, multiple_of: int, hidden_dim: Optional[int],
                 dropout: float, norm_eps: float, w_init_scale: float,
                 depth_scaled_init: bool, cond_type: Literal["adaln","adaln_zero"],
                 embed_input: bool, rope_theta: float, max_seq_len: int,
                 weight_tying: bool, causal: bool, time_scale: float):
        super().__init__()
        self.dim = dim
        self.vocab_size = int(vocab_size)
        self.mask_id = int(mask_id)
        self.max_seq_len = int(max_seq_len)
        self.time_scale = float(time_scale)
        # entrée
        if embed_input:
            self.input = nn.Embedding(vocab_size + 1, dim)
        else:
            self.input = nn.Linear(vocab_size + 1, dim)  # si tu voulais passer des one-hot
        # blocs
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, n_kv_heads, dropout,
                             mlp_multiple_of=multiple_of, hidden_dim=hidden_dim, w_init_scale=w_init_scale,
                             mlp_type=mlp_type, cond_type=cond_type, causal=causal, norm_eps=norm_eps,
                             depth_scaled_init=depth_scaled_init, n_layers=n_layers)
            for _ in range(n_layers)
        ])
        # RoPE
        head_dim = dim // n_heads
        cos_f32, sin_f32 = precompute_freqs_cis(head_dim, max_seq_len, theta=rope_theta, device='cpu', dtype=torch.float32)
        self.register_buffer("rope_cos_f32", cos_f32, persistent=False)
        self.register_buffer("rope_sin_f32", sin_f32, persistent=False)
        # sortie
        self.cond_type = cond_type
        if cond_type in ("adaln", "adaln_zero"):
            self.output_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=norm_eps)
            self.out_ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2*dim, bias=True))
            if cond_type == "adaln_zero":
                nn.init.zeros_(self.out_ada[1].weight); nn.init.zeros_(self.out_ada[1].bias)
        else:
            self.output_norm = RMSNorm(dim, eps=norm_eps)
            self.out_ada = None
        self.to_logits = nn.Linear(dim, vocab_size, bias=False)
        # init logits à zéro (comme JAX)
        nn.init.zeros_(self.to_logits.weight)
        # tying optionnel
        self.weight_tying = bool(weight_tying)
        if self.weight_tying and isinstance(self.input, nn.Embedding):
            self.to_logits.weight = self.input.weight  # tie
        # embedding de temps
        self.time_cond = nn.Identity()  # on prépare un vecteur cond via CondEmbedding externe

    def forward(self, z: torch.Tensor, t: Optional[torch.Tensor], cond_vec: Optional[torch.Tensor],
                training: bool = False) -> torch.Tensor:
        # z: [B,T] (ids)
        B, T = z.shape
        x = self.input(z)
        freqs_cos = self.rope_cos_f32[:T]
        freqs_sin = self.rope_sin_f32[:T]
        for blk in self.blocks:
            x = blk(x, freqs_cos, freqs_sin, cond_vec=cond_vec, training=training)
        if self.out_ada is not None and cond_vec is not None:
            shift_out, scale_out = self.out_ada(cond_vec).chunk(2, dim=-1)
            h = self.output_norm(x) * (1.0 + scale_out) + shift_out
        else:
            h = self.output_norm(x)
        logits = self.to_logits(h)
        return logits

    def prepare_rope(self, device=None, dtype=None):
        return  # rien à faire pour cette architecture, mais on retourne None pour la cohérence avec BaseDenoiser

class DenoiserLlama2Like(BaseDenoiser):
    def __init__(self,
                 vocab_size: int, mask_id: int,
                 d_model: int, n_heads: int, n_layers: int, n_kv_heads: Optional[int],
                 mlp_type: str, multiple_of: int, ff_mult: float, dropout: float,
                 norm_eps: float, w_init_scale: float, depth_scaled_init: bool,
                 cond_type: Literal["adaln","adaln_zero"],
                 embed_input: bool, rope_theta: float, max_seq_len: int,
                 weight_tying: bool, causal: bool, time_scale: float):
        super().__init__()
        hidden_dim = None
        if ff_mult is not None:
            # option pour forcer une taille cachée personnalisée
            hidden_dim = int(d_model * ff_mult)
        self.cond_mlp = CondEmbedding(d_model)
        self.net = TransformerLlama2Like(
            vocab_size=vocab_size, mask_id=mask_id,
            dim=d_model, n_layers=n_layers, n_heads=n_heads, n_kv_heads=n_kv_heads,
            mlp_type=mlp_type, multiple_of=multiple_of, hidden_dim=hidden_dim,
            dropout=dropout, norm_eps=norm_eps, w_init_scale=w_init_scale,
            depth_scaled_init=depth_scaled_init, cond_type=cond_type,
            embed_input=embed_input, rope_theta=rope_theta, max_seq_len=max_seq_len,
            weight_tying=weight_tying, causal=causal, time_scale=time_scale
        )
        self.backbone = self.net
        self.time_scale = float(time_scale)

    def forward(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # compatibilité : on garde t * time_scale (par défaut 1000.0)
        cond_vec = self.cond_mlp(t.view(-1) * self.time_scale).unsqueeze(1)  # [B,1,dim]
        return self.net(xt, t, cond_vec, training=self.training)

    def rope_target(self):
        return self.backbone

    # (facultatif mais utile pour rétro-compat)
    @property
    def classifier(self):
        return self.backbone