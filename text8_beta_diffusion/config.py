from dataclasses import dataclass
from typing import Optional

@dataclass
class DataConfig:
    text8_path: str = "data/text8.zip"
    seq_len: int = 256
    train_frac: float = 0.98
    batch_size: int = 512
    num_workers: int = 8
    lowercase_only: bool = True
    chars: str = "abcdefghijklmnopqrstuvwxyz "
    mask_char: str = "█"

@dataclass
class NGramConfig:
    order: int = 7
    min_count: int = 1

@dataclass
class InfoConfig:
    normalize_mode: str = "seq_minmax"
    global_clip_quantile: float = 0.99

@dataclass
class MLPConfig:
    hidden: int = 512
    depth: int = 3
    dropout: float = 0.1
    lr: float = 1e-3
    batch_size_MLP: int = 64
    wd: float = 0.0
    steps: int = 500
    t_samples_per_step: int = 32
    beta_min: float = 0.1
    beta_max: float = 20.0
    alpha_min: float = 0.1
    alpha_max: float = 20.0

@dataclass
class DiffusionConfig:
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    ff_mult: float = 4.0
    dropout: float = 0.1
    lr: float = 3e-5
    batch_size_denoiser: int = 256
    wd: float = 0.01
    steps: int = 20_000
    warmup: int = 1_000
    grad_clip: float = 1.0
    log_interval: int = 100
    eval_interval: int = 1_000
    ema: bool = False
    ema_decay: float = 0.999
    ema_start: int = 0
    weighted_loss: bool = True
    sample_every: int = 500
    
@dataclass
class PrecisionConfig:
    bf16_only: bool = True
    use_autocast: bool = False
    upcast_softmax_to_fp32: bool = True

@dataclass
class SamplingConfig:
    grid: str = "cosine"
    schedule: str = "linear"
    steps: int = 1000
    temperature: float = 1.0
    snapshots_every: int = 100
    top_p: float = 1.0
    prefix: Optional[str] = None
    n_samples: int = 1
