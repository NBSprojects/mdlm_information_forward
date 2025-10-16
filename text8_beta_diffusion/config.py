from dataclasses import dataclass
from typing import Optional, Literal


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
class InfoProviderDenoiserConfig:
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    ff_mult: float = 4.0
    dropout: float = 0.1
    max_seq_len: int = 256
    # Poids pré-entraînés (chemin où les trouver)
    weights_path: Optional[str] = "checkpoints/denoiser_final.pt"
    # Si True, on échoue si le fichier n'existe pas
    require_weights: bool = True
    lr: float = 3e-5
    batch_size_denoiser: int =  512
    wd: float = 0.01
    steps: int = 20000
    warmup: int = 1000
    grad_clip: float = 1.0
    log_interval: int = 100
    eval_interval: int = 1000
    ema_decay: float = 0.999  # décroissance EMA
    ema_start: int = 0

@dataclass
class InfoProviderRuntimeConfig:
    # Réglages "runtime" de DenoiserInfoProvider
    K: int = 4
    ensure_coverage: bool = False
    normalize_sum_to_one: bool = True
    agg_mode: Literal["naive", "hajek", "ht"] = "hajek"
    mix_weight: float = 0.5
    use_exp_perplexity: bool = False
    max_forward_batch: Optional[int] = 1024
    time_from_pi: bool = True
    ht_exclude_forced: bool = False



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
    # === Nouveaux chemins/flags ===
    # Où SAUVER le scheduler MLP après son entraînement
    save_path: str = "checkpoints/scheduler_mlp.pt"
    # Où CHARGER le scheduler MLP quand on lance train_denoiser_cli.py
    load_path_for_info: Optional[str] = "checkpoints/scheduler_mlp.pt"
    # Si True, on échoue si le fichier n'existe pas
    required_for_info: bool = True


@dataclass
class TransformerConfig:
    # Choix d’archi
    arch: Literal["compat", "llama2"] = "compat"
    # Dimensions
    d_model: int = 288
    n_heads: int = 6
    n_layers: int = 6
    n_kv_heads: Optional[int] = None  # GQA: si None => = n_heads
    # MLP
    mlp_type: Literal["swiglu", "geglu", "glu"] = "swiglu"
    multiple_of: int = 32
    ff_mult: float = 4.0              # facteur si tu veux dévier du (2/3)*4d
    # Normalisation / init
    norm_eps: float = 1e-5
    dropout: float = 0.0
    qkv_bias: bool = False
    weight_tying: bool = False
    w_init_scale: float = 1.0
    depth_scaled_init: bool = False
    # Conditionnement
    cond_type: Literal["adaln", "adaln_zero"] = "adaln"
    # Entrée / sortie / RoPE
    embed_input: bool = True          # True = Embedding; False = Linear(x)
    rope_theta: float = 10_000.0
    max_seq_len: int = 256
    # Autres
    causal: bool = False
    time_scale: float = 1000.0        # pour garder compat : t -> t*1000 avant embedding


@dataclass
class DiffusionConfig:
    transformer: TransformerConfig = TransformerConfig()
    lr: float = 3e-5
    batch_size_denoiser: int = 256
    wd: float = 0.01
    steps: int = 20_000
    warmup: int = 1_000
    grad_clip: float = 1.0
    log_interval: int = 100
    eval_interval: int = 1_000
    verbose_batch: bool = True   # <-- NOUVEAU: active le dump du sous-batch en logs
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


@dataclass
class CheckpointConfig:
    # Chemin vers un checkpoint de débruiteur (optionnel)
    load_denoiser_from: Optional[str] = None
    # Chemin vers un checkpoint de MLP (optionnel) -> state_dict() attendu
    load_mlp_from: Optional[str] = None

@dataclass
class ValidationConfig:
    # 1) Sampling depuis les poids chargés
    enable_sampling_from_checkpoint: bool = False
    sampling_n_samples: int = 2
    sampling_steps: int = 400
    sampling_grid: str = "cosine"
    sampling_temperature: float = 1.0
    sampling_top_p: float = 1.0
    sampling_prefix: Optional[str] = None
    # 2) Statistiques d'information
    run_info_stats: bool = False
    info_stats_batches: int = 1
    info_stats_compute_kl: bool = True
    info_stats_compute_ot: bool = False  # calcul OT (POT) plus coûteux
    info_stats_entropy_in_bits: bool = True
    # 3) Statistiques des paramètres (a,b) du MLP
    run_mlp_param_stats: bool = False
    mlp_param_stats_batches: int = 1
    mlp_param_stats_threshold: float = 19.9
    mlp_param_stats_positions_per_seq: int = 10
    mlp_param_stats_max_seqs_to_show: int = 10
    # 4) Évaluation du gain MSE intégré (Beta vs baseline p(t)=t)
    run_mse_gain_eval: bool = False
    mse_gain_batches: int = 10
    mse_gain_t_samples: int = 64