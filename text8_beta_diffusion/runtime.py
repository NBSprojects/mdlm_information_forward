import torch
from .config import DataConfig, DiffusionConfig, MLPConfig, PrecisionConfig
from .setup_torch import set_seed, enable_fast_kernels, add_torchversion_safe_globals
from .utils.precision import PrecisionPolicy, to_model_dtype
from .data import make_dataloaders
from .models.transformer_compat import DenoiserCompat
from .models.scheduler_mlp import SchedulerMLP
from .info.denoiser_info import DenoiserInfoProvider

def build_everything(data_cfg: DataConfig, diff_cfg: DiffusionConfig, mlp_cfg: MLPConfig,
                     precision_cfg: PrecisionConfig, seed: int = 1234):
    set_seed(seed); enable_fast_kernels(); add_torchversion_safe_globals()
    precision = PrecisionPolicy(bf16_only=precision_cfg.bf16_only, use_autocast=precision_cfg.use_autocast,
                                upcast_softmax_to_fp32=precision_cfg.upcast_softmax_to_fp32)
    train_loader, val_loader, vocab_size, mask_id = make_dataloaders(data_cfg, diff_cfg.batch_size_denoiser)
    tcfg = diff_cfg.transformer
    denoiser = DenoiserCompat(
        vocab_size=vocab_size,
        d_model=tcfg.d_model,
        n_heads=tcfg.n_heads,
        n_layers=tcfg.n_layers,
        ff_mult=tcfg.ff_mult,
        dropout=tcfg.dropout,
        max_seq_len=tcfg.max_seq_len,
        mask_id=mask_id,
    )
    denoiser.classifier.prepare_rope(device=next(denoiser.parameters()).device, dtype=torch.float32)
    mlp_model = SchedulerMLP(seq_len=data_cfg.seq_len, hidden=mlp_cfg.hidden, depth=mlp_cfg.depth, dropout=mlp_cfg.dropout,
                             alpha_min=mlp_cfg.alpha_min, alpha_max=mlp_cfg.alpha_max,
                             beta_min=mlp_cfg.beta_min, beta_max=mlp_cfg.beta_max)
    to_model_dtype(mlp_model, precision)
    info_provider = DenoiserInfoProvider(
        denoiser=denoiser, mask_id=mask_id, device=next(denoiser.parameters()).device, K=4,
        ensure_coverage=False, normalize_sum_to_one=True, agg_mode="hajek",
        mix_weight=0.5, use_exp_perplexity=False, max_forward_batch=None,
        precision=precision, max_seq_len=data_cfg.seq_len
    )
    return dict(
        precision=precision, train_loader=train_loader, val_loader=val_loader,
        denoiser=denoiser, mlp_model=mlp_model, info_provider=info_provider,
        vocab_size=vocab_size, mask_id=mask_id
    )
