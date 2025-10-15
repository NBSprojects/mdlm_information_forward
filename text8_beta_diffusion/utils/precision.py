import torch, contextlib
from dataclasses import dataclass

@dataclass
class PrecisionPolicy:
    bf16_only: bool = False
    use_autocast: bool = True
    upcast_softmax_to_fp32: bool = True

def autocast_ctx(policy: PrecisionPolicy):
    enabled = torch.cuda.is_available() and policy.use_autocast
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)

def maybe_upcast_for_stability(t: torch.Tensor, policy: PrecisionPolicy):
    return t.float() if policy.upcast_softmax_to_fp32 else t

def to_model_dtype(module: torch.nn.Module, policy: PrecisionPolicy):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if policy.bf16_only and torch.cuda.is_available():
        module = module.to(device=device, dtype=torch.bfloat16)
    else:
        module = module.to(device=device)
    return module
