from text8_beta_diffusion.utils.precision import PrecisionPolicy, maybe_upcast_for_stability
import torch

def test_upcast():
    p = PrecisionPolicy(bf16_only=False, use_autocast=False, upcast_softmax_to_fp32=True)
    x = torch.randn(2, 3, dtype=torch.bfloat16)
    y = maybe_upcast_for_stability(x, p)
    assert y.dtype == torch.float32
