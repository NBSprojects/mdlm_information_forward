import torch, random, numpy as np

def set_seed(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def enable_fast_kernels():
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        from torch.backends.cuda import sdp_kernel
        sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)
    except Exception:
        pass

def add_torchversion_safe_globals():
    try:
        from torch.serialization import add_safe_globals
        from torch.torch_version import TorchVersion
        add_safe_globals([TorchVersion])
    except Exception:
        pass
