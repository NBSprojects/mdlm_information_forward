# base_denoiser.py
import torch
import torch.nn as nn
from typing import Optional

class BaseDenoiser(nn.Module):
    def rope_target(self) -> Optional[nn.Module]:
        """Sous-classes : retourner le sous-module qui expose prepare_rope(), ou None si N/A."""
        return None

    def prepare_rope(self, device=None, dtype=None):
        """Point d'entrée unique pour tous les denoisers (éventuellement no-op)."""
        target = self.rope_target()
        if target is None or not hasattr(target, "prepare_rope"):
            return  # pas de RoPE pour cette archi
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype
        return target.prepare_rope(device=device, dtype=dtype)
