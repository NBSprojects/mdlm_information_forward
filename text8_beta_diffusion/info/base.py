import torch

class BaseInfoProvider:
    def compute_info(self, batch_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class UniformInfoProvider(BaseInfoProvider):
    def compute_info(self, batch_ids: torch.Tensor) -> torch.Tensor:
        return torch.full_like(batch_ids, 1.0, dtype=torch.float32)
