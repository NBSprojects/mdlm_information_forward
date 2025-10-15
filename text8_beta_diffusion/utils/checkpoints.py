import torch

def load_state_dict_from_checkpoint(path, device="cpu", trust_checkpoint=True):
    try:
        from torch.serialization import add_safe_globals
        from torch.torch_version import TorchVersion
        add_safe_globals([TorchVersion])
    except Exception:
        pass
    try:
        obj = torch.load(path, map_location=device, weights_only=True)
    except Exception as e1:
        if not trust_checkpoint:
            raise RuntimeError("weights_only=True failed and trust_checkpoint=False.") from e1
        obj = torch.load(path, map_location=device, weights_only=False)
    def _extract_state_dict(o):
        if isinstance(o, dict):
            for k in ("model", "state_dict", "model_state_dict", "ema", "ema_state_dict", "denoiser", "net"):
                if k in o and isinstance(o[k], dict):
                    return o[k]
            if len(o) > 0 and all(isinstance(v, torch.Tensor) for v in o.values()):
                return o
        if hasattr(o, "state_dict") and callable(getattr(o, "state_dict")):
            return o.state_dict()
        raise ValueError("Could not extract state_dict from checkpoint.")
    sd = _extract_state_dict(obj)
    if len(sd) and next(iter(sd)).startswith("module."):
        sd = {k[len("module."):]: v for k, v in sd.items()}
    return sd
