import copy, torch

class ModelEMA:
    def __init__(self, model, decay: float = 0.999):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = float(decay)
    @torch.no_grad()
    def update(self, model):
        d = self.decay
        ema_params = [p for p in self.ema.parameters() if p.dtype.is_floating_point]
        src_params = [p.detach() for p in model.parameters() if p.dtype.is_floating_point]
        torch._foreach_mul_(ema_params, d)
        torch._foreach_add_(ema_params, src_params, alpha=1.0 - d)
    def state_dict(self):
        return self.ema.state_dict()
    def load_state_dict(self, sd):
        self.ema.load_state_dict(sd)
