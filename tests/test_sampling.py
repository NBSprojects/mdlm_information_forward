import torch
from text8_beta_diffusion.training.sampling import _nucleus_filter

def test_nucleus_filter():
    p = torch.tensor([[0.4, 0.3, 0.2, 0.1]], dtype=torch.float32)
    q = _nucleus_filter(p, top_p=0.6)
    assert q.shape == p.shape
    assert torch.allclose(q.sum(dim=-1), torch.ones(1))
