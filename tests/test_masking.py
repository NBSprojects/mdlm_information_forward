import torch
from text8_beta_diffusion.training.masking import beta_pdf_cdf

def test_beta_pdf_cdf_shapes():
    t = torch.tensor([[0.3, 0.6]], dtype=torch.float32)
    a = torch.full_like(t, 2.0)
    b = torch.full_like(t, 5.0)
    pdf, cdf = beta_pdf_cdf(t, a, b)
    assert pdf.shape == t.shape and cdf.shape == t.shape
