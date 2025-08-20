import torch
from torch import Tensor

from jaxtyping import Float
from .linear import Linear
from .silu import silu

class SwiGLU(torch.nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.linear_1 = Linear(d_in=d_model, d_out=d_ff, device=device, dtype=dtype)
        self.linear_2 = Linear(d_in=d_ff, d_out=d_model, device=device, dtype=dtype)
        self.linear_3 = Linear(d_in=d_model, d_out=d_ff, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        return self.linear_2(silu(self.linear_1(x)) * self.linear_3(x))
