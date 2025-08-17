import math
import torch
from torch import Tensor
from einops import einsum

from jaxtyping import Float

class Linear(torch.nn.Module):

    def __init__(self, d_in: int, d_out: int, device=None, dtype=None):
        super().__init__()
        w: Float[Tensor, "d_out d_in"] = torch.empty((d_out, d_in), device=device, dtype=dtype)
        sigma = math.sqrt(2 / (d_in + d_out))
        torch.nn.init.trunc_normal_(
            w,
            mean=0.0,
            std=sigma,
            a=-3.0 * sigma,
            b=3.0 * sigma,
        )
        self.weight = torch.nn.Parameter(w, requires_grad=True)

    def forward(self, in_features: Float[Tensor, " ... d_in"]) -> Float[Tensor, " ... d_out"]:
        return einsum(self.weight, in_features, "d_out d_in, ... d_in -> ... d_out")

