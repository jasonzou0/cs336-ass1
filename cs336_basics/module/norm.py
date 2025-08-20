import math
import torch
from torch import Tensor
from einops import einsum, reduce, rearrange

from jaxtyping import Float

class RmsNorm(torch.nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        # This is a learnable "gain" parameter in RMS normalization.
        self.gain: Float[Tensor, "d_model"] = torch.nn.Parameter(torch.ones((d_model,), device=device, dtype=dtype))
        # eps is a non-learnable parameter, so we register it as a buffer
        self.register_buffer("eps", torch.tensor(eps, device=device, dtype=dtype))
       

    def forward(self, x: Float[Tensor, "batch seq d_model"]) -> Float[Tensor, "batch seq d_model"]:
        def rms(x: Float[Tensor, "d_model"], reduced_axis) -> Float[Tensor, ""]:
            """Root Mean Square calculation."""
            return torch.sqrt(torch.mean(x ** 2, axis=reduced_axis) + self.eps)

        in_dtype = x.dtype
        # Upcast to float32 to prevent overflow
        x = x.to(torch.float32)
        norm: Float[Tensor, "batch seq"] = reduce(x, "batch seq d_model -> batch seq", reduction=rms)
        res: Float[Tensor, "batch seq d_model"] = x * rearrange(self.gain, "d_model -> 1 1 d_model") / rearrange(norm, "batch seq -> batch seq 1")
        return res.to(in_dtype)


