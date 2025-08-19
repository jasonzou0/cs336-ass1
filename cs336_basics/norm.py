import math
import torch
from torch import Tensor
from einops import einsum, reduce, rearrange

from jaxtyping import Float

class RmsNorm(torch.nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.g: Float[Tensor, "d_model"] = torch.nn.Parameter(torch.ones((d_model,), device=device, dtype=dtype))
        # eps is a non-learnable parameter, so we register it as a buffer
        self.register_buffer("eps", torch.tensor(eps, device=device, dtype=dtype))
       

    def forward(self, x: Float[Tensor, "batch seq d_model"]) -> Float[Tensor, "batch seq d_model"]:
        def rms_inverse(x: Float[Tensor, "d_model"], reduced_axis) -> Float[Tensor, ""]:
            return 1 / torch.sqrt(torch.mean(x ** 2, axis=reduced_axis) + self.eps)

        in_dtype = x.dtype
        x = x.to(torch.float32) # Upcast to float32 to prevent overflow
        norm_inverse: Float[Tensor, "batch seq"] = reduce(x, "batch seq d_model -> batch seq", reduction=rms_inverse)
        res: Float[Tensor, "batch seq d_model"] = x * rearrange(norm_inverse, "batch seq -> batch seq 1") * rearrange(self.g, "d_model -> 1 1 d_model")
        return res.to(in_dtype)


