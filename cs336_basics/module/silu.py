import torch
from torch import Tensor

from jaxtyping import Float

def silu(x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    """Sigmoid Linear Unit (SiLU) activation function."""
    return x * torch.sigmoid(x)