import torch
from torch import Tensor

from jaxtyping import Float

def softmax(in_features: Float[Tensor, " ..."], dim) -> Float[Tensor, " ..."]:
    """Applies the softmax function to an n-dimensional input Tensor
    rescaling them so that the elements of the n-dimensional output Tensor
    lie in the range [0, 1] and sum to 1.

    Args:
        in_features: input tensor of arbitrary shape
        dim: A dimension along which softmax will be computed

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    exps = torch.exp(in_features - torch.max(in_features, dim=dim, keepdim=True).values)
    return exps / torch.sum(exps, dim=dim, keepdim=True)