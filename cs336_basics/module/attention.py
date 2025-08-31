import torch
from torch import Tensor
from einops import einsum
from .softmax import softmax

from jaxtyping import Float, Bool

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    assert K.shape[-2] == V.shape[-2], f"Expected the number of keys in K to be equal to the number of values in V, but got {K.shape[-2]} and {V.shape[-2]}"
    scaled_qk: Float[Tensor, " ... queries keys"] = \
        einsum(Q, K, " ... queries d_k, ... keys d_k -> ... queries keys") / (Q.shape[-1] ** 0.5)
    if mask is not None:
        scaled_qk = scaled_qk.masked_fill(~mask, float('-inf'))
    return einsum(softmax(scaled_qk, dim=-1), V, " ... queries keys, ... keys d_v -> ... queries d_v")
