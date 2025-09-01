import torch
from torch import Tensor
from einops import einsum, rearrange
from .softmax import softmax
from .linear import Linear

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
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor, with True values indicating positions to attend to
            The mask tensor's shape just needs to be broadcastable to the shape of the
            attention scores
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of attention in the shape of (batch dims), queries (sequence dim), d_v (d_model)
    """
    if not torch.jit.is_scripting() and not torch._dynamo.is_compiling():
        if K.shape[-2] != V.shape[-2]:
            raise ValueError(f"Expected the number of keys in K to equal the number of values in V, but got {K.shape} and {V.shape}")
    attn_scores: Float[Tensor, " ... queries keys"] = \
        einsum(Q, K, " ... queries d_k, ... keys d_k -> ... queries keys") / (Q.shape[-1] ** 0.5)
    if mask is not None:
        attn_scores.masked_fill_(~mask, float('-inf'))
    return einsum(softmax(attn_scores, dim=-1), V, " ... queries keys, ... keys d_v -> ... queries d_v")


class CasualMultiheadSelfAttention(torch.nn.Module):
    # TODO: add max_seq_len and rope support
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int=256, device=None, dtype=None):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.num_heads = num_heads

        if d_model % num_heads != 0:
            raise ValueError(f"d_model must be multiples of num_heads, but got d_model={d_model} and num_heads={num_heads}")

        d_k = d_v = d_model // num_heads
        # TODO: merge all three input projections into one.
        self.q_proj = Linear(d_in=d_model, d_out=d_k*num_heads, device=self.device, dtype=dtype)
        self.k_proj = Linear(d_in=d_model, d_out=d_k*num_heads, device=self.device, dtype=dtype)
        self.v_proj = Linear(d_in=d_model, d_out=d_v*num_heads, device=self.device, dtype=dtype)
        # Output projection
        self.o_proj = Linear(d_in=d_v*num_heads, d_out=d_model, device=self.device, dtype=dtype)
        self.attn_mask = torch.tril(torch.ones((max_seq_len, max_seq_len), dtype=torch.bool, device=self.device))

    def forward(self, x: Float[Tensor, " ... sequence_length d_model"]) -> Float[Tensor, " ... sequence_length d_model"]:
        Q: Float[Tensor, " ... h sequence_length dk"] = rearrange(self.q_proj(x), "... sequence_length (h dk) -> ... h sequence_length dk", h=self.num_heads)
        K: Float[Tensor, " ... h sequence_length dk"] = rearrange(self.k_proj(x), "... sequence_length (h dk) -> ... h sequence_length dk", h=self.num_heads)
        V: Float[Tensor, " ... h sequence_length dv"] = rearrange(self.v_proj(x), "... sequence_length (h dv) -> ... h sequence_length dv", h=self.num_heads)
        seq_len = x.shape[-2]
        attn_output: Float[Tensor, " ... h sequence_length dv"] = scaled_dot_product_attention(Q, K, V, self.attn_mask[:seq_len, :seq_len])
        return self.o_proj(rearrange(attn_output, "... h sequence_length dv -> ... sequence_length (h dv)"))



