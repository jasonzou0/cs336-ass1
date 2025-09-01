import torch
from torch import Tensor
from .linear import Linear
from .attention import CasualMultiheadSelfAttention
from .norm import RmsNorm
from .rope import Rope
from .swiglu import SwiGLU

from jaxtyping import Float

class TransformerBlock(torch.nn.Module):
    """A single Transformer block consisting of a multi-head self-attention layer
    followed by a feedforward neural network (FFN) with SwiGLU activation.

    Each sub-layer starts with RMS normalization and ends with a residual connection.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len: int, device=None, dtype=None):
        super().__init__()

        rope = Rope(theta=theta, d_k=d_model // num_heads, max_seq_len=max_seq_len, device=device)
        self.attn = CasualMultiheadSelfAttention(d_model=d_model, num_heads=num_heads, rope_module=rope, max_seq_len=max_seq_len, device=device, dtype=dtype)
        self.rms1 = RmsNorm(d_model=d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        self.rms2 = RmsNorm(d_model=d_model, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, " batch sequence_length d_model"]) -> Float[Tensor, "... sequence_length d_model"]:
        """Pass the input through the transformer block.

        Args:
            x: input tensor of shape (..., sequence_length, d_model)
        Returns:
            Tensor of same shape as x
        """
        x += self.attn(self.rms1(x))
        x += self.ffn(self.rms2(x))
        return x