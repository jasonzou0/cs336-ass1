import torch
from torch import Tensor
from .linear import Linear
from .attention import CasualMultiheadSelfAttention
from .norm import RmsNorm
from .rope import Rope
from .swiglu import SwiGLU
from .embedding import Embedding
from .softmax import softmax

from jaxtyping import Float, Int

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


class Transformer(torch.nn.Module):
    """A decoder-only Transformer using RoPE and SwiGLU activations."""
    def __init__(
        self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float, device=None, dtype=None):
        """Create a Transformer model using RoPE and SwiGLU.

        Args:
            vocab_size: size of the vocabulary
            context_length: maximum sequence length supported
            d_model: dimension of the model
            num_layers: number of transformer blocks
            num_heads: number of attention heads in each block
            d_ff: dimension of the feedforward network in each block
            rope_theta: base frequency / rotation angle for RoPE
            device: device to store the model parameters
            dtype: data type for the model parameters
        """
        super().__init__()
        self.token_emb = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)
        self.layers = torch.nn.ModuleList([
            TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, theta=rope_theta, max_seq_len=context_length, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.rms_final = RmsNorm(d_model=d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_in=d_model, d_out=vocab_size, device=device, dtype=dtype)

    def forward(self, in_indices: Int[Tensor, " batch_size sequence_length"]) -> Float[Tensor, "batch_size sequence_length vocab_size"]:
        """Pass the input through the transformer..

        Args:
            in_indices: Tensor with tokenized indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.
        Returns:
            Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized next-word distribution for each token.
        """
        x = self.token_emb(in_indices)
        for layer in self.layers:
            x = layer(x)
        return softmax(self.lm_head(self.rms_final(x)), dim=-1)