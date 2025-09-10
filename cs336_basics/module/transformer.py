import torch
from torch import Tensor
from .linear import Linear
from .attention import CasualMultiheadSelfAttention
from .norm import RmsNorm
from .rope import Rope
from .swiglu import SwiGLU
from .embedding import Embedding

from jaxtyping import Float, Int
from dataclasses import dataclass


@dataclass
class TransformerConfig:
    vocab_size: int = 10000
    context_length: int = 256
    d_model: int = 512
    d_ff: int = None
    rope_theta: float = 10000.0
    num_layers: int = 4
    num_heads: int = 16

    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = int(8 * self.d_model / 3)


class TransformerBlock(torch.nn.Module):
    """A single Transformer block consisting of a multi-head self-attention layer
    followed by a feedforward neural network (FFN) with SwiGLU activation.

    Each sub-layer starts with RMS normalization and ends with a residual connection.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len: int, device=None, dtype=None):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(f"d_model must be multiples of num_heads, but got d_model={d_model} and num_heads={num_heads}")

        rope = Rope(theta=theta, d_k=d_model // num_heads, max_seq_len=max_seq_len, device=device, dtype=dtype)
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
    
    @staticmethod
    def from_config(config: TransformerConfig, device=None, dtype=torch.float32):
        """Create a Transformer instance from a TransformerConfig.
        
        Args:
            config: TransformerConfig instance containing model parameters
            device: device to store the model parameters
            dtype: data type for the model parameters (default: torch.float32)
        
        Returns:
            Transformer: A new Transformer instance
        """
        return Transformer(
            vocab_size=config.vocab_size,
            context_length=config.context_length,
            d_model=config.d_model,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            rope_theta=config.rope_theta,
            device=device,
            dtype=dtype
        )
    
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
            Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted *UN-NORMALIZED* next-word distribution for each token.
        """
        x = self.token_emb(in_indices)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.rms_final(x))