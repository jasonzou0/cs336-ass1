import math
import torch
from torch import Tensor
from einops import einsum, repeat, rearrange

from jaxtyping import Float, Int

class Rope(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=None):
        """Initialize the RoPE module.

        Args:
            theta: base frequency / rotation angle
            d_k: dimension of the key/query vectors (must be even)
            max_seq_len: maximum sequence length supported
            device: device to store the precomputed cache
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype
        assert d_k % 2 == 0, "d_k must be even"
        self._build_cache()

    def _build_cache(self):
        """build the cos and sin cache used in the forward pass"""
        d_half = self.d_k // 2
        # Create indices for each pair: 1, 1, 2, 2, ..., d_half, d_half
        indices: Float[Tensor, "d_k"] = repeat(
            torch.arange(1, d_half+1, device=self.device, dtype=self.dtype),
            "d_half -> (d_half 2)",
        )
        # The theta_i's from the original paper: 1 / (theta^(2(i-1)/d_k))
        inv_freq: Float[Tensor, "d_k"] = 1.0 / (self.theta ** ((2 * indices -2) / self.d_k))
        # Each row stores the "m * theta_i" for a particular sequence position m
        m_inv_freq: Float[Tensor, "seq_len d_k"] = rearrange(torch.arange(self.max_seq_len, device=self.device, dtype=self.dtype), "seq_len -> seq_len 1") \
            * rearrange(inv_freq, "d_k -> 1 d_k")
        self.register_buffer("cos_cache", torch.cos(m_inv_freq), persistent=False)
        self.register_buffer("sin_cache", torch.sin(m_inv_freq), persistent=False)

    def forward(
        self, 
        x: Float[Tensor, "... seq_len d_k"],
        token_positions: Float[Tensor, "... seq_len"] | None = None,
    ) -> Float[Tensor, "... seq_len d_k"]:
        """Apply RoPE to the input tensor x.

        Args:
            x: input tensor of shape (..., seq_len, d_k)
            token_positions: tensor of shape (..., seq_len) indicating the position of each token in the sequence.
                If None, assumes positions are [0, 1, 2, ..., seq_len-1] for each sequence in the batch.
        Returns:
            Tensor of same shape as x with RoPE applied
        """
        *batch_dims, seq_len, d_k = x.shape
        assert d_k == self.d_k, f"Expected last dimension to be {self.d_k}, got {d_k}"
        assert seq_len <= self.max_seq_len, f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"

        # Get the cos and sin values for the given token positions
        cos_pos: Float[Tensor, "... seq_len d_k"] = self.cos_cache[token_positions] if token_positions is not None else self.cos_cache[:seq_len]
        sin_pos: Float[Tensor, "... seq_len d_k"] = self.sin_cache[token_positions] if token_positions is not None else self.sin_cache[:seq_len]
        
        # Using the "Computational efficient realization of rotary matrix multiplication" idea 
        # from the original RoPE paper: https://arxiv.org/pdf/2104.09864
        return (x * cos_pos) + (self._transform_input(x) * sin_pos)


    def _transform_input(self, x: Float[Tensor, "... d_k"],) -> Float[Tensor, "... d_k"]:
        """transform the input tensor by swapping and negating elements; returns a tensor of the same shape as input"""
        if x.shape[-1] % 2 != 0:
            raise ValueError("The last dimension must have an even number of elements.")

        # Reshape the last dimension into pairs.
        reshaped_x: Float[Tensor, "... dk_half r"] = rearrange(x, "... (dk_half r) -> ... dk_half r", r=2)
        
        # Extract and reorder the pairs
        output_pairs: Float[Tensor, "... dk_half r"] = torch.empty_like(reshaped_x)
        output_pairs[..., 0] = -reshaped_x[..., 1]
        output_pairs[..., 1] = reshaped_x[..., 0]
        
        # Return output to original shape
        return rearrange(output_pairs, "... dk_half r -> ... (dk_half r)")
