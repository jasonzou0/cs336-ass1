import torch
import torch.nn as nn
from einops import einsum, repeat
from jaxtyping import Float, Int, Bool
from torch import Tensor
from collections.abc import Callable, Iterable
from typing import Optional
import math
class LinearModule(nn.Module):
    weight: nn.Parameter

    def __init__(self, 
                in_features:int, out_features: int,
                device: torch.device=torch.device('cpu'),
                dtype: torch.dtype=torch.float32):
        super(LinearModule,self).__init__()
        # Initialize weights with normal distribution
        # Using standard deviation of 1/sqrt(in_features) which is common for linear layers
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        torch.nn.init.normal_(self.weight, mean=0.0, std=1.0 / (in_features ** 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return torch.matmul(x, self.W.T)  # This is the correct way to perform matrix multiplication in PyTorch
        return x@(self.weight.T)

class EmbeddingModule(nn.Module):
    weight: nn.Parameter

    def __init__(self, 
                num_embeddings:int, 
                embedding_dim:int,
                device: torch.device=torch.device('cpu'),
                dtype: torch.dtype=torch.float32):
        super(EmbeddingModule,self).__init__()
        # Initialize weights with normal distribution
        # Using standard deviation of 0.02 which is common for transformer embeddings
        print(f"!!!dtype=={dtype}")
        self.weight = nn.Parameter(torch.zeros((num_embeddings, embedding_dim),device=device,dtype=dtype))
        torch.nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]  # This is the correct way to index into the embedding matrix

        
class RMSNormModule(nn.Module):
    weight: nn.Parameter
    def __init__(self, 
                d_model: int, 
                eps: float = 1e-6,
                device: torch.device=torch.device('cpu'),
                dtype: torch.dtype=torch.float32):
        super(RMSNormModule, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))
        torch.nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype=x.dtype
        x=x.to(torch.float32)
        rms = torch.sqrt((x*x).mean(dim=-1, keepdim=True) + self.eps)
        result=x/rms*self.weight
        return result.to(in_dtype)   # This is the correct way to apply RMS normalization

class SwiGLUModule(nn.Module):
    """SwiGLU module as described in PaLM paper.
    
    Projects input to a larger dimension, applies SwiGLU activation,
    then projects back to the original dimension.
    """
    weight1: nn.Parameter
    weight2: nn.Parameter
    weight3: nn.Parameter
    
    def __init__(self, 
                d_model: int, 
                d_ff: int,
                device: torch.device=torch.device('cpu'),
                dtype: torch.dtype=torch.float32):
        super(SwiGLUModule, self).__init__()
        # Ensure d_ff is at least 2 to prevent division by zero
        self.d_ff = max(2, d_ff)
        self.d_model = max(1, d_model)
        
        # For compatibility with test adapters
        self.weight1 = nn.Parameter(torch.empty((self.d_ff, self.d_model), device=device, dtype=dtype))
        self.weight2 = nn.Parameter(torch.empty((self.d_model, self.d_ff), device=device, dtype=dtype))
        self.weight3 = nn.Parameter(torch.empty((self.d_ff, self.d_model), device=device, dtype=dtype))
        
        # Initialize weights
        torch.nn.init.normal_(self.weight1, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.weight2, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.weight3, mean=0.0, std=0.02)

    @staticmethod
    def SiLU(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU transformation.
        
        Args:
            x: Input tensor of shape (..., d_model)
            
        Returns:
            Tensor of shape (..., d_model)
        """
        # Handle batch dimensions properly
        orig_shape = x.shape
        # Flatten all dimensions except last for matrix multiplication
        x_2d = x.view(-1, orig_shape[-1])
        
        # Project to intermediate dimension
        gate = x_2d @ self.weight1.T  # (..., d_ff)
        value = x_2d @ self.weight3.T  # (..., d_ff)
        
        # Apply SwiGLU: SiLU(gate) * value
        hidden = self.SiLU(gate) * value
        
        # Project back to model dimension
        out = hidden @ self.weight2.T  # (..., d_model)
        
        # Restore original shape
        return out.view(orig_shape)
    
class RoPE(torch.nn.Module):
    def __init__(self, theta: float, 
                 d_k: int, 
                 max_seq_len: int, 
                 device: torch.device=torch.device('cpu'), 
                 dtype: torch.dtype=torch.float32):
        """ the RoPE module and create buffers if needed. Rotary Position Embeddings.

        Args:
            theta: float Θ value for the RoPE (TODO: the video says should be different for each different vector??)
            d_k: int dimension of query and key vectors (embedding dimension)
            max_seq_len: int Maximum sequence length that will be inputted
            device: torch.device | None = None Device to store the buffer on
            dtype: torch.dtype | None = None Dtype to store the buffer on

        TODO: store sin/cos as buffer since they would be used in real case????
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype

        # Precompute the frequencies
        ### TEST different theta??######################################
        # x=random.random()
        # base_freqs = 1.0 / ((theta+x) ** (torch.arange(0, d_k, 2, device=device, dtype=dtype) / d_k))
        ################################################################

        base_freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device, dtype=dtype) / d_k))
        base_freqs = base_freqs.unsqueeze(0)  # [1, d_k/2]

        # Calculate outer product to get [seq_len, d_k/2]
        t = torch.arange(max_seq_len, device=device, dtype=dtype)  # [seq_len]
        freqs = einsum(t, base_freqs[0], 'i, j -> i j')
        # OR USE BELOW: (but we should use einsum as per requirement)
        # t=t.unsqueeze(-1)  # then t.shape=[seq_len,1]
        # freqs = t @ base_freqs  

        cos = torch.cos(freqs)  # [seq_len, d_k/2]
        sin = torch.sin(freqs)  # [seq_len, d_k/2]

        # self.register_buffer("freqs", freqs)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def get_cached_cos_sin(self, seq_len: int) -> tuple[Float[Tensor, "seq_len d_k/2"], Float[Tensor, "seq_len d_k/2"]]:
        """ Get the cached cos and sin matrices for RoPE. 
        Might be needed if we want to use the precomputed buffers as per the video in realworld use case.

        Args:
            seq_len: int Length of the sequence to get the cos and sin matrices for

        Returns:
            tuple: (cos, sin) each of shape (seq_len, d_k/2)
        """
        assert seq_len <= self.max_seq_len, f"Input sequence length {seq_len} exceeds maximum {self.max_seq_len}"
        # getattr to get the buffer tensors
        cos = getattr(self, "cos")[:seq_len, :]
        sin = getattr(self, "sin")[:seq_len, :]
        return cos, sin

    def forward(self, 
                x: Float[Tensor, "batch seq_len d_k"], 
                token_positions: Int[Tensor, "... seq_len"] | None = None) -> Float[Tensor, "batch seq_len d_k"]:
        """ Apply RoPE to the input tensor x.
        Args:
            x: Float[Tensor, "batch seq_len d_k"] Input tensor to apply RoPE to
            token_positions: Int[Tensor, "... seq_len"] Optional tensor with token positions to use
        Returns:
            Float[Tensor, "batch seq_len d_k"] Tensor after applying RoPE
        """
        batch, seq_len, d_k = x.shape

        # Check dimensions
        assert d_k == self.d_k, f"Input dimension {d_k} does not match initialized dimension {self.d_k}"

        # Get cached cos and sin values
        if token_positions is None:
            # Use sequential positions if no positions provided
            assert seq_len <= self.max_seq_len, f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}"
            cos, sin = self.get_cached_cos_sin(seq_len)
            # Expand to match batch dimension
            cos = repeat(cos, 'seq_len d -> batch seq_len d', batch=batch)
            sin = repeat(sin, 'seq_len d -> batch seq_len d', batch=batch)
        else:
            # Validate token positions
            assert token_positions.shape[-1] == seq_len, f"token_positions length {token_positions.shape[-1]} does not match sequence length {seq_len}"
            assert torch.max(token_positions) < self.max_seq_len, f"Position {torch.max(token_positions)} exceeds maximum {self.max_seq_len}"
            # Get the full cached tensors
            cos_full, sin_full = self.get_cached_cos_sin(self.max_seq_len)
            # Use token positions to index into cached tensors
            cos = cos_full[token_positions]
            sin = sin_full[token_positions]

        # Split x into even and odd parts
        # ... means as much : as possible for prior dimensions
        # x:y:z means x to z-1, step y
        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        # Apply RoPE transformation, not matrix multiplication!!!
        x_rotated_1 = x1 * cos - x2 * sin
        x_rotated_2 = x1 * sin + x2 * cos

        # Interleave the rotated parts back together
        x_rotated = torch.zeros_like(x)
        x_rotated[..., ::2] = x_rotated_1
        x_rotated[..., 1::2] = x_rotated_2

        return x_rotated

def softmax(x: Float[Tensor," ..."], dim: int) -> Float[Tensor, " ..."]:
    """ Numerically stable softmax implementation that prevents overflow/underflow issues.
    Args:
        x: Input tensor to apply softmax to
        dim: Dimension along which to apply softmax
    Returns:
        Tensor after applying softmax along the specified dimension
    """
    # Subtract the max for numerical stability
    # origin_type=x.dtype
    # x=x.to(torch.float64)
    x_max = torch.max(x, dim=dim, keepdim=True).values
    # temp_val=x-x_max
    # e_x = torch.exp(temp_val)
    e_x = torch.exp(x - x_max)
    sum_e_x = torch.sum(e_x, dim=dim, keepdim=True)
    result=(e_x / sum_e_x)
    # result=(e_x / sum_e_x).to(origin_type)
    return result

def scaled_dot_product_attention(
    Q: Float[Tensor, "... query d_k"],
    K: Float[Tensor, "... key d_k"],
    V: Float[Tensor, "... value d_v"],
    mask: Bool[Tensor, "... query key"] | None = None,
) -> Float[Tensor, "... query d_v"]:
    """ Compute scaled dot-product attention.
    Args:
        Q: Query tensor of shape (..., query_len, d_k)
        K: Key tensor of shape (..., key_len, d_k)
        V: Value tensor of shape (..., key_len, d_v)
        mask: Optional boolean mask tensor of shape (..., query_len, key_len)
              where True indicates positions to be masked (not attended to).
    Returns:
        Tensor of shape (..., query_len, d_v) after applying attention
    """
    # Get the key dimension
    d_k = Q.shape[-1]
    
    # Calculate attention scores (Q @ K^T) / sqrt(d_k)
    # einsum handles the batch and sequence dimensions automatically
    # '...qd,...kd->...qk' means:
    # - ... matches any number of batch/head dimensions
    # - q is the query sequence length
    # - k is the key sequence length
    # - d is the embedding dimension (d_k)
    # print(Q.shape,K.shape,V.shape)
    # scores=Q@K.transpose(-2,-1)
    # scores = scores / (d_k ** 0.5)
    scores = torch.einsum('...qd,...kd->...qk',Q,K)
    scores = scores / (d_k ** 0.5)
    
    # Apply mask if provided (True values will be masked)
    if mask is not None:
        # print(mask)
        mask=mask.logical_not()
        scores = scores.masked_fill(mask, float('-inf'))
    
    # Apply softmax to get attention weights
    attn_weights = softmax(scores, dim=-1)  # (..., query_len, key_len)
    
    # Calculate weighted sum of values
    # '...qk,...kv->...qv' means:
    # - q is query sequence length
    # - k is key sequence length
    # - v is value embedding dimension
    # output=attn_weights@V
    output = torch.einsum('...qk,...kd->...qd',attn_weights,V)
    
    return output

class MultiHeadSelfAttention(nn.Module):
    """Multi-head self attention module with causal masking.

    Attributes:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_head: Dimension per head (d_model // num_heads)
        q_proj: Query projection layer
        k_proj: Key projection layer
        v_proj: Value projection layer
        o_proj: Output projection layer
    """
    def __init__(self,
                d_model: int,
                num_heads: int,
                device: torch.device = torch.device('cpu'),
                dtype: torch.dtype = torch.float32):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Initialize projection layers
        self.q_proj = LinearModule(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = LinearModule(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = LinearModule(d_model, d_model, device=device, dtype=dtype)
        self.o_proj = LinearModule(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, "... sequence_length d_model"]) -> Float[Tensor, "... sequence_length d_model"]:
        """
        Compute multi-head self-attention with causal masking.

        Args:
            x: Input tensor of shape (..., sequence_length, d_model)

        Returns:
            Output tensor of shape (..., sequence_length, d_model)
        """
        # Get shape info
        batch_size, seq_len, _ = x.shape

        # Linear projections
        Q = self.q_proj(x)  # (batch, seq, d_model)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape into heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_head).permute(0, 2, 1, 3)  # (batch, head, seq, d_head)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_head).permute(0, 2, 1, 3)

        # Create causal mask
        causal_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).triu(diagonal=1).logical_not()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
        causal_mask = causal_mask.expand(batch_size, self.num_heads, seq_len, seq_len)

        # Compute attention
        attn_output = scaled_dot_product_attention(Q, K, V, mask=causal_mask)  # (batch, head, seq, d_head)

        # Reshape and project to output
        attn_output = attn_output.permute(0, 2, 1, 3)  # (batch, seq, head, d_head)
        attn_output = attn_output.flatten(-2, -1)  # (batch, seq, d_model)
        
        return self.o_proj(attn_output)
    
class MultiHeadSelfAttentionWithRoPE(MultiHeadSelfAttention):
    """Multi-head self attention with Rotary Position Embedding (RoPE).
    
    Extends MultiHeadSelfAttention to add RoPE functionality. RoPE is applied to queries and keys
    after their projection but before attention computation.

    Attributes:
        rope: Rotary Position Embedding module
        max_seq_len: Maximum sequence length for RoPE computation
        theta: RoPE parameter for frequency calculation
    """
    def __init__(self,
                d_model: int,
                num_heads: int,
                max_seq_len: int,
                theta: float,
                device: torch.device = torch.device('cpu'),
                dtype: torch.dtype = torch.float32):
        super().__init__(d_model, num_heads, device=device, dtype=dtype)
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.rope = RoPE(theta=theta, d_k=self.d_head, max_seq_len=max_seq_len, 
                        device=device, dtype=dtype)

    def forward(self, 
                x: Float[Tensor, "... sequence_length d_model"],
                token_positions: Int[Tensor, "... sequence_length"] | None = None
               ) -> Float[Tensor, "... sequence_length d_model"]:
        """
        Compute multi-head self-attention with RoPE and causal masking.

        Args:
            x: Input tensor of shape (..., sequence_length, d_model)
            token_positions: Optional tensor with token positions for RoPE

        Returns:
            Output tensor of shape (..., sequence_length, d_model)
        """
        # Get shape info
        batch_size, seq_len, _ = x.shape

        # Linear projections
        Q = self.q_proj(x)  # (batch, seq, d_model)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape Q, K, V for multihead attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_head)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_head)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_head)

        # Prepare Q, K for RoPE
        # (batch, seq, num_heads, d_head) -> (batch * num_heads, seq, d_head)
        Q = Q.transpose(1, 2).reshape(batch_size * self.num_heads, seq_len, self.d_head)
        K = K.transpose(1, 2).reshape(batch_size * self.num_heads, seq_len, self.d_head)
        V = V.transpose(1, 2)  # Just transpose V: (batch, num_heads, seq, d_head)

        # Apply RoPE to Q and K
        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)

        # Reshape back to multihead format
        Q = Q.view(batch_size, self.num_heads, seq_len, self.d_head)
        K = K.view(batch_size, self.num_heads, seq_len, self.d_head)

        # Create causal mask
        causal_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).triu(diagonal=1).logical_not()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
        causal_mask = causal_mask.expand(batch_size, self.num_heads, seq_len, seq_len)

        # Compute attention
        attn_output = scaled_dot_product_attention(Q, K, V, mask=causal_mask)

        # Reshape and project to output
        attn_output = attn_output.transpose(1, 2)  # (batch, seq, head, d_head)
        attn_output = attn_output.flatten(-2, -1)  # (batch, seq, d_model)

        return self.o_proj(attn_output)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with standard architecture.

    Detailed architecture:
    LayerNorm -> Self Attention -> Skip Connection -> LayerNorm -> FFN -> Skip Connection

    Attributes:
        attention: Multi-head self attention layer with RoPE
        ln1: First layer normalization
        ln2: Second layer normalization
        ff: Feed-forward network with SwiGLU activation
    """
    def __init__(self,
                d_model: int,
                num_heads: int,
                d_ff: int,
                max_seq_len: int,
                theta: float,
                device: torch.device=torch.device('cpu'),
                dtype: torch.dtype=torch.float32):
        super().__init__()
        self.attention = MultiHeadSelfAttentionWithRoPE(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype
        )
        self.ln1 = RMSNormModule(d_model=d_model, device=device, dtype=dtype)
        self.ln2 = RMSNormModule(d_model=d_model, device=device, dtype=dtype)
        self.ff = SwiGLUModule(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def load_from_weights(self, weights: dict[str, Tensor]) -> None:
        """Load weights from a state dictionary.

        Args:
            weights: Dictionary containing the weights for this transformer block
        """
        self.ln1.load_state_dict({'weight': weights['ln1.weight']})
        self.ln2.load_state_dict({'weight': weights['ln2.weight']})
        self.ff.load_state_dict({
            'weight1': weights['ffn.w1.weight'],
            'weight2': weights['ffn.w2.weight'],
            'weight3': weights['ffn.w3.weight']
        })
        self.attention.q_proj.load_state_dict({'weight': weights['attn.q_proj.weight']})
        self.attention.k_proj.load_state_dict({'weight': weights['attn.k_proj.weight']})
        self.attention.v_proj.load_state_dict({'weight': weights['attn.v_proj.weight']})
        self.attention.o_proj.load_state_dict({'weight': weights['attn.output_proj.weight']})

    def forward(self, x: Float[Tensor, "batch sequence_length d_model"]) -> Float[Tensor, "batch sequence_length d_model"]:
        # First sub-block: Multi-head self attention with Add & Norm
        ln1_out = self.ln1(x)
        attn_out = self.attention(ln1_out)
        res1_out = x + attn_out

        # Second sub-block: Feed forward network with Add & Norm
        ln2_out = self.ln2(res1_out)
        ff_out = self.ff(ln2_out)
        res2_out = res1_out + ff_out

        return res2_out

class TransformerLM(nn.Module):
    """Transformer language model with RoPE positional embeddings.

    Implements a transformer-based language model using Rotary Position Embeddings (RoPE)
    for position encoding. The architecture follows the standard transformer design with
    pre-norm blocks and RMSNorm for layer normalization.

    Architecture:
        1. Token embeddings lookup table
        2. Stack of transformer blocks with:
           - Multi-head self attention with RoPE
           - Feed-forward network with SwiGLU activation
        3. Final layer normalization
        4. Linear projection to vocabulary size

    Args:
        vocab_size: Number of tokens in the vocabulary
        context_length: Maximum sequence length the model can process
        d_model: Size of the model's hidden dimensions
        num_layers: Number of transformer blocks in the stack
        num_heads: Number of attention heads in each block
        d_ff: Dimension of the feed-forward network's hidden layer
        rope_theta: Base value for RoPE frequency calculations
        device: Computation device (default: 'cpu')
        dtype: Model's data type (default: torch.float32)

    Example:
        >>> model = TransformerLM(
        ...     vocab_size=50257,
        ...     context_length=1024,
        ...     d_model=768,
        ...     num_layers=12,
        ...     num_heads=12,
        ...     d_ff=3072,
        ...     rope_theta=10000.0
        ... )
        >>> tokens = torch.randint(0, vocab_size, (1, 512))  # Batch of 1, length 512
        >>> logits = model(tokens)  # Shape: (1, 512, 50257)
    """
    def __init__(self,
                vocab_size: int,
                context_length: int,
                d_model: int,
                num_layers: int,
                num_heads: int,
                d_ff: int,
                rope_theta: float,
                device: torch.device=torch.device('cpu'),
                dtype: torch.dtype=torch.float32):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        # Token embeddings
        self.embed = EmbeddingModule(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                device=device,
                dtype=dtype
            )
            for _ in range(num_layers)
        ])

        # Final layer normalization
        self.ln_final = RMSNormModule(d_model=d_model, device=device, dtype=dtype)

        # Output projection to vocabulary
        self.lm_head = LinearModule(in_features=d_model, out_features=vocab_size, device=device, dtype=dtype)

    def load_from_weights(self, weights: dict[str, Tensor]) -> None:
        """Load weights from a state dictionary.

        Args:
            weights: Dictionary containing model weights
        """
        # Load token embeddings
        self.embed.load_state_dict({'weight': weights['token_embeddings.weight']})
        
        # Load transformer layers
        for i, layer in enumerate(self.layers):
            layer_prefix = f'layers.{i}.'
            layer_weights = {
                k.replace(layer_prefix, ''): v 
                for k, v in weights.items() 
                if k.startswith(layer_prefix)
            }
            layer.load_from_weights(layer_weights)

        # Load final layer norm and output projection
        self.ln_final.load_state_dict({'weight': weights['ln_final.weight']})
        self.lm_head.load_state_dict({'weight': weights['lm_head.weight']})

    def forward(self, in_indices: Int[Tensor, "batch_size sequence_length"]) -> Float[Tensor, "batch_size sequence_length vocab_size"]:
        """Run the transformer language model forward pass.

        Args:
            in_indices: Input token indices of shape (batch_size, sequence_length)

        Returns:
            Next-token logits of shape (batch_size, sequence_length, vocab_size)
        """
        # Token embeddings
        hidden = self.embed(in_indices)

        # Process through transformer layers
        for layer in self.layers:
            hidden = layer(hidden)

        # Final normalization and projection
        hidden = self.ln_final(hidden)
        logits = self.lm_head(hidden)

        return logits


def log_softmax(x: Float[Tensor," ..."], dim: int) -> Float[Tensor, " ..."]:
    """ Numerically stable log-softmax implementation.

    Args:
        x (Float[Tensor, "..."]): Input tensor
        dim (int): Dimension along which to apply log-softmax

    Returns:
        Float[Tensor, "..."]: Tensor after applying log-softmax
    """
    x_max = torch.max(x,dim=dim, keepdim=True).values
    log_sum_exp = torch.log(torch.sum(torch.exp(x - x_max), dim=dim, keepdim=True)) 
    result = (x-x_max) - log_sum_exp  
    return result

def cross_entropy(
    predicted: Float[Tensor, " batch_size vocab_size"],
    target: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Compute cross-entropy loss between logits and target indices.

    Args:
        predicted (Float[Tensor, "batch_size vocab_size"]): Model output predictions
        target (Int[Tensor, "batch_size"]): real value

    Returns:
        Float[Tensor, ""]: cross-entropy loss
    """
    #TODO: optimize to get rid of the flatten steps, AI version don't need that somehow
    # Flatten logits and targets for loss computation
    # vocab_size = predicted.shape[-1]
    vocab_size = predicted.shape[-1]
    predicted_flat = predicted.view(-1, vocab_size)  # (batch_size , vocab_size)
    flatten_dim_0 = predicted_flat.shape[0]  #it's actually batch_size*sequence_length

    # following may not be needed since softmax already did the max deduction
    # predicted_flat=predicted_flat-predicted_flat.max(dim=1, keepdim=True).values 
    # print(f"predicted_flat={predicted_flat}")
    targets_1d=target.view(-1)  # (batch_size,)
    targets_flat=torch.zeros_like(predicted_flat)
    targets_flat[torch.arange(flatten_dim_0), targets_1d]=1
    # print(targets_flat)

    # Compute log probabilities
    # predicted_log_p1=torch.log_softmax(predicted_flat, dim=-1)
    predicted_log_p=log_softmax(predicted_flat, dim=-1)
    # predicted_log_p=torch.log(softmax(predicted_flat, dim=-1))  # (batch_size , vocab_size)
    # print(f"==========predicted_log_p1:{predicted_log_p1}============")
    # print(f"==========predicted_log_p:{predicted_log_p}============")

    # predicted_log_p = torch.log_softmax(predicted_flat, dim=-1)  # (batch_size , vocab_size)
    # print(predicted_log_p)

    # Gather log probabilities of the target indices
    # target_log_p = predicted_log_p[torch.arange(batch_size), targets_flat]  # (batch_size, vocab_size)
    target_log_p = (predicted_log_p*targets_flat)
    # print(f"==========target_log_p:{target_log_p}============")
    target_log_p = -target_log_p.sum(dim=-1)
    # print(f"==========target_log_p:{target_log_p}============")

    # Compute negative log likelihood loss
    # loss = -target_log_p/target_log_p.sum(dim=-1)  # Scalar
    loss=target_log_p.mean()  

    return loss

class AdamW(torch.optim.Optimizer):
    """Implements AdamW optimizer (Adam with decoupled weight decay).
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for computing running averages of gradient and its square (default: (0.9, 0.999))
        eps: Term added to denominator to improve numerical stability (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0.01)
        correct_bias: Whether to correct bias in Adam moments (default: True)
    """
    def __init__(
        self, 
        params, 
        lr=1e-3, 
        betas=(0.9, 0.999), 
        eps=1e-8,
        weight_decay=0.01,
        correct_bias=True
    ):
        defaults = dict(
            lr=lr, 
            betas=betas, 
            eps=eps,
            weight_decay=weight_decay,
            correct_bias=correct_bias
        )
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        """Performs a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None if closure is None else closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Get parameters
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                step = state['step']
                
                # Bias correction
                if group['correct_bias']:
                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step
                    step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                else:
                    step_size = group['lr']

                # Apply weight decay (decoupled from gradient update)
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Compute ratio for update
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                
                # Update parameters
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

def learning_rate_schedule(it: int,
                           max_learning_rate: float,
                           min_learning_rate: float,
                           warmup_iters: int,
                           cosine_cycle_iters: int) -> float: 
    """ Learning rate schedule with linear warmup and cosine decay.
    Args:
        t (int): Current training step
        alpha_max (float): Maximum learning rate after warmup
        alpha_min (float): Minimum learning rate at the end of training
        Tw (int): Number of warmup steps
        Tc (int): Total number of training steps
    Returns:
        float: Learning rate at step t  
    """
    if it < warmup_iters:
        # Linear warmup
        return max_learning_rate * (it / warmup_iters)
    elif it <= cosine_cycle_iters:
        # Cosine decay
        decay_steps = it - warmup_iters
        total_decay_steps = cosine_cycle_iters - warmup_iters
        cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_steps / total_decay_steps))
        return min_learning_rate + (max_learning_rate - min_learning_rate) * cosine_decay
    else:
        # After Tc, keep learning rate at alpha_min
        return min_learning_rate  
    
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    """ Clips gradients of the given parameters to a maximum L2 norm.
        parameters (Iterable[torch.nn.Parameter]): An iterable of model parameters whose gradients will be clipped.
        max_l2_norm (float): The maximum allowed L2 norm for the gradients.
    Returns:
        None
    """
    # print(f"Clipping gradients with max_l2_norm={max_l2_norm}")        
    total_norm:float =0.0
    for parameter in parameters:
        # print(f"parameter={parameter}")
        # print(f"parameter.grad={parameter.grad}")
        if parameter.grad is not None:
            grad_norm = parameter.grad.data.norm(2)
            total_norm += grad_norm ** 2
    total_norm = total_norm ** 0.5

    for parameter in parameters:
        if parameter.grad is not None:
            # Clip gradients in-place to the specified max L2 norm
            grad_norm = parameter.grad.data.norm(2)
            if grad_norm > max_l2_norm:
                parameter.grad.data = parameter.grad.data * (max_l2_norm / (total_norm+eps))
            
