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
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
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
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=torch.float32):
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

    def forward(self, x: Float[Tensor, "batch seq_len d_k"], token_positions: Int[Tensor, "... seq_len"] | None = None) -> Float[Tensor, "batch seq_len d_k"]:
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
        # TODO: AI code, how does it work? Every 2 items picked from 0 and from 1?
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

def multihead_self_attention(
    d_model: int,
    num_heads: int, 
    q_proj_weight: Float[Tensor, "d_k d_in"],
    k_proj_weight: Float[Tensor, "d_k d_in"],
    v_proj_weight: Float[Tensor, "d_v d_in"],
    o_proj_weight: Float[Tensor, "d_model d_v"],
    in_features: Float[Tensor, "... sequence_length d_in"],
) -> Float[Tensor, "... sequence_length d_out"]:
    """
    Compute multi-head self-attention with optimized batched implementation.

    Args:
        d_model: Dimensionality of the model input/output
        num_heads: Number of attention heads
        q_proj_weight: Query projection weights (d_k x d_in) 
        k_proj_weight: Key projection weights (d_k x d_in)
        v_proj_weight: Value projection weights (d_v x d_in)
        o_proj_weight: Output projection weights (d_model x d_v)
        in_features: Input tensor

    Returns:
        Tensor with the same shape as input but last dimension is d_out
    """
    # Get shape info and validate dimensions
    batch_size, seq_len, d_in = in_features.shape
    d_head = d_model // num_heads
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

    # Linear projections in batch for all heads at once
    Q = in_features @ q_proj_weight.T  # (batch, seq, d_model)
    K = in_features @ k_proj_weight.T
    V = in_features @ v_proj_weight.T

    # Q = in_features.transpose(-2,-1) @ q_proj_weight  # (batch, seq, d_model)
    # K = in_features.transpose(-2,-1) @ k_proj_weight
    # V = in_features.transpose(-2,-1) @ v_proj_weight


    # Reshape Q, K, V into heads
    Q = Q.view(batch_size, seq_len, num_heads, d_head).permute(0, 2, 1, 3)  # (batch, head, seq, d_head)
    K = K.view(batch_size, seq_len, num_heads, d_head).permute(0, 2, 1, 3)
    V = V.view(batch_size, seq_len, num_heads, d_head).permute(0, 2, 1, 3)

    # Create causal mask to prevent attending to future tokens
    # Shape: (1, 1, seq_len, seq_len)
    causal_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=Q.device).triu(diagonal=1).logical_not()
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
    causal_mask = causal_mask.expand(batch_size, num_heads, seq_len, seq_len)

    # Compute attention
    attn_output = scaled_dot_product_attention(Q, K, V, mask=causal_mask)  # (batch, head, seq, d_head)

    # Reshape back and project to output dimension
    attn_output = attn_output.permute(0, 2, 1, 3) # (batch, seq, head, d_v)
    attn_output=attn_output.flatten(-2,-1) #(batch,seq,d_v*head)     
    # attn_output = attn_output.view(batch_size, seq_len, d_model)  # (batch, seq, d_model)

    # Final output projection 
    output = attn_output @ o_proj_weight.T  # (batch, seq, d_model)
    return output
    
def multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, "d_k d_in"],
    k_proj_weight: Float[Tensor, "d_k d_in"],
    v_proj_weight: Float[Tensor, "d_v d_in"],
    o_proj_weight: Float[Tensor, "d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """Same as multihead_self_attention but applies RoPE to Q and K after projection.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_v d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Input tensor
        token_positions (Int[Tensor, "... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, "... sequence_length d_out"]: Output tensor with same shape as input but d_out dimension
    """
    # Get shape info and validate dimensions
    batch_size, seq_len, d_in = in_features.shape
    d_head = d_model // num_heads
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

    # Linear projections in batch for all heads at once
    Q = in_features @ q_proj_weight.T  # (batch, seq, d_model)
    K = in_features @ k_proj_weight.T  # (batch, seq, d_model)
    V = in_features @ v_proj_weight.T  # (batch, seq, d_model)

    # Reshape Q, K, V for multihead attention:
    # (batch, seq, d_model) -> (batch, seq, num_heads, d_head)
    Q = Q.view(batch_size, seq_len, num_heads, d_head)
    K = K.view(batch_size, seq_len, num_heads, d_head)
    V = V.view(batch_size, seq_len, num_heads, d_head)

    # Apply RoPE to each head's queries and keys
    # Initialize ROPE module for the head dimension
    rope = RoPE(theta=theta, d_k=d_head, max_seq_len=max_seq_len, device=Q.device, dtype=Q.dtype)

    # Reshape to apply RoPE independently to each head
    # (batch, seq, num_heads, d_head) -> (batch * num_heads, seq, d_head)
    Q = Q.transpose(1, 2).reshape(batch_size * num_heads, seq_len, d_head)
    K = K.transpose(1, 2).reshape(batch_size * num_heads, seq_len, d_head)
    V = V.transpose(1, 2)  # Just need to transpose V to match Q,K shape: (batch, num_heads, seq, d_head)

    # Apply RoPE
    Q = rope(Q,token_positions)  # Apply rotary positional embeddings to queries
    K = rope(K,token_positions)  # Apply rotary positional embeddings to keys

    # Reshape back to multihead format
    # (batch * num_heads, seq, d_head) -> (batch, num_heads, seq, d_head)
    Q = Q.view(batch_size, num_heads, seq_len, d_head)
    K = K.view(batch_size, num_heads, seq_len, d_head)

    # Create causal mask
    causal_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=Q.device).triu(diagonal=1).logical_not()
    causal_mask = causal_mask.view(1, 1, seq_len, seq_len)
    causal_mask = causal_mask.expand(batch_size, num_heads, seq_len, seq_len)

    # Compute attention with RoPE-enhanced Q and K
    attn_output = scaled_dot_product_attention(Q, K, V, mask=causal_mask)  # (batch, num_heads, seq, d_head)

    # Reshape attention output and project to output dimension
    attn_output = attn_output.transpose(1, 2).flatten(-2,-1)  # (batch, seq, num_heads*d_head)

    # Final output projection
    output = attn_output @ o_proj_weight.T  # (batch, seq, d_model)
    return output


def transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, "batch sequence_length d_model"],
) -> Float[Tensor, "batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    Args:
        d_model (int): Dimensionality of the feedforward input/output
        num_heads (int): Number of attention heads
        d_ff (int): Feed-forward inner dimension
        max_seq_len (int): Maximum sequence length for RoPE
        theta (float): RoPE parameter
        weights (dict[str, Tensor]): State dict of a transformer block
        in_features (Float[Tensor, "batch sequence_length d_model"]): Input tensor

    Returns:
        Float[Tensor, "batch sequence_length d_model"]: Output after transformer block
    """

    # initialize modules
    # ln1 = RMSNormModule(d_model=d_model, device=in_features.device, dtype=in_features.dtype)
    # ln2 = RMSNormModule(d_model=d_model, device=in_features.device, dtype=in_features.dtype)
    # ff = SwiGLUModule(d_model=d_model, d_ff=d_ff, device=in_features.device, dtype=in_features.dtype)
    
    ln1 = RMSNormModule(d_model=d_model, device=in_features.device, dtype=torch.float32)
    ln2 = RMSNormModule(d_model=d_model, device=in_features.device, dtype=torch.float32)
    ff = SwiGLUModule(d_model=d_model, d_ff=d_ff, device=in_features.device, dtype=torch.float32)

    # load weights
    # TODO: use load_state_dict to do the weights loading???
    # TODO: trace dtype for every variable
    # TODO: side by side compare of outputs of each layer from 2 models
    ln1.weight.data = weights['ln1.weight']
    ln2.weight.data = weights['ln2.weight']
    ff.weight1.data = weights['ffn.w1.weight']
    ff.weight2.data = weights['ffn.w2.weight']
    ff.weight3.data = weights['ffn.w3.weight']

    # normalize 1
    ln1_out = ln1(in_features)

    # Multi-head attention with RoPE
    multihead_attention_out = multihead_self_attention_with_rope(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        theta=theta,
        q_proj_weight=weights['attn.q_proj.weight'],
        k_proj_weight=weights['attn.k_proj.weight'],
        v_proj_weight=weights['attn.v_proj.weight'],
        o_proj_weight=weights['attn.output_proj.weight'],
        in_features=ln1_out,
        token_positions=None 
    )
    
    # RESNET1
    res1_out = in_features + multihead_attention_out
    # return res1_out

    # normalize 2
    ln2_out = ln2(res1_out)
    
    # FFN
    ff_out = ff(ln2_out)
    
    # RESNET2
    res2_out = res1_out + ff_out

    return res2_out

def transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Run the full transformer language model forward pass.

    Args:
        vocab_size (int): Size of vocabulary
        context_length (int): Maximum sequence length
        d_model (int): Model dimension
        num_layers (int): Number of transformer layers
        num_heads (int): Number of attention heads per layer
        d_ff (int): Feed-forward inner dimension
        rope_theta (float): RoPE parameter
        weights (dict[str, Tensor]): Model weights dictionary
        in_indices (Int[Tensor, "batch_size sequence_length"]): Input token indices

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Next-token logits
    """
    # Token embeddings
    embed = EmbeddingModule(
        num_embeddings=vocab_size, 
        embedding_dim=d_model,
        device=in_indices.device,
        dtype=torch.float32
    )
    embed.weight.data = weights['token_embeddings.weight']
    hidden = embed(in_indices)

    # Process through all transformer layers
    for i in range(num_layers):
        layer_prefix = f'layers.{i}.'
        layer_weights = {
            k.replace(layer_prefix, ''): v 
            for k, v in weights.items() 
            if k.startswith(layer_prefix)
        }

        hidden = transformer_block(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            max_seq_len=context_length,
            theta=rope_theta,
            weights=layer_weights,
            in_features=hidden
        )

    # Final layer normalization
    ln_final = RMSNormModule(d_model=d_model, device=hidden.device, dtype=hidden.dtype)
    ln_final.weight.data = weights['ln_final.weight']
    hidden = ln_final(hidden)

    # Project to vocabulary
    lm_head = LinearModule(in_features=d_model, out_features=vocab_size, device=hidden.device, dtype=hidden.dtype)
    lm_head.weight.data = weights['lm_head.weight']
    logits = lm_head(hidden)

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
    batch_size = predicted_flat.shape[0]

    # following may not be needed since softmax already did the max deduction
    # predicted_flat=predicted_flat-predicted_flat.max(dim=1, keepdim=True).values 
    # print(f"predicted_flat={predicted_flat}")

    targets_flat=torch.zeros_like(predicted_flat)
    targets_flat[torch.arange(batch_size), target]=1
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


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]   # Get the learning rate.

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]         # Get state associated with p.
                t = state.get("t", 0)         # Get iteration number from the state, or initial value.
                grad = p.grad.data            # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad   # Update weight tensor in-place.
                state["t"] = t + 1            # Increment iteration number.
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

    def step(self):
        """Performs a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None

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