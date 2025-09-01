from __future__ import annotations

import os
from typing import IO, Any, BinaryIO
from collections.abc import Iterable
from jaxtyping import Float, Int
import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from cs336_basics.my_tokenizer import BpeTokenizer

def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """

    if weights.ndim !=2 or weights.shape != (d_out,d_in):
        raise ValueError(f" weights mismatch")
    if in_features.shape[-1] != d_in:
        raise ValueError(f" in_features shape error")
    # Ensure device & dtype alignment
    if weights.device != in_features.device or weights.dtype != in_features.dtype:
        weights = weights.to(device=in_features.device, dtype=in_features.dtype)

    # Core: y = x @ W^T (no bias)
    # Works for any leading batch dims "..."
    return in_features @ weights.transpose(0, 1)
    # Equivalent alternatives:
    # return torch.einsum("...i,oi->...o", in_features, weights)
    # return torch.nn.functional.linear(in_features, weights, bias=None)

def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """

        # Basic shape checks
    if weights.ndim != 2 or tuple(weights.shape) != (vocab_size, d_model):
        raise ValueError(f"weights must be [{vocab_size}, {d_model}], got {tuple(weights.shape)}")

    # Embedding expects int64 (long) indices; keep everything on the same device
    token_ids = token_ids.to(device=weights.device, dtype=torch.long)

    # Optional: bounds check to catch OOV IDs early
    if torch.any(token_ids < 0) or torch.any(token_ids >= vocab_size):
        bad = token_ids[(token_ids < 0) | (token_ids >= vocab_size)]
        raise IndexError(f"Token id(s) out of range [0, {vocab_size}): {bad[:10].tolist()}...")

    # Core: gather rows from the embedding matrix.
    # Two equivalent ways:

    # 1) Simple advanced indexing (most readable):
    out = weights[token_ids]                 # shape: [..., d_model]

    # 2) Or via index_select on flattened IDs (explicit):
    # flat = token_ids.reshape(-1)
    # out = weights.index_select(0, flat).reshape(*token_ids.shape, d_model)

    return out


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    if w1_weight.shape != (d_ff, d_model):
        raise ValueError(f"w1_weight must be [{d_ff}, {d_model}], got {tuple(w1_weight.shape)}")
    if w3_weight.shape != (d_ff, d_model):
        raise ValueError(f"w3_weight must be [{d_ff}, {d_model}], got {tuple(w3_weight.shape)}")
    if w2_weight.shape != (d_model, d_ff):
        raise ValueError(f"w2_weight must be [{d_model}, {d_ff}], got {tuple(w2_weight.shape)}")
    if in_features.shape[-1] != d_model:
        raise ValueError(f"in_features last dim must be d_model={d_model}, got {in_features.shape[-1]}")

    # ---- align device & dtype to inputs ----
    dev, dt = in_features.device, in_features.dtype
    w1 = w1_weight.to(dev, dt)
    w2 = w2_weight.to(dev, dt)
    w3 = w3_weight.to(dev, dt)

    x = in_features
    a = x @ w1.transpose(0, 1)      # [..., d_ff]
    b = x @ w3.transpose(0, 1)      # [..., d_ff]
    g = torch.nn.functional.silu(a) * b               # SwiGLU gating
    y = g @ w2.transpose(0, 1)      # [..., d_model]
    return y


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """

    # Attention(Q, K, V) = softmax( (Q • Kᵀ) / √dₖ ) • V
    # Get the dimension of the key vectors for scaling
    # the dimension of the key vectors (dₖ).
    d_k = Q.shape[-1]
    
    # Compute attention scores: Q @ K^T
    # Shape: (..., queries, keys)
    # Dot-Product (Q • Kᵀ):
    scores = Q @ K.transpose(-2, -1)
    
    # Scale by sqrt(d_k) to prevent softmax saturation
    # We divide the scores by the square root of the dimension of the key vectors (dₖ).
    scores = scores / (d_k ** 0.5)
    
    # Apply mask if provided
    if mask is not None:
        # Use a large negative value where mask is 0 (or False)
        # This makes softmax output ~0 for these positions
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Apply softmax to get attention weights
    # Shape: (..., queries, keys)
    attention_weights = torch.softmax(scores, dim=-1)
    
    # Apply attention weights to values: A @ V
    # Shape: (..., queries, d_v)
    output = attention_weights @ V
    
    return output


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.

多头自注意力 是Transformer的核心。

自注意力 ：模型的查询、键、值都来自自身的上一层输出，让序列内部各个部分相互关联。

多头：并行运行多个独立的注意力机制（头），每个头关注信息的不同方面。

工作原理：

拆分：将输入通过不同的线性变换，为每个头生成独立的Q, K, V。

并行注意力：每个头独立计算注意力，专注于不同类型的模式（如语法、指代、语义）。

合并：将所有头的输出拼接起来。

最终投影：通过一个线性层融合所有头的信息。

为什么强大？
它像一个专家团队一起分析句子。有的专家专攻指代（解决“它”指代什么），有的专攻语法结构，有的专攻逻辑关系。最后将所有人的见解汇总，得到最全面的理解。这种设计让模型能够同时捕捉多种复杂的语言现象，是其强大能力的根本原因。        
    """
    # Get dimensions
    d_k = d_model // num_heads  # dimension per head
    batch_dims = in_features.shape[:-2]
    seq_len = in_features.shape[-2]
    d_in = in_features.shape[-1]
    
    # Ensure device & dtype alignment
    device, dtype = in_features.device, in_features.dtype
    q_proj_weight = q_proj_weight.to(device=device, dtype=dtype)
    k_proj_weight = k_proj_weight.to(device=device, dtype=dtype)
    v_proj_weight = v_proj_weight.to(device=device, dtype=dtype)
    o_proj_weight = o_proj_weight.to(device=device, dtype=dtype)
    
    # Project to Q, K, V using batched matrix multiplication
    # The weights are concatenated for all heads: shape [d_model, d_model]
    # Shape: (..., seq_len, d_model)
    Q = in_features @ q_proj_weight.transpose(-2, -1)
    K = in_features @ k_proj_weight.transpose(-2, -1)
    V = in_features @ v_proj_weight.transpose(-2, -1)
    
    # Reshape and split into multiple heads
    # Shape: (..., seq_len, d_model) -> (..., seq_len, num_heads, d_k) -> (..., num_heads, seq_len, d_k)
    Q = Q.view(*batch_dims, seq_len, num_heads, d_k).transpose(-3, -2)
    K = K.view(*batch_dims, seq_len, num_heads, d_k).transpose(-3, -2)
    V = V.view(*batch_dims, seq_len, num_heads, d_k).transpose(-3, -2)
    
    # Apply scaled dot-product attention for each head
    # For language models, we need a causal mask to prevent attending to future tokens
    # Shape: (..., num_heads, seq_len, d_k)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    attn_output = run_scaled_dot_product_attention(Q, K, V, mask=~causal_mask)
    
    # Concatenate heads: (..., num_heads, seq_len, d_k) -> (..., seq_len, d_model)
    attn_output = attn_output.transpose(-3, -2).contiguous().view(*batch_dims, seq_len, d_model)
    
    # Apply output projection
    # Shape: (..., seq_len, d_model)
    output = attn_output @ o_proj_weight.transpose(-2, -1)
    
    return output


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    # Get dimensions
    d_k = d_model // num_heads  # dimension per head
    batch_dims = in_features.shape[:-2]
    seq_len = in_features.shape[-2]
    d_in = in_features.shape[-1]
    
    # Ensure device & dtype alignment
    device, dtype = in_features.device, in_features.dtype
    q_proj_weight = q_proj_weight.to(device=device, dtype=dtype)
    k_proj_weight = k_proj_weight.to(device=device, dtype=dtype)
    v_proj_weight = v_proj_weight.to(device=device, dtype=dtype)
    o_proj_weight = o_proj_weight.to(device=device, dtype=dtype)
    
    # Generate token positions if not provided
    if token_positions is None:
        # Default to sequential positions [0, 1, 2, ..., seq_len-1]
        token_positions = torch.arange(seq_len, device=device, dtype=torch.long)
        # Expand to match batch dimensions
        for _ in batch_dims:
            token_positions = token_positions.unsqueeze(0)
        token_positions = token_positions.expand(*batch_dims, seq_len)
    
    # Project to Q, K, V using batched matrix multiplication
    # The weights are concatenated for all heads: shape [d_model, d_model]
    # Shape: (..., seq_len, d_model)
    Q = in_features @ q_proj_weight.transpose(-2, -1)
    K = in_features @ k_proj_weight.transpose(-2, -1)
    V = in_features @ v_proj_weight.transpose(-2, -1)
    
    # Reshape and split into multiple heads
    # Shape: (..., seq_len, d_model) -> (..., seq_len, num_heads, d_k) -> (..., num_heads, seq_len, d_k)
    Q = Q.view(*batch_dims, seq_len, num_heads, d_k).transpose(-3, -2)
    K = K.view(*batch_dims, seq_len, num_heads, d_k).transpose(-3, -2)
    V = V.view(*batch_dims, seq_len, num_heads, d_k).transpose(-3, -2)
    
    # Apply RoPE to Q and K (but not V)
    # We need to apply RoPE to each head separately
    # Shape after transpose: (..., num_heads, seq_len, d_k)
    Q_roped = torch.zeros_like(Q)
    K_roped = torch.zeros_like(K)
    
    for h in range(num_heads):
        # Extract Q and K for this head: (..., seq_len, d_k)
        Q_h = Q[..., h, :, :]
        K_h = K[..., h, :, :]
        
        # Apply RoPE (d_k is the embedding dimension for each head)
        Q_roped[..., h, :, :] = run_rope(d_k, theta, max_seq_len, Q_h, token_positions)
        K_roped[..., h, :, :] = run_rope(d_k, theta, max_seq_len, K_h, token_positions)
    
    # Apply scaled dot-product attention for each head with causal mask
    # For language models, we need a causal mask to prevent attending to future tokens
    # Shape: (..., num_heads, seq_len, d_k)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    attn_output = run_scaled_dot_product_attention(Q_roped, K_roped, V, mask=~causal_mask)
    
    # Concatenate heads: (..., num_heads, seq_len, d_k) -> (..., seq_len, d_model)
    attn_output = attn_output.transpose(-3, -2).contiguous().view(*batch_dims, seq_len, d_model)
    
    # Apply output projection
    # Shape: (..., seq_len, d_model)
    output = attn_output @ o_proj_weight.transpose(-2, -1)
    
    return output


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    device = in_query_or_key.device
    dtype = in_query_or_key.dtype
    
    # Get the shape
    seq_len = in_query_or_key.shape[-2]
    
    # Create frequency for each dimension pair
    # For dimension i, the frequency is 1 / (theta^(2i/d_k))
    freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device, dtype=dtype) / d_k))
    
    # Create position encodings
    # Shape: (seq_len, d_k//2)
    t = token_positions.to(device=device, dtype=dtype)  # Convert positions to float
    freqs = t.unsqueeze(-1) * freqs.unsqueeze(-2)  # Broadcasting to get (batch_or_seq, seq_len, d_k//2)
    
    # Create cos and sin for rotation
    cos_freqs = torch.cos(freqs)  # Shape: (..., seq_len, d_k//2)
    sin_freqs = torch.sin(freqs)  # Shape: (..., seq_len, d_k//2)
    
    # Split input into even and odd dimensions
    # Shape: (..., seq_len, d_k//2)
    x_even = in_query_or_key[..., 0::2]  # Even indices: 0, 2, 4, ...
    x_odd = in_query_or_key[..., 1::2]   # Odd indices: 1, 3, 5, ...
    
    # Apply rotation: R * [x_even; x_odd] = [cos*x_even - sin*x_odd; sin*x_even + cos*x_odd]
    rotated_even = cos_freqs * x_even - sin_freqs * x_odd
    rotated_odd = sin_freqs * x_even + cos_freqs * x_odd
    
    # Interleave back to original shape
    output = torch.zeros_like(in_query_or_key)
    output[..., 0::2] = rotated_even
    output[..., 1::2] = rotated_odd
    
    return output


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    raise NotImplementedError


def run_transformer_lm(
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
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    raise NotImplementedError


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    raise NotImplementedError


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    raise NotImplementedError


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.

    """

    N = int(len(dataset))
    if N < context_length +1:
        raise ValueError(f"Dataset too short")
    starts = np.random.randint(0, N - context_length, size = batch_size)
    x_np= np.stack([dataset[s : s+context_length] for s in starts])
    y_np = np.stack([dataset[s+1: s+1+context_length] for s in starts])
    x = torch.as_tensor(x_np, dtype= torch.long, device = device)
    y = torch.as_tensor(y_np, dtype= torch.long, device  = device)
    return x,y



def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    raise NotImplementedError


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    raise NotImplementedError


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    raise NotImplementedError


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    raise NotImplementedError


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    raise NotImplementedError


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    raise NotImplementedError


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    raise NotImplementedError


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.

    """
    return BpeTokenizer(vocab, merges, special_tokens or [])



def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    raise NotImplementedError
