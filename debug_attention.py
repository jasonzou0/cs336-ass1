#!/usr/bin/env python3

import torch
import torch.nn.functional as F
import numpy as np

def compare_with_pytorch():
    # Set up test data
    batch_size, seq_len, d_model, num_heads = 1, 4, 64, 4
    d_k = d_model // num_heads
    
    # Create test input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Create weight matrices
    q_weight = torch.randn(d_model, d_model)
    k_weight = torch.randn(d_model, d_model) 
    v_weight = torch.randn(d_model, d_model)
    o_weight = torch.randn(d_model, d_model)
    
    # My implementation
    def my_multihead_attention(x, q_w, k_w, v_w, o_w):
        # Project
        Q = x @ q_w.T
        K = x @ k_w.T  
        V = x @ v_w.T
        
        # Reshape for heads
        Q = Q.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
        
        # Attention
        scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
        
        # Try with and without causal mask
        print("Testing with causal mask:")
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores_masked = scores.masked_fill(mask, -float('inf'))
        attn_weights = F.softmax(scores_masked, dim=-1)
        out_masked = attn_weights @ V
        out_masked = out_masked.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        out_masked = out_masked @ o_w.T
        
        print("Testing without causal mask:")
        attn_weights = F.softmax(scores, dim=-1)
        out_unmasked = attn_weights @ V
        out_unmasked = out_unmasked.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        out_unmasked = out_unmasked @ o_w.T
        
        return out_masked, out_unmasked
    
    # PyTorch implementation
    def pytorch_multihead_attention(x, q_w, k_w, v_w, o_w):
        mha = torch.nn.MultiheadAttention(d_model, num_heads, bias=False, batch_first=True)
        
        # Set weights manually
        with torch.no_grad():
            mha.in_proj_weight.copy_(torch.cat([q_w, k_w, v_w], dim=0))
            mha.out_proj.weight.copy_(o_w)
        
        # Without causal mask
        out_no_mask, _ = mha(x, x, x, need_weights=False)
        
        # With causal mask  
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        out_causal, _ = mha(x, x, x, attn_mask=causal_mask, need_weights=False)
        
        return out_no_mask, out_causal
    
    # Compare
    my_masked, my_unmasked = my_multihead_attention(x, q_weight, k_weight, v_weight, o_weight)
    pt_no_mask, pt_causal = pytorch_multihead_attention(x, q_weight, k_weight, v_weight, o_weight)
    
    print(f"My masked vs PyTorch causal: {torch.allclose(my_masked, pt_causal, atol=1e-5)}")
    print(f"My unmasked vs PyTorch no mask: {torch.allclose(my_unmasked, pt_no_mask, atol=1e-5)}")
    
if __name__ == "__main__":
    compare_with_pytorch()
