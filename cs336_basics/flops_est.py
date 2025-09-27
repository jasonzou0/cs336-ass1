# Calculate FLOPs for the last training run
# Model: d_model=256, num_layers=2, num_heads=8, batch_size=4, context_length=1024, iterations=5

import math

# Model parameters from the run
d_model = 256
num_layers = 2
num_heads = 8
d_ff = 3072
vocab_size = 50257
context_length = 1024
batch_size = 4
iterations = 5

print('=== FLOP Calculation for Last Training Run ===')
print(f'Model: d_model={d_model}, layers={num_layers}, heads={num_heads}')
print(f'Training: batch_size={batch_size}, context_length={context_length}, iterations={iterations}')
print()

# Calculate FLOPs per forward pass
def calculate_forward_flops(batch_size, seq_len, d_model, num_layers, num_heads, d_ff, vocab_size):
    flops = 0
    
    # 1. Token embedding lookup (no FLOPs, just indexing)
    
    # 2. For each transformer layer
    for layer in range(num_layers):
        # Pre-norm (RMSNorm): ~2 * B * T * d_model
        flops += 2 * batch_size * seq_len * d_model
        
        # Multi-head attention
        d_head = d_model // num_heads
        
        # Q, K, V projections: 3 * (B * T * d_model * d_model)
        flops += 3 * batch_size * seq_len * d_model * d_model
        
        # Attention scores: B * num_heads * T * T * d_head
        flops += batch_size * num_heads * seq_len * seq_len * d_head
        
        # Attention softmax: ~4 * B * num_heads * T * T (exp, sum, div, etc.)
        flops += 4 * batch_size * num_heads * seq_len * seq_len
        
        # Attention output: B * num_heads * T * T * d_head
        flops += batch_size * num_heads * seq_len * seq_len * d_head
        
        # Output projection: B * T * d_model * d_model
        flops += batch_size * seq_len * d_model * d_model
        
        # Second pre-norm (RMSNorm): ~2 * B * T * d_model
        flops += 2 * batch_size * seq_len * d_model
        
        # SwiGLU FFN:
        # W1 (gate): B * T * d_model * d_ff
        flops += batch_size * seq_len * d_model * d_ff
        # W3 (up): B * T * d_model * d_ff  
        flops += batch_size * seq_len * d_model * d_ff
        # SiLU activation: ~3 * B * T * d_ff (sigmoid + multiply)
        flops += 3 * batch_size * seq_len * d_ff
        # Element-wise multiply: B * T * d_ff
        flops += batch_size * seq_len * d_ff
        # W2 (down): B * T * d_ff * d_model
        flops += batch_size * seq_len * d_ff * d_model
    
    # 3. Final layer norm: ~2 * B * T * d_model
    flops += 2 * batch_size * seq_len * d_model
    
    # 4. Language model head: B * T * d_model * vocab_size
    flops += batch_size * seq_len * d_model * vocab_size
    
    return flops

# Calculate forward pass FLOPs
forward_flops = calculate_forward_flops(batch_size, context_length, d_model, num_layers, num_heads, d_ff, vocab_size)
print(f'Forward pass FLOPs: {forward_flops:,}')
print(f'Forward pass FLOPs: {forward_flops/1e9:.2f} GFLOPs')

# Backward pass is approximately 2x forward pass
backward_flops = 2 * forward_flops
total_flops_per_iteration = forward_flops + backward_flops

print(f'Backward pass FLOPs: {backward_flops:,}')
print(f'Backward pass FLOPs: {backward_flops/1e9:.2f} GFLOPs')
print(f'Total per iteration: {total_flops_per_iteration:,}')
print(f'Total per iteration: {total_flops_per_iteration/1e9:.2f} GFLOPs')

# Total for all iterations
total_training_flops = total_flops_per_iteration * iterations
print()
print(f'=== TOTAL FOR {iterations} ITERATIONS ===')
print(f'Total training FLOPs: {total_training_flops:,}')
print(f'Total training FLOPs: {total_training_flops/1e9:.2f} GFLOPs')
print(f'Total training FLOPs: {total_training_flops/1e12:.2f} TFLOPs')

# Add evaluation FLOPs (3 evaluations with 200 iterations each)
eval_iterations = 3 * 200  # 3 evaluations × 200 eval_iters each
eval_flops = forward_flops * eval_iterations
total_flops_with_eval = total_training_flops + eval_flops

print()
print(f'=== INCLUDING EVALUATION ===')
print(f'Evaluation FLOPs: {eval_flops:,}')
print(f'Evaluation FLOPs: {eval_flops/1e9:.2f} GFLOPs')
print(f'Total with eval: {total_flops_with_eval:,}')
print(f'Total with eval: {total_flops_with_eval/1e9:.2f} GFLOPs')
print(f'Total with eval: {total_flops_with_eval/1e12:.2f} TFLOPs')

# Estimate actual time and FLOP/s
total_time_seconds = (259.84 + 4.59 + 284.67 + 5.29 + 288.88)  # From the log output
flops_per_second = total_flops_with_eval / total_time_seconds

print()
print(f'=== PERFORMANCE METRICS ===')
print(f'Total runtime: {total_time_seconds:.1f} seconds')
print(f'Average FLOP/s: {flops_per_second/1e9:.2f} GFLOP/s')
print(f'Average FLOP/s: {flops_per_second/1e12:.3f} TFLOP/s')

# Model utilization (rough estimate)
# Modern CPUs can do ~100-1000 GFLOP/s depending on vectorization
print()
print('=== HARDWARE UTILIZATION ESTIMATE ===')
print(f'Achieved: {flops_per_second/1e9:.2f} GFLOP/s')
print('Typical CPU peak: 100-1000 GFLOP/s')
print(f'Estimated utilization: {(flops_per_second/1e9)/500*100:.1f}% (assuming 500 GFLOP/s peak)')
