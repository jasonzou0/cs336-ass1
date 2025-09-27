#!/usr/bin/env python3
"""
GPU Memory Capacity Calculator for Transformer Models
Calculates maximum model size that can fit in available GPU memory
"""

import torch
import math

def calculate_transformer_params(vocab_size, d_model, num_layers, context_length):
    """Calculate approximate parameter count for transformer model"""
    
    # Embedding layer
    embedding_params = vocab_size * d_model
    
    # Each transformer layer
    # - Multi-head attention: 4 * d_model^2 (Q, K, V, O projections)
    # - FFN: ~8 * d_model^2 (assuming FFN hidden size = 4 * d_model)  
    # - Layer norms: 2 * d_model
    layer_params = (4 * d_model**2) + (8 * d_model**2) + (2 * d_model)
    transformer_params = num_layers * layer_params
    
    # Final layer norm
    final_norm_params = d_model
    
    # Output projection (often tied to embedding, but counting separately)
    output_params = vocab_size * d_model
    
    total_params = embedding_params + transformer_params + final_norm_params + output_params
    
    return {
        'embedding': embedding_params,
        'transformer_layers': transformer_params, 
        'final_norm': final_norm_params,
        'output': output_params,
        'total': total_params
    }

def calculate_memory_usage(params, batch_size, context_length, d_model, num_layers):
    """Calculate GPU memory usage for training"""
    
    # Model weights (fp32)
    model_memory = params * 4  # 4 bytes per parameter
    
    # Gradients (fp32)
    grad_memory = params * 4
    
    # Optimizer states (AdamW: momentum + variance)
    optimizer_memory = params * 8
    
    # Activations (rough estimate)
    # Each layer stores activations: batch_size * context_length * d_model
    activation_memory = batch_size * context_length * d_model * num_layers * 4
    
    # Temporary buffers and overhead (~20%)
    base_memory = model_memory + grad_memory + optimizer_memory + activation_memory
    overhead = base_memory * 0.2
    
    total_memory = base_memory + overhead
    
    return {
        'model': model_memory,
        'gradients': grad_memory,
        'optimizer': optimizer_memory,
        'activations': activation_memory,
        'overhead': overhead,
        'total': total_memory
    }

def find_max_model_size(gpu_memory_gb, vocab_size=50257, target_batch_size=4, context_length=1024):
    """Find maximum d_model and num_layers that fit in GPU memory"""
    
    gpu_memory_bytes = gpu_memory_gb * 1e9
    available_memory = gpu_memory_bytes * 0.9  # Leave 10% buffer
    
    print(f"GPU Memory: {gpu_memory_gb:.2f} GB ({available_memory/1e9:.2f} GB usable)")
    print(f"Target config: vocab_size={vocab_size}, batch_size={target_batch_size}, context_length={context_length}")
    print("\nSearching for maximum model size...\n")
    
    best_config = None
    
    # Test different model sizes
    for num_layers in [4, 6, 8, 12, 16, 24]:
        for d_model in [256, 384, 512, 768, 1024, 1280, 1536, 2048]:
            
            # Calculate parameters
            param_info = calculate_transformer_params(vocab_size, d_model, num_layers, context_length)
            total_params = param_info['total']
            
            # Calculate memory usage
            memory_info = calculate_memory_usage(
                total_params, target_batch_size, context_length, d_model, num_layers
            )
            total_memory = memory_info['total']
            
            if total_memory <= available_memory:
                config = {
                    'd_model': d_model,
                    'num_layers': num_layers,
                    'parameters': total_params,
                    'memory_gb': total_memory / 1e9,
                    'memory_breakdown': memory_info
                }
                
                if best_config is None or total_params > best_config['parameters']:
                    best_config = config
                    
                print(f"✅ d_model={d_model:4d}, layers={num_layers:2d} -> "
                      f"{total_params/1e6:6.1f}M params, {total_memory/1e9:5.2f} GB")
            else:
                print(f"❌ d_model={d_model:4d}, layers={num_layers:2d} -> "
                      f"{total_params/1e6:6.1f}M params, {total_memory/1e9:5.2f} GB (too big)")
    
    return best_config

def main():
    # Get GPU info
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("No CUDA GPU detected!")
        return
    
    print("="*70)
    print("🧮 TRANSFORMER MODEL CAPACITY CALCULATOR")
    print("="*70)
    
    # Find maximum model size
    best_config = find_max_model_size(gpu_memory)
    
    if best_config:
        print("\n" + "="*50)
        print("🎯 MAXIMUM RECOMMENDED CONFIGURATION:")
        print("="*50)
        print(f"d_model: {best_config['d_model']}")
        print(f"num_layers: {best_config['num_layers']}")
        print(f"num_heads: {best_config['d_model'] // 64}")  # Common ratio
        print(f"Total parameters: {best_config['parameters']/1e6:.1f}M")
        print(f"Memory usage: {best_config['memory_gb']:.2f} GB")
        
        print(f"\n📊 Memory Breakdown:")
        mem = best_config['memory_breakdown']
        print(f"  Model weights: {mem['model']/1e9:.2f} GB")
        print(f"  Gradients:     {mem['gradients']/1e9:.2f} GB") 
        print(f"  Optimizer:     {mem['optimizer']/1e9:.2f} GB")
        print(f"  Activations:   {mem['activations']/1e9:.2f} GB")
        print(f"  Overhead:      {mem['overhead']/1e9:.2f} GB")
        
        print(f"\n🚀 Command to run:")
        print(f"python cs336_basics/my_training.py --data-path data/train.bin \\")
        print(f"  --d-model {best_config['d_model']} --num-layers {best_config['num_layers']} \\")
        print(f"  --num-heads {best_config['d_model'] // 64} --batch-size 4 --max-iters 10")

if __name__ == "__main__":
    main()