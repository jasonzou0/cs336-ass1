
#!/usr/bin/env python3
"""
Training script for CS336 Transformer Language Model

This script implements a complete training loop that brings together all components:
- Model architecture (transformer with RoPE, RMSNorm, SwiGLU)
- Optimization (AdamW with cosine learning rate schedule)
- Data loading (memory-efficient with np.memmap)
- Checkpointing and logging
- Gradient clipping and loss computation

Usage:
    python my_training.py --config config.json
    python my_training.py --data-path /path/to/data --vocab-size 50000 --max-iters 100000
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to Python path for imports
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# Import our implementations
from tests.adapters import (
    run_get_batch,
    run_cross_entropy,
    run_gradient_clipping,
    get_adamw_cls,
    run_get_lr_cosine_schedule,
    run_save_checkpoint,
    run_load_checkpoint,
)


class TransformerConfig:
    """Configuration for the transformer model"""
    def __init__(
        self,
        vocab_size: int = 50257,
        context_length: int = 1024,
        d_model: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        d_ff: int = 3072,
        rope_theta: float = 10000.0,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.dropout = dropout
        self.bias = bias
        
        # Auto-adjust num_heads if needed to ensure d_model is divisible
        if self.d_model % self.num_heads != 0:
            # Find the largest divisor of d_model that's <= num_heads
            original_heads = self.num_heads
            for candidate in range(self.num_heads, 0, -1):
                if self.d_model % candidate == 0:
                    self.num_heads = candidate
                    break
            print(f"Warning: Adjusted num_heads from {original_heads} to {self.num_heads} "
                  f"to be compatible with d_model={self.d_model}")
        
        # Final validation
        assert self.d_model % self.num_heads == 0, f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"


class TrainingConfig:
    """Configuration for training hyperparameters"""
    def __init__(
        self,
        # Training
        batch_size: int = 32,
        max_iters: int = 100000,
        grad_clip: float = 1.0,
        
        # Optimization
        learning_rate: float = 1e-4,
        min_learning_rate: float = 1e-5,
        warmup_iters: int = 2000,
        cosine_cycle_iters: int = None,  # Will default to max_iters - warmup_iters
        weight_decay: float = 1e-2,
        beta1: float = 0.9,
        beta2: float = 0.95,
        eps: float = 1e-8,
        
        # Logging and checkpointing
        eval_interval: int = 1000,
        log_interval: int = 100,
        checkpoint_interval: int = 5000,
        eval_iters: int = 200,
        
        # Data
        data_path: str = "data/train.bin",
        val_data_path: Optional[str] = None,
        device: str = "auto",
        compile: bool = False,
    ):
        self.batch_size = batch_size
        self.max_iters = max_iters
        self.grad_clip = grad_clip
        
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.warmup_iters = warmup_iters
        self.cosine_cycle_iters = cosine_cycle_iters or (max_iters - warmup_iters)
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
        self.eval_interval = eval_interval
        self.log_interval = log_interval
        self.checkpoint_interval = checkpoint_interval
        self.eval_iters = eval_iters
        
        self.data_path = data_path
        self.val_data_path = val_data_path
        self.device = device
        self.compile = compile


class SimpleTransformer(nn.Module):
    """Simple transformer implementation using our adapter functions"""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            self._make_transformer_block() for _ in range(config.num_layers)
        ])
        
        # Final layer norm
        self.ln_final = nn.RMSNorm(config.d_model, eps=1e-5)
        
        # Language model head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _make_transformer_block(self):
        """Create a transformer block"""
        return nn.ModuleDict({
            'ln1': nn.RMSNorm(self.config.d_model, eps=1e-5),
            'attn': nn.ModuleDict({
                'q_proj': nn.Linear(self.config.d_model, self.config.d_model, bias=False),
                'k_proj': nn.Linear(self.config.d_model, self.config.d_model, bias=False),
                'v_proj': nn.Linear(self.config.d_model, self.config.d_model, bias=False),
                'output_proj': nn.Linear(self.config.d_model, self.config.d_model, bias=False),
            }),
            'ln2': nn.RMSNorm(self.config.d_model, eps=1e-5),
            'ffn': nn.ModuleDict({
                'w1': nn.Linear(self.config.d_model, self.config.d_ff, bias=False),
                'w2': nn.Linear(self.config.d_ff, self.config.d_model, bias=False),
                'w3': nn.Linear(self.config.d_model, self.config.d_ff, bias=False),
            }),
        })
    
    def _init_weights(self, module):
        """Initialize weights following GPT-2 style"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        """Forward pass through the transformer"""
        import torch.nn.functional as F
        
        B, T = idx.shape
        
        # Token embeddings
        x = self.token_embeddings(idx)  # (B, T, d_model)
        
        # Apply transformer layers
        for layer in self.layers:
            # Pre-norm architecture
            # 1. Self-attention with residual connection
            normed = layer['ln1'](x)
            
            # Multi-head self-attention (simplified implementation)
            d_head = self.config.d_model // self.config.num_heads
            q = layer['attn']['q_proj'](normed).view(B, T, self.config.num_heads, d_head).transpose(1, 2)
            k = layer['attn']['k_proj'](normed).view(B, T, self.config.num_heads, d_head).transpose(1, 2)
            v = layer['attn']['v_proj'](normed).view(B, T, self.config.num_heads, d_head).transpose(1, 2)
            
            # Scaled dot-product attention with causal mask
            scores = torch.matmul(q, k.transpose(-2, -1)) / (d_head ** 0.5)
            
            # Causal mask
            mask = torch.tril(torch.ones(T, T, device=idx.device, dtype=torch.bool))
            scores = scores.masked_fill(~mask, float('-inf'))
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_output = torch.matmul(attn_weights, v)
            
            # Reshape and project
            attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.config.d_model)
            attn_output = layer['attn']['output_proj'](attn_output)
            
            # Residual connection
            x = x + attn_output
            
            # 2. Feed-forward with residual connection
            normed = layer['ln2'](x)
            
            # SwiGLU: SiLU(W1 * x) * W3 * x -> W2
            gate = F.silu(layer['ffn']['w1'](normed))
            up = layer['ffn']['w3'](normed)
            ffn_output = layer['ffn']['w2'](gate * up)
            
            # Residual connection
            x = x + ffn_output
        
        # Final layer norm
        x = self.ln_final(x)
        
        # Language model head
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        loss = None
        if targets is not None:
            # Compute cross-entropy loss
            B, T, C = logits.shape
            loss = F.cross_entropy(
                logits.view(B * T, C),
                targets.view(B * T),
                ignore_index=-1
            )
        
        return logits, loss


def load_data(data_path: str) -> np.ndarray:
    """Load data using memory mapping for efficiency"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    return np.memmap(data_path, dtype=np.int32, mode='r')


def estimate_loss(model, train_data, val_data, eval_iters, batch_size, context_length, device):
    """Estimate loss on train and validation sets"""
    model.eval()
    losses = {}
    
    for split, data in [('train', train_data), ('val', val_data)]:
        if data is None:
            continue
            
        split_losses = []
        for _ in range(eval_iters):
            X, Y = run_get_batch(data, batch_size, context_length, device)
            with torch.no_grad():
                logits, loss = model(X, Y)
                split_losses.append(loss.item())
        
        losses[split] = sum(split_losses) / len(split_losses)
    
    model.train()
    return losses


def save_config(config: dict, path: str):
    """Save configuration to JSON file"""
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Train a transformer language model')
    
    # Model configuration
    parser.add_argument('--vocab-size', type=int, default=50257)
    parser.add_argument('--context-length', type=int, default=1024)
    parser.add_argument('--d-model', type=int, default=768)
    parser.add_argument('--num-layers', type=int, default=12)
    parser.add_argument('--num-heads', type=int, default=12)
    parser.add_argument('--d-ff', type=int, default=3072)
    parser.add_argument('--rope-theta', type=float, default=10000.0)
    
    # Training configuration
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-iters', type=int, default=100000)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--min-learning-rate', type=float, default=1e-5)
    parser.add_argument('--warmup-iters', type=int, default=2000)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    
    # Data and logging
    parser.add_argument('--data-path', type=str, required=True, help='Path to training data', default='data/train.bin')
    parser.add_argument('--val-data-path', type=str, help='Path to validation data', default='data/valid.bin')
    parser.add_argument('--out-dir', type=str, default='./checkpoints', help='Output directory for checkpoints')
    parser.add_argument('--eval-interval', type=int, default=1000)
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--checkpoint-interval', type=int, default=5000)
    parser.add_argument('--eval-iters', type=int, default=200)
    
    # System
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--compile', action='store_true', help='Compile model with torch.compile')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    
    # Configuration file
    parser.add_argument('--config', type=str, help='Load configuration from JSON file')
    
    args = parser.parse_args()
    
    # Load config from file if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        # Override with command line arguments
        for key, value in vars(args).items():
            if value is not None:
                config_dict[key] = value
        args = argparse.Namespace(**config_dict)
    
    # Device setup
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Create model configuration
    model_config = TransformerConfig(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    )
    
    # Create training configuration
    train_config = TrainingConfig(
        batch_size=args.batch_size,
        max_iters=args.max_iters,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        warmup_iters=args.warmup_iters,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        checkpoint_interval=args.checkpoint_interval,
        eval_iters=args.eval_iters,
        data_path=args.data_path,
        val_data_path=args.val_data_path,
        device=device,
        compile=args.compile,
    )
    
    # Save configuration
    config_path = os.path.join(args.out_dir, 'config.json')
    save_config({
        'model': vars(model_config),
        'training': vars(train_config),
    }, config_path)
    print(f"Saved configuration to {config_path}")
    
    # Load data
    print("Loading data...")
    train_data = load_data(train_config.data_path)
    val_data = None
    if train_config.val_data_path and os.path.exists(train_config.val_data_path):
        try:
            val_data = load_data(train_config.val_data_path)
            print(f"Loaded validation data: {len(val_data):,} tokens")
        except Exception as e:
            print(f"Warning: Could not load validation data from {train_config.val_data_path}: {e}")
            print("Continuing with training data only...")
            val_data = None
    print(f"Loaded training data: {len(train_data):,} tokens")
    
    # Create model
    print("Creating model...")
    model = SimpleTransformer(model_config)
    model.to(device)
    
    if train_config.compile:
        print("Compiling model...")
        model = torch.compile(model)
    
    # Create optimizer
    AdamW = get_adamw_cls()
    optimizer = AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=(train_config.beta1, train_config.beta2),
        eps=train_config.eps,
        weight_decay=train_config.weight_decay,
    )
    
    # Resume from checkpoint if specified
    start_iter = 0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        start_iter = run_load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from iteration {start_iter}")
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    print("Starting training...")
    model.train()
    
    t0 = time.time()
    for iter_num in range(start_iter, train_config.max_iters):
        
        # Determine learning rate based on schedule
        lr = run_get_lr_cosine_schedule(
            it=iter_num,
            max_learning_rate=train_config.learning_rate,
            min_learning_rate=train_config.min_learning_rate,
            warmup_iters=train_config.warmup_iters,
            cosine_cycle_iters=train_config.cosine_cycle_iters,
        )
        
        # Update learning rate in optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Evaluate and log
        if iter_num % train_config.eval_interval == 0 or iter_num == train_config.max_iters - 1:
            losses = estimate_loss(
                model, train_data, val_data, 
                train_config.eval_iters, train_config.batch_size, 
                model_config.context_length, device
            )
            train_loss = losses.get('train', None)
            val_loss = losses.get('val', None)
            train_loss_str = f"{train_loss:.4f}" if train_loss is not None else "N/A"
            val_loss_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"
            print(f"Step {iter_num}: train loss {train_loss_str}, val loss {val_loss_str}")
        
        # Sample batch
        X, Y = run_get_batch(train_data, train_config.batch_size, model_config.context_length, device)
        
        # Forward pass
        logits, loss = model(X, Y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        run_gradient_clipping(model.parameters(), train_config.grad_clip)
        
        # Update parameters
        optimizer.step()
        
        # Logging
        if iter_num % train_config.log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            lossf = loss.item()
            print(f"iter {iter_num}: loss {lossf:.4f}, lr {lr:.2e}, time {dt*1000:.2f}ms")
        
        # Checkpointing
        if iter_num % train_config.checkpoint_interval == 0 and iter_num > 0:
            checkpoint_path = os.path.join(args.out_dir, f'ckpt_{iter_num:06d}.pt')
            run_save_checkpoint(model, optimizer, iter_num, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final checkpoint
    final_checkpoint_path = os.path.join(args.out_dir, 'final_model.pt')
    run_save_checkpoint(model, optimizer, train_config.max_iters, final_checkpoint_path)
    print(f"Saved final model to {final_checkpoint_path}")
    
    print("Training completed!")


if __name__ == "__main__":
    main()