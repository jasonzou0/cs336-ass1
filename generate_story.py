#!/usr/bin/env python3
"""
Story Generation Application for CS336 Trained Model

This script loads a trained transformer model and generates tiny stories
from user prompts using text generation techniques.

Usage:
    python generate_story.py --checkpoint checkpoints/ckpt_000010.pt --prompt "Once upon a time"
    python generate_story.py --config checkpoints/config.json --interactive
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add project root to Python path
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

# Import our implementations
from tests.adapters import (
    run_load_checkpoint,
    get_tokenizer
)


class StoryGenerator:
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """Encode text prompt to token IDs"""
        token_ids = self.tokenizer.encode(prompt)
        return torch.tensor(token_ids, dtype=torch.long, device=self.device).unsqueeze(0)
    
    def decode_tokens(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs back to text"""
        if token_ids.dim() > 1:
            token_ids = token_ids.squeeze(0)
        return self.tokenizer.decode(token_ids.tolist())
    
    @torch.no_grad()
    def generate(self, 
                prompt: str,
                max_new_tokens: int = 200,
                temperature: float = 0.8,
                top_k: Optional[int] = 50,
                top_p: float = 0.9,
                repetition_penalty: float = 1.1) -> str:
        """
        Generate text continuation from prompt
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens for sampling
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repeating tokens
        """
        
        # Encode the prompt
        input_ids = self.encode_prompt(prompt)
        generated_ids = input_ids.clone()
        
        print(f"Generating story from prompt: '{prompt}'")
        print("=" * 60)
        
        for i in tqdm(range(max_new_tokens), desc="Generating"):
            # Get model predictions
            with torch.no_grad():
                logits = self.model(generated_ids)
                
                # Get logits for next token (last position)
                next_token_logits = logits[0, -1, :]  # [vocab_size]
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for token_id in set(generated_ids[0].tolist()):
                        if next_token_logits[token_id] < 0:
                            next_token_logits[token_id] *= repetition_penalty
                        else:
                            next_token_logits[token_id] /= repetition_penalty
                
                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                
                # Apply top-k filtering
                if top_k is not None and top_k > 0:
                    top_k_actual = min(top_k, next_token_logits.size(-1))
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k_actual)
                    next_token_logits[next_token_logits < top_k_logits[-1]] = -float('inf')
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Keep at least one token
                    sorted_indices_to_remove[0] = False
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Check for end-of-text or story completion
                if self._should_stop(next_token, generated_ids):
                    break
                
                # Append to sequence
                generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
        
        # Decode the full generated sequence
        generated_text = self.decode_tokens(generated_ids)
        
        return generated_text
    
    def _should_stop(self, next_token: torch.Tensor, generated_ids: torch.Tensor) -> bool:
        """Check if generation should stop"""
        # Stop if we hit end-of-text token (if it exists)
        # You might need to adjust this based on your tokenizer
        if hasattr(self.tokenizer, 'eos_token_id'):
            if next_token.item() == self.tokenizer.eos_token_id:
                return True
        
        # Stop if sequence gets too long
        if generated_ids.size(1) > 1024:  # Max context length
            return True
            
        return False


def load_model_and_config(checkpoint_path: str, config_path: Optional[str] = None):
    """Load trained model and configuration"""
    
    # Load configuration
    if config_path is None:
        config_path = os.path.join(os.path.dirname(checkpoint_path), 'config.json')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Extract model config from nested structure
    model_config = config.get('model', config)  # Handle both nested and flat configs
    
    print(f"Loading model configuration:")
    print(f"  d_model: {model_config['d_model']}")
    print(f"  num_layers: {model_config['num_layers']}")  
    print(f"  num_heads: {model_config['num_heads']}")
    print(f"  vocab_size: {model_config['vocab_size']}")
    print(f"  context_length: {model_config['context_length']}")
    
    # Load checkpoint directly
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create a simple tokenizer for now
    print("⚠️  Using simple character tokenizer for generation")
    tokenizer = SimpleCharTokenizer()
    
    # Create model using the same architecture as in training
    # We'll recreate the model class from the training script
    import torch.nn as nn
    
    class TransformerLM(nn.Module):
        def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta=10000.0, dropout=0.0, bias=True):
            super().__init__()
            self.vocab_size = vocab_size
            self.context_length = context_length
            self.d_model = d_model
            
            # Token embedding
            self.token_embedding = nn.Embedding(vocab_size, d_model)
            
            # Transformer layers (simplified - you may need to match your exact architecture)
            self.layers = nn.ModuleList([
                nn.TransformerDecoderLayer(
                    d_model=d_model,
                    nhead=num_heads,
                    dim_feedforward=d_ff,
                    dropout=dropout,
                    batch_first=True
                )
                for _ in range(num_layers)
            ])
            
            # Output head
            self.ln_f = nn.LayerNorm(d_model)
            self.lm_head = nn.Linear(d_model, vocab_size, bias=bias)
        
        def forward(self, x, targets=None):
            B, T = x.shape
            
            # Token embeddings
            x = self.token_embedding(x)  # (B, T, d_model)
            
            # Create causal mask
            mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            mask = ~mask  # Invert for PyTorch convention
            
            # Apply transformer layers
            for layer in self.layers:
                x = layer(x, x, tgt_mask=mask)
            
            # Final layer norm and output projection
            x = self.ln_f(x)
            logits = self.lm_head(x)
            
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                return logits, loss
            
            return logits
    
    model = TransformerLM(
        vocab_size=model_config['vocab_size'],
        context_length=model_config['context_length'],
        d_model=model_config['d_model'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        d_ff=model_config.get('d_ff', 4 * model_config['d_model']),
        rope_theta=model_config.get('rope_theta', 10000.0),
        dropout=0.0,  # Set to 0 for inference
        bias=model_config.get('bias', True)
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    return model, tokenizer, config


class SimpleCharTokenizer:
    """Simple character-level tokenizer as fallback"""
    def __init__(self):
        # Create a simple character vocabulary
        chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'\"-()[]{}@#$%^&*+=<>/\\|`~_")
        self.char_to_id = {char: i for i, char in enumerate(chars)}
        self.id_to_char = {i: char for i, char in enumerate(chars)}
        self.vocab_size = len(chars)
    
    def encode(self, text: str) -> List[int]:
        return [self.char_to_id.get(char, 0) for char in text]
    
    def decode(self, token_ids: List[int]) -> str:
        return ''.join(self.id_to_char.get(id, '') for id in token_ids)


def interactive_mode(generator: StoryGenerator):
    """Interactive story generation mode"""
    print("\n🎭 Interactive Story Generator")
    print("=" * 50)
    print("Enter prompts to generate tiny stories!")
    print("Type 'quit' or 'exit' to stop")
    print("Type 'help' for generation options")
    print("=" * 50)
    
    # Generation parameters
    params = {
        'max_new_tokens': 200,
        'temperature': 0.8,
        'top_k': 50,
        'top_p': 0.9,
        'repetition_penalty': 1.1
    }
    
    while True:
        try:
            prompt = input("\n📝 Enter your story prompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit']:
                print("👋 Goodbye! Happy storytelling!")
                break
            
            if prompt.lower() == 'help':
                print("\n⚙️ Current generation parameters:")
                for key, value in params.items():
                    print(f"  {key}: {value}")
                print("\nTip: Try prompts like:")
                print("  • 'Once upon a time there was a'")
                print("  • 'The little girl found a'")
                print("  • 'In the magical forest,'")
                continue
            
            if not prompt:
                print("Please enter a prompt!")
                continue
            
            # Generate story
            story = generator.generate(prompt, **params)
            
            print(f"\n📚 Generated Story:")
            print("-" * 40)
            print(story)
            print("-" * 40)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error generating story: {e}")


def main():
    parser = argparse.ArgumentParser(description='Generate stories with trained CS336 model')
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to model checkpoint (.pt file)')
    parser.add_argument('--config', type=str, 
                       help='Path to config.json (auto-detected if not provided)')
    parser.add_argument('--prompt', type=str, 
                       help='Text prompt for story generation')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--max-tokens', type=int, default=200,
                       help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature (0.1-2.0)')
    parser.add_argument('--top-k', type=int, default=50,
                       help='Top-k sampling parameter')
    parser.add_argument('--top-p', type=float, default=0.9,
                       help='Top-p (nucleus) sampling parameter')
    
    args = parser.parse_args()
    
    # Validate checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint file not found: {args.checkpoint}")
        return 1
    
    try:
        # Load model
        print("🔄 Loading trained model...")
        model, tokenizer, config = load_model_and_config(args.checkpoint, args.config)
        
        # Create generator
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        generator = StoryGenerator(model, tokenizer, device)
        print(f"✅ Model loaded successfully on {device}")
        
        if args.interactive:
            # Interactive mode
            interactive_mode(generator)
        
        elif args.prompt:
            # Single prompt mode
            print(f"\n🎭 Generating story from prompt...")
            story = generator.generate(
                args.prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p
            )
            
            print(f"\n📚 Generated Story:")
            print("=" * 60)
            print(story)
            print("=" * 60)
        
        else:
            print("❌ Please provide either --prompt or --interactive mode")
            return 1
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())