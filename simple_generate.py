#!/usr/bin/env python3
"""
Simple Story Generator for CS336 Trained Model

A simplified version that loads the exact model architecture used in training.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add project root to Python path
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

# Import the exact model from training script
try:
    from cs336_basics.my_training import SimpleTransformer, TransformerConfig
except ImportError:
    print("❌ Could not import model from training script")
    sys.exit(1)

class SimpleCharTokenizer:
    """Simple character-level tokenizer"""
    def __init__(self):
        # Create a simple character vocabulary  
        chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'\"-()[]{}@#$%^&*+=<>/\\|`~_\n")
        self.char_to_id = {char: i for i, char in enumerate(chars)}
        self.id_to_char = {i: char for i, char in enumerate(chars)}
        self.vocab_size = len(chars)
        self.eos_token_id = len(chars) - 1  # Use newline as EOS
    
    def encode(self, text: str) -> list[int]:
        return [self.char_to_id.get(char, 0) for char in text]
    
    def decode(self, token_ids: list[int]) -> str:
        return ''.join(self.id_to_char.get(id, '') for id in token_ids)

@torch.no_grad()
def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 200, temperature: float = 0.8, device='cuda'):
    """Generate text from prompt using the trained model"""
    
    model.eval()
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    print(f"Generating from prompt: '{prompt}'")
    print("=" * 50)
    
    generated = input_tensor.clone()
    
    for i in tqdm(range(max_new_tokens), desc="Generating"):
        # Get model output
        if generated.size(1) >= 1024:  # Use max context length
            # Truncate if too long
            input_seq = generated[:, -1024:]
        else:
            input_seq = generated
        
        logits, _ = model(input_seq)  # Model returns (logits, loss)
        
        # Get logits for next token (last position)
        next_logits = logits[0, -1, :] / temperature
        
        # Sample next token
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Check for stopping conditions
        if next_token.item() == tokenizer.eos_token_id:
            break
        
        # Append to sequence
        generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
    
    # Decode generated text
    generated_text = tokenizer.decode(generated[0].tolist())
    
    # Also show the token IDs for debugging
    print(f"\nGenerated token IDs: {generated[0].tolist()[-10:]}")  # Last 10 tokens
    
    return generated_text

def main():
    parser = argparse.ArgumentParser(description='Generate stories with trained model')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/final_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--prompt', type=str, default="Once upon a time",
                       help='Text prompt for generation')
    parser.add_argument('--max-tokens', type=int, default=200,
                       help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature')
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return 1
    
    # Load config
    config_path = os.path.join(os.path.dirname(args.checkpoint), 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model_config = config['model']
    
    print("🔄 Loading model...")
    print(f"Model: d_model={model_config['d_model']}, layers={model_config['num_layers']}, heads={model_config['num_heads']}")
    
    # Create model with exact same architecture
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model config
    transformer_config = TransformerConfig(
        vocab_size=model_config['vocab_size'],
        context_length=model_config['context_length'], 
        d_model=model_config['d_model'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        d_ff=model_config['d_ff'],
        rope_theta=model_config['rope_theta'],
        dropout=0.0,  # No dropout for inference
        bias=model_config['bias']
    )
    
    model = SimpleTransformer(transformer_config)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"✅ Model loaded on {device}")
    
    # Create simple tokenizer
    tokenizer = SimpleCharTokenizer()
    print(f"⚠️  Using simple character tokenizer (vocab_size={tokenizer.vocab_size})")
    
    # Generate story
    try:
        generated_story = generate_text(
            model, tokenizer, args.prompt, 
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            device=device
        )
        
        print(f"\n📚 Generated Story:")
        print("=" * 60)
        print(generated_story)
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())