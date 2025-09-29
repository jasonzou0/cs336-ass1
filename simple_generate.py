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

def load_bpe_tokenizer():
    """Load the BPE tokenizer used during training"""
    from cs336_basics.my_tokenizer import BpeTokenizer
    
    vocab_file = 'data/tokenizer/vocab.json'
    merges_file = 'data/tokenizer/merges.txt'
    
    # Load vocabulary
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab_dict = json.load(f)
    
    # Convert vocab back to the format expected by BpeTokenizer
    id_to_bytes = {}
    for k, v in vocab_dict.items():
        token_id = int(k)
        if token_id < 256:
            # Base bytes: convert back to single byte
            try:
                # Try latin-1 first (for proper base bytes)
                id_to_bytes[token_id] = v.encode('latin-1')
            except UnicodeEncodeError:
                # Fallback: if it's a replacement character, use the token_id as byte value
                id_to_bytes[token_id] = bytes([token_id])
        else:
            # Merged tokens and special tokens: use UTF-8
            id_to_bytes[token_id] = v.encode('utf-8')
    
    # Load merges
    with open(merges_file, 'r', encoding='utf-8') as f:
        merges_lines = f.read().strip().split('\n')
    
    # Convert merges back to tuple format
    merges = []
    for line in merges_lines:
        if line.strip():
            parts = line.split(' ', 1)
            if len(parts) == 2:
                merges.append((parts[0].encode('utf-8'), parts[1].encode('utf-8')))
    
    tokenizer = BpeTokenizer(
        id_to_bytes=id_to_bytes,
        merges=merges,
        special_tokens=['<|endoftext|>']
    )
    
    return tokenizer

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
        
        # Check for stopping conditions (optional since BPE doesn't have explicit EOS)
        # We'll just continue until max_new_tokens
        
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
    
    # Load the actual BPE tokenizer
    tokenizer = load_bpe_tokenizer()
    print(f"✅ Using BPE tokenizer (vocab_size={len(tokenizer.id_to_bytes)})")
    
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