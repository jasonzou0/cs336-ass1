#!/usr/bin/env python3
"""
Simple Story Generator for CS336 Model

This script generates stories using the trained final_model.pt
"""

import argparse
import json
import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Import the model architecture
import sys
sys.path.append('cs336_basics')
from my_training import SimpleTransformer, TransformerConfig


def create_random_tokenizer():
    """Create a simple tokenizer that works with vocab_size=50257"""
    # This is a placeholder - in reality we need the exact tokenizer from training
    # For demonstration, we'll create mappings for basic text
    
    # Load a sample of the training data to understand the token distribution
    data = np.fromfile('data/train.bin', dtype=np.int32)
    unique_tokens = np.unique(data[:10000])  # Sample first 10k tokens
    
    print(f"Found {len(unique_tokens)} unique tokens in training data sample")
    print(f"Token range: {unique_tokens.min()} to {unique_tokens.max()}")
    
    # Create a simple character-to-token mapping
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'\"-()[]{}@#$%^&*+=<>/\\|`~_\n\t"
    
    # Use some of the actual tokens from training data
    token_to_char = {}
    char_to_token = {}
    
    # Map first few unique tokens to common characters
    for i, char in enumerate(chars):
        if i < len(unique_tokens):
            token_id = int(unique_tokens[i])
            token_to_char[token_id] = char
            char_to_token[char] = token_id
    
    return char_to_token, token_to_char


def encode_text(text, char_to_token, max_length=100):
    """Encode text using our simple tokenizer"""
    tokens = []
    for char in text[:max_length]:  # Limit length
        token_id = char_to_token.get(char, list(char_to_token.values())[0])  # Use first token as default
        tokens.append(token_id)
    return tokens


def decode_tokens(tokens, token_to_char):
    """Decode tokens back to text"""
    text = ""
    for token_id in tokens:
        if isinstance(token_id, torch.Tensor):
            token_id = token_id.item()
        char = token_to_char.get(token_id, '')
        text += char
    return text


def load_model():
    """Load the trained model"""
    
    checkpoint_path = 'checkpoints/final_model.pt'
    config_path = 'checkpoints/config.json'
    
    # Load config
    with open(config_path, 'r') as f:
        full_config = json.load(f)
    
    model_config = full_config['model']
    
    print("Loading model...")
    print(f"  Vocab size: {model_config['vocab_size']:,}")
    print(f"  Context length: {model_config['context_length']}")
    print(f"  Model dimension: {model_config['d_model']}")
    print(f"  Layers: {model_config['num_layers']}")
    print(f"  Heads: {model_config['num_heads']}")
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    transformer_config = TransformerConfig(
        vocab_size=model_config['vocab_size'],
        context_length=model_config['context_length'], 
        d_model=model_config['d_model'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        d_ff=model_config['d_ff'],
        rope_theta=model_config['rope_theta'],
        dropout=0.0,
        bias=model_config['bias']
    )
    
    model = SimpleTransformer(transformer_config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded on {device}")
    return model, device


@torch.no_grad()
def generate_story(model, prompt, char_to_token, token_to_char, device, max_tokens=50, temperature=0.8):
    """Generate a story from a prompt"""
    
    print(f"🎭 Generating from prompt: '{prompt}'")
    
    # Encode prompt
    input_tokens = encode_text(prompt, char_to_token)
    if not input_tokens:
        input_tokens = [list(char_to_token.values())[0]]  # Fallback
    
    input_ids = torch.tensor([input_tokens], dtype=torch.long, device=device)
    
    print(f"Input token IDs: {input_ids[0][:10].tolist()}... (showing first 10)")
    
    generated = input_ids.clone()
    
    for i in tqdm(range(max_tokens), desc="Generating"):
        # Limit context length
        if generated.size(1) > 400:
            generated = generated[:, -400:]
        
        # Forward pass
        try:
            logits = model(generated)
            if isinstance(logits, tuple):
                logits = logits[0]  # Extract logits from (logits, loss) tuple
        except Exception as e:
            print(f"Model error: {e}")
            break
        
        # Get next token logits
        next_logits = logits[0, -1, :] / temperature
        
        # Simple sampling - just take most likely tokens
        probs = F.softmax(next_logits, dim=-1)
        
        # Sample from top tokens to add some randomness
        top_k = 100
        top_probs, top_indices = torch.topk(probs, top_k)
        next_token = top_indices[torch.multinomial(top_probs, 1)]
        
        # Append token
        generated = torch.cat([generated, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
    
    # Decode generated text
    all_tokens = generated[0].cpu().tolist()
    generated_text = decode_tokens(all_tokens, token_to_char)
    
    return generated_text


def main():
    parser = argparse.ArgumentParser(description='Generate stories with CS336 model')
    parser.add_argument('--prompt', type=str, default='Once upon a time, there was a little girl', 
                       help='Story prompt')
    parser.add_argument('--max-tokens', type=int, default=50,
                       help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature')
    
    args = parser.parse_args()
    
    try:
        # Load model
        model, device = load_model()
        
        # Create tokenizer
        print("Creating simple tokenizer...")
        char_to_token, token_to_char = create_random_tokenizer()
        
        print(f"Tokenizer created with {len(char_to_token)} mappings")
        
        # Generate story
        story = generate_story(
            model=model,
            prompt=args.prompt,
            char_to_token=char_to_token,
            token_to_char=token_to_char,
            device=device,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )
        
        print(f"\n📚 Generated Story:")
        print("=" * 60)
        print(story)
        print("=" * 60)
        
        print("\n💡 Note: This is using a simplified tokenizer.")
        print("For best results, use the exact tokenizer from training.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())