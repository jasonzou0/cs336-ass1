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


def load_bpe_tokenizer():
    """Load the trained BPE tokenizer from data/tokenizer folder"""
    from cs336_basics.my_tokenizer import BpeTokenizer
    
    vocab_file = 'data/tokenizer/vocab.json'
    merges_file = 'data/tokenizer/merges.txt'
    
    print(f"Loading BPE tokenizer from {os.path.dirname(vocab_file)}")
    
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
    
    print(f"✅ BPE tokenizer loaded with vocab size: {len(id_to_bytes)}")
    return tokenizer


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
def generate_story(model, tokenizer, prompt, device, max_tokens=50, temperature=0.8):
    """Generate a story from a prompt using BPE tokenizer"""
    
    print(f"🎭 Generating from prompt: '{prompt}'")
    
    # Encode prompt using BPE tokenizer
    input_tokens = tokenizer.encode(prompt)
    if not input_tokens:
        input_tokens = [0]  # Fallback to first token
    
    input_ids = torch.tensor([input_tokens], dtype=torch.long, device=device)
    
    print(f"Input tokens: {len(input_tokens)}")
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
        
        # Apply top-k sampling
        top_k = 50
        if top_k > 0:
            top_k_logits, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits[next_logits < top_k_logits[-1]] = -float('inf')
        
        # Sample next token
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        
        # Append token
        generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
    
    # Decode generated text using BPE tokenizer
    all_tokens = generated[0].cpu().tolist()
    generated_text = tokenizer.decode(all_tokens)
    
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
        
        # Load BPE tokenizer
        print("Loading BPE tokenizer...")
        tokenizer = load_bpe_tokenizer()
        
        # Generate story
        story = generate_story(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            device=device,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )
        
        print(f"\n📚 Generated Story:")
        print("=" * 60)
        print(story)
        print("=" * 60)
        
        print("\n✅ Generated using trained BPE tokenizer!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())