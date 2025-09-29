#!/usr/bin/env python3
"""
Story Generation Script for CS336 Final Model

This script uses the trained final_model.pt to generate stories from simple prompts.
Since the model was trained with vocab_size=50257, we'll use GPT-2's tokenizer which matches this size.
"""

import argparse
import json
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    # Try to use transformers library for GPT-2 tokenizer
    from transformers import GPT2Tokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("⚠️  transformers library not available, falling back to character tokenizer")


def load_model_from_checkpoint(checkpoint_path):
    """Load the trained model from checkpoint"""
    
    # Import the model architecture
    import sys
    sys.path.append('cs336_basics')
    from my_training import SimpleTransformer, TransformerConfig
    
    # Load checkpoint and config
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config_path = os.path.join(os.path.dirname(checkpoint_path), 'config.json')
    with open(config_path, 'r') as f:
        full_config = json.load(f)
    
    model_config = full_config['model']
    
    print(f"Loading model:")
    print(f"  Vocabulary size: {model_config['vocab_size']:,}")
    print(f"  Context length: {model_config['context_length']}")
    print(f"  Model dimension: {model_config['d_model']}")
    print(f"  Layers: {model_config['num_layers']}")
    print(f"  Attention heads: {model_config['num_heads']}")
    
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
    
    # Create and load model
    model = SimpleTransformer(transformer_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded on {device}")
    return model, device, model_config


def get_tokenizer():
    """Get appropriate tokenizer for vocab_size=50257"""
    
    if HAS_TRANSFORMERS:
        # Use GPT-2 tokenizer which has vocab_size=50257
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"✅ Using GPT-2 tokenizer (vocab_size={tokenizer.vocab_size})")
        return tokenizer
    else:
        # Fallback to simple character tokenizer (this won't work well but at least runs)
        print("⚠️  Using fallback character tokenizer - install transformers for better results")
        return SimpleCharTokenizer()


class SimpleCharTokenizer:
    """Fallback character tokenizer - not ideal for vocab_size=50257"""
    def __init__(self):
        chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'\"-()[]{}@#$%^&*+=<>/\\|`~_\n")
        self.char_to_id = {char: i for i, char in enumerate(chars)}
        self.id_to_char = {i: char for i, char in enumerate(chars)}
        self.vocab_size = 50257  # Pretend to match model vocab size
        self.eos_token_id = 50256
    
    def encode(self, text):
        return [self.char_to_id.get(char, 0) for char in text]
    
    def decode(self, token_ids):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return ''.join(self.id_to_char.get(id % len(self.id_to_char), '') for id in token_ids)


@torch.no_grad()
def generate_story(model, tokenizer, prompt, max_new_tokens=200, temperature=0.8, top_k=50, device='cuda'):
    """Generate a story from a prompt"""
    
    model.eval()
    
    # Encode the prompt
    if HAS_TRANSFORMERS and hasattr(tokenizer, 'encode'):
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    else:
        token_ids = tokenizer.encode(prompt)
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    
    print(f"🎭 Generating story from: '{prompt}'")
    print(f"📝 Input tokens: {input_ids.size(1)}")
    print("=" * 60)
    
    generated = input_ids.clone()
    
    for i in tqdm(range(max_new_tokens), desc="Generating"):
        # Ensure we don't exceed context length
        if generated.size(1) >= 512:  # Model's context length
            # Keep only the most recent tokens
            generated = generated[:, -400:]
        
        # Forward pass
        try:
            if hasattr(model, 'forward') and len(model.forward.__code__.co_varnames) > 2:
                # Model expects (input_ids, targets) - pass None for targets during inference
                logits = model(generated, None)
                if isinstance(logits, tuple):
                    logits = logits[0]  # Extract logits from (logits, loss) tuple
            else:
                logits = model(generated)
        except Exception as e:
            print(f"Model forward error: {e}")
            break
        
        # Get next token logits
        next_token_logits = logits[0, -1, :] / temperature
        
        # Apply top-k filtering
        if top_k > 0:
            top_k_logits, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
            next_token_logits[next_token_logits < top_k_logits[-1]] = -float('inf')
        
        # Sample next token
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Stop at EOS token (for GPT-2 tokenizer)
        if HAS_TRANSFORMERS and hasattr(tokenizer, 'eos_token_id'):
            if next_token.item() == tokenizer.eos_token_id:
                break
        
        # Append to sequence
        generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
    
    # Decode generated text
    if HAS_TRANSFORMERS and hasattr(tokenizer, 'decode'):
        generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    else:
        generated_text = tokenizer.decode(generated[0])
    
    return generated_text


def main():
    parser = argparse.ArgumentParser(description='Generate stories with the trained CS336 model')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/final_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--prompt', type=str, 
                       default="Once upon a time, there was a little girl who",
                       help='Story prompt')
    parser.add_argument('--max-tokens', type=int, default=200,
                       help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature (0.1-2.0)')
    parser.add_argument('--top-k', type=int, default=50,
                       help='Top-k sampling')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive mode for multiple prompts')
    
    args = parser.parse_args()
    
    # Check checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return 1
    
    try:
        # Load model
        print("🔄 Loading model...")
        model, device, model_config = load_model_from_checkpoint(args.checkpoint)
        
        # Get tokenizer
        print("🔄 Loading tokenizer...")
        tokenizer = get_tokenizer()
        
        if args.interactive:
            # Interactive mode
            print("\n🎭 Interactive Story Generator")
            print("=" * 50)
            print("Enter prompts to generate stories!")
            print("Type 'quit' to exit")
            print("=" * 50)
            
            while True:
                try:
                    prompt = input("\n📝 Enter your story prompt: ").strip()
                    
                    if prompt.lower() in ['quit', 'exit', 'q']:
                        print("👋 Goodbye!")
                        break
                    
                    if not prompt:
                        print("Please enter a prompt!")
                        continue
                    
                    # Generate story
                    story = generate_story(
                        model, tokenizer, prompt,
                        max_new_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        device=device
                    )
                    
                    print(f"\n📚 Generated Story:")
                    print("-" * 40)
                    print(story)
                    print("-" * 40)
                    
                except KeyboardInterrupt:
                    print("\n👋 Goodbye!")
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")
        
        else:
            # Single prompt mode
            story = generate_story(
                model, tokenizer, args.prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                device=device
            )
            
            print(f"\n📚 Generated Story:")
            print("=" * 60)
            print(story)
            print("=" * 60)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())