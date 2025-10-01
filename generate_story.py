#!/usr/bin/env python3
"""
Story Generation Application for CS336 Trained Model

This script loads a trained transformer model and generates tiny stories
from user prompts using text generation techniques.

Usage:
    python generate_story.py --checkpoint checkpoints/final_model.pt --prompt "Once upon a time"
    python generate_story.py --checkpoint checkpoints/final_model.pt --interactive
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

# Import the actual model from training
from cs336_basics.my_training import SimpleTransformer, TransformerConfig


# Import your custom tokenizer
from cs336_basics.my_tokenizer import BpeTokenizer


def load_bpe_tokenizer(vocab_file: str, merges_file: str):
    """Load trained BPE tokenizer from vocab and merges files"""
    import json
    
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


class StoryGenerator:
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        self.eot_token_id = self._resolve_special_token_id("<|endoftext|>")
    
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
                    output = self.model(generated_ids)
                    
                    # Handle both single logits and (logits, loss) tuple
                    if isinstance(output, tuple):
                        logits = output[0]  # Extract logits from tuple
                    else:
                        logits = output
                    
                    # Get logits for next token (last position)
                    next_token_logits = logits[0, -1, :]  # [vocab_size]                # Apply repetition penalty
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
                
                # Safety check: ensure token ID is within valid range
                if next_token.item() >= len(self.tokenizer.id_to_bytes if hasattr(self.tokenizer, 'id_to_bytes') else range(50257)):
                    print(f"⚠️  Generated token ID {next_token.item()} is out of range, stopping generation")
                    break
                
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
        token_id = next_token.item()

        eos_token_id = getattr(self.tokenizer, 'eos_token_id', None)
        if eos_token_id is not None and token_id == eos_token_id:
            return True

        if self.eot_token_id is not None and token_id == self.eot_token_id:
            return True

        # Stop if sequence gets too long
        if generated_ids.size(1) > 1024:  # Max context length
            return True

        return False

    def _resolve_special_token_id(self, token_text: str) -> Optional[int]:
        """Attempt to find the ID for a special token by text."""
        token_bytes = token_text.encode('utf-8')

        if hasattr(self.tokenizer, 'bytes_to_id'):
            token_id = self.tokenizer.bytes_to_id.get(token_bytes)
            if token_id is not None:
                return token_id

        if hasattr(self.tokenizer, 'encode'):
            try:
                encoded = self.tokenizer.encode(token_text)
            except Exception:
                encoded = []
            if len(encoded) == 1:
                return encoded[0]

        eos_token_id = getattr(self.tokenizer, 'eos_token_id', None)
        if token_text == '<|endoftext|>' and eos_token_id is not None:
            return eos_token_id

        return None


def load_model_and_config(checkpoint_path: str, config_path: Optional[str] = None, tokenizer_path: Optional[str] = None, vocab_size_override: Optional[int] = None):
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
    
    # Load the BPE tokenizer used during training
    vocab_file = None
    merges_file = None
    
    if tokenizer_path:
        # Use explicitly provided tokenizer path
        if os.path.exists(tokenizer_path):
            vocab_file = os.path.join(tokenizer_path, 'vocab.json')
            merges_file = os.path.join(tokenizer_path, 'merges.txt')
    else:
        # Auto-detect tokenizer files
        checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
        
        # Check for tokenizer subdirectory
        tokenizer_dirs = [
            os.path.join(checkpoint_dir, 'tokenizer'),
            os.path.join(checkpoint_dir, '..', 'tokenizer'),
            os.path.join(checkpoint_dir, '..', 'data', 'tokenizer_50k'),
            os.path.join(checkpoint_dir, '..', 'data', 'tokenizer_10000'),
            'data/tokenizer_50k',
            'data/tokenizer_10000',
        ]
        
        for tokenizer_dir in tokenizer_dirs:
            if os.path.exists(tokenizer_dir):
                potential_vocab = os.path.join(tokenizer_dir, 'vocab.json')
                potential_merges = os.path.join(tokenizer_dir, 'merges.txt')
                if os.path.exists(potential_vocab) and os.path.exists(potential_merges):
                    vocab_file = potential_vocab
                    merges_file = potential_merges
                    break
    
    if vocab_file and merges_file and os.path.exists(vocab_file) and os.path.exists(merges_file):
        print(f"📚 Loading BPE tokenizer from:")
        print(f"  Vocab: {vocab_file}")
        print(f"  Merges: {merges_file}")
        tokenizer = load_bpe_tokenizer(vocab_file, merges_file)
    else:
        print("⚠️  BPE tokenizer files not found, using fallback character tokenizer")
        print("   Please ensure vocab.json and merges.txt are available")
        print("   Use --tokenizer argument to specify tokenizer directory")
        tokenizer = SimpleCharTokenizer()
    
    # Use vocab_size override if provided, otherwise use model config
    actual_vocab_size = vocab_size_override if vocab_size_override is not None else model_config['vocab_size']
    
    if vocab_size_override is not None:
        print(f"⚠️  Overriding vocab_size: {model_config['vocab_size']} → {vocab_size_override}")
    
    # Use the actual model architecture from training
    transformer_config = TransformerConfig(
        vocab_size=actual_vocab_size,
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
    
    # Load model weights with vocab size handling
    state_dict = checkpoint['model_state_dict']
    
    if vocab_size_override is not None and vocab_size_override != model_config['vocab_size']:
        print(f"⚠️  Adjusting embedding weights for vocab size mismatch")
        
        # Handle token embeddings
        original_embeddings = state_dict['token_embeddings.weight']
        original_vocab_size = original_embeddings.size(0)
        
        if vocab_size_override < original_vocab_size:
            # Truncate embeddings
            state_dict['token_embeddings.weight'] = original_embeddings[:vocab_size_override]
            print(f"   Truncated token embeddings: {original_vocab_size} → {vocab_size_override}")
        else:
            # Pad embeddings with random values
            padding_size = vocab_size_override - original_vocab_size
            padding = torch.randn(padding_size, original_embeddings.size(1)) * 0.1
            state_dict['token_embeddings.weight'] = torch.cat([original_embeddings, padding], dim=0)
            print(f"   Padded token embeddings: {original_vocab_size} → {vocab_size_override}")
        
        # Handle lm_head weights
        original_lm_head = state_dict['lm_head.weight']
        
        if vocab_size_override < original_vocab_size:
            # Truncate lm_head
            state_dict['lm_head.weight'] = original_lm_head[:vocab_size_override]
            print(f"   Truncated lm_head: {original_vocab_size} → {vocab_size_override}")
        else:
            # Pad lm_head with random values
            padding_size = vocab_size_override - original_vocab_size
            padding = torch.randn(padding_size, original_lm_head.size(1)) * 0.1
            state_dict['lm_head.weight'] = torch.cat([original_lm_head, padding], dim=0)
            print(f"   Padded lm_head: {original_vocab_size} → {vocab_size_override}")
    
    model.load_state_dict(state_dict)
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
    parser.add_argument('--tokenizer', type=str,
                       help='Path to tokenizer directory (containing vocab.json and merges.txt)')
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
    parser.add_argument('--vocab-size', type=int, 
                       help='Override vocab size for tokenizer compatibility (e.g., 10000, 50257)')
    
    args = parser.parse_args()
    
    # Validate checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint file not found: {args.checkpoint}")
        return 1
    
    try:
        # Load model
        print("🔄 Loading trained model...")
        model, tokenizer, config = load_model_and_config(args.checkpoint, args.config, args.tokenizer, args.vocab_size)
        
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
