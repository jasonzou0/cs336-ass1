#!/usr/bin/env python3
"""
Train BPE tokenizer with configurable vocab_size (default: 50257 to match the model architecture)
"""

import os
import json
import argparse
from tests.adapters import run_train_bpe

def train_large_bpe_tokenizer(vocab_size=50257):
    """Train BPE tokenizer with configurable vocab_size (default: 50257)"""
    
    # Input data file
    input_file = "data/TinyStoriesV2-GPT4-train.txt"
    
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return
    
    # Output directory (adjust name based on vocab size)
    if vocab_size == 50257:
        output_dir = "data/tokenizer_50k"
    else:
        output_dir = f"data/tokenizer_{vocab_size}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🔄 Training BPE tokenizer with vocab_size={vocab_size}...")
    print(f"Input: {input_file}")
    print(f"Output: {output_dir}")
    
    # Train BPE tokenizer
    special_tokens = ["<|endoftext|>"]
    vocab, merges = run_train_bpe(
        input_path=input_file,
        vocab_size=vocab_size,
        special_tokens=special_tokens
    )
    
    print(f"✅ Training completed! Vocabulary size: {len(vocab)}")
    
    # Save vocabulary
    vocab_file = os.path.join(output_dir, "vocab.json")
    vocab_dict = {}
    for token_id, token_bytes in vocab.items():
        if token_id < 256:
            # Base bytes: encode as latin-1 for proper byte representation
            try:
                vocab_dict[str(token_id)] = token_bytes.decode('latin-1')
            except UnicodeDecodeError:
                # Fallback for problematic bytes
                vocab_dict[str(token_id)] = chr(token_id)
        else:
            # Merged tokens and special tokens: decode as UTF-8
            try:
                vocab_dict[str(token_id)] = token_bytes.decode('utf-8')
            except UnicodeDecodeError:
                vocab_dict[str(token_id)] = token_bytes.decode('utf-8', errors='replace')
    
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Vocabulary saved: {vocab_file}")
    
    # Save merges
    merges_file = os.path.join(output_dir, "merges.txt")
    with open(merges_file, 'w', encoding='utf-8') as f:
        for merge_pair in merges:
            token1, token2 = merge_pair
            try:
                token1_str = token1.decode('utf-8')
                token2_str = token2.decode('utf-8')
                f.write(f"{token1_str} {token2_str}\n")
            except UnicodeDecodeError:
                # Skip problematic merge pairs
                continue
    
    print(f"✅ Merges saved: {merges_file}")
    print(f"✅ Total merges: {len(merges)}")
    
    return vocab_file, merges_file

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description="Train BPE tokenizer with configurable vocabulary size")
    parser.add_argument(
        "--vocab-size", 
        type=int, 
        default=50257, 
        help="Vocabulary size for the BPE tokenizer (default: 50257)"
    )
    
    args = parser.parse_args()
    
    try:
        vocab_file, merges_file = train_large_bpe_tokenizer(vocab_size=args.vocab_size)
        print("\n🎉 BPE tokenizer training completed successfully!")
        print(f"📁 Vocabulary: {vocab_file}")
        print(f"📁 Merges: {merges_file}")
        
        # Adjust the output directory name in the usage example
        if args.vocab_size == 50257:
            tokenizer_dir = "data/tokenizer_50k"
        else:
            tokenizer_dir = f"data/tokenizer_{args.vocab_size}"
            
        print("\nYou can now use this tokenizer with:")
        print(f"python prepare_data.py --method bpe --reuse-tokenizer {tokenizer_dir}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()