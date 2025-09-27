#!/usr/bin/env python3
"""
Data preparation script for CS336 training

This script tokenizes text files and converts them to binary format
expected by the training script.

Usage:
    python prepare_data.py --input data/TinyStoriesV2-GPT4-train.txt --output data/train.bin --vocab-size 50000
    python prepare_data.py --input data/owt_train.txt --output data/train.bin --vocab-size 50000
"""

import argparse
import os
import json
import numpy as np
from pathlib import Path
import tempfile
from tqdm import tqdm

# Import our tokenizer implementations
from tests.adapters import run_train_bpe


def train_bpe_tokenizer(text_file: str, vocab_size: int, output_dir: str):
    """Train a BPE tokenizer on the text file"""
    print(f"Training BPE tokenizer with vocab_size={vocab_size} on {text_file}")
    
    # Create temporary files for BPE training
    merges_file = os.path.join(output_dir, "merges.txt")
    vocab_file = os.path.join(output_dir, "vocab.json")
    
    # Train BPE tokenizer
    special_tokens = ["<|endoftext|>"]  # Common special token for language models
    vocab, merges = run_train_bpe(
        input_path=text_file,
        vocab_size=vocab_size,
        special_tokens=special_tokens
    )
    
    # Save vocabulary
    vocab_dict = {token_id: token.decode('utf-8', errors='ignore') for token_id, token in vocab.items()}
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
    
    # Save merges
    with open(merges_file, 'w', encoding='utf-8') as f:
        for merge_pair in merges:
            token1, token2 = merge_pair
            f.write(f"{token1.decode('utf-8', errors='ignore')} {token2.decode('utf-8', errors='ignore')}\n")
    
    return vocab_file, merges_file


def load_bpe_tokenizer(vocab_file: str, merges_file: str):
    """Load trained BPE tokenizer"""
    from cs336_basics.my_tokenizer import Tokenizer
    
    tokenizer = Tokenizer()
    
    # Load vocabulary
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    # Load merges
    with open(merges_file, 'r', encoding='utf-8') as f:
        merges_lines = f.read().strip().split('\n')
    
    # Set up tokenizer (this is a simplified version - you may need to adjust based on your tokenizer implementation)
    tokenizer.vocab = vocab
    tokenizer.merges = merges_lines
    
    return tokenizer


def tokenize_file(input_file: str, tokenizer, output_file: str, chunk_size: int = 10000):
    """Tokenize a text file and save as binary"""
    print(f"Tokenizing {input_file} -> {output_file}")
    
    # Get file size for progress bar
    file_size = os.path.getsize(input_file)
    
    all_tokens = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        with tqdm(total=file_size, unit='B', unit_scale=True, desc="Tokenizing") as pbar:
            chunk_tokens = []
            
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                
                # Tokenize the chunk
                try:
                    if hasattr(tokenizer, 'encode'):
                        tokens = tokenizer.encode(chunk)
                    else:
                        # Fallback - you may need to adjust this based on your tokenizer interface
                        tokens = tokenizer.tokenize(chunk)
                    
                    chunk_tokens.extend(tokens)
                    
                    # If we have a lot of tokens, save them and clear memory
                    if len(chunk_tokens) > 1000000:  # 1M tokens
                        all_tokens.extend(chunk_tokens)
                        chunk_tokens = []
                    
                except Exception as e:
                    print(f"Error tokenizing chunk: {e}")
                    # Skip problematic chunks
                    pass
                
                pbar.update(len(chunk.encode('utf-8')))
            
            # Add remaining tokens
            if chunk_tokens:
                all_tokens.extend(chunk_tokens)
    
    print(f"Total tokens: {len(all_tokens):,}")
    
    # Convert to numpy array and save as binary
    tokens_array = np.array(all_tokens, dtype=np.int32)
    tokens_array.tofile(output_file)
    
    print(f"Saved {len(tokens_array):,} tokens to {output_file}")
    return len(tokens_array)


def simple_whitespace_tokenize(input_file: str, output_file: str, max_vocab_size: int = 50000):
    """Simple whitespace tokenization as fallback"""
    print(f"Using simple whitespace tokenization for {input_file}")
    
    # Build vocabulary
    print("Building vocabulary...")
    word_counts = {}
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Counting words"):
            words = line.strip().split()
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
    
    # Get most frequent words
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    vocab_words = [word for word, count in sorted_words[:max_vocab_size-4]]  # Reserve space for special tokens
    
    # Add special tokens
    special_tokens = ['<pad>', '<unk>', '<bos>', '<eos>']
    vocab = {token: i for i, token in enumerate(special_tokens + vocab_words)}
    
    print(f"Vocabulary size: {len(vocab)}")
    
    # Tokenize file
    print("Tokenizing...")
    all_tokens = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Tokenizing"):
            words = line.strip().split()
            tokens = [vocab.get(word, vocab['<unk>']) for word in words]
            all_tokens.extend(tokens)
    
    # Save tokens
    tokens_array = np.array(all_tokens, dtype=np.int32)
    tokens_array.tofile(output_file)
    
    print(f"Saved {len(tokens_array):,} tokens to {output_file}")
    
    # Save vocabulary
    vocab_file = output_file.replace('.bin', '_vocab.json')
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, indent=2)
    
    print(f"Saved vocabulary to {vocab_file}")
    return len(tokens_array)


def main():
    parser = argparse.ArgumentParser(description='Prepare training data by tokenizing text files')
    
    parser.add_argument('--input', type=str, required=True, help='Input text file')
    parser.add_argument('--output', type=str, required=True, help='Output binary file')
    parser.add_argument('--vocab-size', type=int, default=50000, help='Vocabulary size for BPE')
    parser.add_argument('--method', type=str, choices=['bpe', 'whitespace'], default='bpe',
                       help='Tokenization method')
    parser.add_argument('--reuse-tokenizer', type=str, help='Directory with existing vocab.json and merges.txt')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    # Create output directory
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        if args.method == 'bpe':
            if args.reuse_tokenizer and os.path.exists(args.reuse_tokenizer):
                # Reuse existing tokenizer
                vocab_file = os.path.join(args.reuse_tokenizer, 'vocab.json')
                merges_file = os.path.join(args.reuse_tokenizer, 'merges.txt')
                
                if os.path.exists(vocab_file) and os.path.exists(merges_file):
                    print(f"Reusing tokenizer from {args.reuse_tokenizer}")
                else:
                    print(f"Error: vocab.json or merges.txt not found in {args.reuse_tokenizer}")
                    return 1
            else:
                # Train new tokenizer
                tokenizer_dir = os.path.join(output_dir, 'tokenizer')
                os.makedirs(tokenizer_dir, exist_ok=True)
                
                vocab_file, merges_file = train_bpe_tokenizer(
                    args.input, args.vocab_size, tokenizer_dir
                )
            
            # Load tokenizer and tokenize
            try:
                tokenizer = load_bpe_tokenizer(vocab_file, merges_file)
                num_tokens = tokenize_file(args.input, tokenizer, args.output)
            except Exception as e:
                print(f"Error with BPE tokenization: {e}")
                print("Falling back to whitespace tokenization...")
                num_tokens = simple_whitespace_tokenize(args.input, args.output, args.vocab_size)
        
        else:  # whitespace
            num_tokens = simple_whitespace_tokenize(args.input, args.output, args.vocab_size)
        
        print(f"\nSuccess! Processed {num_tokens:,} tokens")
        print(f"Output file: {args.output}")
        print(f"File size: {os.path.getsize(args.output) / 1024 / 1024:.1f} MB")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())