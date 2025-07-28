#!/usr/bin/env python3
"""
CLI tool for training BPE tokenizer using the cs336_basics.train_bpe module.
"""

import argparse
import json
import os
import sys

from cs336_basics.train_bpe import train_bpe, save_bpe


def main():
    parser = argparse.ArgumentParser(
        description="Train a BPE tokenizer and save results to disk",
        epilog="""Examples:
  %(prog)s --input-path data/corpus.txt --vocab-size 1000 --output-dir ./tokenizer_output
  %(prog)s --input-path data/corpus.txt --vocab-size 500 --output-dir ./out --special-tokens '[\"<|endoftext|>\", \"<|pad|>\"]'""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--input-path", 
        type=str, 
        required=True,
        help="Path to the input corpus file"
    )
    
    parser.add_argument(
        "--vocab-size", 
        type=int, 
        required=True,
        help="Target vocabulary size"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        required=True,
        help="Directory to save vocab.pkl and merges.pkl"
    )
    
    parser.add_argument(
        "--special-tokens", 
        type=str, 
        default='["<|endoftext|>"]',
        help="JSON list of special tokens (default: [\"<|endoftext|>\"])"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input_path):
        print(f"Error: Input file '{args.input_path}' does not exist.", file=sys.stderr)
        sys.exit(1)
    
    # Validate vocab size
    if args.vocab_size <= 0:
        print(f"Error: Vocabulary size must be positive, got {args.vocab_size}.", file=sys.stderr)
        sys.exit(1)
    
    # Parse special tokens JSON
    try:
        special_tokens = json.loads(args.special_tokens)
        if not isinstance(special_tokens, list) or not all(isinstance(token, str) for token in special_tokens):
            raise ValueError("Special tokens must be a list of strings")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error: Invalid special tokens JSON: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    try:
        os.makedirs(args.output_dir, exist_ok=True)
    except OSError as e:
        print(f"Error: Could not create output directory '{args.output_dir}': {e}", file=sys.stderr)
        sys.exit(1)
    
    # Train the BPE tokenizer
    try:
        print(f"Training BPE tokenizer...")
        print(f"  Input path: {args.input_path}")
        print(f"  Vocabulary size: {args.vocab_size}")
        print(f"  Special tokens: {special_tokens}")
        print(f"  Output directory: {args.output_dir}")
        print()
        
        vocab, merges = train_bpe(
            input_path=args.input_path,
            vocab_size=args.vocab_size,
            special_tokens=special_tokens
        )
        
        print(f"Training completed. Vocabulary size: {len(vocab)}, Merges: {len(merges)}")
        
        # Save the results to disk
        save_bpe(vocab, merges, args.output_dir)
        print("BPE tokenizer saved successfully!")
        
    except Exception as e:
        print(f"Error during training: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
