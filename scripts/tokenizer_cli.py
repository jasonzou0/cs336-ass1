#!/usr/bin/env python3
"""
CLI tool for tokenizing text using a trained BPE tokenizer.

Usage:
    python tokenizer_cli.py --artifact_dir <tokenizer_dir> --input_text <text_file> --output_tokens <output_file>
"""

import argparse
import numpy as np
import time
from pathlib import Path
from cs336_basics.train_bpe import load_bpe
from cs336_basics.tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize text using a trained BPE tokenizer and save as numpy array"
    )
    parser.add_argument("--artifact_dir", required=True, 
                        help="Directory containing tokenizer artifacts (vocab.pkl and merges.pkl)")
    parser.add_argument("--input_text", required=True, help="Path to the text file to tokenize")
    parser.add_argument("--output_tokens", required=True, help="Path to save the tokenized output as numpy array")
    parser.add_argument("--progress_interval", type=int, default=100000,
                        help="Print progress every N lines (default: 100000)")
    
    args = parser.parse_args()
    
    # Construct paths to vocab and merges files
    artifact_dir = Path(args.artifact_dir)
    vocab_file = artifact_dir / "vocab.pkl"
    merges_file = artifact_dir / "merges.pkl"
    
    # Load BPE vocabulary and merges
    print(f"Loading BPE from {vocab_file} and {merges_file}")
    vocab, merges = load_bpe(vocab_file, merges_file)
    
    # Create tokenizer instance
    tokenizer = Tokenizer(vocab, merges, special_tokens=[])
    
    # Get file size for throughput calculation
    file_size_bytes = Path(args.input_text).stat().st_size
    
    # Tokenize the text using streaming interface
    print(f"Tokenizing text from {args.input_text} (size: {file_size_bytes:,} bytes)")
    print(f"Progress will be reported every {args.progress_interval} lines")
    
    start_time = time.time()
    token_list = []
    with open(args.input_text, 'r', encoding='utf-8') as f:
        # Use encode_iterable with progress reporting
        for token_id in tokenizer.encode_iterable(f, args.progress_interval):
            token_list.append(token_id)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Calculate throughput
    throughput_bytes_per_sec = file_size_bytes / elapsed_time if elapsed_time > 0 else 0
    throughput_mb_per_sec = throughput_bytes_per_sec / (1024 * 1024)
    
    # Convert to numpy array with dtype uint16
    tokens_array = np.array(token_list, dtype=np.uint16)
    
    # Save to output file
    print(f"Saving {len(tokens_array)} tokens to {args.output_tokens}")
    np.save(args.output_tokens, tokens_array)
    
    print(f"Successfully tokenized {file_size_bytes:,} bytes into {len(tokens_array):,} tokens")
    print(f"Tokenization time: {elapsed_time:.2f} seconds")
    print(f"Throughput: {throughput_mb_per_sec:.2f} MB/s ({throughput_bytes_per_sec:,.0f} bytes/s)")
    print(f"Saved to {args.output_tokens}")


if __name__ == "__main__":
    main()