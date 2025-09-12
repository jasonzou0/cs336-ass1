#!/usr/bin/env python3
"""
CLI tool for tokenizing text using a trained BPE tokenizer with checkpointing support.

Usage:
    python tokenizer_cli.py --artifact_dir <tokenizer_dir> --input_text <text_file> --output_directory <output_dir>
    python tokenizer_cli.py --artifact_dir <tokenizer_dir> --input_text <text_file> --output_directory <output_dir> --resume_checkpoint
"""

import argparse
import numpy as np
import time
import pickle
import os
from pathlib import Path
from cs336_basics.bpe_utils import load_bpe
from cs336_basics.tokenizer import Tokenizer


def save_checkpoint(checkpoint_path: Path, tokens: list, lines_processed: int, byte_position: int):
    """Save checkpoint with tokens and position information."""
    checkpoint_data = {
        'tokens': tokens,
        'lines_processed': lines_processed,
        'byte_position': byte_position,
        'timestamp': time.time()
    }
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    print(f"Checkpoint saved: {lines_processed:,} lines, {len(tokens):,} tokens, offset: {byte_position:,} bytes")


def load_checkpoint(checkpoint_path: Path):
    """Load checkpoint data if it exists."""
    if checkpoint_path.exists():
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded checkpoint: {data['lines_processed']:,} lines, {len(data['tokens']):,} tokens, offset: {data['byte_position']:,} bytes")
        return data
    return None


def get_output_paths(output_directory: Path, input_text_path: Path):
    """Generate standardized output paths based on input filename."""
    input_basename = Path(input_text_path).stem  # filename without extension
    tokens_path = output_directory / f"{input_basename}_tokens.npy"
    checkpoint_path = output_directory / f"{input_basename}_checkpoint.pkl"
    return tokens_path, checkpoint_path


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize text using a trained BPE tokenizer and save as numpy array"
    )
    parser.add_argument("--artifact_dir", required=True, 
                        help="Directory containing tokenizer artifacts (vocab.pkl and merges.pkl)")
    parser.add_argument("--input_text", required=True, help="Path to the text file to tokenize")
    parser.add_argument("--output_directory", required=True, help="Directory to save tokenized output and checkpoints")
    parser.add_argument("--progress_interval", type=int, default=100000,
                        help="Print progress every N lines (default: 100000)")
    parser.add_argument("--checkpoint_interval", type=int, default=10000000,
                        help="Save checkpoint every N lines (default: 10000000)")
    parser.add_argument("--resume_checkpoint", action="store_true",
                        help="Resume from existing checkpoint if available")
    parser.add_argument("--cache_size", type=int, default=16384,
                        help="LRU cache size for tokenization (default: 16384, use 0 to disable)")
    parser.add_argument("--no_cython", action="store_true",
                        help="Disable Cython optimization, use Python implementation")
    
    args = parser.parse_args()
    
    # Construct paths
    artifact_dir = Path(args.artifact_dir)
    vocab_file = artifact_dir / "vocab.pkl"
    merges_file = artifact_dir / "merges.pkl"
    special_tokens_file = artifact_dir / "special_tokens.pkl"
    output_dir = Path(args.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output file paths
    tokens_path, checkpoint_path = get_output_paths(output_dir, args.input_text)
    
    # Load BPE vocabulary and merges
    print(f"Loading BPE from {vocab_file} and {merges_file}")
    vocab, merges, special_tokens = load_bpe(vocab_file, merges_file, special_tokens_file)
    print(f"Loaded vocab size: {len(vocab)}, merges size: {len(merges)}, special tokens: {special_tokens}")
    
    # Determine Cython usage (default is to use it if available, unless explicitly disabled)
    # Don't pass use_cython unless explicitly set by user, let tokenizer decide
    tokenizer_kwargs = {
        'progress_interval': args.progress_interval,
        'cache_size': args.cache_size,
    }

    if args.no_cython:
        tokenizer_kwargs['use_cython'] = False

    # Create tokenizer instance with progress tracking and cache configuration
    tokenizer = Tokenizer(vocab, merges, special_tokens=special_tokens, **tokenizer_kwargs)
    
    # Get file size for display
    file_size_bytes = Path(args.input_text).stat().st_size
    
    # Initialize or load checkpoint
    token_list = []
    lines_processed = 0
    start_byte_offset = 0
    
    if args.resume_checkpoint:
        checkpoint_data = load_checkpoint(checkpoint_path)
        if checkpoint_data:
            token_list = checkpoint_data['tokens']
            lines_processed = checkpoint_data['lines_processed']
            start_byte_offset = checkpoint_data['byte_position']
            print(f"Resuming from line {lines_processed:,} at byte offset {start_byte_offset:,}")
        else:
            print("No checkpoint found, starting from beginning")
    
    # Tokenize the text using streaming interface with checkpointing
    print(f"Tokenizing text from {args.input_text} (size: {file_size_bytes:,} bytes)")
    print(f"Progress will be reported every {args.progress_interval} encode() calls")
    print(f"Checkpoints will be saved every {args.checkpoint_interval:,} lines")
    
    start_time = time.time()
    current_lines = 0
    session_bytes_processed = 0  # Track bytes processed in this session
    
    with open(args.input_text, 'r', encoding='utf-8') as f:
        # Seek directly to resume position if resuming
        if start_byte_offset > 0:
            print(f"Seeking directly to byte offset {start_byte_offset:,}")
            f.seek(start_byte_offset)
        
        while True:
            line = f.readline()
            if not line:  # EOF
                break
                
            current_lines += 1
            total_lines = lines_processed + current_lines
            
            # Track bytes processed in this session
            session_bytes_processed += len(line.encode('utf-8'))
            
            # Tokenize the line (progress tracking handled inside encode())
            line_tokens = tokenizer.encode(line)
            token_list.extend(line_tokens)
            
            # Checkpoint saving - capture position after processing this line
            if total_lines % args.checkpoint_interval == 0:
                elapsed_time = time.time() - start_time
                throughput_mb_per_sec = (session_bytes_processed / elapsed_time) / (1024 * 1024) if elapsed_time > 0 else 0
                print(f"Tokenization throughput: {throughput_mb_per_sec:.2f} MB/s")
                current_position = f.tell()  # Get position after reading this line
                save_checkpoint(checkpoint_path, token_list, total_lines, current_position)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Calculate throughput based on bytes processed in this session
    throughput_bytes_per_sec = session_bytes_processed / elapsed_time if elapsed_time > 0 else 0
    throughput_mb_per_sec = throughput_bytes_per_sec / (1024 * 1024)
    
    # Convert to numpy array with dtype uint16
    tokens_array = np.array(token_list, dtype=np.uint16)
    
    # Save to output file
    print(f"Saving {len(tokens_array)} tokens to {tokens_path}")
    np.save(tokens_path, tokens_array)
    
    total_lines_processed = lines_processed + current_lines
    print(f"Successfully tokenized {total_lines_processed:,} lines into {len(tokens_array):,} tokens")
    print(f"Session processed {session_bytes_processed:,} bytes in {elapsed_time:.2f} seconds")
    print(f"Total throughput: {throughput_mb_per_sec:.2f} MB/s ({throughput_bytes_per_sec:,.0f} bytes/s)")
    print(f"Saved to {tokens_path}")


if __name__ == "__main__":
    main()