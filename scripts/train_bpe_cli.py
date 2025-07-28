#!/usr/bin/env python3
"""
CLI tool for training BPE tokenizer using the cs336_basics.train_bpe module.
"""

import argparse
import cProfile
import json
import os
import pstats
import sys
import time

from cs336_basics.train_bpe import train_bpe, save_bpe


def print_basic_profile_stats(profiler):
    """Print basic profiling statistics."""
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    print("\nTop 20 functions by cumulative time:")
    stats.print_stats(20)


def generate_profile_svg(profiler, output_dir):
    """Generate SVG profiling output using gprof2dot and graphviz."""
    profile_path = os.path.join(output_dir, "profile.prof")
    svg_path = os.path.join(output_dir, "profile.svg")

    # Save profile data
    profiler.dump_stats(profile_path)

    try:
        import subprocess
        result = subprocess.run([
            'gprof2dot', '-f', 'pstats', profile_path
        ], capture_output=True, text=True, check=True)

        dot_output = result.stdout

        # Convert dot to SVG using graphviz
        svg_result = subprocess.run([
            'dot', '-Tsvg'
        ], input=dot_output, capture_output=True, text=True, check=True)

        with open(svg_path, 'w') as f:
            f.write(svg_result.stdout)

        print(f"Profiling SVG saved to: {svg_path}")
        print(f"Open in browser: file://{os.path.abspath(svg_path)}")
        return True

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Warning: Could not generate SVG profiling output: {e}")
        print("Make sure 'gprof2dot' and 'graphviz' are installed:")
        print("  pip install gprof2dot")
        print("  brew install graphviz  # on macOS")
        return False


def run_with_profiling(training_func, output_dir):
    """Run a function with profiling enabled and generate SVG visualization."""
    profiler = cProfile.Profile()
    profiler.enable()

    try:
        result = training_func()
    finally:
        profiler.disable()

    # Always try to generate SVG, fall back to basic stats if it fails
    if not generate_profile_svg(profiler, output_dir):
        print_basic_profile_stats(profiler)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Train a BPE tokenizer and save results to disk",
        epilog="""Examples:
  %(prog)s --input-path data/corpus.txt --vocab-size 1000 --output-dir ./tokenizer_output
  %(prog)s --input-path data/corpus.txt --vocab-size 500 --output-dir ./out --special-tokens '[\"<|endoftext|>\", \"<|pad|>\"]'
  %(prog)s --input-path data/corpus.txt --vocab-size 500 --output-dir ./out --profile""",
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

    parser.add_argument(
        "--stop-at-merge-num",
        type=int,
        default=None,
        help="Stop training after this many merges (default: None, which means train until vocab size is reached)"
    )

    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling of the training process and generate SVG visualization"
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
    def run_training():
        print(f"Training BPE tokenizer...")
        print(f"  Input path: {args.input_path}")
        print(f"  Vocabulary size: {args.vocab_size}")
        print(f"  Special tokens: {special_tokens}")
        print(f"  Output directory: {args.output_dir}")
        print(f"  Stop at merge number: {args.stop_at_merge_num if args.stop_at_merge_num is not None else 'None'}")
        if args.profile:
            print(f"  Profiling enabled: SVG will be generated")
        print()

        start_time = time.time()
        vocab, merges = train_bpe(
            input_path=args.input_path,
            vocab_size=args.vocab_size,
            special_tokens=special_tokens,
            stop_at_merge_num=args.stop_at_merge_num,
        )
        execution_time = time.time() - start_time
        actual_vocab_size = len(vocab)
        print(f"BPE Training completed:")
        print(f"  Input vocab size: {args.vocab_size}")
        print(f"  Final vocab size: {actual_vocab_size}")
        print(f"  Execution time: {execution_time:.2f} seconds")

        # Save the results to disk
        save_bpe(vocab, merges, args.output_dir)
        print("BPE tokenizer saved successfully!")

    try:
        if args.profile:
            # Run with profiling
            run_with_profiling(run_training, args.output_dir)
        else:
            # Run without profiling
            run_training()

    except Exception as e:
        print(f"Error during training: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
