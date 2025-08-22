#!/usr/bin/env python3
"""
Benchmarking script for measuring train_bpe performance.
Directly invokes train_bpe() and save_bpe() functions instead of using CLI wrapper.
"""

import argparse
import json
import os
import sys
import time
from typing import List

from cs336_basics.train_bpe import train_bpe, save_bpe




def ensure_pretokenization(input_path: str, output_dir: str):
    """
    Ensure pretokenization.pickle exists in output_dir.
    If not, create it by calling train_bpe with vocab_size=256.
    """
    pretok_path = os.path.join(output_dir, "pretokenization.pickle")
    
    if os.path.exists(pretok_path):
        print(f"Pretokenization file already exists: {pretok_path}")
        return
    
    print(f"Creating pretokenization file at {pretok_path}...")
    
    # Create pretokenization with a large enough vocab_size
    special_tokens = ["<|endoftext|>"]
    
    try:
        print(f"Training BPE with vocab_size=10000 to create pretokenization...")
        vocab, merges = train_bpe(
            input_path=input_path,
            vocab_size=260,
            special_tokens=special_tokens,
            save_pretokenization_path=pretok_path,
            use_optimization=True,
            debug=False
        )
        print(f"Pretokenization file created successfully at {pretok_path}")
    except Exception as e:
        print(f"Error creating pretokenization: {e}", file=sys.stderr)
        raise


def run_benchmark(input_path: str, vocab_size: int, output_dir: str, use_optimization: bool, num_runs: int = 3) -> List[float]:
    """
    Run the BPE training benchmark multiple times and return execution times.
    """
    times = []
    
    # Set up parameters for train_bpe
    special_tokens = ["<|endoftext|>"]
    pretokenization_path = os.path.join(output_dir, "pretokenization.pickle")
    
    print(f"\nRunning benchmark with:")
    print(f"  Input: {input_path}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Output dir: {output_dir}")
    print(f"  Use optimization: {use_optimization}")
    print(f"  Number of runs: {num_runs}")
    print(f"  Loading pretokenization from: {pretokenization_path}")
    print()
    
    for run_num in range(1, num_runs + 1):
        print(f"Run {run_num}/{num_runs}...")
        
        try:
            # Measure timing of train_bpe call
            start_time = time.time()
            
            vocab, merges = train_bpe(
                input_path=input_path,
                vocab_size=vocab_size,
                special_tokens=special_tokens,
                load_pretokenization_path=pretokenization_path,
                use_optimization=use_optimization,
                debug=False
            )
            
            # Save the BPE results
            save_bpe(vocab, merges, output_dir)
            
            exec_time = time.time() - start_time
            times.append(exec_time)
            print(f"  Execution time: {exec_time:.2f} seconds")
            print(f"  Final vocab size: {len(vocab)}")
            
        except Exception as e:
            print(f"  Error in run {run_num}: {e}", file=sys.stderr)
            continue
    
    return times


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark train_bpe performance",
        epilog="""Examples:
  %(prog)s --output-dir ./bench_results --vocab-size 300
  %(prog)s --output-dir ./bench_results --vocab-size 1000 --use-optimization
  %(prog)s --output-dir ./bench_results --vocab-size 500 --input-dataset data0/owt_train.txt --runs 5""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ.get("BENCHMARK_OUTPUT_DIR", "./benchmark_results"),
        help="Output directory for benchmark results (can also be set via BENCHMARK_OUTPUT_DIR env var)"
    )
    
    parser.add_argument(
        "--use-optimization",
        action="store_true",
        default=os.environ.get("USE_OPTIMIZATION", "").lower() in ("true", "1", "yes"),
        help="Enable --use-optimization flag (can also be set via USE_OPTIMIZATION env var)"
    )
    
    parser.add_argument(
        "--input-dataset",
        type=str,
        default=os.environ.get("INPUT_DATASET", "data0/TinyStoriesV2-GPT4-train.txt"),
        help="Input dataset path (can also be set via INPUT_DATASET env var)"
    )
    
    parser.add_argument(
        "--vocab-size",
        type=int,
        required=True,
        help="Vocabulary size for training"
    )
    
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of benchmark runs (default: 3)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input_dataset):
        print(f"Error: Input dataset '{args.input_dataset}' does not exist.", file=sys.stderr)
        sys.exit(1)
    
    if args.vocab_size <= 0:
        print(f"Error: Vocabulary size must be positive, got {args.vocab_size}.", file=sys.stderr)
        sys.exit(1)
    
    if args.runs <= 0:
        print(f"Error: Number of runs must be positive, got {args.runs}.", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory
    try:
        os.makedirs(args.output_dir, exist_ok=True)
    except OSError as e:
        print(f"Error: Could not create output directory '{args.output_dir}': {e}", file=sys.stderr)
        sys.exit(1)
    
    print("BPE Training Benchmark")
    print("=" * 50)
    
    # Ensure pretokenization file exists
    ensure_pretokenization(args.input_dataset, args.output_dir)
    
    # Run benchmark
    times = run_benchmark(
        args.input_dataset, 
        args.vocab_size, 
        args.output_dir, 
        args.use_optimization, 
        args.runs
    )
    
    # Report results
    if not times:
        print("No successful runs completed.", file=sys.stderr)
        sys.exit(1)
    
    print(f"\nBenchmark Results:")
    print(f"  Completed runs: {len(times)}/{args.runs}")
    for i, t in enumerate(times, 1):
        print(f"  Run {i}: {t:.2f} seconds")
    
    if len(times) > 1:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        print(f"\nSummary:")
        print(f"  Average time: {avg_time:.2f} seconds")
        print(f"  Min time: {min_time:.2f} seconds")
        print(f"  Max time: {max_time:.2f} seconds")


if __name__ == "__main__":
    main()