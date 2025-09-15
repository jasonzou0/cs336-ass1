"""
Utility functions for validating token sequences and special token handling.
"""

import os
from .bpe_utils import load_artifact
import numpy as np
from .tokenizer import Tokenizer


def _contains_subsequence_optimized(main_array: np.array, sub_array: np.array) -> bool:
    """
    Highly optimized subsequence matching using numpy's sliding window approach.
    
    This implementation uses:
    1. Vectorized first-element matching to reduce candidates
    2. Early termination on bounds checking  
    3. Efficient numpy array comparison for final verification
    
    Time complexity: O(n + m*k) where n=len(main_array), m=matches of first element, k=len(sub_array)
    Much better than naive O(n*k) approach.
    
    Args:
        main_array: Array to search in
        sub_array: Subsequence to find
        
    Returns:
        True if sub_array is found as a contiguous subsequence in main_array
    """
    len_main = len(main_array)
    len_sub = len(sub_array)

    if len_sub > len_main or len_sub == 0:
        return False
    
    if len_sub == 1:
        # Single element search - use numpy's optimized search
        return sub_array[0] in main_array
    
    # Vectorized search for first element positions
    first_element = sub_array[0]
    potential_starts = np.where(main_array == first_element)[0]
    
    # Early termination: filter positions that would exceed bounds
    valid_starts = potential_starts[potential_starts <= len_main - len_sub]
    
    # For each valid starting position, check the full pattern
    for start in valid_starts:
        # Vectorized comparison of the entire window
        if np.array_equal(main_array[start:start + len_sub], sub_array):
            return True
    
    return False


def _check_special_tokens_tokenization(
    tokens: np.array,
    vocab: dict[bytes, int], 
    merges: list[tuple[bytes, bytes]], 
    special_tokens: list[str] = ['<|endoftext|>']
) -> dict[str, bool]:
    """
    Check if special tokens are correctly tokenized in a token sequence.
    
    This function creates a tokenizer WITHOUT the special tokens and checks if any
    of the special tokens appear as incorrectly tokenized sequences in the data.
    If special tokens were handled correctly during original tokenization, they
    should NOT appear as these incorrect sequences.
    
    Args:
        tokens: Numpy array of token IDs to validate
        vocab: Vocabulary mapping bytes to token IDs  
        merges: List of merge rules as (bytes, bytes) tuples
        special_tokens: List of special tokens to check (default: ['<|endoftext|>'])
        
    Returns:
        Dictionary mapping special token -> bool (True if incorrectly tokenized found)
    """

    results = {}
    
    # Create tokenizer WITHOUT special tokens to generate bad tokenizations
    tokenizer_without_special = Tokenizer(vocab, merges, special_tokens=[])
    
    for special_token in special_tokens:
        # Generate what the token sequence would look like if tokenized incorrectly
        # (i.e., without proper special token handling)
        bad_token_sequence = np.array(tokenizer_without_special.encode(special_token))
        
        # Check if this bad sequence appears in the data
        found_bad_tokenization = _contains_subsequence_optimized(tokens, bad_token_sequence)
        results[special_token] = found_bad_tokenization
        
    return results


def validate_special_tokens(
    tokens_file: str | os.PathLike,
    tokenizer_artifact_dir: str | os.PathLike,
) -> bool:
    """
    Validate that special tokens are correctly tokenized in a token sequence file.
    
    Args:
        tokens_file: Path to the numpy file containing token IDs
        tokenizer_artifact_dir: Directory containing tokenizer artifacts (vocab.pkl, merges.pkl, special_tokens.pkl)
    """
    # Load tokens
    mmap_mode = None
    if os.path.getsize(tokens_file) > 512 * 1024 * 1024:
        mmap_mode = "r"
    tokens = np.load(tokens_file, mmap_mode=mmap_mode)
    
    # Load tokenizer artifacts
    vocab = load_artifact(os.path.join(tokenizer_artifact_dir, "vocab.pkl"))
    merges = load_artifact(os.path.join(tokenizer_artifact_dir, "merges.pkl"))
    special_tokens = load_artifact(os.path.join(tokenizer_artifact_dir, "special_tokens.pkl"))
    
    # Validate special tokens
    return validate_special_tokens_in_data(tokens, vocab, merges, special_tokens)


def validate_special_tokens_in_data(
    tokens: np.array,
    vocab: dict[bytes, int], 
    merges: list[tuple[bytes, bytes]], 
    special_tokens: list[str] = None,
    verbose: bool = True
) -> bool:
    """
    Validate that special tokens are correctly tokenized in data and optionally print results.
    
    Args:
        tokens: Token data to validate
        vocab, merges: Tokenizer artifacts
        special_tokens: Special tokens to check
        verbose: Whether to print detailed results
        
    Returns:
        True if all special tokens are correctly tokenized (no bad sequences found)
    """
    results = _check_special_tokens_tokenization(tokens, vocab, merges, special_tokens)
    
    all_good = True
    for special_token, has_bad_tokenization in results.items():
        if has_bad_tokenization:
            all_good = False
            if verbose:
                print(f"❌ BAD TOKENIZATION FOUND for '{special_token}'")
        else:
            if verbose:
                print(f"✅ GOOD TOKENIZATION for '{special_token}'")
    
    return all_good