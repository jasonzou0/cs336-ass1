# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

from typing import List
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
def encode_one_tuple_uncached_cy(
    bytes token_bytes,
    dict vocab,
    dict merges
):
    """
    Cython-optimized version of _encode_one_tuple_uncached.
    
    Encodes a single token into its corresponding ID using the vocabulary.
    It iteratively merges byte pairs based on the merges list in the original merge discovery order,
    until no more merges can be applied.
    
    Args:
        token_bytes: Bytes representing the token to encode.
        vocab: Vocabulary mapping bytes to token IDs
        merges: Merges mapping (bytes, bytes) pairs to merge indices
    Returns:
        List of token IDs in the vocabulary.
    """
    # Check if the token already exists in vocab as a complete token (e.g., special tokens)
    if token_bytes in vocab:
        return [vocab[token_bytes]]
    
    # Break up token_bytes into list of single-byte tokens for merging.
    cdef list tokens = []
    cdef int token_len = len(token_bytes)
    cdef int i
    
    for i in range(token_len):
        tokens.append(token_bytes[i:i+1])
    
    # Cython variables for optimization
    cdef int tokens_len
    cdef int best_merge_idx
    cdef int best_pos
    cdef int merge_idx
    cdef tuple pair
    cdef bytes merged_token
    cdef bytes token_i, token_i_plus_1
    
    while len(tokens) > 1:
        # Collect all mergeable pairs in a single pass
        merge_candidates = []
        tokens_len = len(tokens)
        
        for i in range(tokens_len - 1):
            token_i = tokens[i]
            token_i_plus_1 = tokens[i + 1]
            pair = (token_i, token_i_plus_1)
            if pair in merges:
                merge_candidates.append((i, merges[pair]))
        
        if not merge_candidates:
            # No more merges possible
            break
        
        # Find the highest priority (lowest index) merge
        best_pos = merge_candidates[0][0]
        best_merge_idx = merge_candidates[0][1]
        
        for i in range(1, len(merge_candidates)):
            pos, merge_idx = merge_candidates[i]
            if merge_idx < best_merge_idx:
                best_merge_idx = merge_idx
                best_pos = pos
        
        # Perform the merge using list slicing (optimized in Cython)
        merged_token = tokens[best_pos] + tokens[best_pos + 1]
        tokens = tokens[:best_pos] + [merged_token] + tokens[best_pos + 2:]
    
    # Convert the final tokens to IDs
    return [vocab[token] for token in tokens]