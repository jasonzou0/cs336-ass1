import os
import pickle
from collections import defaultdict
from typing import BinaryIO
from multiprocessing import Pool

import regex as re

def get_num_processes() -> int:
    """Get the number of processes to use from environment variable or default to 4."""
    return int(os.environ.get("NUM_PROCESS", 4))

def _find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def _decode_bytes_debug(b: bytes) -> str:
    """
    Decode bytes to a string, replacing errors with a placeholder.
    Useful for debugging byte sequences that may not decode cleanly.
    """
    return b.decode("utf-8", errors="replace")

def get_pretokenizer(tokenizer_name: str="default") -> re.Pattern:
    """
    Get a pretokenizer pattern based on the tokenizer name.
    """
    if tokenizer_name == "ws":
        return re.compile(r"\S+")
    elif tokenizer_name == "default":
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        return re.compile(PAT, flags=re.UNICODE)
    else:
        raise ValueError(f"Unknown tokenizer: {tokenizer_name}")

def _process_chunk(args: tuple) -> dict[tuple, int]:
    """
    Worker function to process a single chunk of the corpus.

    Args:
        args: Tuple containing (input_path, start, end, pretokenizer_name, special_tokens, debug)

    Returns:
        Dictionary mapping tuples of bytes to their counts for this chunk
    """
    input_path, start, end, pretokenizer_name, special_tokens, debug = args

    pretokenizer = get_pretokenizer(pretokenizer_name)

    chunk_tokens_counts: dict[tuple, int] = defaultdict(int)

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

        if debug:
            print(f"Processing chunk from {start} to {end}: {chunk[:100]}...")

        # Run pre-tokenization on the chunk and store the counts for each pre-token
        mini_chunks = re.split("|".join([re.escape(token) for token in special_tokens]), chunk)
        for mini_chunk in mini_chunks:
            for match in pretokenizer.finditer(mini_chunk):
                token = match.group()
                byte_tuple = tuple(bytes([b]) for b in token.encode("utf-8"))
                chunk_tokens_counts[byte_tuple] += 1

    return dict(chunk_tokens_counts)  # Convert defaultdict to regular dict for pickling

def pretokenize_corpus(
    input_path: str | os.PathLike,
    pretokenizer_name: str,
    special_tokens: list[str],
    debug: bool = False,
) -> dict[tuple, int]:
    """
    Pretokenize a corpus and return token counts.
    
    Args:
        input_path: Path to the input corpus file
        pretokenizer: Compiled regex pattern for pretokenization
        special_tokens: List of special tokens to handle separately
        debug: Whether to print debug information

    Returns:
        Dictionary mapping tuples of bytes to their counts
    """
    num_processes = get_num_processes()
    print(f"Using {num_processes} processes for BPE training")

    with open(input_path, "rb") as f:
        boundaries = _find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # Prepare arguments for each worker process
    chunk_args = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        chunk_args.append((
            input_path,
            start,
            end,
            pretokenizer_name,
            special_tokens,
            debug
        ))

    # Process chunks in parallel using multiprocessing
    with Pool(processes=num_processes) as pool:
        chunk_results = pool.map(_process_chunk, chunk_args)

    # Merge results from all processes
    tokens_counts: dict[tuple, int] = defaultdict(int)
    for chunk_tokens_counts in chunk_results:
        for token_tuple, count in chunk_tokens_counts.items():
            tokens_counts[token_tuple] += count

    if debug:
        print(f"Pretokenization results: {len(tokens_counts)} tokens. (Only show first 10)")
        for i, (token_tuple, count) in enumerate(tokens_counts.items()):
            token_strs = [_decode_bytes_debug(token) for token in token_tuple]
            print(f"  {i+1}. '{token_strs}' (count: {count})")
            if i >= 9:  # Only show first 10
                break
        print()
    
    return tokens_counts

def _find_most_common_pair(
    tokens_counts: dict[tuple, int],
    merge_step: int,
    debug: bool = False,
) -> tuple[bytes, bytes] | None:
    """
    Find the most common byte pair in the token counts with optional debug printing.
    
    Args:
        tokens_counts: Dictionary mapping tuples of bytes to their counts
        merge_step: Current merge step number (for debug output)
        debug: Whether to print debug information
        
    Returns:
        Most common byte pair, or None if no pairs found
    """
    # Count the occurrences of each pair of bytes in the token counts
    bytepair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)

    for bytes_tuple, count in tokens_counts.items():
        if len(bytes_tuple) == 1:
            continue  # Skip single-byte tokens
        for i in range(len(bytes_tuple) - 1):
            pair = (bytes_tuple[i], bytes_tuple[i + 1])
            bytepair_counts[pair] += count

    # Find the most common byte pair
    if not bytepair_counts:
        return None  # No more pairs to merge

    # TODO: double check if the tie breaker function is implemented correctly.
    most_common_pair = max(bytepair_counts, key=lambda bytepair: (bytepair_counts[bytepair], bytepair))

    if debug:
        print(f"Merge step {merge_step}:")
        print(f"  Total byte pairs found: {len(bytepair_counts)}")
        for pair, count in bytepair_counts.items():
            pair_str= _decode_bytes_debug(pair[0]) + ", " + _decode_bytes_debug(pair[1])
            print(f"    {pair_str} (count: {count})")
        merged_token_str = _decode_bytes_debug(most_common_pair[0] + most_common_pair[1])
        print(f"  Selected merge: {most_common_pair} -> '{merged_token_str}' (count: {bytepair_counts[most_common_pair]})")
        print()

    return most_common_pair

def _update_vocab_with_merge(
    most_common_pair: tuple[bytes, bytes],
    vocab: dict[bytes, int],
    token_id: int,
    merges: list[tuple[bytes, bytes]],
) -> tuple[bytes, int]:
    """
    Form new token from merge pair and update vocabulary.
    
    Args:
        most_common_pair: The pair of bytes to merge
        vocab: Current vocabulary dictionary
        token_id: Next available token ID
        merges: List of merges to append to
        
    Returns:
        Tuple of (new_token, updated_token_id)
    """
    # Merge the most common pair
    merges.append(most_common_pair)

    # Create a new token by merging the two bytes
    first, second = most_common_pair
    new_token = first + second
    if new_token not in vocab:
        vocab[new_token] = token_id
        token_id += 1
    
    return new_token, token_id

def _update_tokens_counts(
    tokens_counts: dict[tuple, int],
    most_common_pair: tuple[bytes, bytes],
    new_token: bytes,
) -> dict[tuple, int]:
    """
    Update token counts by merging occurrences of the most common pair.
    
    Args:
        tokens_counts: Current token counts dictionary
        most_common_pair: The pair of bytes that was merged
        new_token: The new token created from the merge
        
    Returns:
        Updated token counts dictionary
    """
    # Update the token counts
    new_tokens_counts = {}
    for bytes_tuple, count in list(tokens_counts.items()):
        if len(bytes_tuple) == 1:
            continue
        new_bytes_tuple = []
        i = 0
        merge_happened = False
        while i < len(bytes_tuple):
            if i < len(bytes_tuple) - 1 and (bytes_tuple[i], bytes_tuple[i + 1]) == most_common_pair:
                new_bytes_tuple.append(new_token)
                merge_happened = True
                i += 2
            else:
                new_bytes_tuple.append(bytes_tuple[i])
                i += 1
        if merge_happened:
            new_tokens_counts[tuple(new_bytes_tuple)] = count
        else:
            # If no merge happened, keep the original tuple
            new_tokens_counts[bytes_tuple] = count

    return new_tokens_counts

def perform_bpe_merges(
    tokens_counts: dict[tuple, int],
    vocab: dict[bytes, int],
    vocab_size: int,
    stop_at_merge_num: int | None,
    debug: bool = False,
) -> tuple[dict[bytes, int], list[tuple[bytes, bytes]]]:
    """
    Perform BPE merges on the token counts to build vocabulary and merges.

    Args:
        tokens_counts: Dictionary mapping tuples of bytes to their counts
        vocab: Initial vocabulary (should contain single-byte tokens and special tokens)
        vocab_size: Target vocabulary size
        debug: Whether to print debug information

    Returns:
        Tuple of (final_vocab, merges_list)
    """
    token_id = max(vocab.values()) + 1 if vocab else 0
    merges: list[tuple[bytes, bytes]] = []

    merge_step = 0
    while (stop_at_merge_num is None or merge_step < stop_at_merge_num) and len(vocab) < vocab_size:
        merge_step += 1
        
        # 1) Find the most common pair with debug print
        most_common_pair = _find_most_common_pair(tokens_counts, merge_step, debug)
        if most_common_pair is None:
            break  # No more pairs to merge
        
        # 2) Form new token and update vocab
        new_token, token_id = _update_vocab_with_merge(most_common_pair, vocab, token_id, merges)
        
        # 3) Update / merge the tokens_counts data structure
        tokens_counts = _update_tokens_counts(tokens_counts, most_common_pair, new_token)

    return vocab, merges

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    pretokenizer_name = kwargs.get("pretokenizer_name", "default")
    debug = kwargs.get("debug", False)
    stop_at_merge_num = kwargs.get("stop_at_merge_num", None)
    if stop_at_merge_num is not None and not isinstance(stop_at_merge_num, int):
        raise ValueError("stop_at_merge_num must be an integer or None")

    # Pretokenize the corpus
    tokens_counts: dict[tuple, int] = pretokenize_corpus(input_path, pretokenizer_name, special_tokens, debug)

    # Create the vocabulary and merges based on the token counts
    vocab: dict[bytes, int] = {bytes([i]): i for i in range(256)}  # Initial vocabulary with single-byte tokens
    token_id = len(vocab)  # Start token ID after single-byte tokens
    merges: list[tuple[bytes, bytes]] = []

    # Add special tokens to the vocabulary next
    for token in special_tokens:
        token_bytes = token.encode("utf-8")
        if token_bytes not in vocab:
            vocab[token_bytes] = token_id
            token_id += 1

    assert len(vocab) <= vocab_size, "Vocabulary size exceeds the specified limit"

    # Perform BPE merges
    vocab, merges = perform_bpe_merges(tokens_counts, vocab, vocab_size, stop_at_merge_num, debug)

    # Ensure the vocabulary is limited to the specified size
    if len(vocab) > vocab_size:
        raise ValueError(f"Vocabulary size {len(vocab)} exceeds the specified limit {vocab_size}")

    # Invert vocab and return
    inverted_vocab: dict[int, bytes] = {v: k for k, v in vocab.items()}
    return inverted_vocab, merges


def save_bpe(
    vocab: dict[int, bytes], 
    merges: list[tuple[bytes, bytes]], 
    output_directory: str | os.PathLike
) -> None:
    """
    Save BPE vocabulary and merges to disk as pickled files.
    
    Args:
        vocab: The vocabulary dictionary mapping token IDs to bytes
        merges: List of merge tuples (bytes, bytes)
        output_directory: Directory where to save the vocab.pkl and merges.pkl files
    """
    output_dir = os.path.abspath(output_directory)
    os.makedirs(output_dir, exist_ok=True)
    
    vocab_path = os.path.join(output_dir, "vocab.pkl")
    merges_path = os.path.join(output_dir, "merges.pkl")
    
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    
    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)
    
    print(f"Saved vocabulary to {vocab_path}")
    print(f"Saved merges to {merges_path}")