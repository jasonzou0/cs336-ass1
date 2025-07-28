import os
from collections import defaultdict
from typing import BinaryIO

import regex as re

NUM_PROCESSES = 4  # Number of processes to use for parallel processing

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

def pretokenize_corpus(
    input_path: str | os.PathLike,
    pretokenizer: re.Pattern,
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
    with open(input_path, "rb") as f:
        boundaries = _find_chunk_boundaries(f, NUM_PROCESSES, b"<|endoftext|>")

    # TODO: parallelize this.
    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.
    tokens_counts: dict[tuple, int] = defaultdict(int)

    with open(input_path, "rb") as f:
        # Read the file in chunks based on the boundaries
        # and count the occurrences of each pre-token
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            if debug:
                #input("enter..")
                print(f"Processing chunk from {start} to {end}: {chunk}...")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            mini_chunks = re.split("|".join([re.escape(token) for token in special_tokens]), chunk)
            for mini_chunk in mini_chunks:
                for match in pretokenizer.finditer(mini_chunk):
                    token = match.group()
                    byte_tuple = tuple(bytes([b]) for b in token.encode("utf-8"))
                    tokens_counts[byte_tuple] += 1

    if debug:
        print(f"Pretokenization results: {len(tokens_counts)} tokens. (Only show first 10)")
        for i, (token_tuple, count) in enumerate(tokens_counts.items()):
            token_strs = [_decode_bytes_debug(token) for token in token_tuple]
            print(f"  {i+1}. '{token_strs}' (count: {count})")
            if i >= 9:  # Only show first 10
                break
        print()
    
    return tokens_counts

def perform_bpe_merges(
    tokens_counts: dict[tuple, int],
    vocab: dict[bytes, int],
    vocab_size: int,
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
    while len(vocab) < vocab_size:
        merge_step += 1
        
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
            break  # No more pairs to merge

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

        # TODO: implement indexing of tokens_counts to avoid the need to iterate through it
        # Merge the most common pair
        merges.append(most_common_pair)

        # Create a new token by merging the two bytes
        first, second = most_common_pair
        new_token = first + second
        if new_token not in vocab:
            vocab[new_token] = token_id
            token_id += 1

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

        tokens_counts = new_tokens_counts

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
    pretokenizer = get_pretokenizer(kwargs.get("pretokenizer_name", "default"))
    debug = kwargs.get("debug", False)

    # Pretokenize the corpus
    tokens_counts: dict[tuple, int] = pretokenize_corpus(input_path, pretokenizer, special_tokens, debug)

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
    vocab, merges = perform_bpe_merges(tokens_counts, vocab, vocab_size, debug)

    # Ensure the vocabulary is limited to the specified size
    if len(vocab) > vocab_size:
        raise ValueError(f"Vocabulary size {len(vocab)} exceeds the specified limit {vocab_size}")

    # Invert vocab and return
    inverted_vocab: dict[int, bytes] = {v: k for k, v in vocab.items()}
    return inverted_vocab, merges