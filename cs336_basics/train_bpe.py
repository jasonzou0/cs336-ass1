import os
import pickle
import heapq
import time
from collections import defaultdict
from typing import BinaryIO, Dict, Tuple, List, Optional, Set
from multiprocessing import Pool

import regex as re

MERGE_LOG_INTERVAL = 100 

class MaxHeapItem:
    """Wrapper for heap items that implements max-heap with proper tie-breaking."""

    def __init__(self, count: int, pair: Tuple[bytes, bytes]):
        self.count = count
        self.pair = pair

    def __lt__(self, other):
        # For max-heap, we want larger counts first, so reverse the comparison
        # For tie-breaking, we want lexicographically larger pairs first
        if self.count != other.count:
            return self.count > other.count  # Larger count wins (max-heap)
        return self.pair > other.pair  # Larger pair wins ties

    def __eq__(self, other):
        return self.count == other.count and self.pair == other.pair

    def __repr__(self):
        return f"MaxHeapItem(count={self.count}, pair={self.pair})"

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
        for i in range(len(bytes_tuple) - 1):
            pair = (bytes_tuple[i], bytes_tuple[i + 1])
            bytepair_counts[pair] += count

    # Find the most common byte pair
    if not bytepair_counts:
        return None  # No more pairs to merge

    # TODO: double check if the tie breaker function is implemented correctly.
    most_common_pair = max(bytepair_counts, key=lambda bytepair: (bytepair_counts[bytepair], bytepair))

    if merge_step == 1 or merge_step % MERGE_LOG_INTERVAL == 0:
        print(f"Merge step {merge_step} done: Most common pair is {most_common_pair} with count {bytepair_counts[most_common_pair]}")

    if debug:
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
        bytes_tuple_count = len(bytes_tuple)
        if bytes_tuple_count == 1:
            continue
        new_bytes_tuple = []
        i = 0
        merge_happened = False
        while i < bytes_tuple_count:
            if i < bytes_tuple_count - 1 and (bytes_tuple[i], bytes_tuple[i + 1]) == most_common_pair:
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


class OptimizedBPEMerger:
    """
    High-performance BPE merger using advanced data structures and caching.

    Key Optimizations:
    1. Simplified heap management with lazy deletion
    2. Incremental token updates (only affected tokens) 
    3. Cached pair frequencies (no recomputation)
    4. Minimal memory allocation
    """

    def __init__(self, debug: bool = False):
        self.debug = debug

        # Index-based data structures for better performance
        self.tokens_list: List[tuple] = []                    # [token_tuple, ...]
        self.tokens_counts: List[int] = []                    # [count, ...]

        # Track which token indices contain each pair (use sets for O(1) operations)
        self.pair_positions: Dict[tuple, Set[int]] = defaultdict(set)
        self.pair_counts: Dict[tuple, int] = defaultdict(int)

        # Max-heap for O(log P) pair selection using custom heap items
        self.pair_heap: List[MaxHeapItem] = []
        self.heap_valid: Dict[tuple, bool] = {}               # Track valid heap entries

        # Performance tracking
        self.stats = {
            'total_merges': 0,
            'tokens_scanned': 0,
            'pairs_updated': 0,
            'heap_operations': 0,
        }

    def initialize(self, tokens_counts: Dict[tuple, int]):
        """Initialize the optimizer with initial token counts."""
        if self.debug:
            print(f"🚀 Initializing optimized BPE with {len(tokens_counts)} token types")

        # Convert dictionary to index-based lists
        self.tokens_list.clear()
        self.tokens_counts.clear()

        for token_tuple, count in tokens_counts.items():
            self.tokens_list.append(token_tuple)
            self.tokens_counts.append(count)

        self._build_initial_pair_tracking()

        if self.debug:
            total_pairs = sum(max(0, len(token_tuple) - 1) * count 
                            for token_tuple, count in zip(self.tokens_list, self.tokens_counts))
            print(f"   📊 Total pair instances: {total_pairs:,}")
            print(f"   📊 Unique pairs: {len(self.pair_counts):,}")
            print(f"   📊 Token indices: {len(self.tokens_list):,}")


    def _build_initial_pair_tracking(self):
        """Build initial pair frequency tracking using indices."""
        # Clear existing data
        self.pair_counts.clear()
        self.pair_positions.clear()
        self.pair_heap.clear()
        self.heap_valid.clear()

        # Build pair frequency map and position tracking using token indices
        for token_idx in range(len(self.tokens_list)):
            token_tuple = self.tokens_list[token_idx]
            count = self.tokens_counts[token_idx]

            # Inline pair extraction to avoid 6.6M+ function calls
            token_len = len(token_tuple)
            if token_len >= 2:
                for i in range(token_len - 1):
                    pair = (token_tuple[i], token_tuple[i+1])
                    self.pair_counts[pair] += count
                    self.pair_positions[pair].add(token_idx)

        # Build initial heap from all pairs
        for pair, count in self.pair_counts.items():
            if count > 0:
                heap_item = MaxHeapItem(count, pair)
                heapq.heappush(self.pair_heap, heap_item)
                self.heap_valid[pair] = True

    def _update_pair_count(self, pair: tuple, count_delta: int):
        """Update a pair's count and maintain heap invariants."""
        old_count = self.pair_counts[pair]
        new_count = old_count + count_delta

        if new_count <= 0:
            self.pair_counts[pair] = 0
            # Mark heap entry as invalid instead of removing (lazy deletion)
            self.heap_valid[pair] = False
        else:
            self.pair_counts[pair] = new_count
            # Add new entry to heap (old entries will be lazily removed) 
            heap_item = MaxHeapItem(new_count, pair)
            heapq.heappush(self.pair_heap, heap_item)
            self.heap_valid[pair] = True
            self.stats['heap_operations'] += 1

        self.stats['pairs_updated'] += 1

    def find_most_common_pair(self, merge_step: int = 0) -> Optional[tuple]:
        """
        Find the most common pair using custom heap for O(log P) performance.

        Args:
            merge_step: Current merge step number (for logging)

        Returns:
            Most frequent pair, or None if no valid pairs remain.
        """
        # Use heap with lazy deletion for O(log P) performance
        while self.pair_heap:
            heap_item = heapq.heappop(self.pair_heap)
            pair = heap_item.pair
            count = heap_item.count

            # Check if this heap entry is still valid (lazy deletion)
            if self.heap_valid.get(pair, False) and self.pair_counts[pair] == count:
                # Log merge results every N steps or on first step (configurable interval)
                if merge_step == 1 or merge_step % MERGE_LOG_INTERVAL == 0:
                    print(f"Merge step {merge_step} done: Most common pair is {pair} with count {count}")

                if self.debug:
                    pair_str = f"({pair[0].decode('utf-8', errors='replace')}, {pair[1].decode('utf-8', errors='replace')})"
                    print(f"   🎯 Selected pair: {pair_str} (count: {count})")

                return pair

            # Invalid entry, continue to next
            self.stats['heap_operations'] += 1

        # No valid pairs found
        return None

    def update_tokens_incremental(self, merge_pair: tuple, new_token: bytes) -> int:
        """
        Update tokens in-place - directly modify affected tokens for maximum efficiency.
        No need to add/delete tokens, just update existing ones.

        Returns:
            Number of tokens that were modified.
        """
        if merge_pair not in self.pair_positions:
            return 0

        affected_indices = list(self.pair_positions[merge_pair])  # Convert set to list for iteration
        tokens_modified = 0

        if self.debug:
            print(f"   🔄 Updating {len(affected_indices)} affected tokens")

        # Batch tracking: collect all pair changes before applying them
        pairs_to_remove = defaultdict(list)  # pair -> [count_delta_1, count_delta_2, ...]
        pairs_to_add = defaultdict(list)     # pair -> [count_delta_1, count_delta_2, ...]
        indices_to_remove = defaultdict(set)  # pair -> {idx1, idx2, ...}
        indices_to_add = defaultdict(set)    # pair -> {idx1, idx2, ...}

        # First pass: collect all changes without modifying data structures
        for token_idx in affected_indices:
            old_token_tuple = self.tokens_list[token_idx]
            count = self.tokens_counts[token_idx]

            # Inline merge operation to avoid function calls
            old_token_len = len(old_token_tuple)
            if old_token_len < 2:
                new_token_tuple = old_token_tuple
            else:
                # Inline the merge logic for maximum performance
                result = []
                merge_first, merge_second = merge_pair
                i = 0
                while i < old_token_len:
                    # Check if we can merge at current position
                    if (i < old_token_len - 1 and 
                        old_token_tuple[i] == merge_first and old_token_tuple[i+1] == merge_second):
                        result.append(new_token)
                        i += 2  # Skip both parts of the merged pair
                    else:
                        result.append(old_token_tuple[i])
                        i += 1
                new_token_tuple = tuple(result)

            # Only proceed if the token actually changed
            if new_token_tuple != old_token_tuple:
                # Batch collect old pairs to remove
                if old_token_len >= 2:
                    for i in range(old_token_len - 1):
                        pair = (old_token_tuple[i], old_token_tuple[i+1])
                        pairs_to_remove[pair].append(-count)
                        indices_to_remove[pair].add(token_idx)

                # Batch collect new pairs to add
                new_token_len = len(new_token_tuple)
                if new_token_len >= 2:
                    for i in range(new_token_len - 1):
                        pair = (new_token_tuple[i], new_token_tuple[i+1])
                        pairs_to_add[pair].append(count)
                        indices_to_add[pair].add(token_idx)

                # Update the token in-place
                self.tokens_list[token_idx] = new_token_tuple
                tokens_modified += 1

        # Second pass: batch apply all pair tracking updates
        # Remove old pair tracking
        for pair, count_deltas in pairs_to_remove.items():
            total_delta = sum(count_deltas)
            if total_delta != 0:
                self._update_pair_count(pair, total_delta)
            # Remove indices from position tracking
            for idx in indices_to_remove[pair]:
                self.pair_positions[pair].discard(idx)

        # Add new pair tracking
        for pair, count_deltas in pairs_to_add.items():
            total_delta = sum(count_deltas)
            if total_delta != 0:
                self._update_pair_count(pair, total_delta)
            # Add indices to position tracking
            for idx in indices_to_add[pair]:
                self.pair_positions[pair].add(idx)

        self.stats['tokens_scanned'] += tokens_modified
        return tokens_modified



    def perform_optimized_merges(
        self, 
        tokens_counts: Dict[tuple, int],
        vocab: Dict[bytes, int], 
        vocab_size: int,
        stop_at_merge_num: Optional[int] = None
    ) -> Tuple[Dict[bytes, int], List[Tuple[bytes, bytes]]]:
        """
        Perform BPE merges using optimized algorithm.

        Returns:
            Updated vocabulary and list of merges performed.
        """
        if self.debug:
            print("🚀 Starting optimized BPE merges")

        # Initialize
        self.initialize(tokens_counts)
        token_id = max(vocab.values()) + 1 if vocab else 256
        merges = []

        merge_step = 0
        start_time = time.time()

        while ((stop_at_merge_num is None or merge_step < stop_at_merge_num) and 
               len(vocab) < vocab_size):

            merge_step += 1
            step_start = time.time()

            # Find most common pair (optimized)
            most_common_pair = self.find_most_common_pair(merge_step)
            if most_common_pair is None:
                if self.debug:
                    print("   ℹ️  No more pairs to merge")
                break

            # Create new token
            new_token = most_common_pair[0] + most_common_pair[1]
            if new_token not in vocab:
                vocab[new_token] = token_id
                token_id += 1

            # Record the merge
            merges.append(most_common_pair)

            # Update tokens incrementally (optimized)
            tokens_modified = self.update_tokens_incremental(most_common_pair, new_token)

            step_time = time.time() - step_start

            if self.debug and merge_step % 100 == 0:
                print(f"   📈 Step {merge_step}: {tokens_modified} tokens updated in {step_time*1000:.1f}ms")

            self.stats['total_merges'] += 1

        total_time = time.time() - start_time

        if self.debug:
            print(f"\n📊 OPTIMIZATION RESULTS:")
            print(f"   Merges completed: {self.stats['total_merges']}")
            print(f"   Total time: {total_time:.3f}s")
            print(f"   Avg time per merge: {total_time/max(1,self.stats['total_merges'])*1000:.2f}ms")
            print(f"   Tokens scanned: {self.stats['tokens_scanned']:,}")
            print(f"   Pairs updated: {self.stats['pairs_updated']:,}")
            print(f"   Heap operations: {self.stats['heap_operations']:,}")


        return vocab, merges


def perform_bpe_merges(
    tokens_counts: dict[tuple, int],
    vocab: dict[bytes, int],
    vocab_size: int,
    stop_at_merge_num: int | None,
    debug: bool = False,
    use_optimization: bool = False,
) -> tuple[dict[bytes, int], list[tuple[bytes, bytes]]]:
    """
    Perform BPE merges on the token counts to build vocabulary and merges.

    Args:
        tokens_counts: Dictionary mapping tuples of bytes to their counts
        vocab: Initial vocabulary (should contain single-byte tokens and special tokens)
        vocab_size: Target vocabulary size
        stop_at_merge_num: Stop training after this many merges (None for no limit)
        debug: Whether to print debug information
        use_optimization: Whether to use the optimized heap-based algorithm

    Returns:
        Tuple of (final_vocab, merges_list)
    """
    if use_optimization:
        # Use optimized algorithm
        optimizer = OptimizedBPEMerger(debug=debug)
        return optimizer.perform_optimized_merges(tokens_counts, vocab, vocab_size, stop_at_merge_num)

    # Use original algorithm
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
        save_pretokenization_path (str | os.PathLike, optional): Path to save pretokenization results.
        load_pretokenization_path (str | os.PathLike, optional): Path to load pretokenization results from.
        use_optimization (bool, optional): Whether to use the optimized heap-based BPE algorithm. Default: False.

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
    save_pretokenization_path = kwargs.get("save_pretokenization_path", None)
    load_pretokenization_path = kwargs.get("load_pretokenization_path", None)
    use_optimization = kwargs.get("use_optimization", False)

    if stop_at_merge_num is not None and not isinstance(stop_at_merge_num, int):
        raise ValueError("stop_at_merge_num must be an integer or None")

    # Pretokenize the corpus or load from cache
    if load_pretokenization_path is not None:
        print(f"Loading pretokenization results from {load_pretokenization_path}")
        with open(load_pretokenization_path, "rb") as f:
            tokens_counts: dict[tuple, int] = pickle.load(f)
        print(f"Loaded {len(tokens_counts)} unique token types")
    else:
        tokens_counts: dict[tuple, int] = pretokenize_corpus(input_path, pretokenizer_name, special_tokens, debug)

        # Save pretokenization results if path provided
        if save_pretokenization_path is not None:
            print(f"Saving pretokenization results to {save_pretokenization_path}")
            os.makedirs(os.path.dirname(save_pretokenization_path), exist_ok=True)
            with open(save_pretokenization_path, "wb") as f:
                pickle.dump(tokens_counts, f)
            print(f"Saved {len(tokens_counts)} unique token types")

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
    vocab, merges = perform_bpe_merges(tokens_counts, vocab, vocab_size, stop_at_merge_num, debug, use_optimization)

    # Ensure the vocabulary is limited to the specified size
    if len(vocab) > vocab_size:
        raise ValueError(f"Vocabulary size {len(vocab)} exceeds the specified limit {vocab_size}")

    # Invert vocab and return
    inverted_vocab: dict[int, bytes] = {v: k for k, v in vocab.items()}
    return inverted_vocab, merges
