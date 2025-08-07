import multiprocessing
import os
import heapq
import regex as re
from typing import BinaryIO
from multiprocessing import Pool
from collections import defaultdict, Counter

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
):
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

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


def process_corpus(input_path, special_tokens):
    num_processes = multiprocessing.cpu_count()
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    word_freq = defaultdict(int)
    with open(input_path, "rb") as f:
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            mini_chunks = re.split("|".join([re.escape(token) for token in special_tokens]), chunk)
            for mini_chunk in mini_chunks:
                for match in re.finditer(PAT, mini_chunk):
                    token = match.group()
                    word = tuple(bytes([b]) for b in token.encode("utf-8"))
                    word_freq[word] += 1
    return word_freq


class MaxHeapObj:
    def __init__(self, obj):
        self.obj = obj

    def __lt__(self, other):
        return self.obj > other.obj  # reverse comparison

    def __eq__(self, other):
        return self.obj == other.obj


class BPE:
    def __init__(self, word_freq, vocab, num_merges):
        self.word_freq = word_freq
        self.vocab = vocab
        self.num_merges = num_merges
        self.merges = []

    def get_stats(self):
        pairs = defaultdict(int)
        for word, freq in self.word_freq.items():
            for i in range(len(word) - 1):
                pairs[word[i], word[i + 1]] += freq
        return pairs

    def get_lex_key(self, pair):
        return MaxHeapObj(pair)

    def build_heap(self, stats):
        heap = [(-freq, self.get_lex_key(pair), pair) for pair, freq in stats.items()]
        heapq.heapify(heap)
        return heap

    def merge_vocab(self, pair):
        new_vocab = {}
        for word, freq in self.word_freq.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                    new_word.append(word[i] + word[i+1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_vocab[tuple(new_word)] = freq
        self.word_freq = new_vocab

    def train(self):
        stats = self.get_stats()
        heap = self.build_heap(stats)
        next_id = len(self.vocab)
        for _ in range(self.num_merges):
            if not heap:
                break
            _, _, best = heapq.heappop(heap)
            new_token = best[0] + best[1]
            if new_token not in self.vocab:
                self.vocab[new_token] = next_id
                next_id += 1
            self.merges.append(best)
            self.merge_vocab(best)
            stats = self.get_stats()
            heap = self.build_heap(stats)


def train_bpe(input_path, vocab_size, special_tokens):
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
    word_freq = process_corpus(input_path, special_tokens)
    vocab = {bytes([i]): i for i in range(256)}
    token_id = len(vocab)
    for token in special_tokens:
        token_bytes = token.encode("utf-8")
        if token_bytes not in vocab:
            vocab[token_bytes] = token_id
            token_id += 1
    num_merges = vocab_size - len(vocab)
    bpe = BPE(word_freq, vocab, num_merges)
    bpe.train()
    vocab = {v: k for k, v in bpe.vocab.items()}
    return vocab, bpe.merges

