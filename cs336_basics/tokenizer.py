import pickle
import regex as re
from functools import lru_cache

from typing import Iterable, Iterator
from .train_bpe import get_pretokenizer

# Note: if we want to parallelize tokenization in the future, pay attention to:
# 1. the file chunking algorithm should only chunk at word / line boundaries and 
#    does not split the "<|endoftext|>" special token.
# 2. might need to reduce the size of the cache per process to save memory.
class Tokenizer(object):
    def __init__(
        self, 
        vocab: dict[int, bytes], 
        merges: list[tuple[bytes, bytes]], 
        special_tokens: list[str],
        **kwargs,
    ):
        self._id_to_vocab: dict[int, bytes] = vocab
        self._vocab: dict[bytes, int] = {v: k for k, v in vocab.items()}
        self._merges: dict[tuple[bytes, bytes], int] = {
            bytes_pair: idx for idx, bytes_pair in enumerate(merges)
        }
        self._special_tokens = special_tokens if special_tokens is not None else []
        pretokenizer_name = kwargs.get("pretokenizer_name", "default")
        self._pretokenizer = get_pretokenizer(pretokenizer_name)
        self._debug = kwargs.get("debug", False)
        
        # Progress tracking
        self._progress_interval = kwargs.get("progress_interval", None)
        self._encode_call_count = 0
        
        # Create a cached version of the encoding function
        # 
        # Cache size analysis result from owt_valid_100k.txt:
        #
        #    | Cache Size | Time (s) | Hit Rate | Speed Gain | Memory Est |
        #    |------------|----------|----------|------------|------------|
        #    | 128        | 0.71     | 51.0%    | 1.0x       | 6KB        |
        #    | 512        | 0.61     | 65.8%    | 1.16x      | 25KB       |
        #    | 2048       | 0.50     | 76.3%    | 1.42x      | 100KB      |
        #    | 8192       | 0.40     | 86.2%    | 1.78x      | 400KB      |
        #    | 16384      | 0.36     | 89.4%    | 1.97x      | 800KB      |
        self._encode_cache = lru_cache(maxsize=16384)(self._encode_one_tuple_uncached)

    def _pretokenize(self, text: str) -> list[bytes]:
        """
        Pre-tokenizes the input text into bytes based on special tokens and pretokenizer.
        
        Args:
            text: The input text to pre-tokenize.
        Returns:
            A list of bytes, where each bytes object represents a pretokenized token
            in the original order they appear in the text.
        """
        result: list[bytes] = []

        # Use capturing group in split to keep special tokens in the result
        # Sort special tokens by length (descending) to match longest first for overlapping cases
        sorted_special_tokens = sorted(self._special_tokens, key=len, reverse=True)
        special_pattern = "|".join([re.escape(token) for token in sorted_special_tokens])
        chunks = re.split(f"({special_pattern})", text) if self._special_tokens else [text]

        for chunk in chunks:
            if self._debug:
                print(f"Pretokenizing chunk: {chunk!r}")
            if not chunk:  # Skip empty chunks
                continue
            elif chunk in self._special_tokens:
                # Special token: add as bytes representation
                result.append(chunk.encode("utf-8"))
            else:
                # Regular text: process with pretokenizer
                for match in self._pretokenizer.finditer(chunk):
                    token = match.group()
                    result.append(token.encode("utf-8"))

        return result

    def _encode_one_tuple(self, token_bytes: bytes) -> list[int]:
        """
        Cached wrapper for encoding a single bytes token. Uses LRU cache for performance.
        """
        return self._encode_cache(token_bytes)
    
    def _encode_one_tuple_uncached(self, token_bytes: bytes) -> list[int]:
        """
        Encodes a single token into its corresponding ID using the vocabulary.
        
        It iteratively merges byte pairs based on the merges list in the original merge discovery order,
        until no more merges can be applied.
        
        Args:
            token_bytes: Bytes representing the token to encode.
        Returns:
            List of token IDs in the vocabulary.
        """
        # Check if the token already exists in vocab as a complete token (e.g., special tokens)
        if token_bytes in self._vocab:
            return [self._vocab[token_bytes]]
            
        # Break up token_bytes into list of single-byte tokens for merging.
        tokens: list[bytes] = list(token_bytes[i:i+1] for i in range(len(token_bytes)))
        
        while len(tokens) > 1:
            # Find the best merge (earliest in merge order) in single pass
            best_merge_idx = float('inf')
            best_pos = -1
            
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self._merges:
                    merge_idx = self._merges[pair]
                    if merge_idx < best_merge_idx:
                        best_merge_idx = merge_idx
                        best_pos = i
            
            if best_pos == -1:
                # No more merges possible
                break
                
            # Perform the merge
            merged_token = tokens[best_pos] + tokens[best_pos + 1]
            if self._debug:
                print(f"Merging {tokens[best_pos]} and {tokens[best_pos + 1]} at index {best_pos} into {merged_token}")
            
            # Optimized: Use in-place modification instead of list concatenation
            tokens[best_pos] = merged_token
            del tokens[best_pos + 1]
        
        # Convert the final tokens to IDs
        return [self._vocab[token] for token in tokens]

    def from_file(cls, vocab_file, merges_file, special_tokens=None):
        # open and unpickle vocab and merges files
        with open(vocab_file, "rb") as vf:
            vocab = pickle.load(vf)
        with open(merges_file, "rb") as mf:
            merges = pickle.load(mf)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        # Increment call count for progress tracking
        self._encode_call_count += 1
        
        # Progress reporting if enabled
        if (self._progress_interval is not None and 
            self._encode_call_count % self._progress_interval == 0):
            print(f"Encoded {self._encode_call_count:,} lines")
        
        encoding_per_pretoken = (self._encode_one_tuple(pretoken) for pretoken in self._pretokenize(text))
        # Flatten the list of lists into a single list
        return [item for sublist in encoding_per_pretoken for item in sublist]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Encode an iterable of text strings.
        
        Args:
            iterable: The input text iterable (e.g., file lines)
        
        Yields:
            Token IDs as integers
        """
        for text in iterable:
            for encoded_id in self.encode(text):
                yield encoded_id

    def decode(self, ids: list[int]) -> str:
        return b''.join([self._id_to_vocab[id] for id in ids]).decode("utf-8", errors="replace")