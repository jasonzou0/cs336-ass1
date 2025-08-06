import pickle
import regex as re

from typing import Iterable, Iterator
from .train_bpe import get_pretokenizer

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

    def _pretokenize(self, text: str) -> list[tuple]:
        """
        Pre-tokenizes the input text into byte tuples based on special tokens and pretokenizer.
        
        Args:
            text: The input text to pre-tokenize.
        Returns:
            A list of tuples, where each tuple contains the per-byte representation of a pretokenized token
            in the original order they appear in the text.
        """
        result: list[tuple] = []

        # Use capturing group in split to keep special tokens in the result
        special_pattern = "|".join([re.escape(token) for token in self._special_tokens])
        chunks = re.split(f"({special_pattern})", text)

        for chunk in chunks:
            if not chunk:  # Skip empty chunks
                continue
            elif chunk in self._special_tokens:
                # Special token: add as single tuple with its bytes representation
                byte_tuple = (chunk.encode("utf-8"),)
                result.append(byte_tuple)
            else:
                # Regular text: process with pretokenizer
                for match in self._pretokenizer.finditer(chunk):
                    token = match.group()
                    byte_tuple = tuple(bytes([b]) for b in token.encode("utf-8"))
                    result.append(byte_tuple)

        return result

    # TODO: consider caching the result of this api call for frequent tokens.
    def _encode_one_tuple(self, bytes_tuple: tuple) -> list[int]:
        """
        Encodes a single token into its corresponding ID using the vocabulary.
        
        Args:
            token: A tuple representing the token to encode.
        Returns:
            The ID of the token in the vocabulary.
        """
        while len(bytes_tuple) > 1:
            merge_found = False
            # saves the position to merge, its merge index, and the merged token
            best_merge: tuple[int, int, bytes] = (-1, float('inf'), bytes())
            for i in range(len(bytes_tuple) - 1):
                pair = (bytes_tuple[i], bytes_tuple[i + 1])
                # TODO: decide if we need to consider positions of merges and only apply the first merge found.
                if pair in self._merges:
                    merged_token: bytes = pair[0] + pair[1]
                    merge_found = True
                    merge_index = self._merges[pair]
                    # Check if this merge is better than the best found so far
                    if merge_index < best_merge[1]:
                        best_merge = (i, merge_index, merged_token)
            if merge_found:
                i, _, merged_token = best_merge
                if self._debug:
                    print(f"Merging {bytes_tuple[i]} and {bytes_tuple[i + 1]} at index {i} into {merged_token}")
                bytes_tuple = bytes_tuple[:i] + (merged_token,) + bytes_tuple[i + 2:]
            else:
                if self._debug:
                    print(f"No more merges found for {bytes_tuple}, breaking.")
                break
        # Convert the final bytes tuple to IDs
        return [self._vocab[byte] for byte in bytes_tuple]

    def from_file(cls, vocab_file, merges_file, special_tokens=None):
        # open and unpickle vocab and merges files
        with open(vocab_file, "rb") as vf:
            vocab = pickle.load(vf)
        with open(merges_file, "rb") as mf:
            merges = pickle.load(mf)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        encoding_per_pretoken = (self._encode_one_tuple(pretoken) for pretoken in self._pretokenize(text))
        # Flatten the list of lists into a single list
        return [item for sublist in encoding_per_pretoken for item in sublist]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # TODO: write a different, streaming version of, _pretokenize().
        return NotImplementedError

    def decode(self, ids: list[int]) -> str:
        return b''.join([self._id_to_vocab[id] for id in ids]).decode("utf-8", errors="replace")