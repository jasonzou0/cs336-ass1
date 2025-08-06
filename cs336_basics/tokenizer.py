import pickle

import regex as re

from typing import Iterable, Iterator

from .train_bpe import get_pretokenizer

class Tokenizer(object):
    def __init__(
        self, 
        vocab: dict[int, bytes], 
        merges: list[tuple[bytes, bytes]], 
        special_tokens: list[str] | None = None,
        pretokenizer_name: str ="default",
        **kwargs,
    ):
        self._id_to_vocab: dict[int, bytes] = vocab
        self._vocab: dict[bytes, int] = {v: k for k, v in vocab.items()}
        self._merges: dict[tuple[bytes, bytes], int] = {
            bytes_pair: idx for idx, bytes_pair in enumerate(merges)
        }
        self._special_tokens = special_tokens if special_tokens is not None else []
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

        chunks = re.split("|".join([re.escape(token) for token in self._special_tokens]), text)
        for chunk in chunks:
            for match in self._pretokenizer.finditer(chunk):
                token = match.group()
                byte_tuple = tuple(bytes([b]) for b in token.encode("utf-8"))
                result.append(byte_tuple)

        return result

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
            for i in range(len(bytes_tuple) - 1):
                pair = (bytes_tuple[i], bytes_tuple[i + 1])
                # TODO: decide if we need to consider positions of merges and only apply the first merge found.
                if pair in self._merges:
                    merged_token: bytes = pair[0] + pair[1]
                    bytes_tuple = bytes_tuple[:i] + (merged_token,) + bytes_tuple[i + 2:]
                    merge_found = True
                    if self._debug:
                        print(f"Merging {pair} into {merged_token}, new tuple: {bytes_tuple}")
                    break
            if not merge_found:
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
        result: list[int] = []
        for pretoken in self._pretokenize(text):
            token_ids = self._encode_one_tuple(pretoken)
            result.extend(token_ids)
        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # TODO: write a different, streaming version of, _pretokenize().
        return NotImplementedError

    def decode(self, ids: list[int]) -> str:
        return b''.join([self._id_to_vocab[id] for id in ids]).decode("utf-8", errors="replace")