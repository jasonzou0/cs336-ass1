
# ddimport re
import numbers
from typing import Any, Dict, Iterable, List, Tuple, Iterator
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class BpeTokenizer:
    def __init__(
        self,
        id_to_bytes: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: List[str] | None = None,
    ):
        self.id_to_bytes = dict(id_to_bytes)
        self.bytes_to_id = {v: k for k, v in self.id_to_bytes.items()}

        # Rank of each merge pair: lower rank = applied earlier / higher priority
        self.rank: Dict[Tuple[bytes, bytes], int] = {
            pair: i for i, pair in enumerate(merges)
        }

        # Precompile regex to protect special tokens (as plain strings)
        self.special_tokens = list(special_tokens or [])
        if self.special_tokens:
            # 1) longest-first to handle overlaps like "<eot><eot>"
            self.special_tokens.sort(key=len, reverse=True)
            # 2) compile alternation in that order
            escaped = [re.escape(tok) for tok in self.special_tokens]
            self._special_pat = re.compile("(" + "|".join(escaped) + ")")

            # ensure each special token exists in vocab (as bytes)
            for tok in self.special_tokens:
                b = tok.encode("utf-8")
                if b not in self.bytes_to_id:
                    raise KeyError(f"Special token {tok!r} not found in vocab.")
        else:
            self._special_pat = None
        # Sanity: all 256 single bytes should be in vocab for byte-level BPE
        # (optional but helpful; comment out if your vocab is not byte-level)
        # for b in range(256):
        #     if bytes([b]) not in self.bytes_to_id:
        #         raise KeyError(f"Missing base byte {b} in vocab.")

    # ---- core BPE merge on a list of byte tokens ----
    def _bpe_merge(self, symbols: List[bytes]) -> List[bytes]:
        if len(symbols) <= 1:
            return symbols
        while True:
            # find best pair by rank
            best_pair = None
            best_rank = None
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                r = self.rank.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_pair = r, pair
            if best_pair is None:
                break  # no more applicable merges

            # merge all non-overlapping occurrences of best_pair
            merged: List[bytes] = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best_pair:
                    merged.append(symbols[i] + symbols[i + 1])
                    i += 2
                else:
                    merged.append(symbols[i])
                    i += 1
            symbols = merged
        return symbols

    # ---- public API ----
    def encode_01(self, text: str) -> List[int]:
        pieces: List[int] = []

        # Split by special tokens (keeping them in the result via capture group)
        chunks = (
            self._special_pat.split(text) if self._special_pat is not None else [text]
        )

        for chunk in chunks:
            if not chunk:
                continue

            if self._special_pat is not None and chunk in self.special_tokens:
                b = chunk.encode("utf-8")
                # keep as a single token
                pieces.append(self.bytes_to_id[b])
                continue

            # Normal text: run byte-level BPE
            b = chunk.encode("utf-8")
            # start with single-byte symbols
            symbols = [bytes([byte]) for byte in b]
            tokens = self._bpe_merge(symbols)
            for t in tokens:
                try:
                    pieces.append(self.bytes_to_id[t])
                except KeyError:
                    raise KeyError(
                        f"Token bytes {t!r} not found in vocab. "
                        "Your vocab must include all base bytes and all learned merges."
                    )
        return pieces


    def encode(self, text: str) -> list[int]:
        ids: list[int] = []

        # 1) split out special tokens first (keep them intact)
        chunks = self._special_pat.split(text) if self._special_pat else [text]

        for chunk in chunks:
            if not chunk:
                continue
            # encoding specail tokens
            if self._special_pat and chunk in self.special_tokens:
                ids.append(self.bytes_to_id[chunk.encode("utf-8")])
                continue

            # 2) GPT-2 pretokenize this normal chunk
            for piece in re.findall(PAT,chunk):
            #for piece in re.finditer(PAT,chunk):
                b = piece.encode("utf-8")

                # 3) run BPE *inside* the piece only
                symbols = [bytes([x]) for x in b]
                tokens = self._bpe_merge(symbols)

                for t in tokens:
                    try:
                        ids.append(self.bytes_to_id[t])
                    except KeyError:
                        raise KeyError(
                            f"Token bytes {t!r} not in vocab;"
                            " ensure base bytes + all merges are present."
                        )
        return ids

    def _flatten_ids(self, obj):
        """Yield ints from nested iterables of ints (e.g. [1,2] or [[1,2],[3]])."""
        if isinstance(obj, numbers.Integral):
            yield int(obj)
        elif isinstance(obj, (list, tuple)):
            for x in obj:
                yield from self._flatten_ids(x)
        elif isinstance(obj, (bytes, bytearray, str)):
            # Avoid treating bytes/str as iterables of IDs
            raise TypeError("decode expects ints or sequences of ints, not bytes/str")
        else:
            try:
                it = iter(obj)
            except TypeError:
                raise TypeError(f"Unsupported id element type: {type(obj)}")
            for x in it:
                yield from self._flatten_ids(x)

    def decode(self, ids: Iterable[int] | Iterable[Iterable[int]]) -> str:
        # Accept either a flat list of ints or nested lists (e.g., output of encode_iterable)
        flat_ids = self._flatten_ids(ids)
        bs = b"".join(self.id_to_bytes[i] for i in flat_ids)
        return bs.decode("utf-8", errors="replace")
                
    def decode_01(self, ids: Iterable[int]) -> str:
        bs = b"".join(self.id_to_bytes[i] for i in ids)
        # Use 'utf-8' strict if you know all tokens form valid UTF-8;
        # 'replace' is safer if you’re not sure.
        return bs.decode("utf-8", errors="replace")


    def encode_iterable_01(self, texts: Iterable[str]) -> Iterator[List[int]]:
        """
        Lazily encode an iterable of strings. For each element in `texts`,
        yield a list of token IDs (same as calling `encode` on each item).

        Note:
            - Special tokens are detected *within each item independently*.
              If a special token is split across item boundaries, it won't be
              recognized (typical when streaming line-by-line).
        """
        for text in texts:
            yield self.encode(text)

    def encode_iterable(self, texts: Iterable[str]) -> Iterator[int]:
        """
        Lazily stream token IDs for each input string in `texts`.
        Yields a flat sequence of ints so collecting into a list
        matches tiktoken's flat encode output.
        """
        for text in texts:
            # encode returns List[int]; stream them out flat
            for tid in self.encode(text):
                yield tid
