That output is good news: the test is marked **XFAIL**, meaning it’s *expected to fail*. In this repo they expect `Tokenizer.encode` to exceed a very tight memory cap (\~1 MB), so the failure is recorded as “expected” and doesn’t block you.

If you *do* want to try making `encode` pass a strict memory test, here are the biggest wins:

* Use **iterators instead of materializing lists**

  * Replace `GPT2_PAT.findall(chunk)` with `GPT2_PAT.finditer(chunk)`.
  * Don’t `split` text into a full list of chunks; iterate matches and yield segments as you go.

* Stream inside `encode`
  Build the output list incrementally and avoid large temporaries:

  * Append token IDs directly; avoid creating `tokens = [...]` for the whole piece.

* Avoid per-byte object churn

  * Cache single-byte IDs: `single_byte_id = [bytes([i]) for i in range(256)]` once, reuse these instead of `bytes([byte])` inside loops.
  * Consider `memoryview(b)` to slice the UTF-8 bytes without copying (you’ll still need `bytes` keys for lookup, but you can reduce intermediate objects).

* Keep regex/special-token machinery lean

  * Precompile the special-token regex once (you already do).
  * Sort specials by length (desc) to handle overlaps without backtracking explosions.

* Short-circuit common paths

  * If a piece is all ASCII letters/digits with no merges in your rank table, map bytes → ids directly rather than running the merge loop.

Sketch of lower-memory encode loop:

```python
def encode(self, text: str) -> list[int]:
    ids: list[int] = []
    for is_special, chunk in self._iter_special_chunks(text):  # yield (flag, substring)
        if is_special:
            ids.append(self.bytes_to_id[chunk.encode("utf-8")])
            continue
        for m in GPT2_PAT.finditer(chunk):                     # iterator, not findall
            b = m.group(0).encode("utf-8")
            # Avoid building a large symbols list:
            # Start from cached single-byte objects and merge in-place-ish.
            symbols = [self._single_byte[b_] for b_ in b]      # _single_byte prebuilt
            symbols = self._bpe_merge(symbols)                 # returns much smaller list
            for tok in symbols:
                ids.append(self.bytes_to_id[tok])
    return ids
```

If you want, share your current `encode` and I’ll point out the exact spots causing the biggest allocations and give a drop-in patch.
