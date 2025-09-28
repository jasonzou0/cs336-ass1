from __future__ import annotations
from collections import Counter
from typing import List, Tuple, Dict, Iterable
import os
import regex as re  # pip install regex
import time
from itertools import pairwise
import heapq
# Helper to invert bytes for max-lex tie-breaking using a min-heap
def _invert_bytes_for_tie(b: bytes) -> bytes:
    return bytes(255 - x for x in b)
# GPT-2 pretokenizer (same structure tiktoken uses)
GPT2_PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

# Performance Improvement
# 1. Map base bytes to ids 0..255 and store sequences as list[int].
# 2. Don’t recompute pair counts from scratch

def _piece_to_symbols(b: bytes) -> List[bytes]:
    # start from single bytes
    return [bytes([x]) for x in b]

def _piece_to_sequence(b: bytes) -> List[bytes]:
    return _piece_to_symbols(b)

def _merge_once(seq: List[bytes], pair: Tuple[bytes, bytes]) -> List[bytes]:
    a, b = pair
    out: List[bytes] = []
    i = 0
    L = len(seq)
    while i < L:
        if i+1 < L and seq[i] == a and seq[i+1] == b:
            out.append(a + b)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out

def _merge_once_int(seq: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
    a, b = pair
    # Lazily allocate output only if a merge occurs
    out: List[int] | None = None
    out_append = None  # set when out is created
    i = 0
    L = len(seq)
    while i < L:
        if i + 1 < L and seq[i] == a and seq[i + 1] == b:
            if out is None:
                out = seq[:i]
                out_append = out.append  # type: ignore[assignment]
            out_append(new_id)  # type: ignore[misc]
            i += 2
        else:
            if out is not None:
                out_append(seq[i])  # type: ignore[misc]
            i += 1
    return out if out is not None else seq

def _merge_once_int_pair_counts(
    seq: List[int],
    pair: Tuple[int, int],
    new_id: int,
    pair_counts: Counter[Tuple[int, int]],
    changed_pairs: set[Tuple[int, int]] | None = None,
) -> List[int]:
    a, b = pair
    pc = pair_counts
    pc_get = pc.get
    pc_pop = pc.pop
    add = changed_pairs.add if changed_pairs is not None else None
    out: List[int] | None = None
    out_append = None  # set when out is created
    i = 0
    L = len(seq)
    key_ab = (a, b)
    while i < L:
        if i + 1 < L and seq[i] == a and seq[i + 1] == b:
            if out is None:
                out = seq[:i]
                out_append = out.append  # type: ignore[assignment]
            # previous emitted token (post earlier merges in this pass)
            prev_tok = out[-1] if out else None
            out_append(new_id)  # type: ignore[misc]

            # decrement (a,b) by 1 without creating zero/negative entries
            prev = pc_get(key_ab, 0)
            new = prev - 1
            if add:
                add(key_ab)
            if new <= 0:
                if prev:
                    pc_pop(key_ab, None)
            else:
                pc[key_ab] = new

            # left neighbor update
            if prev_tok is not None:
                key_prev_a = (prev_tok, a)
                prev_pa = pc_get(key_prev_a, 0)
                new_pa = prev_pa - 1
                if add:
                    add(key_prev_a)
                if new_pa <= 0:
                    if prev_pa:
                        pc_pop(key_prev_a, None)
                else:
                    pc[key_prev_a] = new_pa

                key_prev_new = (prev_tok, new_id)
                pc[key_prev_new] = pc_get(key_prev_new, 0) + 1
                if add:
                    add(key_prev_new)

            # right neighbor update
            j = i + 2
            if j < L:
                nxt = seq[j]
                key_b_next = (b, nxt)
                prev_bn = pc_get(key_b_next, 0)
                new_bn = prev_bn - 1
                if add:
                    add(key_b_next)
                if new_bn <= 0:
                    if prev_bn:
                        pc_pop(key_b_next, None)
                else:
                    pc[key_b_next] = new_bn

                key_new_next = (new_id, nxt)
                pc[key_new_next] = pc_get(key_new_next, 0) + 1
                if add:
                    add(key_new_next)

            i += 2
        else:
            if out is not None:
                out_append(seq[i])  # type: ignore[misc]
            i += 1
    return out if out is not None else seq

def _merge_once_int_pair_counts_weighted(
    seq: List[int],
    pair: Tuple[int, int],
    new_id: int,
    pair_counts: Counter[Tuple[int, int]],
    changed_pairs: set[Tuple[int, int]] | None,
    weight: int,
) -> List[int]:
    a, b = pair
    # Fast-path locals to cut attribute lookups in the hot loop
    pc = pair_counts
    pc_get = pc.get
    pc_pop = pc.pop
    add = changed_pairs.add if changed_pairs is not None else None
    w = weight
    # Lazily allocate output buffer only if a merge actually occurs
    out: List[int] | None = None
    out_append = None  # set when out is created
    i = 0
    L = len(seq)
    key_ab = (a, b)  # static within this call
    while i < L:
        if i + 1 < L and seq[i] == a and seq[i + 1] == b:
            # Create output on first merge; copy prefix up to i
            if out is None:
                out = seq[:i]
                out_append = out.append  # type: ignore[assignment]
            # emit merged token
            prev_tok = out[-1] if out else None
            out_append(new_id)  # type: ignore[misc]

            # decrement (a,b) by weight; avoid creating then popping keys
            prev = pc_get(key_ab, 0)
            new = prev - w
            if add:
                add(key_ab)
            if new <= 0:
                if prev:
                    pc_pop(key_ab, None)
            else:
                pc[key_ab] = new

            # left neighbor: (prev_tok, a) -> (prev_tok, new_id)
            if prev_tok is not None:
                key_prev_a = (prev_tok, a)
                prev_pa = pc_get(key_prev_a, 0)
                new_pa = prev_pa - w
                if add:
                    add(key_prev_a)
                if new_pa <= 0:
                    if prev_pa:
                        pc_pop(key_prev_a, None)
                else:
                    pc[key_prev_a] = new_pa

                key_prev_new = (prev_tok, new_id)
                prev_pn = pc_get(key_prev_new, 0)
                pc[key_prev_new] = prev_pn + w
                if add:
                    add(key_prev_new)

            # right neighbor: (b, nxt) -> (new_id, nxt)
            j = i + 2
            if j < L:
                nxt = seq[j]
                key_b_next = (b, nxt)
                prev_bn = pc_get(key_b_next, 0)
                new_bn = prev_bn - w
                if add:
                    add(key_b_next)
                if new_bn <= 0:
                    if prev_bn:
                        pc_pop(key_b_next, None)
                else:
                    pc[key_b_next] = new_bn

                key_new_next = (new_id, nxt)
                prev_nn = pc_get(key_new_next, 0)
                pc[key_new_next] = prev_nn + w
                if add:
                    add(key_new_next)

            i += 2
        else:
            if out is not None:
                out_append(seq[i])  # type: ignore[misc]
            i += 1
    return out if out is not None else seq

def _count_pairs(seqs: Iterable[List[bytes]]) -> Dict[Tuple[bytes, bytes], int]:
    stats = Counter()
    for s in seqs:
        for i in range(len(s)-1):
            stats[(s[i], s[i+1])] += 1
    return stats

def _count_pairs_int(seqs: Iterable[List[List[int]]]) -> Counter[Tuple[int, int]]:
    stats = Counter()
    for s in seqs:
        for i in range(len(s)-1):
            stats[(s[i], s[i+1])] += 1
    return stats


# version 4 , use heap to trains, takes 4.7 seconds
def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: List[str],
    **kwargs,
) -> tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Train byte-level BPE on `input_path` and return:
      - vocab: dict[int, bytes]
      - merges: list[(bytes, bytes)] in creation order

    Notes:
      * Special tokens are NOT protected during training (per spec), but are appended to vocab.
      * No merges across GPT-2 pretokenizer piece boundaries.
    """
    # --- sanity checks ---
    # optional profiling (set CS336_BPE_PROFILE=1 to enable)
    do_profile = os.getenv("CS336_BPE_PROFILE") == "1"
    debug_flag = os.getenv("CS336_BPE_DEBUG") == "1"
    debug_step = int(os.getenv("CS336_BPE_DEBUG_STEP", "-1"))
    if do_profile:
        import cProfile
        profile = cProfile.Profile()
        profile.enable()
    start_time = time.time()
    print(f"Training BPE with vocab_size={vocab_size}, special_tokens={special_tokens}, start_time={start_time}")
    previously_time = start_time
    if vocab_size <= 256 + len(special_tokens):
        raise ValueError(
            f"vocab_size must exceed 256 + len(special_tokens) "
            f"(got {vocab_size}, specials={len(special_tokens)})"
        )

    # --- read corpus ---
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    #cur_time = time.time()
    #print(f"Read corpus in {cur_time - previously_time} seconds")
    #previously_time = cur_time
    # --- build sequences of symbols (bytes) per GPT-2 piece ---
    # Remove occurrences of special token strings from the corpus to avoid
    # learning merges from their internals (e.g., "<|", "oftext", "|>").
    # This keeps specials isolated in the final vocab.
    if special_tokens:
        # Replace each special token occurrence with a single space so we don't
        # accidentally fuse surrounding newlines or words, which was creating
        # long runs like "\n\n\n" after removal.
        for st in special_tokens:
            text = text.replace(st, " ")
    pieces = (m.group(0) for m in GPT2_PAT.finditer(text))
    # Build unique sequences with counts (weights)
    seq_counter: Dict[Tuple[int, ...], int] = {}
    for piece in pieces:
        b = piece.encode("utf-8")
        sequence = _piece_to_sequence(b)
        ints = tuple(x[0] for x in sequence)
        seq_counter[ints] = seq_counter.get(ints, 0) + 1
    # Materialize unique seqs and counts
    seqs_int: List[List[int]] = [list(t) for t in seq_counter.keys()]
    seq_counts: List[int] = list(seq_counter.values())
    id_to_bytes: List[bytes] = []
    id_to_token: Dict[int, bytes] = {}
    # Initial pair counts, weighted by sequence frequency
    pair_counts: Counter[Tuple[int, int]] = Counter()
    for seq, wt in zip(seqs_int, seq_counts):
        if len(seq) < 2:
            continue
        a = seq[0]
        for j in range(1, len(seq)):
            b = seq[j]
            pair_counts[(a, b)] += wt
            a = b
    # Initialize id_to_bytes for single bytes
    id_to_bytes = [bytes([i]) for i in range(256)]
    # Build initial heap: store (-freq, bytes for tie-break, pair) - no inversion for lexicographically smallest
    heap = []
    for pair, freq in pair_counts.items():
        a_bytes = id_to_bytes[pair[0]]
        b_bytes = id_to_bytes[pair[1]]
        heapq.heappush(heap, (-freq, a_bytes, b_bytes, pair))

    #cur_time = time.time()
    #print(f"Processed pieces in {cur_time - previously_time} seconds")
    #previously_time = cur_time

    merges: List[Tuple[int, int]] = []
    target_merges = vocab_size - 256 - len(special_tokens)
    merges_done = 0

    # --- iterative merging --- (with deterministic tie-break on ties)
    while heap and merges_done < target_merges:
        # Look at current top frequency
        neg_top, _, _, _ = heap[0]
        top_freq = -neg_top
        # Collect all valid candidates with this frequency
        same_freq_valid: List[Tuple[int, int]] = []
        popped_items = []
        while heap and heap[0][0] == -top_freq:
            item = heapq.heappop(heap)
            popped_items.append(item)
            _, _, _, cand = item
            if pair_counts.get(cand, 0) == top_freq:
                same_freq_valid.append(cand)
        if not same_freq_valid:
            continue
        # If the only valid candidate is (\n,\n), defer it: push it back and continue
        # only_nl = True
        # for cand in same_freq_valid:
        #     if not (id_to_bytes[cand[0]] == b"\n" and id_to_bytes[cand[1]] == b"\n"):
        #         only_nl = False
        #         break
        # if only_nl:
        #     # push back the valid candidates so they remain in heap
        #     for cand in same_freq_valid:
        #         a_bytes = id_to_bytes[cand[0]]
        #         b_bytes = id_to_bytes[cand[1]]
        #         heapq.heappush(
        #             heap,
        #             (-top_freq, _invert_bytes_for_tie(a_bytes), _invert_bytes_for_tie(b_bytes), cand),
        #         )
        #     continue
        if debug_flag and (merges_done == debug_step):
            dbg = [
                (cand, id_to_bytes[cand[0]] + id_to_bytes[cand[1]], pair_counts.get(cand, 0))
                for cand in same_freq_valid
            ]
            print(f"DEBUG step {merges_done}: candidates at freq {top_freq} -> {dbg}")
            # Specific probes
            sw = (32, 119)
            nd = (110, 100)
            sh = (32, 104)
            sT = (32, 84)
            print(
                "DEBUG counts:",
                {" (' ', 'w')": pair_counts.get(sw, 0), " ('n','d')": pair_counts.get(nd, 0),
                 " (' ', 'h')": pair_counts.get(sh, 0), " (' ', 'T')": pair_counts.get(sT, 0)}
            )
        # Tie-break: pure lexicographic largest (bytes) among equals (to match reference)
        best_pair = max(same_freq_valid, key=lambda p: (id_to_bytes[p[0]], id_to_bytes[p[1]]))
        # Push back the other valid candidates at the same frequency
        for cand in same_freq_valid:
            if cand == best_pair:
                continue
            a_bytes = id_to_bytes[cand[0]]
            b_bytes = id_to_bytes[cand[1]]
            heapq.heappush(
                heap,
                (-top_freq, a_bytes, b_bytes, cand),
            )
        new_bytes = id_to_bytes[best_pair[0]] + id_to_bytes[best_pair[1]]
        merges.append(best_pair)
        id_to_bytes.append(new_bytes)
        # Apply merge to all unique sequences with weights, update pair_counts, and rebuild unique seqs
        changed_pairs: set[Tuple[int, int]] = set()
        new_token_id = len(id_to_bytes) - 1
        new_seq_map: Dict[Tuple[int, ...], int] = {}
        new_seqs: List[List[int]] = []
        new_counts: List[int] = []
        for s, wt in zip(seqs_int, seq_counts):
            if wt == 0:
                continue
            out = _merge_once_int_pair_counts_weighted(s, best_pair, new_token_id, pair_counts, changed_pairs, wt)
            key = tuple(out)
            prev = new_seq_map.get(key)
            if prev is None:
                new_seq_map[key] = len(new_seqs)
                new_seqs.append(out)
                new_counts.append(wt)
            else:
                new_counts[prev] += wt
        seqs_int = new_seqs
        seq_counts = new_counts
        # Push updated counts for changed pairs into heap (lazy invalidation handles old entries)
        for p in changed_pairs:
            f = pair_counts.get(p, 0)
            if f > 0:
                a_bytes = id_to_bytes[p[0]]
                b_bytes = id_to_bytes[p[1]]
                heapq.heappush(
                    heap,
                    (-f, a_bytes, b_bytes, p),
                )
        merges_done += 1

    #cur_time = time.time()
    #print(f"Applied merges in {cur_time - previously_time} seconds")
    #previously_time = cur_time

    # --- build vocab: ids -> bytes ---
    # 0..255: single bytes
    tokens: List[bytes] = [bytes([i]) for i in range(256)]
    
    # then each newly created merged token (concatenation) in order
    for a, b in merges:
        tok = bytes(id_to_bytes[a]) + bytes(id_to_bytes[b])
        if tok not in tokens:
            tokens.append(tok)
    cur_time = time.time()
    print(f"Built vocab in {cur_time - previously_time} seconds")
    previously_time = cur_time
    # finally, append specials
    for tok in special_tokens:
        tokens.append(tok.encode("utf-8"))
    #cur_time = time.time()
    #print(f"Appended special tokens in {cur_time - previously_time} seconds")
    #previously_time = cur_time
    # If we overshot (due to duplicate merged tokens), trim or pad as needed.
    # Prefer trimming merged tokens at the end to match requested vocab_size.
    if len(tokens) > vocab_size:
        tokens = tokens[:vocab_size]
    cur_time = time.time()
    print(f"Trimmed/Padded vocab in {cur_time - previously_time} seconds")
    # Stop and save profiling
    if do_profile:
        import pstats
        profile.disable()
        profile.dump_stats("bpe_profile.prof")
        print("Profiling results saved to bpe_profile.prof")
        # Print top 30 by cumulative time
        stats = pstats.Stats(profile)
        stats.sort_stats("cumulative").print_stats(30)
    vocab = {i: tok for i, tok in enumerate(tokens)}
    # convert merges from (int,int) to (bytes,bytes)
    merges_bytes = [(id_to_bytes[a], id_to_bytes[b]) for a, b in merges]
    return vocab, merges_bytes




# version 4 , use heap to trains, takes 4.7 seconds
def run_train_bpe_heap(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: List[str],
    **kwargs,
) -> tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Train byte-level BPE on `input_path` and return:
      - vocab: dict[int, bytes]
      - merges: list[(bytes, bytes)] in creation order

    Notes:
      * Special tokens are NOT protected during training (per spec), but are appended to vocab.
      * No merges across GPT-2 pretokenizer piece boundaries.
    """
    # --- sanity checks ---
    # timestamp here and print
    start_time = time.time()
    print(f"Training BPE with vocab_size={vocab_size}, special_tokens={special_tokens}, start_time={start_time}")
    previously_time = start_time
    if vocab_size <= 256 + len(special_tokens):
        raise ValueError(
            f"vocab_size must exceed 256 + len(special_tokens) "
            f"(got {vocab_size}, specials={len(special_tokens)})"
        )

    # --- read corpus ---
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    cur_time = time.time()
    print(f"Read corpus in {cur_time - previously_time} seconds")
    previously_time = cur_time
    # --- build sequences of symbols (bytes) per GPT-2 piece ---
    # Important: we don't treat specials specially during training.
    pieces = (m.group(0) for m in GPT2_PAT.finditer(text))
    seqs_int: List[List[int]] = []
    id_to_bytes: List[bytes] = []
    id_to_token: Dict[int, bytes] = {}
    wordcountmap = Counter()
    pair_counts : Counter[Tuple[int,int]] = Counter()
    for piece in pieces:
        b = piece.encode("utf-8")
        sequence = _piece_to_sequence(b)
        ints = [x[0] for x in sequence]
        seqs_int.append(ints)
        wordcountmap[piece] += 1
        pair_counts.update(pairwise(ints))
    # Initialize id_to_bytes for single bytes
    id_to_bytes = [bytes([i]) for i in range(256)]
    # Build initial heap: store (-freq, inverted bytes for tie-break, pair)
    heap = []
    for pair, freq in pair_counts.items():
        a_bytes = id_to_bytes[pair[0]]
        b_bytes = id_to_bytes[pair[1]]
        heapq.heappush(heap, (-freq, _invert_bytes_for_tie(a_bytes), _invert_bytes_for_tie(b_bytes), pair))

    cur_time = time.time()
    print(f"Processed pieces in {cur_time - previously_time} seconds")
    previously_time = cur_time

    merges: List[Tuple[int, int]] = []
    target_merges = vocab_size - 256 - len(special_tokens)
    merges_done = 0

    # --- iterative merging ---
    while heap and merges_done < target_merges:
        # Look at current top frequency
        neg_top, _, _, _ = heap[0]
        top_freq = -neg_top
        # Collect all valid candidates with this frequency
        same_freq_valid: List[Tuple[int, int]] = []
        popped_items = []
        while heap and heap[0][0] == -top_freq:
            item = heapq.heappop(heap)
            popped_items.append(item)
            _, _, _, cand = item
            if pair_counts.get(cand, 0) == top_freq:
                same_freq_valid.append(cand)
        if not same_freq_valid:
            # No valid items at this frequency; continue to next
            continue
        # Tie-break: choose lexicographically largest (by bytes) among equals
        best_pair = max(same_freq_valid, key=lambda p: (id_to_bytes[p[0]], id_to_bytes[p[1]]))
        # Push back the other valid candidates at the same frequency
        for cand in same_freq_valid:
            if cand == best_pair:
                continue
            a_bytes = id_to_bytes[cand[0]]
            b_bytes = id_to_bytes[cand[1]]
            heapq.heappush(
                heap,
                (-top_freq, _invert_bytes_for_tie(a_bytes), _invert_bytes_for_tie(b_bytes), cand),
            )
        new_bytes = id_to_bytes[best_pair[0]] + id_to_bytes[best_pair[1]]
        merges.append(best_pair)
        id_to_bytes.append(new_bytes)
        # Apply merge to all sequences and update pair_counts, tracking changed pairs
        changed_pairs: set[Tuple[int, int]] = set()
        new_token_id = len(id_to_bytes) - 1
        seqs_int = [
            _merge_once_int_pair_counts(s, best_pair, new_token_id, pair_counts, changed_pairs)
            for s in seqs_int
        ]
        # Push updated counts for changed pairs into heap (lazy invalidation handles old entries)
        for p in changed_pairs:
            f = pair_counts.get(p, 0)
            if f > 0:
                a_bytes = id_to_bytes[p[0]]
                b_bytes = id_to_bytes[p[1]]
                heapq.heappush(
                    heap,
                    (-f, _invert_bytes_for_tie(a_bytes), _invert_bytes_for_tie(b_bytes), p),
                )
        merges_done += 1

    cur_time = time.time()
    print(f"Applied merges in {cur_time - previously_time} seconds")
    previously_time = cur_time

    # --- build vocab: ids -> bytes ---
    # 0..255: single bytes
    tokens: List[bytes] = [bytes([i]) for i in range(256)]
    
    # then each newly created merged token (concatenation) in order
    for a, b in merges:
        tok = bytes(id_to_bytes[a]) + bytes(id_to_bytes[b])
        if tok not in tokens:
            tokens.append(tok)
    cur_time = time.time()
    print(f"Built vocab in {cur_time - previously_time} seconds")
    previously_time = cur_time
    # finally, append specials
    for tok in special_tokens:
        tokens.append(tok.encode("utf-8"))
    cur_time = time.time()
    print(f"Appended special tokens in {cur_time - previously_time} seconds")
    previously_time = cur_time
    # If we overshot (due to duplicate merged tokens), trim or pad as needed.
    # Prefer trimming merged tokens at the end to match requested vocab_size.
    if len(tokens) > vocab_size:
        tokens = tokens[:vocab_size]
    elif len(tokens) < vocab_size:
        # Very rare; could happen if corpus is tiny. Pad with unused synthetic tokens.
        # Here we just stop early (keeping size smaller) or raise; tests usually expect exact size:
        raise RuntimeError(f"Could not reach requested vocab_size={vocab_size} (got {len(tokens)})")
    cur_time = time.time()
    print(f"Trimmed/Padded vocab in {cur_time - previously_time} seconds")
    previously_time = cur_time
    vocab = {i: tok for i, tok in enumerate(tokens)}
    # convert merges from (int,int) to (bytes,bytes)
    merges_bytes = [(id_to_bytes[a], id_to_bytes[b]) for a, b in merges]
    return vocab, merges_bytes



# version 3 , use pair counts to trains, takes 5.x seconds
def run_train_bpe_pair_counts(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: List[str],
    **kwargs,
) -> tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Train byte-level BPE on `input_path` and return:
      - vocab: dict[int, bytes]
      - merges: list[(bytes, bytes)] in creation order

    Notes:
      * Special tokens are NOT protected during training (per spec), but are appended to vocab.
      * No merges across GPT-2 pretokenizer piece boundaries.
    """
    # --- sanity checks ---
    # timestamp here and print
    start_time = time.time()
    print(f"Training BPE with vocab_size={vocab_size}, special_tokens={special_tokens}, start_time={start_time}")
    previously_time = start_time
    if vocab_size <= 256 + len(special_tokens):
        raise ValueError(
            f"vocab_size must exceed 256 + len(special_tokens) "
            f"(got {vocab_size}, specials={len(special_tokens)})"
        )

    # --- read corpus ---
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    cur_time = time.time()
    print(f"Read corpus in {cur_time - previously_time} seconds")
    previously_time = cur_time
    # --- build sequences of symbols (bytes) per GPT-2 piece ---
    # Important: we don't treat specials specially during training.
    pieces = (m.group(0) for m in GPT2_PAT.finditer(text))
    seqs_int: List[List[int]] = []
    id_to_bytes: List[bytes] = []
    id_to_token: Dict[int, bytes] = {}
    wordcountmap = Counter()
    pair_counts : Counter[Tuple[int,int]] = Counter()
    for piece in pieces:
        b = piece.encode("utf-8")
        #print(f"Processing piece: {piece} -> {b}")
        sequence = _piece_to_sequence(b)
        ints = [x[0] for x in sequence]
        seqs_int.append(ints)
        wordcountmap[piece]+=1
        pair_counts.update(pairwise(ints))
    # Debug, print wordcount and seqs
    # print(f"seq int: {seqs_int}")
    # print(f"pair_count: {pair_counts}")

    cur_time = time.time()
    print(f"Processed pieces in {cur_time - previously_time} seconds")
    previously_time = cur_time

    merges: List[Tuple[int, int]] = []
    target_merges = vocab_size - 256 - len(special_tokens)
    id_to_bytes = [bytes([i]) for i in range(256)]

    # --- iterative merging ---
    for _ in range(target_merges):
        if not pair_counts:
            break
        # Tie-break deterministically: prefer lexicographically larger pair on equal freq.
        # This matches the reference merges used by the tests.
        # DEBUG: the following commented line is wrong, because it does not account for tie-breaking.
        # best_pair, best_freq = max(stats.items(), key=lambda kv: kv[1])
        # kv[0] is the pair (a, b), kv[1] is the frequency
        best_pair, best_freq = max(pair_counts.items(), key=lambda kv: (kv[1], (id_to_bytes[kv[0][0]], id_to_bytes[kv[0][1]])))
        if best_freq < 1:
            break
        #print(f"Best pair: {best_pair}, freq: {best_freq}")
        new_bytes = id_to_bytes[best_pair[0]] + id_to_bytes[best_pair[1]]
        #print(f"new byte: {new_bytes}")
        merges.append(best_pair)
        id_to_bytes.append(new_bytes)
        # update pair counts
        # Decrement old pairs: (prev,a), (b,next)

        # apply merge to all sequences
        seqs_int = [_merge_once_int_pair_counts(s, best_pair, len(id_to_bytes)-1, pair_counts) for s in seqs_int]

    cur_time = time.time()
    print(f"Applied merges in {cur_time - previously_time} seconds")
    previously_time = cur_time

    # --- build vocab: ids -> bytes ---
    # 0..255: single bytes
    tokens: List[bytes] = [bytes([i]) for i in range(256)]
    
    # then each newly created merged token (concatenation) in order
    for a, b in merges:
        tok = bytes(id_to_bytes[a]) + bytes(id_to_bytes[b])
        if tok not in tokens:
            tokens.append(tok)
    cur_time = time.time()
    print(f"Built vocab in {cur_time - previously_time} seconds")
    previously_time = cur_time
    # finally, append specials
    for tok in special_tokens:
        tokens.append(tok.encode("utf-8"))
    cur_time = time.time()
    print(f"Appended special tokens in {cur_time - previously_time} seconds")
    previously_time = cur_time
    # If we overshot (due to duplicate merged tokens), trim or pad as needed.
    # Prefer trimming merged tokens at the end to match requested vocab_size.
    if len(tokens) > vocab_size:
        tokens = tokens[:vocab_size]
    elif len(tokens) < vocab_size:
        # Very rare; could happen if corpus is tiny. Pad with unused synthetic tokens.
        # Here we just stop early (keeping size smaller) or raise; tests usually expect exact size:
        raise RuntimeError(f"Could not reach requested vocab_size={vocab_size} (got {len(tokens)})")
    cur_time = time.time()
    print(f"Trimmed/Padded vocab in {cur_time - previously_time} seconds")
    previously_time = cur_time
    vocab = {i: tok for i, tok in enumerate(tokens)}
    # convert merges from (int,int) to (bytes,bytes)
    merges_bytes = [(id_to_bytes[a], id_to_bytes[b]) for a, b in merges]
    return vocab, merges_bytes




# version 2: use int to train, takes 8.x seconds
def run_train_bpe_int(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: List[str],
    **kwargs,
) -> tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Train byte-level BPE on `input_path` and return:
      - vocab: dict[int, bytes]
      - merges: list[(bytes, bytes)] in creation order

    Notes:
      * Special tokens are NOT protected during training (per spec), but are appended to vocab.
      * No merges across GPT-2 pretokenizer piece boundaries.
    """
    # --- sanity checks ---
    # timestamp here and print
    start_time = time.time()
    print(f"Training BPE with vocab_size={vocab_size}, special_tokens={special_tokens}, start_time={start_time}")
    previously_time = start_time
    if vocab_size <= 256 + len(special_tokens):
        raise ValueError(
            f"vocab_size must exceed 256 + len(special_tokens) "
            f"(got {vocab_size}, specials={len(special_tokens)})"
        )

    # --- read corpus ---
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    cur_time = time.time()
    print(f"Read corpus in {cur_time - previously_time} seconds")
    previously_time = cur_time
    # --- build sequences of symbols (bytes) per GPT-2 piece ---
    # Important: we don't treat specials specially during training.
    pieces = (m.group(0) for m in GPT2_PAT.finditer(text))
    seqs_int: List[List[int]] = []
    id_to_bytes: List[bytes] = []
    id_to_token: Dict[int, bytes] = {}
    wordcountmap = Counter()
    for piece in pieces:
        b = piece.encode("utf-8")
        #print(f"Processing piece: {piece} -> {b}")
        sequence = _piece_to_sequence(b)
        seqs_int.append([x[0] for x in sequence])
        wordcountmap[piece]+=1
    # Debug, print wordcount and seqs
    # print(f"seq int: {seqs_int}")

    cur_time = time.time()
    print(f"Processed pieces in {cur_time - previously_time} seconds")
    previously_time = cur_time

    merges: List[Tuple[int, int]] = []
    target_merges = vocab_size - 256 - len(special_tokens)
    id_to_bytes = [bytes([i]) for i in range(256)]

    # --- iterative merging ---
    for _ in range(target_merges):
        stats = _count_pairs_int(seqs_int)
        if not stats:
            break
        # Tie-break deterministically: prefer lexicographically larger pair on equal freq.
        # This matches the reference merges used by the tests.
        # DEBUG: the following commented line is wrong, because it does not account for tie-breaking.
        # best_pair, best_freq = max(stats.items(), key=lambda kv: kv[1])
        # kv[0] is the pair (a, b), kv[1] is the frequency
        best_pair, best_freq = max(stats.items(), key=lambda kv: (kv[1], (id_to_bytes[kv[0][0]], id_to_bytes[kv[0][1]])))
        if best_freq < 1:
            break
        #print(f"Best pair: {best_pair}, freq: {best_freq}")
        new_bytes = id_to_bytes[best_pair[0]] + id_to_bytes[best_pair[1]]
        #print(f"new byte: {new_bytes}")
        merges.append(best_pair)
        id_to_bytes.append(new_bytes)
        # apply merge to all sequences
        seqs_int = [_merge_once_int(s, best_pair, len(id_to_bytes)-1) for s in seqs_int]
    cur_time = time.time()
    print(f"Applied merges in {cur_time - previously_time} seconds")
    previously_time = cur_time

    # --- build vocab: ids -> bytes ---
    # 0..255: single bytes
    tokens: List[bytes] = [bytes([i]) for i in range(256)]
    
    # then each newly created merged token (concatenation) in order
    for a, b in merges:
        tok = bytes(id_to_bytes[a]) + bytes(id_to_bytes[b])
        if tok not in tokens:
            tokens.append(tok)
    cur_time = time.time()
    print(f"Built vocab in {cur_time - previously_time} seconds")
    previously_time = cur_time
    # finally, append specials
    for tok in special_tokens:
        tokens.append(tok.encode("utf-8"))
    cur_time = time.time()
    print(f"Appended special tokens in {cur_time - previously_time} seconds")
    previously_time = cur_time
    # If we overshot (due to duplicate merged tokens), trim or pad as needed.
    # Prefer trimming merged tokens at the end to match requested vocab_size.
    if len(tokens) > vocab_size:
        tokens = tokens[:vocab_size]
    elif len(tokens) < vocab_size:
        # Very rare; could happen if corpus is tiny. Pad with unused synthetic tokens.
        # Here we just stop early (keeping size smaller) or raise; tests usually expect exact size:
        raise RuntimeError(f"Could not reach requested vocab_size={vocab_size} (got {len(tokens)})")
    cur_time = time.time()
    print(f"Trimmed/Padded vocab in {cur_time - previously_time} seconds")
    previously_time = cur_time
    vocab = {i: tok for i, tok in enumerate(tokens)}
    # convert merges from (int,int) to (bytes,bytes)
    merges_bytes = [(id_to_bytes[a], id_to_bytes[b]) for a, b in merges]
    return vocab, merges_bytes


# version 1: use bytes to train, takes 9 seconds
def run_train_bpe_bytes(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: List[str],
    **kwargs,
) -> tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Train byte-level BPE on `input_path` and return:
      - vocab: dict[int, bytes]
      - merges: list[(bytes, bytes)] in creation order

    Notes:
      * Special tokens are NOT protected during training (per spec), but are appended to vocab.
      * No merges across GPT-2 pretokenizer piece boundaries.
    """
    # --- sanity checks ---
    # timestamp here and print
    start_time = time.time()
    print(f"Training BPE with vocab_size={vocab_size}, special_tokens={special_tokens}, start_time={start_time}")
    previously_time = start_time
    if vocab_size <= 256 + len(special_tokens):
        raise ValueError(
            f"vocab_size must exceed 256 + len(special_tokens) "
            f"(got {vocab_size}, specials={len(special_tokens)})"
        )

    # --- read corpus ---
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    cur_time = time.time()
    print(f"Read corpus in {cur_time - previously_time} seconds")
    previously_time = cur_time
    # --- build sequences of symbols (bytes) per GPT-2 piece ---
    # Important: we don't treat specials specially during training.
    pieces = (m.group(0) for m in GPT2_PAT.finditer(text))
    seqs: List[List[bytes]] = []
    seqs_int: List[List[int]] = []
    id_to_seqs: Dict[int, List[bytes]] = {}
    id_to_token: Dict[int, bytes] = {}
    seqs_set: set = set()
    wordcountmap = Counter()
    for piece in pieces:
        b = piece.encode("utf-8")
        #print(f"Processing piece: {piece} -> {b}")
        symbols = _piece_to_symbols(b)
        seqs.append(symbols)
        seqs_int.append([x[0] for x in symbols])
        id_to_seqs[len(id_to_seqs)] = symbols
        seqs_set.update(piece)
        wordcountmap[piece]+=1
    # Debug, print wordcount and seqs
    print(f"seq int: {seqs_int}")
    print(f"Sequences: {seqs}")

    cur_time = time.time()
    print(f"Processed pieces in {cur_time - previously_time} seconds")
    previously_time = cur_time

    merges: List[Tuple[bytes, bytes]] = []
    target_merges = vocab_size - 256 - len(special_tokens)

    # --- iterative merging ---
    for _ in range(target_merges):
        stats = _count_pairs(seqs)
        if not stats:
            break
        # Tie-break deterministically: prefer lexicographically larger pair on equal freq.
        # This matches the reference merges used by the tests.
        # DEBUG: the following commented line is wrong, because it does not account for tie-breaking.
        # best_pair, best_freq = max(stats.items(), key=lambda kv: kv[1])
        # kv[0] is the pair (a, b), kv[1] is the frequency
        best_pair, best_freq = max(stats.items(), key=lambda kv: (kv[1], kv[0]))
        if best_freq < 1:
            break
        merges.append(best_pair)
        # apply merge to all sequences
        seqs = [_merge_once(s, best_pair) for s in seqs]
    cur_time = time.time()
    print(f"Applied merges in {cur_time - previously_time} seconds")
    previously_time = cur_time

    # --- build vocab: ids -> bytes ---
    # 0..255: single bytes
    tokens: List[bytes] = [bytes([i]) for i in range(256)]
    # then each newly created merged token (concatenation) in order
    for a, b in merges:
        tok = a + b
        if tok not in tokens:
            tokens.append(tok)
    cur_time = time.time()
    print(f"Built vocab in {cur_time - previously_time} seconds")
    previously_time = cur_time
    # finally, append specials
    for tok in special_tokens:
        tokens.append(tok.encode("utf-8"))
    cur_time = time.time()
    print(f"Appended special tokens in {cur_time - previously_time} seconds")
    previously_time = cur_time
    # If we overshot (due to duplicate merged tokens), trim or pad as needed.
    # Prefer trimming merged tokens at the end to match requested vocab_size.
    if len(tokens) > vocab_size:
        tokens = tokens[:vocab_size]
    elif len(tokens) < vocab_size:
        # Very rare; could happen if corpus is tiny. Pad with unused synthetic tokens.
        # Here we just stop early (keeping size smaller) or raise; tests usually expect exact size:
        raise RuntimeError(f"Could not reach requested vocab_size={vocab_size} (got {len(tokens)})")
    cur_time = time.time()
    print(f"Trimmed/Padded vocab in {cur_time - previously_time} seconds")
    previously_time = cur_time
    vocab = {i: tok for i, tok in enumerate(tokens)}
    return vocab, merges
