import json
import time

from .adapters import run_train_bpe
from .common import FIXTURES_PATH, gpt2_bytes_to_unicode


def test_train_bpe_speed():
    """
    Ensure that BPE training is relatively efficient by measuring training
    time on this small dataset and throwing an error if it takes more than 1.5 seconds.
    This is a pretty generous upper-bound, it takes 0.38 seconds with the
    reference implementation on my laptop. In contrast, the toy implementation
    takes around 3 seconds.
    """
    input_path = FIXTURES_PATH / "corpus.en"
    start_time = time.time()
    _, _ = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    end_time = time.time()
    assert end_time - start_time < 1.5

def test_train_bpe_sennrich_example():
    input_path = FIXTURES_PATH / "sennrich.en"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=263,
        special_tokens=["<|endoftext|>"],
        pretokenizer_name="ws",
        debug=True,
    )
    
    # Verify there are exactly 263 tokens in the vocab
    assert len(vocab) == 263
    
    # The vocab should have:
    # - 256 single-byte tokens (0-255)  
    # - 1 special token "<|endoftext|>" at index 256
    # - 6 merged tokens (257-262)
    
    # Get the last 6 entries in the vocab (the merged tokens)
    last_6_token_ids = sorted(vocab.keys())[-6:]
    last_6_tokens = [vocab[token_id] for token_id in last_6_token_ids]
    
    # Convert to string representations for checking
    # Based on the BPE algorithm, these should be the merged tokens created
    # Note: The current implementation has some issues with byte representation
    # but the core logic should produce the expected merges

    # Expected tokens based on the merges we observed:
    # st, ne, ow, and combinations with spaces and 'e' prefix
    expected_token_strings = ['st', 'est', 'ow', 'low', 'west', 'ne']

    # Verify there are exactly 6 merges
    assert len(merges) == 6

    # Verify the specific merges are as expected
    # Convert merges to string representation for comparison
    merge_strings = []
    for first, second in merges:
        merge_strings.append(f"{first.decode('utf-8')} {second.decode('utf-8')}")

    # Expected merges based on the BPE algorithm
    expected_merges = ['s t', 'e st', 'o w', 'l ow', 'w est', 'n e']

    # For debugging, print what we actually got
    print(f"Actual merges: {merge_strings}")
    print(f"Expected merges: {expected_merges}")

    assert merge_strings == expected_merges, (
        f"Expected merges {expected_merges} but got {merge_strings}"
    )


def test_train_bpe():
    input_path = FIXTURES_PATH / "corpus.en"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )

    # Path to the reference tokenizer vocab and merges
    reference_vocab_path = FIXTURES_PATH / "train-bpe-reference-vocab.json"
    reference_merges_path = FIXTURES_PATH / "train-bpe-reference-merges.txt"

    # Compare the learned merges to the expected output merges
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(reference_merges_path, encoding="utf-8") as f:
        gpt2_reference_merges = [tuple(line.rstrip().split(" ")) for line in f]
        reference_merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_reference_merges
        ]
    assert merges == reference_merges

    # Compare the vocab to the expected output vocab
    with open(reference_vocab_path, encoding="utf-8") as f:
        gpt2_reference_vocab = json.load(f)
        reference_vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_reference_vocab.items()
        }
    # Rather than checking that the vocabs exactly match (since they could
    # have been constructed differently, we'll make sure that the vocab keys and values match)
    assert set(vocab.keys()) == set(reference_vocab.keys())
    assert set(vocab.values()) == set(reference_vocab.values())


def test_train_bpe_special_tokens(snapshot):
    """
    Ensure that the special tokens are added to the vocabulary and not
    merged with other tokens.
    """
    input_path = FIXTURES_PATH / "tinystories_sample_5M.txt"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=1000,
        special_tokens=["<|endoftext|>"],
    )

    # Check that the special token is not in the vocab
    vocabs_without_specials = [word for word in vocab.values() if word != b"<|endoftext|>"]
    for word_bytes in vocabs_without_specials:
        assert b"<|" not in word_bytes

    snapshot.assert_match(
        {
            "vocab_keys": set(vocab.keys()),
            "vocab_values": set(vocab.values()),
            "merges": merges,
        },
    )
