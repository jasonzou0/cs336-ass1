import os
import pickle

def load_artifact(
    artifact_path: str | os.PathLike
) -> dict[int, bytes] | list[tuple[bytes, bytes]]:
    """
    Load BPE vocabulary or merges artifact.

    Args:
        artifact_path: Path to the BPE vocab or merges artifact.

    Returns:
        vocab or merges loaded from the artifact.
    """
    with open(artifact_path, "rb") as f:
        artifact = pickle.load(f)
    return artifact


def load_bpe(
        vocab_path: str | os.PathLike,
        merges_path: str | os.PathLike,
        special_tokens_path: str | os.PathLike = None
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Load BPE vocabulary and merges from pickled files.
    Args:
        vocab_path: Path to the pickled vocabulary file (vocab.pkl)
        merges_path: Path to the pickled merges file (merges.pkl)
        special_tokens_path: Optional path to a text file containing special tokens, one per line.
    Returns:
        Tuple of (vocab, merges, special_tokens)
    """
    return (load_artifact(vocab_path), 
            load_artifact(merges_path), 
            load_artifact(special_tokens_path) if special_tokens_path else [])


def save_bpe(
    vocab: dict[int, bytes], 
    merges: list[tuple[bytes, bytes]], 
    special_tokens: list[str],
    output_directory: str | os.PathLike
) -> None:
    """
    Save BPE vocabulary and merges to disk as pickled files.

    Args:
        vocab: The vocabulary dictionary mapping token IDs to bytes
        merges: List of merge tuples (bytes, bytes)
        special_tokens: List of special tokens to include in the vocabulary
        output_directory: Directory where to save the vocab.pkl and merges.pkl files
    """
    output_dir = os.path.abspath(output_directory)
    os.makedirs(output_dir, exist_ok=True)

    vocab_path = os.path.join(output_dir, "vocab.pkl")
    merges_path = os.path.join(output_dir, "merges.pkl")
    special_tokens_path = os.path.join(output_dir, "special_tokens.pkl")

    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)

    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)

    with open(special_tokens_path, "wb") as f:
        pickle.dump(special_tokens, f)

    print(f"Saved vocabulary to {vocab_path}")
    print(f"Saved merges to {merges_path}")
    print(f"Saved special tokens to {special_tokens_path}")