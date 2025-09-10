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
        merges_path: str | os.PathLike
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Load BPE vocabulary and merges from pickled files.
    Args:
        vocab_path: Path to the pickled vocabulary file (vocab.pkl)
        merges_path: Path to the pickled merges file (merges.pkl)
    Returns:
        Tuple of (vocab, merges)
    """
    return load_artifact(vocab_path), load_artifact(merges_path)


def save_bpe(
    vocab: dict[int, bytes], 
    merges: list[tuple[bytes, bytes]], 
    output_directory: str | os.PathLike
) -> None:
    """
    Save BPE vocabulary and merges to disk as pickled files.

    Args:
        vocab: The vocabulary dictionary mapping token IDs to bytes
        merges: List of merge tuples (bytes, bytes)
        output_directory: Directory where to save the vocab.pkl and merges.pkl files
    """
    output_dir = os.path.abspath(output_directory)
    os.makedirs(output_dir, exist_ok=True)

    vocab_path = os.path.join(output_dir, "vocab.pkl")
    merges_path = os.path.join(output_dir, "merges.pkl")

    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)

    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)

    print(f"Saved vocabulary to {vocab_path}")
    print(f"Saved merges to {merges_path}")