import numpy as np
import numpy.typing as npt
import torch
import random
from typing import Optional

def sample_batch(
    dataset: npt.NDArray, 
    batch_size: int, 
    context_length: int, 
    device: str, 
    random_seed: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.
        random_seed (Optional[int]): Optional random seed for reproducibility. If None, no
            seed is set.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    start_indices = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    x = np.array([dataset[i : i + context_length] for i in start_indices])
    y = np.array([dataset[i + 1 : i + context_length + 1] for i in start_indices])
    return torch.from_numpy(np.stack(x, axis=0)).long().to(device), torch.from_numpy(np.stack(y, axis=0)).long().to(device)
