from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import torch
import random
from typing import Optional, Tuple

@dataclass
class DataLoaderConfig:
    # Path to the dataset file (a .npy file containing a 1D numpy array of integer token IDs).
    dataset_path: str
    batch_size: int = 32
    context_length: int = 256
    num_batches: Optional[int] = None
    random_seed: Optional[int] = None


class DataLoader:
    """
    Simple iterable that repeatedly delegates batch construction to sample_batch().

    Each iteration yields a fresh random batch sampled from the full dataset.

    Example:
        dl = DataLoader(dataset, batch_size=32, context_length=128, device='cpu', num_batches=100, random_seed=42)
        for x, y in dl:  # yields num_batches batches per epoch
            ...

    Reproducibility:
        If random_seed is set, each (epoch, batch_index) pair produces deterministic batches
        by offsetting the base seed.
    """
    @staticmethod
    def from_config(config: DataLoaderConfig, device: str) -> "DataLoader":
        dataset = np.load(config.dataset_path)
        return DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            context_length=config.context_length,
            device=device,
            num_batches=config.num_batches,
            random_seed=config.random_seed,
        )

    def __init__(
        self,
        dataset: npt.NDArray,
        batch_size: int,
        context_length: int,
        device: str,
        num_batches: Optional[int] = None,
        random_seed: Optional[int] = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device
        # If num_batches not provided, default to covering dataset roughly once (heuristic)
        if num_batches is None:
            usable_positions = max(0, len(dataset) - context_length)
            est = usable_positions // batch_size
            self.num_batches = max(1, est)
        else:
            self.num_batches = num_batches
        self.random_seed = random_seed
        self._epoch = 0
        self._batches_yielded = 0

    def __iter__(self) -> "DataLoader":
        self._epoch += 1
        self._batches_yielded = 0
        return self

    def __len__(self) -> int:
        return self.num_batches

    def __next__(self) -> Tuple[torch.LongTensor, torch.LongTensor]:
        if self._batches_yielded >= self.num_batches:
            raise StopIteration
        x, y = sample_batch(
            self.dataset,
            self.batch_size,
            self.context_length,
            self.device,
            random_seed=self.random_seed,
        )
        self._batches_yielded += 1
        return x, y

    def set_num_batches(self, num_batches: int):
        self.num_batches = num_batches


def sample_batch(
    dataset: npt.NDArray, 
    batch_size: int, 
    context_length: int, 
    device: str, 
    random_seed: Optional[int] = None,
) -> tuple[torch.LongTensor, torch.LongTensor]:
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
