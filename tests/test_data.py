import math
from collections import Counter

import numpy as np
import pytest

from .adapters import run_get_batch, run_get_sequential_dataloader

def test_sequence_data_loading():
    """Test that sequential data loading creates non-overlapping batches with context_length spacing."""
    dataset = np.arange(0, 100)
    context_length = 7
    batch_size = 4
    device = "cpu"

    data_loader = run_get_sequential_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        context_length=context_length,
        device=device,
    )

    # Collect all batches and verify sequential pattern
    starting_indices = []
    batch_count = 0
    
    for input_ids, target_ids in iter(data_loader):
        # Verify batch shapes
        assert input_ids.shape == (batch_size, context_length)
        assert target_ids.shape == (batch_size, context_length)

        # Verify target is input shifted by 1
        np.testing.assert_allclose((input_ids + 1).detach().numpy(), target_ids.detach().numpy())

        # Collect starting indices (first token of each sequence in the batch)
        batch_starting_indices = input_ids[:, 0].tolist()
        starting_indices.extend(batch_starting_indices)
        batch_count += 1

    # Expecting 3 complete batches to cover the dataset of size 100 with context_length 7:
    # Batch 1: starts at [0, 7, 14, 21]
    # Batch 2: starts at [28, 35, 42, 49
    # Batch 3: starts at [56, 63, 70, 77]
    # (Batch 4 would be [84, 91, 98, 105] but 105 is out of bounds)
    expected_indices = [0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77]
    assert batch_count == 3, f"Expected 3 batches, but got {batch_count}"
    assert starting_indices == expected_indices, (
        f"Expected starting indices to cover the dataset sequentially, "
        f"but got {starting_indices}, expected {expected_indices}"
    )
    

def test_get_batch():
    dataset = np.arange(0, 100)
    context_length = 7
    batch_size = 32
    device = "cpu"

    # Sanity check to make sure that the random samples are indeed somewhat random.
    starting_indices = Counter()
    num_iters = 1000
    for _ in range(num_iters):
        x, y = run_get_batch(
            dataset=dataset,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )

        # Make sure the shape is correct
        assert x.shape == (batch_size, context_length)
        assert y.shape == (batch_size, context_length)

        # Make sure the y's are always offset by 1
        np.testing.assert_allclose((x + 1).detach().numpy(), y.detach().numpy())

        starting_indices.update(x[:, 0].tolist())

    # Make sure we never sample an invalid start index
    num_possible_starting_indices = len(dataset) - context_length
    assert max(starting_indices) == num_possible_starting_indices - 1
    assert min(starting_indices) == 0
    # Expected # of times that we see each starting index
    expected_count = (num_iters * batch_size) / num_possible_starting_indices
    standard_deviation = math.sqrt(
        (num_iters * batch_size) * (1 / num_possible_starting_indices) * (1 - (1 / num_possible_starting_indices))
    )
    # Range for expected outcomes (mu +/- 5sigma). For a given index,
    # this should happen 99.99994% of the time of the time.
    # So, in the case where we have 93 possible start indices,
    # the entire test should pass with 99.9944202% of the time
    occurrences_lower_bound = expected_count - 5 * standard_deviation
    occurrences_upper_bound = expected_count + 5 * standard_deviation

    for starting_index, count in starting_indices.items():
        if count < occurrences_lower_bound:
            raise ValueError(
                f"Starting index {starting_index} occurs {count} times, but expected at least {occurrences_lower_bound}"
            )
        if count > occurrences_upper_bound:
            raise ValueError(
                f"Starting index {starting_index} occurs {count} times, but expected at most {occurrences_upper_bound}"
            )

    with pytest.raises((RuntimeError, AssertionError)) as excinfo:
        # We're assuming that cuda:99 is an invalid device ordinal.
        # Just adding this here to make sure that the device flag is
        # being handled.
        run_get_batch(
            dataset=dataset,
            batch_size=batch_size,
            context_length=context_length,
            device="cuda:99",
        )
        assert "CUDA error" in str(excinfo.value) or "Torch not compiled with CUDA enabled" in str(excinfo.value)
