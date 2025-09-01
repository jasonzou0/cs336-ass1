import torch
from torch import Tensor

from einops import rearrange
from jaxtyping import Float, Int


def cross_entropy_loss(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    """Compute the average cross-entropy loss between the input logits and target class indices.

    Args:
        inputs: Tensor of shape (batch_size, vocab_size) representing the predicted logits for each class
        targets: Tensor of shape (batch_size,) containing the true class indices for each example

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    # Subtract the max for numerical stability
    inputs -= torch.max(inputs, dim=-1, keepdim=True).values
    # The log normalizer log(sum(exp(inputs))) term is common for all classes in a given example / batch
    log_normalizer: Float[Tensor, " batch_size"] = torch.log(torch.sum(torch.exp(inputs), dim=-1))
    # Select the logit corresponding to the target class for each example / batch
    losses: Float[Tensor, " batch_size"] = log_normalizer - inputs.gather(dim=-1, index=rearrange(targets, " batch_size -> batch_size 1")).squeeze(-1)
    return torch.mean(losses)