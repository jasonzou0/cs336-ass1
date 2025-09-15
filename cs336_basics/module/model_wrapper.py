import torch
from torch import Tensor

from jaxtyping import Float, Int

from .loss import cross_entropy_loss


class ModelWrapperWithCELoss(torch.nn.Module):
    """A wrapper around the Transformer model that computes cross-entropy loss given input and target sequences."""
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, in_indices: Int[Tensor, " batch_size sequence_length"], target_indices: Int[Tensor, " batch_size sequence_length"]) -> Float[Tensor, ""]:
        """Compute the cross-entropy loss between the model's predictions and the target indices.

        Args:
            in_indices: Input token indices of shape (batch_size, sequence_length)
            target_indices: Target token indices of shape (batch_size, sequence_length)
        Returns:
            Float[Tensor, ""]: Scalar tensor representing the cross-entropy loss
        """
        return cross_entropy_loss(
            inputs=self.model(in_indices),
            targets=target_indices,
        )