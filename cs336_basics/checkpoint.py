import os
from typing import BinaryIO, IO

import torch

def _save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration
    }, out)
    


def _load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]

class CheckpointClient:
    """
    A client for saving and loading model and optimizer checkpoints.
    """
    def __init__(
            self, 
            model: torch.nn.Module, 
            optimizer: torch.optim.Optimizer,
            checkpoint_dest: str | None = None):
        """
        Initialize the CheckpointClient.

        Args:
            model (torch.nn.Module): The model to save and load.
            optimizer (torch.optim.Optimizer): The optimizer to save and load.
            checkpoint_dest (str): Directory or file path to save checkpoints to.
        """
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_dest = checkpoint_dest
    
    def save(self, iteration: int) -> None:
        """
        Save a checkpoint.

        Args:
            iteration (int): The current training iteration.
        """
        if os.path.isdir(self.checkpoint_dest):
            checkpoint_path = os.path.join(self.checkpoint_dest, f"checkpoint_step_{iteration}.pt")
        else:
            assert self.checkpoint_dest, "checkpoint_dest must be specified"
            checkpoint_path = self.checkpoint_dest
        _save_checkpoint(self.model, self.optimizer, iteration, checkpoint_path)

    def load(self, checkpoint_src: str | os.PathLike | BinaryIO | IO[bytes]) -> int:
        """
        Load a checkpoint.

        Args:
            checkpoint_path (str): Path to the checkpoint file.

        Returns:
            int: The iteration number stored in the checkpoint.
        """
        return _load_checkpoint(checkpoint_src, self.model, self.optimizer)