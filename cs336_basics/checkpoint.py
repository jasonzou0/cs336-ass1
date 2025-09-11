import os
from typing import BinaryIO, IO

import torch

from cs336_basics.optimizer import CosineScheduler

def _save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
    lr_scheduler: CosineScheduler | None = None
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
    state_dict = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    if lr_scheduler is not None:
        state_dict["lr_scheduler_state_dict"] = lr_scheduler.state_dict()
    torch.save(state_dict, out)


def _load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: CosineScheduler | None = None,
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
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
    return checkpoint["iteration"]


class CheckpointClient:
    """
    A client for saving and loading model and optimizer checkpoints.
    """
    def __init__(
            self, 
            model: torch.nn.Module, 
            optimizer: torch.optim.Optimizer,
            lr_scheduler: CosineScheduler | None = None,
            checkpoint_dest: str | None = None):
        """
        Initialize the CheckpointClient.

        Args:
            model (torch.nn.Module): The model to save and load.
            optimizer (torch.optim.Optimizer): The optimizer to save and load.
            CosineScheduler (CosineScheduler | None): The learning rate scheduler to save and load.
            checkpoint_dest (str): Directory or file path to save checkpoints to.
        """
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_dest = checkpoint_dest
        self.lr_scheduler = lr_scheduler
    
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
        _save_checkpoint(
            model=self.model, 
            optimizer=self.optimizer, 
            iteration=iteration, 
            lr_scheduler=self.lr_scheduler,
            out=checkpoint_path
        )


    def load(self, checkpoint_src: str | os.PathLike | BinaryIO | IO[bytes]) -> int:
        """
        Load a checkpoint.

        Args:
            checkpoint_path (str): Path to the checkpoint file.

        Returns:
            int: The iteration number stored in the checkpoint.
        """
        return _load_checkpoint(
            src=checkpoint_src, 
            model=self.model, 
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler
        )