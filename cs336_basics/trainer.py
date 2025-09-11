import torch

from cs336_basics.optimizer import CosineScheduler
from cs336_basics.data_loader import DataLoader
from cs336_basics.checkpoint import CheckpointClient
from cs336_basics.grad_clipping import grad_clipping
from cs336_basics.module.loss import cross_entropy_loss


class Trainer:
    def __init__(self, 
                 model: torch.nn.Module, 
                 data_loader: DataLoader,
                 optimizer: torch.optim.Optimizer, 
                 scheduler: CosineScheduler, 
                 checkpoint_client: CheckpointClient,
                 checkpoint_interval: int,
                 starting_iteration: int = 0,
                 device: str = None):
        """
        Trainer for training a model.

        Args:
            model (torch.nn.Module): The model to train.
            starting_iteration (int): The iteration to start training from (useful for resuming from checkpoints).
            data_loader (DataLoader): DataLoader providing training data.
            optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
            scheduler (CosineScheduler): Learning rate scheduler.
            checkpoint_client (CheckpointClient): Client for saving and loading checkpoints.
            checkpoint_interval (int): Interval (in steps) to save checkpoints.
            device (str): Device to use for training (e.g., "cpu", "cuda", "mps").
        """
        self.model = model
        self.starting_iteration = starting_iteration
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_client = checkpoint_client
        self.checkpoint_interval = checkpoint_interval
        self.device = device if device is not None else torch.device("cpu")
    
    def train(self):
        """
        Train the model for a specified number of steps.
        Args:
            get_batch_func (callable): Function to get a batch of input and target tensors.
            num_steps (int): Number of training steps to perform.
        """
        t = self.starting_iteration
        self.model.train()
        for input_ids, target_ids in iter(self.data_loader):
            # Forward pass
            logits = self.model(input_ids)
            loss = cross_entropy_loss(logits, target_ids)
            # Backward pass and optimization step
            loss.backward()
            # TODO: expose max_l2_norm as a config parameter
            grad_clipping(self.model.parameters(), max_l2_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            if t % 10 == 0:
                print(f"Step {t}, Loss: {loss.item():.4f}")
            if t % self.checkpoint_interval == 0 and t > self.starting_iteration:
                self.checkpoint_client.save(t)
                print(f"Checkpoint saved at step {t}")
            t += 1
        # Save final checkpoint
        self.checkpoint_client.save(t)
        print(f"Final checkpoint saved at step {t}")

