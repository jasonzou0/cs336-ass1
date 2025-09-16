import torch

from cs336_basics.optimizer import CosineScheduler
from cs336_basics.data_loader import DataLoader, DataLoaderConfig, DataLoadingMode
from cs336_basics.checkpoint import CheckpointClient
from cs336_basics.grad_clipping import grad_clipping
from cs336_basics.optimizer import create_from_config, OptimizerConfig
from cs336_basics.checkpoint import CheckpointClient


def run_training(
        dataset_path: str, 
        model: torch.nn.Module,
        num_batches: int, 
        conxtext_length: int,
        batch_size: int,
        load_checkpoint_path: str | None,
        checkpoint_interval: int,
        checkpoint_dir: str,
        device: str) -> torch.nn.Module:
    """Run the training loop for a Transformer model.

    Args:
        model (nn.Module): The Transformer model to train (wrapped in ModelWrapperWithCELoss).
        dataset_path (str): Path to the training dataset (numpy file).
        vocab_size (int): Size of the vocabulary.
        num_batches (int): Number of training batches.
        load_checkpoint_path (str | None): Path to a checkpoint to load before training.
        checkpoint_interval (int): Interval (in steps) to save checkpoints.
        checkpoint_dir (str): Directory to save checkpoints.
        device (str): Device to use for training (e.g., "cpu", "cuda", "mps").
    Returns:
        model (nn.Module): The trained Transformer model.
    """
    model.to(device)
    # TODO: inspect the compiled code. 
    if device == "mps":
        model = torch.compile(model, backend="aot_eager", fullgraph=True)
    else:
        model = torch.compile(model, fullgraph=True)
    train_data_loader = DataLoader.from_config(
        DataLoaderConfig(dataset_path=dataset_path, num_batches=num_batches, batch_size=batch_size, context_length=conxtext_length, data_loading_mode=DataLoadingMode.RANDOM), 
        device=device)
    optimizer, scheduler = create_from_config(model.parameters(), config=OptimizerConfig(total_iters=num_batches))
    checkpoint_client = CheckpointClient(
        model=model, 
        optimizer=optimizer, 
        lr_scheduler=scheduler,
        checkpoint_dest=checkpoint_dir)
    starting_iteration = 0
    if load_checkpoint_path:
        starting_iteration = checkpoint_client.load(load_checkpoint_path)
        print(f"Loaded checkpoint from {load_checkpoint_path}, starting from iteration {starting_iteration}")
    trainer = Trainer(
        model_with_loss=model, 
        starting_iteration=starting_iteration,
        data_loader=train_data_loader,
        optimizer=optimizer, 
        scheduler=scheduler, 
        checkpoint_client=checkpoint_client,
        checkpoint_interval=checkpoint_interval,
        device=device)
    trainer.train()
    return model


class Trainer:
    def __init__(self, 
                 model_with_loss: torch.nn.Module, 
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
            model_with_loss (torch.nn.Module): The model to train. It should return a single loss value as float.
            starting_iteration (int): The iteration to start training from (useful for resuming from checkpoints).
            data_loader (DataLoader): DataLoader providing training data.
            optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
            scheduler (CosineScheduler): Learning rate scheduler.
            checkpoint_client (CheckpointClient): Client for saving and loading checkpoints.
            checkpoint_interval (int): Interval (in steps) to save checkpoints.
            device (str): Device to use for training (e.g., "cpu", "cuda", "mps").
        """
        self.model_with_loss = model_with_loss
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
        self.model_with_loss.train()
        for input_ids, target_ids in iter(self.data_loader):
            # Forward pass
            loss = self.model_with_loss(input_ids, target_ids)
            # Backward pass and optimization step
            loss.backward()
            # TODO: expose max_l2_norm as a config parameter
            grad_clipping(self.model_with_loss.parameters(), max_l2_norm=1.0)
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

