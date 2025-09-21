import time
import torch
import wandb

from cs336_basics.optimizer import CosineScheduler
from cs336_basics.data_loader import DataLoader, DataLoaderConfig, DataLoadingMode
from cs336_basics.checkpoint import CheckpointClient
from cs336_basics.grad_clipping import grad_clipping
from cs336_basics.optimizer import create_from_config, OptimizerConfig
from cs336_basics.checkpoint import CheckpointClient


def run_training(
        data_loader_config: DataLoaderConfig,
        optimizer_config: OptimizerConfig,
        model: torch.nn.Module,
        load_checkpoint_path: str | None,
        checkpoint_interval: int,
        checkpoint_dir: str,
        device: str,
        wandb: wandb.Run | None,
    ) -> torch.nn.Module:
    """Run the training loop for a Transformer model.

    Args:
        model (nn.Module): The Transformer model to train (wrapped in ModelWrapperWithCELoss).
        load_checkpoint_path (str | None): Path to a checkpoint to load before training.
        checkpoint_interval (int): Interval (in steps) to save checkpoints.
        checkpoint_dir (str): Directory to save checkpoints.
        wandb_run: wandb.Run | None: Weights & Biases run for logging, or None to disable logging.
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
    train_data_loader = DataLoader.from_config(data_loader_config, device=device)
    optimizer, scheduler = create_from_config(model.parameters(), optimizer_config)
    checkpoint_client = CheckpointClient(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        checkpoint_dest=checkpoint_dir)
    starting_iteration = 0
    if load_checkpoint_path:
        starting_iteration = checkpoint_client.load(load_checkpoint_path)
        print(f"Loaded checkpoint from {load_checkpoint_path}, starting from iteration {starting_iteration}")
    start_time = time.time()
    trainer = Trainer(
        model_with_loss=model,
        starting_iteration=starting_iteration,
        data_loader=train_data_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_client=checkpoint_client,
        checkpoint_interval=checkpoint_interval,
        device=device,
        wandb=wandb,
        max_l2_norm=optimizer_config.max_l2_norm,
    )
    trainer.train()
    if wandb is not None:
        wandb.summary["training_duration_min"] = (time.time() - start_time) / 60.0
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
                 max_l2_norm: float = 1.0,
                 device: str = None,
                 wandb: wandb.Run | None = None):
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
            wandb (wandb.Run | None): Weights & Biases run for logging, or None to disable logging.
            device (str): Device to use for training (e.g., "cpu", "cuda", "mps").
            max_l2_norm (float): Maximum L2 norm for gradient clipping.
        """
        self.model_with_loss = model_with_loss
        self.starting_iteration = starting_iteration
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_client = checkpoint_client
        self.checkpoint_interval = checkpoint_interval
        self.device = device if device is not None else torch.device("cpu")
        self.wandb = wandb
        self.max_l2_norm = max_l2_norm

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
            grad_clipping(self.model_with_loss.parameters(), max_l2_norm=self.max_l2_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            if t % 10 == 0:
                print(f"Step {t}, Loss: {loss.item():.4f}")
                if self.wandb is not None:
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    self.wandb.log({"train/loss": loss.item(), "train/lr": current_lr}, step=t)
            if t % self.checkpoint_interval == 0 and t > self.starting_iteration:
                self.checkpoint_client.save(t)
                print(f"Checkpoint saved at step {t}")
            t += 1
        # Save final checkpoint
        self.checkpoint_client.save(t)
        print(f"Final checkpoint saved at step {t}")

