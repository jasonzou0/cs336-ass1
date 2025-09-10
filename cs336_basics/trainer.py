import argparse
import torch
import os

from cs336_basics.module.transformer import Transformer, TransformerConfig
from cs336_basics.optimizer import CosineScheduler, create_from_config, OptimizerConfig
from cs336_basics.data_loader import DataLoader, DataLoaderConfig
from cs336_basics.module.loss import cross_entropy_loss
from cs336_basics.checkpoint import CheckpointClient
from cs336_basics.grad_clipping import grad_clipping

class Trainer:
    def __init__(self, 
                 model: torch.nn.Module, 
                 data_loader: DataLoader,
                 optimizer: torch.optim.Optimizer, 
                 scheduler: CosineScheduler, 
                 checkpoint_client: CheckpointClient,
                 checkpoint_interval: int,
                 device: str = None):
        self.model = model
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
        t = 0
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
            if t % self.checkpoint_interval == 0 and t > 0:
                self.checkpoint_client.save(t)
                print(f"Checkpoint saved at step {t}")
            t += 1
        # Save final checkpoint
        self.checkpoint_client.save(t)
        print(f"Final checkpoint saved at step {t}")
    

def run_training(
        dataset_path: str, 
        num_batches: int, 
        checkpoint_interval: int,
        checkpoint_dir: str,
        device: str):
    model = Transformer.from_config(TransformerConfig())
    model.to(device)
    # TODO: inspect the compiled code. 
    if device == "mps":
        model = torch.compile(model, backend="aot_eager")
    else:
        model = torch.compile(model)
    data_loader = DataLoader.from_config(DataLoaderConfig(dataset_path=dataset_path, num_batches=num_batches), device=device)
    optimizer, scheduler = create_from_config(model.parameters(), config=OptimizerConfig(total_iters=num_batches))
    checkpoint_client = CheckpointClient(
        model=model, 
        optimizer=optimizer, 
        checkpoint_dest=checkpoint_dir)
    trainer = Trainer(
        model=model, 
        data_loader=data_loader, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        checkpoint_client=checkpoint_client,
        checkpoint_interval=checkpoint_interval,
        device=device)
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Transformer model")
    parser.add_argument("--input", required=True, help="Path to the training dataset")
    parser.add_argument("--device", default="cpu", help="Device to use for training (cpu, cuda, mps)")
    parser.add_argument("--num-batches", type=int, default=2000, help="Number of training batches")
    parser.add_argument("--checkpoint-dir", type=str, help="Directory to save checkpoints to (default to {input_dir}/checkpoints)")
    parser.add_argument("--checkpoint-interval", type=int, default=1000, help="Checkpoint interval")
    args = parser.parse_args()

    if not args.checkpoint_dir:
        args.checkpoint_dir = os.path.join(os.path.dirname(args.input), "checkpoints")
    print(f"Starting training with dataset {args.input}, device {args.device}, num_batches {args.num_batches}, checkpoint_dir {args.checkpoint_dir}, checkpoint_interval {args.checkpoint_interval}")
    run_training(
        dataset_path=args.input, 
        num_batches=args.num_batches, 
        device=args.device, 
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval)
