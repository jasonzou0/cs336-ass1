import argparse
import torch

from cs336_basics.module.transformer import Transformer, TransformerConfig
from cs336_basics.optimizer import CosineScheduler, create_from_config, OptimizerConfig
from cs336_basics.data_loader import DataLoader, DataLoaderConfig
from cs336_basics.module.loss import cross_entropy_loss

class Trainer:
    def __init__(self, 
                 model: torch.nn.Module, 
                 data_loader: DataLoader,
                 optimizer: torch.optim.Optimizer, 
                 scheduler: CosineScheduler, 
                 device=None):
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device if device is not None else torch.device("cpu")
    
    def train(self):
        """
        Train the model for a specified number of steps.
        Args:
            get_batch_func (callable): Function to get a batch of input and target tensors.
            num_steps (int): Number of training steps to perform.
        """
        for t, (input_ids, target_ids) in enumerate(iter(self.data_loader)):
            # Forward pass
            logits = self.model(input_ids)
            loss = cross_entropy_loss(logits, target_ids)
            # Backward pass and optimization step
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            if t % 10 == 0:
                print(f"Step {t}, Loss: {loss.item():.4f}")
    

def run_training(dataset_path: str, num_batches: int, device: str):
    model = Transformer.from_config(TransformerConfig())
    model.to(device)
    # TODO: inspect the compiled code. 
    if device == "mps":
        model = torch.compile(model, backend="aot_eager")
    else:
        model = torch.compile(model)
    data_loader = DataLoader.from_config(DataLoaderConfig(dataset_path=dataset_path, num_batches=num_batches), device=device)
    optimizer, scheduler = create_from_config(model.parameters(), config=OptimizerConfig(), cosine_cycle_iters=num_batches)
    trainer = Trainer(model=model, data_loader=data_loader, optimizer=optimizer, scheduler=scheduler, device=device)
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Transformer model")
    parser.add_argument("--input", required=True, help="Path to the training dataset")
    parser.add_argument("--device", default="cpu", help="Device to use for training (cpu, cuda, mps)")
    parser.add_argument("--num-batches", type=int, default=2000, help="Number of training batches")
    args = parser.parse_args()
    run_training(dataset_path=args.input, num_batches=args.num_batches, device=args.device)
