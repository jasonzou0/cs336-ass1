import torch
import argparse
import os

from cs336_basics.trainer import Trainer
from cs336_basics.module.transformer import Transformer, TransformerConfig
from cs336_basics.optimizer import create_from_config, OptimizerConfig
from cs336_basics.data_loader import DataLoader, DataLoaderConfig
from cs336_basics.checkpoint import CheckpointClient
from cs336_basics.bpe_utils import load_artifact
from cs336_basics.evaluator import Evaluator


def run_training(
        dataset_path: str, 
        vocab_size: int,
        num_batches: int, 
        checkpoint_interval: int,
        checkpoint_dir: str,
        device: str) -> torch.nn.Module:
    """Run the training loop for a Transformer model.

    Args:
        dataset_path (str): Path to the training dataset (numpy file).
        vocab_size (int): Size of the vocabulary.
        num_batches (int): Number of training batches.
        checkpoint_interval (int): Interval (in steps) to save checkpoints.
        checkpoint_dir (str): Directory to save checkpoints.
        device (str): Device to use for training (e.g., "cpu", "cuda", "mps").
    Returns:
        model (nn.Module): The trained Transformer model.
    """
    model = Transformer.from_config(TransformerConfig(vocab_size=vocab_size))
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
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Transformer model")
    parser.add_argument("--train_data", required=True, help="Path to the training dataset")
    parser.add_argument("--vocab", required=True, type=str, help="Path to the tokenizer vocab file")
    parser.add_argument("--device", default="cpu", help="Device to use for training (cpu, cuda, mps)")
    parser.add_argument("--num_batches", type=int, default=2000, help="Number of training batches")
    parser.add_argument("--checkpoint_dir", type=str, help="Directory to save checkpoints to (default to {input_dir}/checkpoints)")
    parser.add_argument("--checkpoint_interval", type=int, default=1000, help="Checkpoint interval")
    parser.add_argument("--eval_data", help="Path to the evaluation dataset")
    args = parser.parse_args()

    if not args.checkpoint_dir:
        args.checkpoint_dir = os.path.join(os.path.dirname(args.train_data), "checkpoints")
    vocab_size = len(load_artifact(args.vocab))
    print(f"Starting training with dataset {args.train_data}, vocab_size {vocab_size}, device {args.device}, num_batches {args.num_batches}, checkpoint_dir {args.checkpoint_dir}, checkpoint_interval {args.checkpoint_interval}")
    trained_model = run_training(
        dataset_path=args.train_data, 
        vocab_size=vocab_size,
        num_batches=args.num_batches, 
        device=args.device, 
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval)
    
    if args.eval_data:
        eval_data_loader = DataLoader.from_config(DataLoaderConfig(dataset_path=args.eval_data), device=args.device)
        evaluator = Evaluator(model=trained_model, eval_data_loader=eval_data_loader, device=args.device)
        avg_loss = evaluator.avg_loss()
        print(f"Avg Evaluation Loss: {avg_loss:.4f}")    
    