import torch
import argparse
import os

from cs336_basics.trainer import Trainer
from cs336_basics.module.transformer import Transformer, TransformerConfig
from cs336_basics.optimizer import create_from_config, OptimizerConfig
from cs336_basics.data_loader import DataLoader, DataLoaderConfig, DataLoadingMode
from cs336_basics.checkpoint import CheckpointClient
from cs336_basics.bpe_utils import load_artifact
from cs336_basics.evaluator import Evaluator


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
        model = torch.compile(model, backend="aot_eager")
    else:
        model = torch.compile(model)
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
        model=model, 
        starting_iteration=starting_iteration,
        data_loader=train_data_loader,
        optimizer=optimizer, 
        scheduler=scheduler, 
        checkpoint_client=checkpoint_client,
        checkpoint_interval=checkpoint_interval,
        device=device)
    trainer.train()
    return model


def run_eval(
        model: torch.nn.Module,
        eval_data_path: str,
        context_length: int,
        eval_batch_size: int,
        device: str):
    """Run evaluation on a trained Transformer model.

    Args:
        model (torch.nn.Module): The trained Transformer model.
        eval_data_path (str): Path to the evaluation dataset (numpy file).
        context_length (int): Context length for evaluation.
        device (str): Device to use for evaluation (e.g., "cpu", "cuda", "mps").
        eval_batch_size (int): Batch size for evaluation.
    """
    eval_data_loader = DataLoader.from_config(DataLoaderConfig(
        dataset_path=eval_data_path,
        num_batches=None,
        batch_size=eval_batch_size,
        context_length=context_length,
        data_loading_mode=DataLoadingMode.SEQUENTIAL,
    ), device=device)
    evaluator = Evaluator(model=model, eval_data_loader=eval_data_loader)
    avg_loss = evaluator.avg_loss()
    print(f"Avg Evaluation Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Transformer model")
    parser.add_argument("--train_data", required=True, help="Path to the training dataset")
    parser.add_argument("--vocab", required=True, type=str, help="Path to the tokenizer vocab file")
    parser.add_argument("--context_length", type=int, default=256, help="Context length for training")
    parser.add_argument("--device", default="cpu", help="Device to use for training (cpu, cuda, mps)")
    parser.add_argument("--num_batches", type=int, default=2000, help="Number of batches to train on")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training (and evaluation)")
    parser.add_argument("--checkpoint_dir", type=str, help="Directory to save checkpoints to (default to {input_dir}/checkpoints)")
    parser.add_argument("--checkpoint_interval", type=int, default=1000, help="Checkpoint interval")
    parser.add_argument("--load_checkpoint", help="Path to a checkpoint to load before training")
    parser.add_argument("--eval_data", help="Path to the evaluation dataset")
    args = parser.parse_args()

    if not args.checkpoint_dir:
        args.checkpoint_dir = os.path.join(os.path.dirname(args.train_data), "checkpoints")
    vocab_size = len(load_artifact(args.vocab))
    model = Transformer.from_config(TransformerConfig(
        vocab_size=vocab_size, 
        context_length=args.context_length))
    print(f"Starting training with dataset {args.train_data}, vocab_size {vocab_size}, device {args.device}, num_batches {args.num_batches}, checkpoint_dir {args.checkpoint_dir}, checkpoint_interval {args.checkpoint_interval}")

    trained_model = run_training(
        model=model,
        dataset_path=args.train_data, 
        num_batches=args.num_batches, 
        batch_size=args.batch_size,
        conxtext_length=args.context_length,
        device=args.device, 
        load_checkpoint_path=args.load_checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval
    )
    
    if args.eval_data:
        print(f"Starting evaluation with dataset {args.eval_data}, and eval batch_size {args.batch_size}")
        run_eval(
            model=trained_model,
            eval_data_path=args.eval_data,
            context_length=args.context_length,
            eval_batch_size=args.batch_size,
            device=args.device
        )