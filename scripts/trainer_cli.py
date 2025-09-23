import torch
import argparse
import os
import wandb

from dataclasses import asdict

from cs336_basics.data_loader import DataLoaderConfig, DataLoadingMode
from cs336_basics.optimizer import OptimizerConfig
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.module.transformer import Transformer, TransformerConfig
from cs336_basics.module.model_wrapper import ModelWrapperWithCELoss
from cs336_basics.token_validation import validate_special_tokens
from cs336_basics.trainer import run_training
from cs336_basics.evaluator import run_eval
from cs336_basics.resource_accounting import print_resource_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Transformer model")
    parser.add_argument("--train_data", required=True, help="Path to the training dataset")
    parser.add_argument("--tokenizer_dir", required=True, type=str, help="Path to the tokenizer artifact directory containing vocab.pkl, merges.pkl, and special_tokens.pkl")
    parser.add_argument("--context_length", type=int, default=256, help="Context length for training")
    parser.add_argument("--device", default="cpu", help="Device to use for training (cpu, cuda, mps)")
    parser.add_argument("--iterations", type=int, default=2000, help="Number of iterations to train on")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training (and evaluation)")
    parser.add_argument("--checkpoint_dir", type=str, help="Directory to save checkpoints to (default to {input_dir}/checkpoints)")
    parser.add_argument("--checkpoint_interval", type=int, default=1000, help="Checkpoint interval")
    parser.add_argument("--load_checkpoint", help="Path to a checkpoint to load before training")
    parser.add_argument("--eval_data", help="Path to the evaluation dataset")
    parser.add_argument("--log_to_wandb", action="store_true", help="Whether to log training and evaluation metrics to Weights & Biases")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="The peak learning rate for the optimizer (default: 1e-3)")
    args = parser.parse_args()

    validate_special_tokens(
        tokens_file=args.train_data,
        tokenizer_artifact_dir=args.tokenizer_dir
    )
    if not args.checkpoint_dir:
        args.checkpoint_dir = os.path.join(os.path.dirname(args.train_data), "checkpoints")

    vocab_size=Tokenizer.load_from_directory(args.tokenizer_dir).vocab_size
    model_config = TransformerConfig(
        vocab_size=vocab_size,
        context_length=args.context_length
    )
    model = ModelWrapperWithCELoss(
        Transformer.from_config(model_config)
    )
    train_dataloader_config = DataLoaderConfig(
        dataset_path=args.train_data,
        num_batches=args.iterations,
        batch_size=args.batch_size,
        context_length=args.context_length,
        data_loading_mode=DataLoadingMode.RANDOM
    )
    optimizer_config=OptimizerConfig(
        total_iters=args.iterations,
        learning_rate=args.learning_rate,
    )

    print_resource_summary(model_config)

    print(f"Starting training with dataset {args.train_data}, vocab_size {vocab_size}, device {args.device}, iterations {args.iterations}, checkpoint_dir {args.checkpoint_dir}, checkpoint_interval {args.checkpoint_interval}")

    training_dir = os.path.basename(os.path.dirname(args.train_data))
    wandb_run = None
    if args.log_to_wandb:
        wandb_run = wandb.init(
            project=f"llm-{training_dir}",
            config={
                "train_data": args.train_data,
                "tokenizer_dir": args.tokenizer_dir,
                "iterations": args.iterations,
                "epochs": 1,  # Currently we only support 1 epoch training.
                "batch_size": args.batch_size,
                "vocab_size": vocab_size,
                "model_config": asdict(model_config),
                "data_loader_config": asdict(train_dataloader_config),
                "optimizer_config": asdict(optimizer_config),
                "device": args.device,
                "eval_data": args.eval_data,
                "checkpoint_interval": args.checkpoint_interval,
                "load_checkpoint_from": args.load_checkpoint,
            }
        )

    trained_model = run_training(
        data_loader_config=train_dataloader_config,
        optimizer_config=optimizer_config,
        model=model,
        load_checkpoint_path=args.load_checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        device=args.device,
        wandb=wandb_run,
    )

    if args.eval_data:
        print(f"Starting evaluation with dataset {args.eval_data}, and eval batch_size {args.batch_size}")
        validate_special_tokens(
            tokens_file=args.eval_data,
            tokenizer_artifact_dir=args.tokenizer_dir
        )
        run_eval(
            model=trained_model,
            eval_data_path=args.eval_data,
            context_length=args.context_length,
            eval_batch_size=args.batch_size,
            device=args.device,
            wandb=wandb_run,
        )

    if wandb_run:
        wandb_run.finish()