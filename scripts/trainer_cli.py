import torch
import argparse
import os

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
    parser.add_argument("--num_batches", type=int, default=2000, help="Number of batches to train on")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training (and evaluation)")
    parser.add_argument("--checkpoint_dir", type=str, help="Directory to save checkpoints to (default to {input_dir}/checkpoints)")
    parser.add_argument("--checkpoint_interval", type=int, default=1000, help="Checkpoint interval")
    parser.add_argument("--load_checkpoint", help="Path to a checkpoint to load before training")
    parser.add_argument("--eval_data", help="Path to the evaluation dataset")
    args = parser.parse_args()

    validate_special_tokens(
        tokens_file=args.train_data,
        tokenizer_artifact_dir=args.tokenizer_dir
    )
    if not args.checkpoint_dir:
        args.checkpoint_dir = os.path.join(os.path.dirname(args.train_data), "checkpoints")
    
    vocab_size=Tokenizer.load_from_directory(args.tokenizer_dir).vocab_size
    config = TransformerConfig(
        vocab_size=vocab_size,
        context_length=args.context_length
    )
    model = ModelWrapperWithCELoss(
        Transformer.from_config(config)
    )
    
    print_resource_summary(config)
    
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
        validate_special_tokens(
            tokens_file=args.eval_data,
            tokenizer_artifact_dir=args.tokenizer_dir
        )
        run_eval(
            model=trained_model,
            eval_data_path=args.eval_data,
            context_length=args.context_length,
            eval_batch_size=args.batch_size,
            device=args.device
        )