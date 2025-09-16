#!/usr/bin/env python3
"""
CLI tool for text generation using a trained Transformer model with BPE tokenizer.

Usage:
    python decoder_cli.py --tokenizer_artifact_dir <tokenizer_dir> --model <model_path> --context_length <length> --device <device> --temperature <temp>
"""

import torch
import argparse
import sys

from cs336_basics.decoder import Decoder, ModelWrapperWithDecoder
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.module.transformer import Transformer, TransformerConfig
from cs336_basics.checkpoint import CheckpointClient
from cs336_basics.module.model_wrapper import ModelWrapperWithCELoss


def compile_model(model: torch.nn.Module, device: str) -> torch.nn.Module:
    """Compile the model using torch.compile with appropriate backend."""
    if device == "mps":
        return torch.compile(model, backend="aot_eager", fullgraph=True, dynamic=True)
    else:
        return torch.compile(model, fullgraph=True, dynamic=True)


def load_model_from_checkpoint(model_path: str,
                               vocab_size: int,
                               context_length: int,
                               temperature: float,
                               nucleus_sampling_p: float,
                               device: str) -> torch.nn.Module:
    """Load a trained model from checkpoint, handling torch.compile prefixes."""
    model_wrapper = ModelWrapperWithCELoss(Transformer.from_config(TransformerConfig(
        vocab_size=vocab_size,
        context_length=context_length
    )))
    model_wrapper.to(device)
    # HACK: this torch.compile is needed to match the state dict keys saved during training.
    model_wrapper = compile_model(model_wrapper, device)
    checkpoint = torch.load(model_path)
    model_wrapper.load_state_dict(checkpoint["model_state_dict"])
    # Extracts the underlying model that computes logits instead of loss for decoding.
    model = model_wrapper.model
    model_with_decoder = ModelWrapperWithDecoder(
        model=model, temperature=temperature, nucleus_sampling_p=nucleus_sampling_p, device=device
    )
    model_with_decoder.eval()
    # Compiles the inner model that generates one new token given a prompt.
    model_with_decoder = compile_model(model_with_decoder, device)
    return model_with_decoder


def tokenize_prompt(tokenizer: Tokenizer, prompt: str, context_length: int, device: str) -> torch.Tensor:
    """Tokenize the input prompt and move to the specified device."""
    prompt_tokens = tokenizer.encode(prompt)  # List[int]
    if len(prompt_tokens) > context_length:
        prompt_tokens = prompt_tokens[-context_length:]  # Truncate from the left
    prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.int64, device=device)  # (seq_len,)
    return prompt_tensor


def get_eos_token(tokenizer: Tokenizer) -> int:
    """Get the token ID for the end-of-text token."""
    # Look up the end-of-text token directly in the vocab (it's stored as bytes)
    end_token_bytes = b'<|endoftext|>'
    if end_token_bytes not in tokenizer._vocab:
        raise ValueError(f"End-of-text token {end_token_bytes} not found in vocabulary.")
    return tokenizer._vocab[end_token_bytes]


def generate_from_prompt(decoder: torch.nn.Module, prompt: str, tokenizer: Tokenizer, eos_token: int, args: argparse.Namespace) -> str:
    """Generate text from a given prompt using the decoder and tokenizer."""
    if not prompt:
        raise ValueError("Prompt cannot be empty for generation.")

    prompt_tokens = tokenize_prompt(tokenizer, prompt, args.context_length, args.device)  # (seq_len,)
    decoded_tokens = decoder(prompt_tokens).tolist()
    if eos_token in decoded_tokens:
        decoded_tokens = decoded_tokens[:decoded_tokens.index(eos_token)]
    return tokenizer.decode(decoded_tokens)


def main():
    parser = argparse.ArgumentParser(
        description="Generate text using a trained Transformer model"
    )
    parser.add_argument("--tokenizer_dir", required=True,
                        help="Directory containing tokenizer artifacts (vocab.pkl and merges.pkl)")
    parser.add_argument("--model", required=True,
                        help="Path to the trained model checkpoint")
    # TODO: this should be saved and loaded from the model checkpoint
    parser.add_argument("--context_length", type=int, required=True,
                        help="Context length of the model")
    parser.add_argument("--device", default="cpu",
                        help="Device to use for generation (cpu, cuda, mps)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (default: 1.0)")
    parser.add_argument("--nucleus_sampling_p", type=float, default=0.9,
                        help="Nucleus sampling probability (default: 0.9)")
    parser.add_argument("--max_new_tokens", type=int, default=100,
                        help="Maximum number of new tokens to generate (default: 100)")
    parser.add_argument("--prompt", type=str, default="Once upon a time",
                        help="Initial prompt for text generation")
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = Tokenizer.load_from_directory(args.tokenizer_dir)

    # Load model
    print(f"Loading model from {args.model}")
    model = load_model_from_checkpoint(
        model_path=args.model,
        vocab_size=tokenizer.vocab_size,
        context_length=args.context_length,
        temperature=args.temperature,
        nucleus_sampling_p=args.nucleus_sampling_p,
        device=args.device
    )

    eos_token = get_eos_token(tokenizer)

    # Create decoder
    decoder_model = Decoder(
        model_with_decoder=model,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        eos_token=eos_token,
    )
    # TODO: Investigate why compiling the whole decoder model makes overall
    # text generation slower.
    #decoder_model = compile_model(decoder_model, args.device)

    print(f"Decoder ready! Using device: {args.device}, temperature: {args.temperature}")

    # Prime the model on the initial prompt.
    if args.prompt:
        print(f"Generating for initial prompt: {args.prompt}")
        generated_text = generate_from_prompt(decoder_model, args.prompt, tokenizer, eos_token, args)
        print(f"\nGenerated text:\n{generated_text}")
        print("-" * 50)

    print("Enter prompts to generate text (Ctrl+C to exit):")
    print("-" * 50)

    while True:
        prompt = input("\nPrompt: ")
        if not prompt.strip():
            continue

        print("Generating...")
        generated_text = generate_from_prompt(decoder_model, prompt, tokenizer, eos_token, args)
        print(f"\nGenerated text:\n{generated_text}")
        print("-" * 50)


if __name__ == "__main__":
    main()