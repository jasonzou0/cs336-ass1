import os
import torch
import argparse
from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a trained Transformer Language Model")

    parser.add_argument(
        "--data", 
        type=str, 
        default="data/TinyStoriesV2/",
        help="Path to the directory containing the training data (sample.txt)"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="output/inference_model.bin",
        help="Path to the trained model checkpoint file"    
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable interactive mode for text generation"
    )
    parser.add_argument(
        "--display_token_ids",
        action="store_true",
        help="Display token IDs instead of decoded text"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Arguments: {args}")

    # Load the model from the checkpoint
    print(f"Loading model from {args.model}")
    checkpoint = torch.load(args.model)
    model_args = checkpoint['model_args']
    model = TransformerLM(
        vocab_size=model_args['vocab_size'],
        context_length=model_args['context_length'],
        d_model=model_args['d_model'],
        num_layers=model_args['num_layers'],
        num_heads=model_args['num_heads'],
        d_ff=model_args['d_ff'],
        rope_theta=model_args['rope_theta'],
        device=torch.device('cpu'),  # Always load on CPU first
        dtype=torch.float32
    )

    model.load_state_dict(checkpoint['model_state_dict'])

    # Print out summary
    print(f"Model architecture:")
    for k,v in model_args.items():
        print(f"  {k}: {v}")
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Model parameters:{[name for name, param in model.named_parameters()]}")

    # Initialize tokenizer from the data directory
    tokenizer_dir = os.path.join(args.data,'tokenizer_data/')
    if not os.path.exists(tokenizer_dir):
        raise ValueError(f"Tokenizer directory not found at {tokenizer_dir}")
    
    # Load vocabulary and merges files
    # tokenizer = Tokenizer.load_from_directory(tokenizer_dir,use_cython=False)
    tokenizer = Tokenizer.load_from_directory(tokenizer_dir)

    # Set the model to evaluation mode
    model.eval()

    if args.interactive:
        while True:
            try:
                # Get input from user
                prompt = input("\nEnter prompt (Ctrl+C to exit): ")
                if not prompt:
                    continue

                # Encode input using tokenizer
                token_ids = tokenizer.encode(prompt)
                context = torch.tensor([token_ids], dtype=torch.long)  # Add batch dimension

                # Generate tokens
                with torch.no_grad():
                    for _ in range(100):  # Generate up to 100 tokens
                        # Get model predictions
                        logits = model(context[:, -model_args['context_length']:])
                        probs = torch.softmax(logits[:, -1, :], dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)

                        # Display output
                        if args.display_token_ids:
                            print(next_token.item(), end=' ')
                        else:
                            # Convert tensor to list of ints for decoder
                            token_list = [next_token.item()]
                            print(tokenizer.decode(token_list), end='', flush=True)

                        # Update context
                        context = torch.cat([context, next_token], dim=1)

            except KeyboardInterrupt:
                print("\nExiting interactive mode...")
                break
            except Exception as e:
                print(f"\nError: {e}")
                continue
    else:
        # Use a default prompt
        prompt = "Once upon a time"
        token_ids = tokenizer.encode(prompt)
        context = torch.tensor([token_ids], dtype=torch.long)  # Add batch dimension

        print(f"\nGenerating text from prompt: '{prompt}'")
        
        # Generate tokens
        with torch.no_grad():
            for _ in range(500):  # Generate 500 tokens
                logits = model(context[:, -model_args['context_length']:])
                probs = torch.softmax(logits[:, -1, :], dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                if args.display_token_ids:
                    print(next_token.item(), end=' ')
                else:
                    # Convert tensor to list of ints for decoder
                    token_list = [next_token.item()]
                    print(tokenizer.decode(token_list), end='', flush=True)

                context = torch.cat([context, next_token], dim=1)

        print("\nDone.")

if __name__ == "__main__":
    main()
