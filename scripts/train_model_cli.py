import datetime
import os.path
import argparse
import torch
# from tokenizers import Tokenizer
from cs336_basics.model import LinearModule,EmbeddingModule, RMSNormModule, SwiGLUModule, RoPE, AdamW
from cs336_basics.model import scaled_dot_product_attention, \
softmax, \
multihead_self_attention, \
multihead_self_attention_with_rope, \
transformer_block, \
transformer_lm, \
cross_entropy, \
learning_rate_schedule, \
gradient_clipping 
from cs336_basics.dataloader import get_batch
from cs336_basics.checkpoint import save_checkpoint, load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Transformer language model.")

    # data and checkpointing
    parser.add_argument('--data', type=str, default=f"data/{str(datetime.datetime.now()).replace(':','_')}", help='Path to the training data file.')

    # data hyperparameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--context_length', type=int, default=128, help='Context length for the model.')
    parser.add_argument('--vocab_size', type=int, default=50257, help='Vocabulary size.')

    # model hyperparameters
    parser.add_argument('--n_layers', type=int, default=6, help='Number of transformer layers.')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads.')
    parser.add_argument('--d_model', type=int, default=512, help='Dimension of model embeddings.')
    parser.add_argument('--d_ff', type=int, default=2048, help='Dimension of feedforward network.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate.')

    # optimization hyperparameters
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate.')
    parser.add_argument('--lr_schedule_max_lr', type=float, default=1e-3, help='Maximum learning rate for learning rate schedule.')
    parser.add_argument('--lr_schedule_min_lr', type=float, default=1e-5, help='Minimum learning rate for learning rate schedule.')
    parser.add_argument('--lr_schedule_warmup_iters', type=int, default=1000, help='Number of warmup iterations for learning rate schedule.')
    parser.add_argument('--lr_schedule_total_iters', type=int, default=10000, help='Total number of iterations for learning rate schedule.')

    # other hyperparameters
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Maximum gradient norm for clipping.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility.')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for training (cpu or cuda).')

    # training control
    parser.add_argument('--max_iters', type=int, default=10000, help='Maximum number of training iterations.')
    parser.add_argument('--eval_every', type=int, default=500, help='Evaluate model every N iterations.')
    parser.add_argument('--save_every', type=int, default=1000, help='Save checkpoint every N iterations.')
    parser.add_argument('--log_every', type=int, default=100, help='Log training metrics every N iterations.')
    parser.add_argument('--resume', type=str, default=None, help='Path to a checkpoint to resume training from.')

    return parser.parse_args()

def main():
    # Parse arguments (e.g., using argparse)
    args=parse_args()

    # create directory for saving checkpoints if it doesn't exist
    dir_data=os.path.dirname(args.data)
    dir_checkpoints=os.path.join(dir_data, 'checkpoints')
    if not os.path.exists(dir_checkpoints):
        os.makedirs(dir_checkpoints)

    # Set random seed for reproducibility
    if args.seed!=0:
        torch.manual_seed(args.seed)
        if args.device.startswith('cuda'):
            torch.cuda.manual_seed_all(args.seed)

    # Create Model
    # TODO: update model parameters to match the model
    model=transformer_lm(vocab_size=args.vocab_size,  # Example vocab size
                         context_length=args.context_length,
                         n_layers=args.n_layers,
                         n_heads=args.n_heads,
                         d_model=args.d_model,
                         d_ff=args.d_ff,
                         dropout=args.dropout,
                         device=args.device)
    model.to(args.device)
    model=torch.compile(model)  # Optional: Compile the model for performance

    # Create optimizer
    optimizer=AdamW(model.parameters(), lr=args.learning_rate)
    
    # Load Checkpoint?
    if args.resume is not None:
        file_checkpoints_resume=os.path.join(dir_checkpoints, args.resume)
        # Load model and optimizer state from checkpoint
        load_checkpoint(src=file_checkpoints_resume, model=model, optimizer=optimizer)
        
    # Prepare Data
    file_data=os.path.join(dir_data, "sample.txt")
    with open(file_data, 'r', encoding='utf-8') as f:
        text = f.read()
    
    #tokenize using BPE tokenizer from assignment 1
    #TODO: need further updates
    tokenizer = Tokenizer.from_file("tokenizer.json")  # Adjust path as necessary
    tokens = tokenizer.encode(text).ids
    dataset = tokens  # Use token IDs as dataset
    
    # set model to training mode
    model.train()
    # Model Training Loop
    iter=0
    while iter<=args.max_iters:
        if iter%args.log_every==0:
            # Log training metrics
            pass
        if iter%args.eval_every==0:
            # Evaluate model on validation set
            pass
        if iter%args.save_every==0:
            # Save model checkpoint
            file_checkpoints=os.path.join(dir_checkpoints, f'context{args.context_length}_layers{args.n_layers}_heads{args.n_heads}_dmodel{args.d_model}_dff{args.d_ff}_dropout{args.dropout}_batch{args.batch_size}_lr{args.learning_rate}_{str(datetime.datetime.now()).replace(":","_")}.bin')
            save_checkpoint(model=model, optimizer=optimizer, iteration=iter, out=file_checkpoints)
        # Get batch of data
        x, y = get_batch(dataset=dataset, context_length=args.context_length, batch_size=args.batch_size, device=args.device)
        # Forward pass
        logits = model(x)
        loss = cross_entropy(logits, y)
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        # Update learning rate if using a scheduler
        learning_rate_schedule(optimizer=optimizer, 
                               iteration=iter,
                               min_learning_rate=args.lr_schedule_min_lr,
                               max_learning_rate=args.lr_schedule_max_lr,
                               warmup_iters=args.lr_schedule_warmup_iters,
                               total_iters=args.lr_schedule_total_iters)   
        # Gradient clipping
        gradient_clipping(model.parameters(), max_l2_norm=args.max_grad_norm)

        # update model weights
        optimizer.step()

        # Increment iteration counter 
        iter+=1
    
if __name__ == "__main__":
    main()
    