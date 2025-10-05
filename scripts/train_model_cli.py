import pickle
import datetime
import os.path
import argparse
import torch
import torch.nn as nn
# from tokenizers import Tokenizer
from cs336_basics.model import LinearModule,EmbeddingModule, RMSNormModule, SwiGLUModule, RoPE, AdamW, TransformerLM
from cs336_basics.model import \
cross_entropy, \
learning_rate_schedule, \
gradient_clipping 
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.dataloader import get_batch
from cs336_basics.checkpoint import save_checkpoint, load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Transformer language model.")

    # data directory and training/eval files
    parser.add_argument('--data', type=str, default="data/TinyStoriesV2/", help='Path to the training data file.')
    parser.add_argument('--training_data', type=str, default='TinyStoriesV2-GPT4-train.txt', help='Training data file name inside the data directory.')
    parser.add_argument('--eval_data', type=str, default='TinyStoriesV2-GPT4-valid.txt', help='Evaluation data file name inside the data directory.')

    # tokenizer directory
    parser.add_argument('--tokenizer_data', type=str, default='tokenizer_data/', help='Subdir within data dir that contains containing tokenizer data: vocab.pkl, merges.pkl, special_tokens.pkl and tokenized_data.pkl')

    # output directory
    parser.add_argument('--output', type=str, default=f"output/{str(datetime.datetime.now()).replace(':','_')}/", help='Directory to save checkpoints.')

    # data hyperparameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--context_length', type=int, default=128, help='Context length for the model.')
    parser.add_argument('--vocab_size', type=int, default=50257, help='Vocabulary size.')

    # model hyperparameters
    parser.add_argument('--n_layers', type=int, default=6, help='Number of transformer layers.')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads.')
    parser.add_argument('--d_model', type=int, default=512, help='Dimension of model embeddings.')
    parser.add_argument('--d_ff', type=int, default=2048, help='Dimension of feedforward network.')

    # optimization hyperparameters
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate.')
    parser.add_argument('--lr_schedule_max_lr', type=float, default=1e-3, help='Maximum learning rate for learning rate schedule.')
    parser.add_argument('--lr_schedule_min_lr', type=float, default=1e-5, help='Minimum learning rate for learning rate schedule.')
    parser.add_argument('--lr_schedule_warmup_iters', type=int, default=150, help='Number of warmup iterations for learning rate schedule.')
    parser.add_argument('--lr_schedule_total_iters', type=int, default=1000, help='Total number of iterations for learning rate schedule.')

    # other hyperparameters
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Maximum gradient norm for clipping.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility.')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for training (cpu or cuda).')
    parser.add_argument('--dtype', type=str, default='float32', help='Data type for model parameters (float32, float16, bfloat16).')

    # training control
    parser.add_argument('--max_iters', type=int, default=3000, help='Maximum number of training iterations.')
    parser.add_argument('--log_every', type=int, default=5, help='Log training metrics every N iterations.')
    parser.add_argument('--min_loss', type=float, default=0.8, help='Minimum loss value to stop the training loop.')  

    parser.add_argument('--eval_every', type=int, default=50, help='Evaluate model every N iterations.')
    parser.add_argument('--eval_iters', type=int, default=5, help='Evaluate over this many batches during each evaluation.')

    # checkpoint save/load
    parser.add_argument('--save_every', type=int, default=50, help='Save checkpoint every N iterations.')
    parser.add_argument('--ckpt', type=str, default=None, help='Path to a checkpoint to resume training from.')

    return parser.parse_args()

def load_data_and_tokenize(data_file: str, tokenizer: Tokenizer, tokenized_data_file: str) -> list[int]:
    if os.path.exists(tokenized_data_file):
        # tokenized data already exists
        print(f"Tokenized data found in {tokenized_data_file}")
        dataset = pickle.load(open(tokenized_data_file, 'rb'))
        print(f"Loaded tokenized data with {len(dataset)} tokens")
        return dataset
    else:
        # tokenized data does not exist
        # Tokenize data and save tokenized version for future use
        print(f"Tokenized data not found at {tokenized_data_file}, tokenizing from raw data.")
        with open(data_file, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"Loaded training data from {data_file} with {len(text)} characters.")
    
        #TODO: write my own tokenizer to understand the performance tuning better
        tokens = tokenizer.encode(text)
        dataset = tokens  # Use token IDs as dataset
        #dump tokenized data for next time usage
        pickle.dump(dataset, open(tokenized_data_file, 'wb'))
        print(f"Saved tokenized data to {tokenized_data_file}")

def main():
    # Parse arguments (e.g., using argparse)
    args=parse_args()

    # create directory for saving checkpoints if it doesn't exist
    dir_data=os.path.dirname(args.data)
    dir_output=os.path.dirname(args.output)
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)

    # Set random seed for reproducibility
    if args.seed!=0:
        torch.manual_seed(args.seed)
        if args.device.startswith('cuda'):
            torch.cuda.manual_seed_all(args.seed)
    
    # Output hyperparameters
    print("Training configuration:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    # Create Model
    model=TransformerLM(vocab_size=args.vocab_size,  # Example vocab size
                        context_length=args.context_length,
                        num_layers=args.n_layers,
                        num_heads=args.n_heads,
                        d_model=args.d_model,
                        d_ff=args.d_ff,
                        rope_theta=100000,
                        device=torch.device(args.device),
                        dtype={'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}[args.dtype] )

    # TODO: fix the compile error:
    # Traceback (most recent call last):
    #   File "/DATA/Sync/Files/Programming/AI_ML/CS336/github/cs336-ass1/scripts/train_model_cli.py", line 147, in <module>
    #     main()
    #   File "/DATA/Sync/Files/Programming/AI_ML/CS336/github/cs336-ass1/scripts/train_model_cli.py", line 85, in main
    #     model=torch.compile(model)  # Optional: Compile the model for performance
    #           ^^^^^^^^^^^^^^^^^^^^
    #   File "/DATA/Sync/Files/Programming/AI_ML/CS336/github/cs336-ass1/.venv/lib/python3.11/site-packages/torch/__init__.py", line 2565, in compile
    #     return torch._dynamo.optimize(
    #            ^^^^^^^^^^^^^^^^^^^^^^^
    #   File "/DATA/Sync/Files/Programming/AI_ML/CS336/github/cs336-ass1/.venv/lib/python3.11/site-packages/torch/_dynamo/eval_frame.py", line 512, in __call__
    #     assert callable(fn)
    # AssertionError
    #
    # model=torch.compile(model)  # Optional: Compile the model for performance

    # AdamW optimizer
    optimizer=AdamW(model.parameters(), lr=args.learning_rate)
    
    # Load Checkpoint?
    if args.ckpt is not None:
        file_checkpoint=os.path.join(".", args.ckpt)
        # Load model and optimizer state from checkpoint
        loaded_iter=load_checkpoint(src=file_checkpoint, model=model, optimizer=optimizer)
        print(f"Resumed model and optimizer state from checkpoint {file_checkpoint}")
    else:
        loaded_iter=None

    # Print out summary
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Optimizer has {sum(p.numel() for p in optimizer.param_groups[0]['params'])} parameters")
    print(f"Model parameters:{[name for name, param in model.named_parameters()]}")
        
    # Compile model
    # QUESTION: when to compile? before or after loading checkpoint?
    model=torch.compile(model, fullgraph=True)

    # Prepare Data
    # see if tokenized data already exists
    tokenized_data_path = os.path.join(dir_data, args.tokenizer_data)

    tokenized_data_file_train = os.path.join(tokenized_data_path, 'tokenized_data_train.pkl')
    data_file_train=os.path.join(dir_data, args.training_data)  # Assuming the data file is named 'data_train.txt' 
    tokenizer=Tokenizer.load_from_directory(tokenized_data_path, use_cython=False)
    dataset_train=load_data_and_tokenize(data_file=data_file_train, tokenizer=tokenizer, tokenized_data_file=tokenized_data_file_train)

    # set model to training mode
    model.train()

    # Model Training Loop
    if loaded_iter is not None:  # continue from loaded iteration 
        iter=loaded_iter+1
    else:
        iter=0

    # initialize learning rate parm used to decide whether to update lr in optimizer for each iteration
    lr_old=args.learning_rate
    loss=99.99 

    print(f"Entering training loop...")
    while True:
        t_it_start=datetime.datetime.now()

        # End loop conditions
        if iter>args.max_iters or \
           loss<args.min_loss:  
            print(f"Stopping training at iteration {iter} with loss {loss}")
            break

        # Get batch of data
        x, y = get_batch(dataset=dataset_train, context_length=args.context_length, batch_size=args.batch_size, device=args.device)

        # Forward pass
        logits = model(x)
        loss = cross_entropy(logits, y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        gradient_clipping(model.parameters(), max_l2_norm=args.max_grad_norm)

        # Update model weights
        optimizer.step()

        # Save checkpoint
        if iter%args.save_every==0 and iter>0:
            # Save model checkpoint
            file_checkpoints=os.path.join(dir_output, f'iter{iter}_loss{loss}_context{args.context_length}_layers{args.n_layers}_heads{args.n_heads}_dmodel{args.d_model}_dff{args.d_ff}__batch{args.batch_size}_lr{args.learning_rate}_{str(datetime.datetime.now()).replace(":","_")}.bin')
            save_checkpoint(model=model, optimizer=optimizer, iteration=iter, out=file_checkpoints)
            print(f"Saved checkpoint to {file_checkpoints}")

        # Evaluation
        if iter%args.eval_every==0 and iter>0:
            # Evaluate model on validation set
            # TODO: implement eval
            tokenized_data_file_eval = os.path.join(tokenized_data_path, 'tokenized_data_eval.pkl')
            data_file_eval=os.path.join(dir_data, args.eval_data)  # Assuming the data file is named 'data_train.txt' 
            dataset_eval=load_data_and_tokenize(data_file=data_file_eval, tokenizer=tokenizer, tokenized_data_file=tokenized_data_file_eval)
            loss_list=[]
            model.eval()
            for _ in range(args.eval_iters):
                x,y=get_batch(dataset=dataset_eval, context_length=args.context_length, batch_size=args.batch_size, device=args.device)
                logits=model(x)
                loss=cross_entropy(logits, y)
                loss_list.append(loss)
                print(f"Intermediate eval loss: {loss} ")
            print(f"Evaluation loss at iteration {iter}: {sum(loss_list)/len(loss_list)}")
            model.train()

        # Update learning rate for next iteration if using a scheduler
        lr_new=learning_rate_schedule(it=iter, 
                                      min_learning_rate=args.lr_schedule_min_lr,
                                      max_learning_rate=args.lr_schedule_max_lr,
                                      warmup_iters=args.lr_schedule_warmup_iters,
                                      cosine_cycle_iters=args.lr_schedule_total_iters)   
        if lr_old!=lr_new:
            # TODO: is this the right way to update learning rate for AdamW?
            for param_group in optimizer.param_groups:
                # print(f"learning rate update for next iteration {iter + 1}: {param_group['lr']}->{lr_new}")
                param_group['lr'] = lr_new
            # optimizer.lr=lr_new
        lr_old=lr_new

        # Logging
        t_it_end=datetime.datetime.now()
        t_it_elapsed=t_it_end-t_it_start
        if iter%args.log_every==0:
            # Log training metrics
            # TODO: better logging
            print(f"{datetime.datetime.now()} - " \
                  f"iteration={iter}," \
                  f"elapse={t_it_elapsed.total_seconds():.3f}s," \
                  f"loss={loss if 'loss' in locals() else 'N/A'},"\
                  f"lr={optimizer.param_groups[0]['lr']}")

        # Increment iteration counter 
        iter+=1


    
if __name__ == "__main__":
    main()
    