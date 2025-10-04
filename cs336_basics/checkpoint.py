import torch

def strip_prefix(state_dict, prefix="_orig_mod."):
    return { (k[len(prefix):] if k.startswith(prefix) else k): v
             for k, v in state_dict.items() }

def clean_compiled_state_dict(state_dict):
    # If keys in sd have the prefix, remove it:
    first_key = next(iter(state_dict))
    if first_key.startswith("_orig_mod."):
        print("state_dict is from a compiled model, stripping _orig_mod. prefix from state_dict keys")
        state_dict = strip_prefix(state_dict, "_orig_mod.")
    return state_dict

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str) -> None:
    """
    Save the model state, optimizer state, and hyperparameters to a checkpoint file.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        iteration (int): Current training iteration.
        out (str): The path to the checkpoint file.
    """
    # Get model hyperparameters
    model_args = {
        'vocab_size': model.vocab_size if hasattr(model, 'vocab_size') else None,
        'context_length': model.context_length if hasattr(model, 'context_length') else None,
        'd_model': model.d_model if hasattr(model, 'd_model') else None,
        'num_layers': model.num_layers if hasattr(model, 'num_layers') else None,
        'num_heads': model.num_heads if hasattr(model, 'num_heads') else None,
        'd_ff': model.d_ff if hasattr(model, 'd_ff') else None,
        'rope_theta': model.rope_theta if hasattr(model, 'rope_theta') else None
    }

    # Get optimizer hyperparameters
    optim_args = optimizer.state_dict()['param_groups'][0].copy()
    # Remove the 'params' key as it's not needed for reconstruction
    optim_args.pop('params', None)

    checkpoint_dict = {
        'model_state_dict': model.state_dict(),
        'model_args': model_args,
        'optimizer_state_dict': optimizer.state_dict(),
        'optimizer_args': optim_args,
        'iteration': iteration
    }
    torch.save(checkpoint_dict, out)
    
def load_checkpoint(src: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:  
    """
    Load the model state, optimizer state, and hyperparameters from a checkpoint file.

    Args:
        src (str): The path to the checkpoint file.
        model (torch.nn.Module): The model to load the state into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state into.

    Returns:
        int: The last saved iteration number.

    Raises:
        ValueError: If the checkpoint's model architecture doesn't match the provided model.
    """
    checkpoint_dict = torch.load(src)
    
    # Verify model architecture matches (check if hyperparameters are same)
    if 'model_args' in checkpoint_dict:
        saved_args = checkpoint_dict['model_args']
        for key, value in saved_args.items():
            if value is not None and hasattr(model, key):
                current_value = getattr(model, key)
                if current_value != value:
                    raise ValueError(
                        f"Model architecture mismatch: {key} differs "
                        f"(checkpoint: {value}, current: {current_value})"
                    )

    # Load states for model
    # Strip _orig_mod. prefix if present
    print("Cleaning state_dicts for model from compiled model if necessary")
    checkpoint_dict['model_state_dict'] = clean_compiled_state_dict(checkpoint_dict['model_state_dict'])
    model.load_state_dict(checkpoint_dict['model_state_dict'])

    if optimizer is not None:
        # Update optimizer hyperparameters if available
        if 'optimizer_args' in checkpoint_dict:
            for group in optimizer.param_groups:
                saved_args = checkpoint_dict['optimizer_args']
                # Update all optimizer parameters except 'params'
                for key in set(saved_args.keys()) - {'params'}:
                    if key in group:
                        group[key] = saved_args[key]

        # Load states for optimizer
        print("Cleaning state_dicts for optimizer from compiled model if necessary")
        checkpoint_dict['optimizer_state_dict'] = clean_compiled_state_dict(checkpoint_dict['optimizer_state_dict'])
        optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])

    return checkpoint_dict['iteration']

