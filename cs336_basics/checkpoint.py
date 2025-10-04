import torch

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str) -> None:
    """
    Save the model and optimizer state to a checkpoint file.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        out (str): The path to the checkpoint file.
    """
    checkpoint_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint_dict, out)
    
def load_checkpoint(src: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:  
    """
    Load the model and optimizer state from a checkpoint file.

    Args:
        model (torch.nn.Module): The model to load the state into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state into.
        filepath (str): The path to the checkpoint file.

    Returns:
        tuple[int, float]: A tuple containing the epoch number and loss value.
    """
    checkpoint_dict = torch.load(src)
    model.load_state_dict(checkpoint_dict['model_state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
    iteration= checkpoint_dict['iteration']
    return iteration