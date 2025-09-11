import torch

from cs336_basics.data_loader import DataLoader
from cs336_basics.module.loss import cross_entropy_loss


class Evaluator:
    def __init__(self, 
                 model: torch.nn.Module,
                 eval_data_loader: DataLoader,
                 device: str) -> None:
        """Evaluator for computing average loss on an evaluation dataset."""
        self.model = model
        self.eval_data_loader = eval_data_loader
        self.device = device if device is not None else torch.device("cpu")


    def avg_loss(self) -> float:
        """Compute the average loss over the evaluation dataset (averaged over all tokens)."""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for input_ids, target_ids in self.eval_data_loader:
                logits = self.model(input_ids)
                loss = cross_entropy_loss(logits, target_ids)
                n_tokens = target_ids.numel()
                total_loss += loss.item() * n_tokens
                total_tokens += n_tokens
        avg_loss = total_loss / total_tokens
        return avg_loss