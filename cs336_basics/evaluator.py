import os
import torch
import numpy as np

from cs336_basics.data_loader import DataLoader, DataLoadingMode
from cs336_basics.module.loss import cross_entropy_loss


class Evaluator:
    def __init__(self, 
                 model: torch.nn.Module,
                 eval_data_loader: DataLoader = None) -> None:
        """Evaluator for computing average loss on an evaluation dataset."""
        self.model = model
        self.eval_data_loader = eval_data_loader

    def avg_loss(self) -> float:
        """Compute the average loss over the evaluation dataset (averaged over all tokens)."""
        self.model.eval()

        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            t = 0
            for input_ids, target_ids in self.eval_data_loader:
                logits = self.model(input_ids)
                loss = cross_entropy_loss(logits, target_ids)
                
                # Accumulate loss weighted by number of tokens
                batch_tokens = target_ids.numel()
                total_loss += loss.item() * batch_tokens
                total_tokens += batch_tokens
                if t % 100 == 0:
                    print(f"Loss at eval batch {t}: {loss.item():.4f}")
                t += 1
        
        return total_loss / total_tokens if total_tokens > 0 else 0.0