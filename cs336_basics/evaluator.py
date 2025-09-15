import torch

from cs336_basics.data_loader import DataLoader, DataLoaderConfig, DataLoadingMode


def run_eval(
        model: torch.nn.Module,
        eval_data_path: str,
        context_length: int,
        eval_batch_size: int,
        device: str):
    """Run evaluation on a trained Transformer model.

    Args:
        model (torch.nn.Module): The trained Transformer model.
        eval_data_path (str): Path to the evaluation dataset (numpy file).
        context_length (int): Context length for evaluation.
        device (str): Device to use for evaluation (e.g., "cpu", "cuda", "mps").
        eval_batch_size (int): Batch size for evaluation.
    """
    eval_data_loader = DataLoader.from_config(DataLoaderConfig(
        dataset_path=eval_data_path,
        num_batches=None,
        batch_size=eval_batch_size,
        context_length=context_length,
        data_loading_mode=DataLoadingMode.SEQUENTIAL,
    ), device=device)
    evaluator = Evaluator(model_with_loss=model, eval_data_loader=eval_data_loader)
    avg_loss = evaluator.avg_loss()
    print(f"Avg Evaluation Loss: {avg_loss:.4f}")


class Evaluator:
    def __init__(self, 
                 model_with_loss: torch.nn.Module,
                 eval_data_loader: DataLoader = None) -> None:
        """Evaluator for computing average loss on an evaluation dataset."""
        self.model_with_loss = model_with_loss
        self.eval_data_loader = eval_data_loader

    def avg_loss(self) -> float:
        """Compute the average loss over the evaluation dataset (averaged over all tokens)."""
        self.model_with_loss.eval()

        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            t = 0
            for input_ids, target_ids in self.eval_data_loader:
                loss = self.model_with_loss(input_ids, target_ids)
                
                # Accumulate loss weighted by number of tokens
                batch_tokens = target_ids.numel()
                total_loss += loss.item() * batch_tokens
                total_tokens += batch_tokens
                if t % 100 == 0:
                    print(f"Loss at eval batch {t}: {loss.item():.4f}")
                t += 1
        
        return total_loss / total_tokens if total_tokens > 0 else 0.0