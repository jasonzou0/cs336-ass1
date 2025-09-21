from math import exp

import torch
import wandb

from cs336_basics.data_loader import DataLoader, DataLoaderConfig, DataLoadingMode


def run_eval(
        model: torch.nn.Module,
        eval_data_path: str,
        context_length: int,
        eval_batch_size: int,
        device: str,
        wandb: wandb.Run | None) -> None:
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
    print(f"Avg Cross Entropy Loss: {avg_loss:.4f}")
    print(f"Avg Perplexity: {exp(avg_loss):.4f}")
    if wandb is not None:
        wandb.log({"eval/avg_loss": avg_loss, "eval/perplexity": exp(avg_loss)})


class Evaluator:
    def __init__(self,
                 model_with_loss: torch.nn.Module,
                 eval_data_loader: DataLoader = None) -> None:
        """Evaluator for computing average loss on an evaluation dataset."""
        self.model_with_loss = model_with_loss
        self.eval_data_loader = eval_data_loader

    @torch.no_grad()
    def avg_loss(self) -> float:
        """Compute the average loss over the evaluation dataset (averaged over all tokens)."""
        self.model_with_loss.eval()

        losses = []
        token_counts = []

        for t, (input_ids, target_ids) in enumerate(self.eval_data_loader):
            loss = self.model_with_loss(input_ids, target_ids)
            batch_tokens = target_ids.shape[0] * target_ids.shape[1]  # batch_size * seq_len

            # Keep tensors on GPU, accumulate in lists
            losses.append(loss * batch_tokens)
            token_counts.append(batch_tokens)

        # Single GPU-CPU sync at the end
        if losses:
            total_loss = torch.stack(losses).sum().item()
            total_tokens = sum(token_counts)
            return total_loss / total_tokens
        return 0.0