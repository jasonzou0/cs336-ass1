from cs336_basics.module.softmax import softmax
import torch
from torch import Tensor

from jaxtyping import Float, Int


def nucleus_sampling(
        logits: Float[Tensor, "vocab_size"],
        temperature: float,
        nucleus_sampling_p: float) -> Int[Tensor, "1"]:
    """Sample the next token ID from the model's output logits using temperature scaling and nucleus sampling.
    Args:
        logits (Float[Tensor, "vocab_size"]): The model's output logits for the next token.
        temperature (float): The temperature for scaling the logits.
        nucleus_sampling_p (float): The cumulative probability threshold for nucleus sampling.
    Returns:
        Int[Tensor, "1"]: The sampled token ID as a tensor of shape (1,).
    """
    # Apply temperature scaling
    scaled_logits = logits / temperature
    # Nucleus sampling
    probabilities = softmax(scaled_logits, dim=-1)
    sorted_prob, sorted_indices = torch.sort(probabilities, descending=True)
    cumulative_probs = sorted_prob.cumsum(dim=-1)
    sorted_indices_to_remove = cumulative_probs >= nucleus_sampling_p
    # Ensure at least one token is kept - shift mask to keep the first index that is >= nucleus_sampling_p
    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
    sorted_indices_to_remove[0] = False
    # Remove probabilities that are not in the nucleus
    mask_to_remove = torch.zeros_like(logits, dtype=torch.bool)
    mask_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    probabilities = probabilities.masked_fill(mask_to_remove, 0.0)
    # Sample from the filtered distribution (torch.multinomial re-normalizes the probabilities)
    next_token_id = torch.multinomial(probabilities, num_samples=1)
    return next_token_id


class ModelWrapperWithDecoder(torch.nn.Module):
    """A wrapper around the Transformer model that provides an autoregressive decoder."""
    def __init__(self,
                 model: torch.nn.Module,
                 temperature: float,
                 nucleus_sampling_p: float,
                 device: str):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.nucleus_sampling_p = nucleus_sampling_p
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def forward(self, input_tokens: Int[Tensor, " seq_len"]) -> Int[Tensor, " 1"]:
        """Generate text autoregressively given a prompt.

        Args:
            prompt_tokens (Int[Tensor, " seq_len"]): Input token IDs of shape (seq_len,).
        Returns:
            A single new token ID as an Int[Tensor, " 1"] tensor.
        """

        logits = self.model(input_tokens)  # (seq_len, vocab_size)
        next_token_logits = logits[-1]  # (vocab_size,)
        return nucleus_sampling(
            logits=next_token_logits,
            temperature=self.temperature,
            nucleus_sampling_p=self.nucleus_sampling_p
        )


class Decoder(torch.nn.Module):
    def __init__(
            self,
            model_with_decoder: torch.nn.Module,
            max_new_tokens: int,
            device: str,
            eos_token: int | None = None,
    ):
        """A simple autoregressive decoder for a Transformer model.

        Args:
            tokenizer (Tokenizer): Tokenizer for encoding and decoding text.
            model (torch.nn.Module): The trained Transformer model.
            temperature (float): Sampling temperature.
            nucleus_sampling_p (float): Nucleus sampling probability.
            device (str): Device to run the model on (e.g., "cpu", "cuda", "mps").
        """
        super().__init__()
        self.model = model_with_decoder
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.eos_token = eos_token
        self.model.eval()

    @torch.no_grad()
    def forward(self, prompt_tokens: Int[Tensor, " seq_len"]) -> Int[Tensor, " new_seq_len"]:
        """Generate text autoregressively given a prompt.

        Args:
            prompt_tokens (Int[Tensor, " seq_len"]): Input token IDs of shape (seq_len,).
        Returns:
            Int[Tensor, " new_seq_len"]: Generated token IDs of shape (new_seq_len,).
        """
        for _ in range(self.max_new_tokens):
            next_token_id = self.model(prompt_tokens)  # (1,)
            if self.eos_token is not None and next_token_id.item() == self.eos_token:
                break
            prompt_tokens = torch.cat([prompt_tokens, next_token_id], dim=-1)  # (seq_len + 1,)
        return prompt_tokens

