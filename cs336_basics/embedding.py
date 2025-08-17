import math
import torch
from torch import Tensor

from jaxtyping import Float, Int

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.emb: Float[Tensor, "n_emb emb_dim"] = torch.nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(
            self.emb, 
            mean=0.0, 
            std=1.0,
            a=-3.0,
            b=3.0,
        )

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        # TODO: handle batch dimension in token_ids
        return self.emb[token_ids]