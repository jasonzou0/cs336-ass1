import torch
import torch.nn as nn
class LinearModule(nn.Module):
    W: nn.Parameter

    def __init__(self, 
                in_features:int, out_features: int,
                device: torch.device=torch.device('cpu'),
                dtype: torch.dtype=torch.float32):
        super(LinearModule,self).__init__()
        # super().__init__()
        #TODO: change to normal distribution initialization
        self.W=nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return torch.matmul(x, self.W.T)  # This is the correct way to perform matrix multiplication in PyTorch
        return x@(self.W.T)

class EmbeddingModule(nn.Module):
    W: nn.Parameter

    def __init__(self, 
                num_embeddings:int, 
                embedding_dim:int,
                device: torch.device=torch.device('cpu'),
                dtype: torch.dtype=torch.float32):
        super(EmbeddingModule,self).__init__()
        #TODO: initialize with normal distribution
        #TODO: user initialization function
        self.W=nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.W[token_ids]  # This is the correct way to index into the embedding matrix