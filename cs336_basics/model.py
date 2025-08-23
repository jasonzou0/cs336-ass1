import torch
import torch.nn as nn
class LinearModule(nn.Module):
    weight: nn.Parameter

    def __init__(self, 
                in_features:int, out_features: int,
                device: torch.device=torch.device('cpu'),
                dtype: torch.dtype=torch.float32):
        super(LinearModule,self).__init__()
        # Initialize weights with normal distribution
        # Using standard deviation of 1/sqrt(in_features) which is common for linear layers
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        torch.nn.init.normal_(self.weight, mean=0.0, std=1.0 / (in_features ** 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return torch.matmul(x, self.W.T)  # This is the correct way to perform matrix multiplication in PyTorch
        return x@(self.weight.T)

class EmbeddingModule(nn.Module):
    weight: nn.Parameter

    def __init__(self, 
                num_embeddings:int, 
                embedding_dim:int,
                device: torch.device=torch.device('cpu'),
                dtype: torch.dtype=torch.float32):
        super(EmbeddingModule,self).__init__()
        # Initialize weights with normal distribution
        # Using standard deviation of 0.02 which is common for transformer embeddings
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        torch.nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]  # This is the correct way to index into the embedding matrix

        
class RMSNormModule(nn.Module):
    weight: nn.Parameter
    def __init__(self, 
                d_model: int, 
                eps: float = 1e-6,
                device: torch.device=torch.device('cpu'),
                dtype: torch.dtype=torch.float32):
        super(RMSNormModule, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))
        torch.nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt((x*x).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight   # This is the correct way to apply RMS normalization

class SwiGLUModule(nn.Module):
    weight1: nn.Parameter
    weight2: nn.Parameter
    weight3: nn.Parameter
    def __init__(self, 
                d_model: int, 
                d_ff: int,
                device: torch.device=torch.device('cpu'),
                dtype: torch.dtype=torch.float32):
        super(SwiGLUModule, self).__init__()
        self.weight1 = nn.Parameter(torch.empty((d_ff,d_model), device=device, dtype=dtype))
        self.weight2 = nn.Parameter(torch.empty((d_model,d_ff), device=device, dtype=dtype))
        self.weight3 = nn.Parameter(torch.empty((d_ff,d_model), device=device, dtype=dtype))
        torch.nn.init.normal_(self.weight1, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.weight2, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.weight3, mean=0.0, std=0.02)

    @staticmethod
    def SiLU(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x) 

    def GLU(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.weight1@x)*(self.weight2@x)

    def SwiGLU(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight2@(self.SiLU(self.weight1@x) * (self.weight3@x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.SwiGLU(x)  
    