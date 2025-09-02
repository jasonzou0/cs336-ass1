import torch
import math
from typing import Optional, Callable

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float, weight_decay: float, betas: tuple[float, float], eps: float):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0 <= betas[0] < 1):
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not (0 <= betas[1] < 1):
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps,
        }
        super().__init__(params, defaults)
        
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                # Iteration number
                t = state.get("t", 1) 
                # Get stored 1st and 2nd moments.
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                # Update moments and store back into parameter state.
                m = beta1 * m + (1 - beta1) * p.grad.data
                v = beta2 * v + (1 - beta2) * (p.grad.data ** 2)
                state["m"] = m 
                state["v"] = v
                alpha = lr * (math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t))
                p.data -= alpha * m / (torch.sqrt(v) + eps)
                p.data -= lr * weight_decay * p.data

                # Update step in parameter state.
                state["t"] = t + 1

        return loss