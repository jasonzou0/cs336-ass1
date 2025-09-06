import torch
import math
from typing import Optional, Callable

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    assert cosine_cycle_iters > warmup_iters, "cosine_cycle_iters must be larger than warmup_iters, but got " \
        f"cosine_cycle_iters={cosine_cycle_iters} and warmup_iters={warmup_iters}"
    
    if it < warmup_iters:
        return max_learning_rate * (it / warmup_iters)
    elif it > cosine_cycle_iters:
        return min_learning_rate
    else:
        cos_inner = math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (1 + math.cos(cos_inner))

    

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
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            lr_m_weight_decay = lr * weight_decay

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
                m.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
                state["m"] = m 
                state["v"] = v
                alpha = lr * (math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t))
                # Perform parameter update.                
                p.addcdiv_(m, torch.sqrt(v).add_(eps), value=-alpha)
                p.sub_(p, alpha=lr_m_weight_decay)
                # Update step in parameter state.
                state["t"] = t + 1

        return loss