import torch
import math
from typing import Optional, Callable
from dataclasses import dataclass


@dataclass
class OptimizerConfig:
    # The total number of training iterations.
    total_iters: int
    # Number of iterations to linearly warm up the learning rate.
    warmup_iters: int | None = None
    # The initial learning rate after warmup.
    learning_rate: float = 1e-3
    # The weight decay to apply.
    weight_decay: float = 0.01
    # The beta coefficients used for computing running averages of gradient and its square.
    betas: tuple[float, float] = (0.9, 0.999)
    # The final learning rate after cosine annealing.
    min_learning_rate: float = 1e-5

    def __post_init__(self):
        if self.warmup_iters is None:
            self.warmup_iters = max(1, self.total_iters // 10)
        if self.total_iters <= self.warmup_iters:
            raise ValueError("total_iters must be larger than warmup_iters, but got "
                             f"total_iters={self.total_iters} and warmup_iters={self.warmup_iters}")


def create_from_config(params, config: OptimizerConfig) -> tuple[torch.optim.Optimizer, "CosineScheduler"]:
    """Create a coupled AdamW optimizer and CosineScheduler from a configuration.

    Args:
        params: Iterable of model parameters.
        config (OptimizerConfig): Optimizer hyperparameter configuration.
        cosine_cycle_iters (int): Total iterations for the warmup + cosine schedule.

    Returns:
        (optimizer, scheduler): The initialized AdamW optimizer and CosineScheduler.
    """
    optimizer = AdamW(
        params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
    )
    scheduler = CosineScheduler(
        min_learning_rate=config.min_learning_rate,
        optimizer=optimizer,
        warmup_iters=config.warmup_iters,
        cosine_cycle_iters=config.total_iters,
    )
    return optimizer, scheduler


class CosineScheduler:
    """
    A cosine learning rate scheduler with linear warmup.
    """
    def __init__(
        self,
        min_learning_rate: float,
        optimizer: torch.optim.Optimizer,
        warmup_iters: int, 
        cosine_cycle_iters: int,
    ):
        """
        Initialize the cosine scheduler.
        
        Args:
            torch.optim.Optimizer: optimizer to attach the scheduler to. 
                Its initial learning rate is the learning rate warmup target.
            min_learning_rate (float): the minimum / final learning rate.
            warmup_iters (int): the number of iterations to linearly warm-up learning rate to .
            cosine_cycle_iters (int): the number of cosine annealing iterations.
            
        """
        assert cosine_cycle_iters > warmup_iters, "cosine_cycle_iters must be larger than warmup_iters, but got " \
            f"cosine_cycle_iters={cosine_cycle_iters} and warmup_iters={warmup_iters}"
        
        self.optimizer = optimizer
        self.max_learning_rate = self.optimizer.param_groups[0]["lr"]
        self.min_learning_rate = min_learning_rate
        self.warmup_iters = warmup_iters
        self.cosine_cycle_iters = cosine_cycle_iters
        self.step_count = 0
    
    def step(self):
        self.step_count += 1
        for group in self.optimizer.param_groups:
            group['lr'] = self.get_lr_at_iter(self.step_count)

    def get_lr_at_iter(self, it: int) -> float:
        """
        Get the learning rate for the given iteration.
        
        Args:
            it (int): Iteration number to get learning rate for.
            
        Returns:
            Learning rate at the given iteration under the specified schedule.
        """
        if it < self.warmup_iters:
            return self.max_learning_rate * (it / self.warmup_iters)
        elif it > self.cosine_cycle_iters:
            return self.min_learning_rate
        else:
            cos_inner = math.pi * (it - self.warmup_iters) / (self.cosine_cycle_iters - self.warmup_iters)
            return self.min_learning_rate + 0.5 * (self.max_learning_rate - self.min_learning_rate) * (1 + math.cos(cos_inner))


class AdamW(torch.optim.Optimizer):
    def __init__(self, 
                 params, 
                 lr: float, 
                 weight_decay: float, 
                 betas: tuple[float, float], 
                 eps: float = 1e-8):
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
