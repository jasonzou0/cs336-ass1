import torch
from typing import Iterable

def grad_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    with torch.no_grad():
        grads = [p.grad for p in parameters if p.grad is not None]
        first_device = grads[0].device
        # TODO: check if all grads are on the same device
        # NOTE: we have to use torch.norm instead of torch.linalg.norm for compatibility with PyTorch MPS
        total_norm = torch.norm(torch.stack([torch.norm(g, p=2) for g in grads]), p=2)
        clip_coef = max_l2_norm / (total_norm + 1e-6)
        clip_coef = clip_coef.to(first_device)
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)