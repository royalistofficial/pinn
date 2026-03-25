from __future__ import annotations
import torch

def domain_integral(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.sum(values * weights)

def boundary_integral(values: torch.Tensor, weights: torch.Tensor,
                      mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is not None:
        return torch.sum(values * weights * mask)
    return torch.sum(values * weights)
