from __future__ import annotations
from typing import Tuple
import torch

def gradient(v: torch.Tensor, xy: torch.Tensor, create_graph: bool = True) -> torch.Tensor:

    return torch.autograd.grad(v, xy, torch.ones_like(v), create_graph=create_graph)[0]

def laplacian(v: torch.Tensor, xy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

    gv = torch.autograd.grad(v, xy, torch.ones_like(v), create_graph=True)[0]
    d2x = torch.autograd.grad(gv[:, 0:1], xy, torch.ones_like(gv[:, 0:1]), create_graph=True)[0][:, 0:1]
    d2y = torch.autograd.grad(gv[:, 1:2], xy, torch.ones_like(gv[:, 1:2]), create_graph=True)[0][:, 1:2]
    return gv, d2x + d2y

def normal_derivative(v_bd: torch.Tensor, xy_bd: torch.Tensor,
                      normals: torch.Tensor, create_graph: bool = True) -> torch.Tensor:

    gv = gradient(v_bd, xy_bd, create_graph=create_graph)
    return (gv * normals).sum(-1, keepdim=True)
