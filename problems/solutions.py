from __future__ import annotations
import abc
import math
from typing import Tuple
import torch

class AnalyticalSolution(abc.ABC):
    @abc.abstractmethod
    def eval(self, xy: torch.Tensor) -> torch.Tensor: ...

    @abc.abstractmethod
    def grad(self, xy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: ...

    @abc.abstractmethod
    def rhs(self, xy: torch.Tensor) -> torch.Tensor: ...

    def grad_vector(self, xy: torch.Tensor) -> torch.Tensor:
        ux, uy = self.grad(xy)
        return torch.cat([ux, uy], dim=1)

    def neumann_data(self, xy: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
        g = self.grad_vector(xy)
        return (g * normals).sum(dim=1, keepdim=True)

    @property
    def regularity_order(self) -> float:
        return 2.0  

    @property
    def sobolev_index(self) -> float:
        return self.regularity_order

class SineSolution(AnalyticalSolution):
    def eval(self, xy):
        return (torch.sin(math.pi*xy[:,0]) * torch.sin(math.pi*xy[:,1])).unsqueeze(-1)

    def grad(self, xy):
        sx = torch.sin(math.pi*xy[:,0]); sy = torch.sin(math.pi*xy[:,1])
        cx = torch.cos(math.pi*xy[:,0]); cy = torch.cos(math.pi*xy[:,1])
        return (math.pi*cx*sy).unsqueeze(-1), (math.pi*sx*cy).unsqueeze(-1)

    def rhs(self, xy):
        return (2*math.pi**2 * torch.sin(math.pi*xy[:,0]) * torch.sin(math.pi*xy[:,1])).unsqueeze(-1)

    @property
    def regularity_order(self) -> float:
        return float('inf')  

class ExponentialSolution(AnalyticalSolution):
    def eval(self, xy):
        return torch.exp(xy[:,0]+xy[:,1]).unsqueeze(-1)

    def grad(self, xy):
        e = torch.exp(xy[:,0]+xy[:,1]).unsqueeze(-1)
        return e, e

    def rhs(self, xy):
        return -2.0 * torch.exp(xy[:,0]+xy[:,1]).unsqueeze(-1)

    @property
    def regularity_order(self) -> float:
        return float('inf')

class PolynomialSolution(AnalyticalSolution):
    def eval(self, xy):
        return (xy[:,0]**3 * xy[:,1]**3).unsqueeze(-1)

    def grad(self, xy):
        x, y = xy[:,0], xy[:,1]
        return (3*x**2 * y**3).unsqueeze(-1), (3*x**3 * y**2).unsqueeze(-1)

    def rhs(self, xy):
        x, y = xy[:,0], xy[:,1]
        return (-6*x*y*(x**2+y**2)).unsqueeze(-1)

    @property
    def regularity_order(self) -> float:
        return float('inf')

SOLUTIONS = {
    "sine": SineSolution,
    "exponential": ExponentialSolution,
    "polynomial": PolynomialSolution,
}
