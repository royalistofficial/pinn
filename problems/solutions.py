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

class HighFreqSineSolution(AnalyticalSolution):
    def __init__(self, k: float = 5.0):
        self.k = k

    def eval(self, xy):
        return (torch.sin(self.k * math.pi * xy[:,0]) * torch.sin(self.k * math.pi * xy[:,1])).unsqueeze(-1)

    def grad(self, xy):
        sx = torch.sin(self.k * math.pi * xy[:,0])
        sy = torch.sin(self.k * math.pi * xy[:,1])
        cx = torch.cos(self.k * math.pi * xy[:,0])
        cy = torch.cos(self.k * math.pi * xy[:,1])
        return (self.k * math.pi * cx * sy).unsqueeze(-1), (self.k * math.pi * sx * cy).unsqueeze(-1)

    def rhs(self, xy):
        return (2 * (self.k * math.pi)**2 * torch.sin(self.k * math.pi * xy[:,0]) * torch.sin(self.k * math.pi * xy[:,1])).unsqueeze(-1)

    @property
    def regularity_order(self) -> float:
        return float('inf')

class SteepPeakSolution(AnalyticalSolution):
    def __init__(self, a: float = 100.0):
        self.a = a

    def eval(self, xy):
        r2 = xy[:,0]**2 + xy[:,1]**2
        return torch.exp(-self.a * r2).unsqueeze(-1)

    def grad(self, xy):
        r2 = xy[:,0]**2 + xy[:,1]**2
        u = torch.exp(-self.a * r2)
        ux = -2 * self.a * xy[:,0] * u
        uy = -2 * self.a * xy[:,1] * u
        return ux.unsqueeze(-1), uy.unsqueeze(-1)

    def rhs(self, xy):
        r2 = xy[:,0]**2 + xy[:,1]**2
        u = torch.exp(-self.a * r2)

        return (4 * self.a * (1.0 - self.a * r2) * u).unsqueeze(-1)

    @property
    def regularity_order(self) -> float:
        return float('inf')

class TanhLayerSolution(AnalyticalSolution):
    def __init__(self, k: float = 10.0):
        self.k = k

    def eval(self, xy):
        return torch.tanh(self.k * (xy[:,0] - xy[:,1])).unsqueeze(-1)

    def grad(self, xy):
        u = torch.tanh(self.k * (xy[:,0] - xy[:,1]))
        sech2 = 1.0 - u**2
        ux = self.k * sech2
        uy = -self.k * sech2
        return ux.unsqueeze(-1), uy.unsqueeze(-1)

    def rhs(self, xy):
        u = torch.tanh(self.k * (xy[:,0] - xy[:,1]))

        return (4 * self.k**2 * u * (1.0 - u**2)).unsqueeze(-1)

    @property
    def regularity_order(self) -> float:
        return float('inf')

class LowRegularitySolution(AnalyticalSolution):
    def eval(self, xy):

        r = torch.sqrt(xy[:,0]**2 + xy[:,1]**2 + 1e-12)
        return (r**3).unsqueeze(-1)

    def grad(self, xy):
        r = torch.sqrt(xy[:,0]**2 + xy[:,1]**2 + 1e-12)
        ux = 3 * xy[:,0] * r
        uy = 3 * xy[:,1] * r
        return ux.unsqueeze(-1), uy.unsqueeze(-1)

    def rhs(self, xy):
        r = torch.sqrt(xy[:,0]**2 + xy[:,1]**2 + 1e-12)
        return (-9.0 * r).unsqueeze(-1)

    @property
    def regularity_order(self) -> float:
        return 2.5

SOLUTIONS = {
    "sine": SineSolution,
    "exponential": ExponentialSolution,
    "polynomial": PolynomialSolution,
    "high_freq": HighFreqSineSolution,
    "steep_peak": SteepPeakSolution,
    "tanh_layer": TanhLayerSolution,
    "low_reg": LowRegularitySolution,
}