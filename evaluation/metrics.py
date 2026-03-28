from __future__ import annotations
import torch
from typing import Dict

from problems.solutions import AnalyticalSolution
from functionals.integrals import domain_integral

class MetricsCalculator:
    def __init__(self, solution: AnalyticalSolution):
        self.solution = solution

    def energy_error(
                self,
                grad_v: torch.Tensor,
                xy: torch.Tensor,
                vol_w: torch.Tensor,
            ) -> torch.Tensor:
        exact_grad = self.solution.grad_vector(xy)
        err_sq = ((grad_v - exact_grad) ** 2).sum(dim=1, keepdim=True)
        return domain_integral(err_sq, vol_w)

    def relative_l2_error(
                self,
                v: torch.Tensor,
                xy: torch.Tensor,
                vol_w: torch.Tensor,
            ) -> torch.Tensor:
        u_exact = self.solution.eval(xy)
        l2_err_sq = domain_integral((v - u_exact) ** 2, vol_w)
        l2_norm_sq = domain_integral(u_exact ** 2, vol_w)
        return torch.sqrt(l2_err_sq / l2_norm_sq)

    def relative_energy_error(
                self,
                grad_v: torch.Tensor,
                xy: torch.Tensor,
                vol_w: torch.Tensor,
            ) -> torch.Tensor:
        exact_grad = self.solution.grad_vector(xy)
        err_sq = ((grad_v - exact_grad) ** 2).sum(dim=1, keepdim=True)
        norm_sq = (exact_grad ** 2).sum(dim=1, keepdim=True)
        return torch.sqrt(
            domain_integral(err_sq, vol_w) / domain_integral(norm_sq, vol_w)
        )

    def compute_all(
                self,
                v: torch.Tensor,
                grad_v: torch.Tensor,
                xy: torch.Tensor,
                vol_w: torch.Tensor,
            ) -> Dict[str, float]:
        with torch.no_grad():
            return {
                "energy": self.energy_error(grad_v, xy, vol_w).item(),
                "rel_l2": self.relative_l2_error(v, xy, vol_w).item(),
                "rel_energy": self.relative_energy_error(grad_v, xy, vol_w).item(),
            }