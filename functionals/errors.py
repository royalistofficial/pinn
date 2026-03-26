from __future__ import annotations
import torch
from problems.solutions import AnalyticalSolution
from functionals.integrals import domain_integral

def energy_error(grad_v, xy, vol_w, solution):
    exact_grad = solution.grad_vector(xy)
    err_sq = ((grad_v - exact_grad)**2).sum(1, keepdim=True)
    return domain_integral(err_sq, vol_w)

def relative_l2_error(v, xy, vol_w, solution):
    u_exact = solution.eval(xy)
    l2_err_sq = domain_integral((v - u_exact)**2, vol_w)
    l2_norm_sq = domain_integral(u_exact**2, vol_w)
    return torch.sqrt(l2_err_sq / l2_norm_sq)

def relative_energy_error(grad_v, xy, vol_w, solution):
    exact_grad = solution.grad_vector(xy)
    err_sq = ((grad_v - exact_grad)**2).sum(1, keepdim=True)
    norm_sq = (exact_grad**2).sum(1, keepdim=True)
    return torch.sqrt(domain_integral(err_sq, vol_w) / domain_integral(norm_sq, vol_w))
