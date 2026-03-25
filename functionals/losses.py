from __future__ import annotations
from typing import Tuple
import torch
from functionals.integrals import domain_integral, boundary_integral
from functionals.operators import gradient, normal_derivative

def pde_residual_loss(lap_v: torch.Tensor, f: torch.Tensor, vol_w: torch.Tensor) -> torch.Tensor:
    return domain_integral((-lap_v - f)**2, vol_w)

def dirichlet_loss(v_bd: torch.Tensor, g_D: torch.Tensor,
                   surf_w: torch.Tensor, bc_mask: torch.Tensor) -> torch.Tensor:
    return boundary_integral((v_bd - g_D)**2, surf_w, mask=bc_mask)

def neumann_loss(dvdn: torch.Tensor, g_N: torch.Tensor,
                 surf_w: torch.Tensor, bc_mask: torch.Tensor) -> torch.Tensor:
    return boundary_integral((dvdn - g_N)**2, surf_w, mask=(1.0 - bc_mask))

def bc_losses(model: torch.nn.Module, xy_bd: torch.Tensor,
              normals: torch.Tensor, g_D: torch.Tensor, g_N: torch.Tensor,
              bc_mask: torch.Tensor, surf_w: torch.Tensor
              ) -> Tuple[torch.Tensor, torch.Tensor]:
    v_bd = model(xy_bd)
    loss_dir = dirichlet_loss(v_bd, g_D, surf_w, bc_mask)
    dvdn = normal_derivative(v_bd, xy_bd, normals, create_graph=True)
    loss_neu = neumann_loss(dvdn, g_N, surf_w, bc_mask)
    return loss_dir, loss_neu
