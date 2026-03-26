from __future__ import annotations
from typing import Tuple
import torch
from functionals.integrals import domain_integral, boundary_integral
from functionals.operators import gradient, normal_derivative

def pde_residual_loss(lap_v, f, vol_w):
    return domain_integral((-lap_v - f)**2, vol_w)

def dirichlet_loss(v_bd, g_D, surf_w, bc_mask):
    return boundary_integral((v_bd - g_D)**2, surf_w, mask=bc_mask)

def neumann_loss(dvdn, g_N, surf_w, bc_mask):
    return boundary_integral((dvdn - g_N)**2, surf_w, mask=(1.0 - bc_mask))

def bc_losses(model, xy_bd, normals, g_D, g_N, bc_mask, surf_w):
    v_bd = model(xy_bd)
    loss_dir = dirichlet_loss(v_bd, g_D, surf_w, bc_mask)
    dvdn = normal_derivative(v_bd, xy_bd, normals, create_graph=True)
    loss_neu = neumann_loss(dvdn, g_N, surf_w, bc_mask)
    return loss_dir, loss_neu
