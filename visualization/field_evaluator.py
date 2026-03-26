from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import matplotlib.tri as mtri

from config import DEVICE
from problems.solutions import AnalyticalSolution
from functionals.operators import gradient

@dataclass
class FieldValues:
    v: np.ndarray
    u_exact: np.ndarray
    abs_error: np.ndarray
    energy_error_density: np.ndarray

def refine_mesh(mesh: Dict[str, np.ndarray], subdiv: int = 3
                ) -> Tuple[mtri.Triangulation, np.ndarray]:
    pts = mesh["points"]
    tris = mesh["triangles"]
    tri_orig = mtri.Triangulation(pts[:, 0], pts[:, 1], tris)
    refiner = mtri.UniformTriRefiner(tri_orig)
    tri_refi = refiner.refine_triangulation(subdiv=subdiv)
    pts_refi = np.column_stack([tri_refi.x, tri_refi.y])
    return tri_refi, pts_refi

def evaluate_fields(xy, pinn, solution):
    xy_req = xy.clone().requires_grad_(True)
    with torch.no_grad():
        ue = solution.eval(xy_req).squeeze(-1).cpu().numpy()
        eg = solution.grad_vector(xy_req).cpu().numpy()
    with torch.enable_grad():
        v = pinn(xy_req)
        gv = gradient(v, xy_req, create_graph=False)
    vn = v.detach().squeeze(-1).cpu().numpy()
    gn = gv.detach().cpu().numpy()
    en_err = np.clip(((gn - eg)**2).sum(-1), 0, None)
    return FieldValues(v=vn, u_exact=ue, abs_error=np.abs(vn - ue),
                       energy_error_density=en_err)
