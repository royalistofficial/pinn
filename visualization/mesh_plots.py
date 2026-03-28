from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from config import DEVICE
from problems.solutions import AnalyticalSolution
from functionals.operators import gradient

@dataclass
class FieldValues:
    v: np.ndarray
    u_exact: np.ndarray
    abs_error: np.ndarray
    energy_density: np.ndarray

def refine_mesh(
        mesh: Dict[str, np.ndarray],
        subdiv: int = 3,
    ) -> Tuple[mtri.Triangulation, np.ndarray]:
    pts = mesh["points"]
    tris = mesh["triangles"]
    tri_orig = mtri.Triangulation(pts[:, 0], pts[:, 1], tris)
    refiner = mtri.UniformTriRefiner(tri_orig)
    tri_refi = refiner.refine_triangulation(subdiv=subdiv)
    pts_refi = np.column_stack([tri_refi.x, tri_refi.y])
    return tri_refi, pts_refi

def evaluate_fields(
        xy: torch.Tensor,
        pinn: nn.Module,
        solution: AnalyticalSolution,
    ) -> FieldValues:
    xy_req = xy.clone().requires_grad_(True)

    with torch.no_grad():
        ue = solution.eval(xy_req).squeeze(-1).cpu().numpy()
        eg = solution.grad_vector(xy_req).cpu().numpy()

    with torch.enable_grad():
        v = pinn(xy_req)
        gv = gradient(v, xy_req, create_graph=False)

    vn = v.detach().squeeze(-1).cpu().numpy()
    gn = gv.detach().cpu().numpy()
    en_err = np.clip(((gn - eg) ** 2).sum(-1), 0, None)

    return FieldValues(
        v=vn,
        u_exact=ue,
        abs_error=np.abs(vn - ue),
        energy_density=en_err,
    )

def plot_mesh(
        mesh: Dict[str, np.ndarray],
        domain_name: str,
        bc_type_func,
        save_path: str,
    ) -> None:
    pts = mesh["points"]
    tris = mesh["triangles"]
    bv = mesh["boundary_vertices"]
    bs = mesh["boundary_segments"]

    fig, ax = plt.subplots(figsize=(8, 8))
    tri_obj = mtri.Triangulation(pts[:, 0], pts[:, 1], tris)
    ax.triplot(tri_obj, linewidth=0.3, color="steelblue")

    has_dir = has_neu = False
    for ei, (i0, i1) in enumerate(bs):
        p0, p1 = bv[i0], bv[i1]
        if bc_type_func(ei) > 0.5:
            color, has_dir = "red", True
        else:
            color, has_neu = "green", True
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, linewidth=2.0)

    ax.set_aspect("equal")
    ax.set_title(f"Mesh: {domain_name} ({len(tris)} triangles)")

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color="steelblue", lw=0.5, label="Interior")]
    if has_dir:
        handles.append(Line2D([0], [0], color="red", lw=2, label="Dirichlet"))
    if has_neu:
        handles.append(Line2D([0], [0], color="green", lw=2, label="Neumann"))
    ax.legend(handles=handles, loc="upper right")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)