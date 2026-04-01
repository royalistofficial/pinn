from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, Tuple, Any

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
    grad_abs_err = np.sqrt(np.clip(((gn - eg) ** 2).sum(-1), 0, None))

    return FieldValues(
        v=vn,
        u_exact=ue,
        abs_error=np.abs(vn - ue),
        energy_density=grad_abs_err,
    )

def plot_mesh(
        mesh: Dict[str, np.ndarray],
        domain_name: str,
        bc_type_func,
        save_path: str,
        quad: Any = None,  
    ) -> None:
    pts = mesh["points"]
    tris = mesh["triangles"]
    bv = mesh["boundary_vertices"]
    bs = mesh["boundary_segments"]

    fig, ax = plt.subplots(figsize=(8, 8))
    tri_obj = mtri.Triangulation(pts[:, 0], pts[:, 1], tris)

    ax.triplot(tri_obj, linewidth=0.3, color="steelblue", alpha=0.5)

    has_dir = has_neu = False
    for ei, (i0, i1) in enumerate(bs):
        p0, p1 = bv[i0], bv[i1]
        if bc_type_func(ei) > 0.5:
            color, has_dir = "red", True
        else:
            color, has_neu = "green", True
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, linewidth=2.0)

    ax.set_aspect("equal")
    title = f"Сетка: {domain_name} ({len(tris)} треугольников)"

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color="steelblue", lw=0.5, label="Внутренняя область")]
    if has_dir:
        handles.append(Line2D([0], [0], color="red", lw=2, label="Граница Дирихле"))
    if has_neu:
        handles.append(Line2D([0], [0], color="green", lw=2, label="Граница Неймана"))

    if quad is not None:
        xy_in = quad.xy_in.detach().cpu().numpy()
        xy_bd = quad.xy_bd.detach().cpu().numpy()
        bc_mask = quad.bc_mask.detach().cpu().numpy().flatten()

        mask_dir = bc_mask > 0.5
        mask_neu = bc_mask <= 0.5

        xy_dir = xy_bd[mask_dir]
        xy_neu = xy_bd[mask_neu]

        ax.scatter(xy_in[:, 0], xy_in[:, 1], color="darkorange", s=3, zorder=3)

        if len(xy_dir) > 0:
            ax.scatter(xy_dir[:, 0], xy_dir[:, 1], color="red", s=12, zorder=4, marker="o", edgecolors="black", linewidths=0.3)
        if len(xy_neu) > 0:
            ax.scatter(xy_neu[:, 0], xy_neu[:, 1], color="green", s=12, zorder=4, marker="s", edgecolors="black", linewidths=0.3)

        title += f"\nТочки коллокации: {len(xy_in)} (внутр.), {len(xy_bd)} (гран.)"
        handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='darkorange', markersize=5, label=f'Внутр. точки ({len(xy_in)})'))

        if len(xy_dir) > 0:
            handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markeredgecolor='black', markeredgewidth=0.3, markersize=7, label=f'Точки Дирихле ({len(xy_dir)})'))
        if len(xy_neu) > 0:
            handles.append(Line2D([0], [0], marker='s', color='w', markerfacecolor='green', markeredgecolor='black', markeredgewidth=0.3, markersize=7, label=f'Точки Неймана ({len(xy_neu)})'))

    ax.set_title(title, fontsize=13, pad=15)
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=10)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)