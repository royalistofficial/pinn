from __future__ import annotations
import os
from typing import Dict, Callable
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.colors as mcolors

def _tripcolor_panel(ax, tri, values, title, cmap, fig, clip_zero=False):
    valid = ~np.isnan(values)
    if not np.any(valid):
        values = np.zeros_like(values)
        vmin, vmax = 0.0, 1e-8
    elif clip_zero:
        values = np.clip(values, 0.0, None)
        vmin, vmax = 0.0, np.nanmax(values)
    else:
        vmin, vmax = np.nanmin(values), np.nanmax(values)
    if np.isnan(vmax) or vmax <= vmin:
        vmax = vmin + 1e-8
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    safe_values = np.where(np.isnan(values), vmin, values)
    shading = "flat" if clip_zero else "gouraud"
    tc = ax.tripcolor(tri, safe_values, shading=shading, cmap=cmap, norm=norm)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12)
    fig.colorbar(tc, ax=ax, shrink=0.82)

def plot_training_fields(epoch, tri, v, u_exact, abs_error, energy_density, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    panels = [
        (v, f"v (Epoch {epoch})", "viridis", False),
        (u_exact, "Exact Solution u", "viridis", False),
        (abs_error, "Absolute Error |u - v|", "turbo", False),
        (energy_density, r"True Energy Error $|\nabla e|^2$", "turbo", False),
    ]
    for ax, (vals, title, cmap, cz) in zip(axes.flat, panels):
        _tripcolor_panel(ax, tri, vals, title, cmap, fig, clip_zero=cz)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_mesh(mesh, domain_name, bc_type_func, save_path):
    pts, tris = mesh["points"], mesh["triangles"]
    bv, bs = mesh["boundary_vertices"], mesh["boundary_segments"]
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
    handles = [Line2D([0], [0], color="steelblue", lw=0.5, label="interior")]
    if has_dir: handles.append(Line2D([0], [0], color="red", lw=2, label="Dirichlet"))
    if has_neu: handles.append(Line2D([0], [0], color="green", lw=2, label="Neumann"))
    ax.legend(handles=handles, loc="upper right")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
