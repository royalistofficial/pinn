from __future__ import annotations

import os
from typing import Optional, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Wedge, FancyBboxPatch

from networks.ntk_utils import (
    compute_jacobian,
    compute_pde_jacobian,
    compute_bc_jacobian,
    compute_ntk_from_jacobian,
    split_boundary_points,
    _compute_condition_number,
    _compute_effective_rank,
)

PALETTE = {
    "K": "#2563EB",       
    "K_L": "#DC2626",     
    "dirichlet": "#059669", 
    "neumann": "#D97706",   
    "grid": "#CBD5E1",      
    "bg": "#F8FAFC",        
    "text": "#1E293B",      
    "accent": "#7C3AED",    
    "good": "#10B981",      
    "warning": "#F59E0B",   
    "critical": "#EF4444",  
}

CMAP_K = "viridis"
CMAP_KL = "plasma"
CMAP_HEALTH = "RdYlGn"

def _ax_style(ax: plt.Axes, title: str = "", xlabel: str = "",
              ylabel: str = "", fontsize: int = 11) -> None:
    ax.set_facecolor(PALETTE["bg"])
    ax.set_title(title, fontsize=fontsize, fontweight="600",
                 color=PALETTE["text"], pad=8)
    ax.set_xlabel(xlabel, fontsize=9, color=PALETTE["text"])
    ax.set_ylabel(ylabel, fontsize=9, color=PALETTE["text"])
    ax.tick_params(colors=PALETTE["text"], labelsize=8)
    ax.grid(True, alpha=0.4, color=PALETTE["grid"], linewidth=0.5, zorder=0)
    for sp in ax.spines.values():
        sp.set_color(PALETTE["grid"])
        sp.set_linewidth(0.7)

def _matrix_panel(fig: plt.Figure, ax: plt.Axes, M: np.ndarray,
                  title: str, cmap: str, node_labels: bool = True,
                  label_step: int = 1, show_colorbar: bool = True,
                  square: bool = True) -> None:
    N = M.shape[0]
    if N == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
        ax.set_title(title, fontsize=10, fontweight="600")
        return

    vmax = np.abs(M).max()
    if vmax < 1e-30:
        vmax = 1e-8

    if M.min() < -1e-10 * vmax:
        norm = mcolors.TwoSlopeNorm(vmin=M.min(), vcenter=0, vmax=vmax)
        cmap_use = "RdBu_r"
    else:
        norm = mcolors.Normalize(vmin=0, vmax=vmax)
        cmap_use = cmap

    aspect = "equal" if square else "auto"
    im = ax.imshow(M, cmap=cmap_use, norm=norm, aspect=aspect, interpolation="nearest")

    for k in range(N + 1):
        ax.axhline(k - 0.5, color="white", linewidth=0.4, alpha=0.6)
        ax.axvline(k - 0.5, color="white", linewidth=0.4, alpha=0.6)

    ticks = np.arange(0, N, label_step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    if node_labels and N <= 50:
        ax.set_xticklabels([str(t) for t in ticks], rotation=90, fontsize=max(5, 8 - N // 8))
        ax.set_yticklabels([str(t) for t in ticks], fontsize=max(5, 8 - N // 8))
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    ax.set_xlabel("Node j", fontsize=9, color=PALETTE["text"])
    ax.set_ylabel("Node i", fontsize=9, color=PALETTE["text"])
    ax.set_title(title, fontsize=10, fontweight="600", color=PALETTE["text"], pad=8)

    if show_colorbar:
        cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
        cb.ax.tick_params(labelsize=7)
        cb.set_label("Value", fontsize=8)

def _subsample(X: torch.Tensor, n: int) -> torch.Tensor:
    N = len(X)
    if N <= n:
        return X
    idx = torch.linspace(0, N - 1, n, device=X.device).long()
    return X[idx]

def _compute_spectrum_stats(eigenvalues: np.ndarray) -> Dict[str, float]:
    eig_pos = eigenvalues[eigenvalues > 1e-10]
    if len(eig_pos) < 2:
        cond = float("inf")
    else:
        cond = float(eig_pos[0] / eig_pos[-1])

    if len(eig_pos) == 0:
        eff_rank = 0.0
    else:
        p = eig_pos / eig_pos.sum()
        eff_rank = float(np.exp(-np.sum(p * np.log(p + 1e-30))))

    return {"condition_number": cond, "effective_rank": eff_rank, "trace": float(eigenvalues.sum())}

def _order_xy(X_np: np.ndarray) -> np.ndarray:
    n_bins = max(1, int(np.sqrt(len(X_np))))
    x_bin = np.round(X_np[:, 0] * n_bins) / n_bins
    return np.lexsort((X_np[:, 1], x_bin))

def _get_health_status(value: float, thresholds: tuple) -> str:
    good_t, warn_t = thresholds
    if value >= good_t:
        return "good"
    elif value >= warn_t:
        return "warning"
    return "critical"

def plot_ntk_pde(
            model: nn.Module,
            epoch: int,
            X_interior: torch.Tensor,
            X_dirichlet: Optional[torch.Tensor] = None,
            X_neumann: Optional[torch.Tensor] = None,
            normals_neumann: Optional[torch.Tensor] = None,
            n_pts: int = 64,
            output_dir: str = "data/ntk_plots",
            node_order: str = "xy",
        ) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    device = next(model.parameters()).device

    X = _subsample(X_interior.to(device), n_pts)
    X_np = X.detach().cpu().numpy()
    N = len(X)

    print(f"[NTK PDE] Epoch {epoch}: Computing PDE Jacobian (N={N})...")
    J_L = compute_pde_jacobian(model, X)
    K_L = compute_ntk_from_jacobian(J_L).cpu().numpy()

    eig_KL = np.sort(np.linalg.eigvalsh(K_L))[::-1].clip(0)
    stats_KL = _compute_spectrum_stats(eig_KL)

    order_idx = _order_xy(X_np) if node_order == "xy" else np.arange(N)
    K_L_sorted = K_L[np.ix_(order_idx, order_idx)]

    bc_results = {}
    if X_dirichlet is not None and len(X_dirichlet) > 0:
        X_D = _subsample(X_dirichlet.to(device), n_pts // 4)
        if len(X_D) > 0:
            J_D = compute_bc_jacobian(model, X_D, bc_type="dirichlet")
            K_D = compute_ntk_from_jacobian(J_D).cpu().numpy()
            eig_D = np.sort(np.linalg.eigvalsh(K_D))[::-1].clip(0)
            bc_results["dirichlet"] = {
                "eigenvalues": eig_D,
                "condition_number": _compute_condition_number(eig_D),
                "effective_rank": _compute_effective_rank(eig_D),
            }

    if X_neumann is not None and len(X_neumann) > 0 and normals_neumann is not None:
        X_N = _subsample(X_neumann.to(device), n_pts // 4)
        normals_N = _subsample(normals_neumann.to(device), n_pts // 4)
        if len(X_N) > 0:
            J_N = compute_bc_jacobian(model, X_N, normals_N, "neumann")
            K_N = compute_ntk_from_jacobian(J_N).cpu().numpy()
            eig_N = np.sort(np.linalg.eigvalsh(K_N))[::-1].clip(0)
            bc_results["neumann"] = {
                "eigenvalues": eig_N,
                "condition_number": _compute_condition_number(eig_N),
                "effective_rank": _compute_effective_rank(eig_N),
            }

    fig = plt.figure(figsize=(20, 14), facecolor="white")
    gs = GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.30,
                  width_ratios=[1.2, 1.2, 1, 1])

    ax = fig.add_subplot(gs[0, 0:2])
    _matrix_panel(fig, ax, K_L_sorted,
                  title=r"$K_{\mathcal{L}}(x_i, x_j)$ — PDE NTK" + f"\nEpoch {epoch}, N={N}",
                  cmap=CMAP_KL, node_labels=N <= 50, label_step=max(1, N // 10),
                  square=True)

    ax = fig.add_subplot(gs[0, 2])
    sc = ax.scatter(X_np[:, 0], X_np[:, 1], c=np.arange(N), cmap="plasma",
                    s=50, edgecolors="k", linewidths=0.4, zorder=5)
    for i in range(min(N, 25)):
        ax.annotate(str(i), (X_np[i, 0], X_np[i, 1]),
                    textcoords="offset points", xytext=(3, 3), fontsize=6)
    cb = fig.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label("Index", fontsize=8)
    ax.set_aspect("equal")
    _ax_style(ax, title="Interior Points", xlabel="x", ylabel="y")

    ax = fig.add_subplot(gs[0, 3])
    _draw_health_panel(ax, stats_KL, bc_results)

    ax = fig.add_subplot(gs[1, 0])
    modes = np.arange(1, N + 1)
    ax.semilogy(modes, eig_KL, "o-", markersize=4, color=PALETTE["K_L"],
                linewidth=1.8, label=f"κ={stats_KL['condition_number']:.1e}")
    ax.fill_between(modes, eig_KL, alpha=0.2, color=PALETTE["K_L"])
    _ax_style(ax, title=r"Spectrum $K_{\mathcal{L}}$", xlabel="Mode k", ylabel="λ_k")
    ax.legend(fontsize=9)
    ax.set_xlim(left=1)

    ax = fig.add_subplot(gs[1, 1])
    ax.bar(np.arange(N), np.diag(K_L_sorted), color=PALETTE["K_L"], alpha=0.85,
           edgecolor="white", linewidth=0.4)
    _ax_style(ax, title=r"Diagonal $K_{\mathcal{L}}(x_i, x_i)$",
              xlabel="Node i", ylabel=r"$(K_{\mathcal{L}})_{ii}$")

    ax = fig.add_subplot(gs[1, 2])
    n_modes = min(25, N)
    m_idx = np.arange(1, n_modes + 1)
    rates = 1.0 - np.exp(-eig_KL[:n_modes].clip(0))
    ax.bar(m_idx, rates, color=PALETTE["K_L"], alpha=0.85, edgecolor="white")
    ax.axhline(0.5, color="grey", ls="--", lw=1.5, alpha=0.7, label="50%")
    ax.axhline(0.9, color=PALETTE["good"], ls="--", lw=1.5, alpha=0.7, label="90%")
    _ax_style(ax, title=r"Convergence: $1-e^{-\lambda_k}$", xlabel="Mode k", ylabel="Rate")
    ax.legend(fontsize=8)
    ax.set_xlim(0.5, n_modes + 0.5)
    ax.set_ylim(0, 1.05)

    ax = fig.add_subplot(gs[1, 3])
    if bc_results:
        if "dirichlet" in bc_results:
            ax.semilogy(np.arange(1, len(bc_results["dirichlet"]["eigenvalues"]) + 1),
                       bc_results["dirichlet"]["eigenvalues"], "^-", 
                       color=PALETTE["dirichlet"], markersize=4, linewidth=1.5, label="K_D")
        if "neumann" in bc_results:
            ax.semilogy(np.arange(1, len(bc_results["neumann"]["eigenvalues"]) + 1),
                       bc_results["neumann"]["eigenvalues"], "v-",
                       color=PALETTE["neumann"], markersize=4, linewidth=1.5, label="K_N")
        _ax_style(ax, title="BC NTK Spectra", xlabel="Mode k", ylabel="λ_k")
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "No BC data", ha="center", va="center", fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.suptitle(
        f"PDE NTK Analysis | Epoch {epoch}\n"
        f"κ = {stats_KL['condition_number']:.2e}, rank_eff = {stats_KL['effective_rank']:.1f}",
        fontsize=13, fontweight="700", color=PALETTE["text"], y=1.01
    )

    out_path = os.path.join(output_dir, f"ntk_pde_epoch{epoch:04d}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[NTK PDE] Saved → {out_path}")

    return {
        "K_L": K_L,
        "eigenvalues": eig_KL,
        "condition_number": stats_KL["condition_number"],
        "effective_rank": stats_KL["effective_rank"],
        "trace": stats_KL["trace"],
        "bc_results": bc_results,
    }

def _draw_health_panel(ax, stats_main: Dict, bc_results: Dict) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Health Indicators", fontsize=11, fontweight="600", color=PALETTE["text"])

    rect = FancyBboxPatch((0.2, 0.2), 9.6, 9.6, boxstyle="round,pad=0.1",
                          facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"], linewidth=1)
    ax.add_patch(rect)

    y_pos = 9.0
    line_height = 1.4

    ax.text(5, y_pos, "PDE NTK Health", ha="center", fontsize=10, fontweight="bold")
    y_pos -= line_height * 1.5

    kappa = stats_main["condition_number"]
    status = _get_health_status(-np.log10(kappa), (2, 0))  
    color = PALETTE[status]
    ax.text(1, y_pos, "κ(K_L):", fontsize=9, fontweight="500")
    ax.text(6, y_pos, f"{kappa:.1e}", fontsize=9, color=color, fontweight="bold")
    ax.plot(8.5, y_pos, "o", color=color, markersize=10)
    y_pos -= line_height

    rank = stats_main["effective_rank"]
    status = _get_health_status(rank / 50, (0.5, 0.2))  
    color = PALETTE[status]
    ax.text(1, y_pos, "rank_eff:", fontsize=9, fontweight="500")
    ax.text(6, y_pos, f"{rank:.1f}", fontsize=9, color=color, fontweight="bold")
    ax.plot(8.5, y_pos, "o", color=color, markersize=10)
    y_pos -= line_height

    if bc_results:
        ax.text(5, y_pos, "─── BC ───", ha="center", fontsize=8, color=PALETTE["text"])
        y_pos -= line_height

        if "dirichlet" in bc_results:
            kappa_D = bc_results["dirichlet"]["condition_number"]
            status = _get_health_status(-np.log10(kappa_D), (2, 0))
            color = PALETTE[status]
            ax.text(1, y_pos, "κ(K_D):", fontsize=9, fontweight="500")
            ax.text(6, y_pos, f"{kappa_D:.1e}", fontsize=9, color=color, fontweight="bold")
            ax.plot(8.5, y_pos, "o", color=color, markersize=10)
            y_pos -= line_height

        if "neumann" in bc_results:
            kappa_N = bc_results["neumann"]["condition_number"]
            status = _get_health_status(-np.log10(kappa_N), (2, 0))
            color = PALETTE[status]
            ax.text(1, y_pos, "κ(K_N):", fontsize=9, fontweight="500")
            ax.text(6, y_pos, f"{kappa_N:.1e}", fontsize=9, color=color, fontweight="bold")
            ax.plot(8.5, y_pos, "o", color=color, markersize=10)

    y_pos = 0.8
    ax.plot(1.5, y_pos, "o", color=PALETTE["good"], markersize=8)
    ax.text(2.2, y_pos, "Good", fontsize=8, va="center")
    ax.plot(4.5, y_pos, "o", color=PALETTE["warning"], markersize=8)
    ax.text(5.2, y_pos, "Warning", fontsize=8, va="center")
    ax.plot(7.5, y_pos, "o", color=PALETTE["critical"], markersize=8)
    ax.text(8.2, y_pos, "Critical", fontsize=8, va="center")

def plot_ntk_full(
            model: nn.Module,
            epoch: int,
            X_interior: torch.Tensor,
            X_boundary: Optional[torch.Tensor] = None,
            normals: Optional[torch.Tensor] = None,
            bc_mask: Optional[torch.Tensor] = None,
            n_interior: int = 48,
            n_boundary: int = 24,
            output_dir: str = "data/ntk_plots",
        ) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    device = next(model.parameters()).device

    X_in = _subsample(X_interior.to(device), n_interior)
    X_bd = _subsample(X_boundary.to(device), n_boundary) if X_boundary is not None else torch.empty(0, 2, device=device)

    n_in = len(X_in)
    n_bd = len(X_bd)

    print(f"[NTK Full] Epoch {epoch}: Interior={n_in}, Boundary={n_bd}")

    J_in = compute_jacobian(model, X_in)
    J_L_in = compute_pde_jacobian(model, X_in)

    J_all_list = [J_in]
    block_labels = ["Interior"]
    block_sizes = [n_in]
    block_colors = [PALETTE["K"]]

    all_points = [X_in.cpu().numpy()]
    point_labels = [f"Interior ({n_in})"]
    point_colors = [PALETTE["K"]]

    n_D, n_N = 0, 0
    xy_D, xy_N = None, None
    result_dirichlet, result_neumann = None, None

    if n_bd > 0 and bc_mask is not None:
        normals_sub = _subsample(normals.to(device), n_boundary) if normals is not None else None
        bc_mask_sub = _subsample(bc_mask.to(device), n_boundary)

        xy_D, xy_N, idx_D, idx_N = split_boundary_points(X_bd, bc_mask_sub)
        n_D, n_N = len(xy_D), len(xy_N)

        if n_D > 0:
            normals_D = normals_sub[idx_D] if normals_sub is not None else None
            J_D = compute_bc_jacobian(model, xy_D, normals_D, "dirichlet")
            J_all_list.append(J_D)
            block_labels.append("Dirichlet")
            block_sizes.append(n_D)
            block_colors.append(PALETTE["dirichlet"])
            all_points.append(xy_D.cpu().numpy())
            point_labels.append(f"Dirichlet ({n_D})")
            point_colors.append(PALETTE["dirichlet"])

            K_D = compute_ntk_from_jacobian(J_D).cpu().numpy()
            eig_D = np.sort(np.linalg.eigvalsh(K_D))[::-1].clip(0)
            stats_D = _compute_spectrum_stats(eig_D)
            result_dirichlet = {"eigenvalues": eig_D, **stats_D}

        if n_N > 0:
            normals_N = normals_sub[idx_N] if normals_sub is not None else None
            J_N = compute_bc_jacobian(model, xy_N, normals_N, "neumann")
            J_all_list.append(J_N)
            block_labels.append("Neumann")
            block_sizes.append(n_N)
            block_colors.append(PALETTE["neumann"])
            all_points.append(xy_N.cpu().numpy())
            point_labels.append(f"Neumann ({n_N})")
            point_colors.append(PALETTE["neumann"])

            K_N = compute_ntk_from_jacobian(J_N).cpu().numpy()
            eig_N = np.sort(np.linalg.eigvalsh(K_N))[::-1].clip(0)
            stats_N = _compute_spectrum_stats(eig_N)
            result_neumann = {"eigenvalues": eig_N, **stats_N}

    J_all = torch.cat(J_all_list, dim=0)
    K_full = (J_all @ J_all.T).cpu().numpy()
    K_L = (J_L_in @ J_L_in.T).cpu().numpy()

    eig_full = np.sort(np.linalg.eigvalsh(K_full))[::-1].clip(0)
    eig_KL = np.sort(np.linalg.eigvalsh(K_L))[::-1].clip(0)

    stats_full = _compute_spectrum_stats(eig_full)
    stats_KL = _compute_spectrum_stats(eig_KL)

    fig = plt.figure(figsize=(22, 16), facecolor="white")
    gs = GridSpec(3, 4, figure=fig, hspace=0.40, wspace=0.30,
                  width_ratios=[1.2, 1.2, 1, 1])

    ax00 = fig.add_subplot(gs[0, 0:2])
    _matrix_panel(fig, ax00, K_full,
                  title=f"Full NTK Matrix K\nEpoch {epoch}, N={len(J_all)}",
                  cmap="viridis", node_labels=False, square=True)

    cumsum = 0
    for size, label, color in zip(block_sizes, block_labels, block_colors):
        ax00.axhline(cumsum - 0.5, color="red", linewidth=2.5, alpha=0.9)
        ax00.axvline(cumsum - 0.5, color="red", linewidth=2.5, alpha=0.9)
        if size > 3:
            mid = cumsum + size // 2
            ax00.text(-5, mid, label, fontsize=9, va="center", rotation=90,
                      color=color, fontweight="bold")
        cumsum += size
    ax00.axhline(cumsum - 0.5, color="red", linewidth=2.5, alpha=0.9)
    ax00.axvline(cumsum - 0.5, color="red", linewidth=2.5, alpha=0.9)

    ax02 = fig.add_subplot(gs[0, 2])
    _matrix_panel(fig, ax02, K_L,
                  title=f"PDE NTK $K_{{\\mathcal{{L}}}}$\n(Interior, N={n_in})",
                  cmap=CMAP_KL, node_labels=n_in <= 40, square=True)

    ax03 = fig.add_subplot(gs[0, 3])
    for pts, label, color in zip(all_points, point_labels, point_colors):
        ax03.scatter(pts[:, 0], pts[:, 1], c=color, s=45,
                     label=label, edgecolors="k", linewidths=0.3, zorder=5)
    ax03.set_aspect("equal")
    ax03.legend(fontsize=8, loc="best")
    _ax_style(ax03, title="All Analysis Points", xlabel="x", ylabel="y")

    ax10 = fig.add_subplot(gs[1, 0])
    ax10.semilogy(np.arange(1, len(eig_full) + 1), eig_full, "o-",
                  color=PALETTE["K"], markersize=4, linewidth=1.8,
                  label=f"K (κ={stats_full['condition_number']:.1e})")
    ax10.fill_between(np.arange(1, len(eig_full) + 1), eig_full, alpha=0.15, color=PALETTE["K"])
    _ax_style(ax10, title="Full K Spectrum", xlabel="Mode k", ylabel="λ_k")
    ax10.legend(fontsize=9)
    ax10.set_xlim(left=1)

    ax11 = fig.add_subplot(gs[1, 1])
    ax11.semilogy(np.arange(1, len(eig_KL) + 1), eig_KL, "s-",
                  color=PALETTE["K_L"], markersize=4, linewidth=1.8,
                  label=f"K_L (κ={stats_KL['condition_number']:.1e})")
    ax11.fill_between(np.arange(1, len(eig_KL) + 1), eig_KL, alpha=0.15, color=PALETTE["K_L"])
    _ax_style(ax11, title=r"PDE NTK $K_{\mathcal{L}}$ Spectrum", xlabel="Mode k", ylabel="λ_k")
    ax11.legend(fontsize=9)
    ax11.set_xlim(left=1)

    ax12 = fig.add_subplot(gs[1, 2])
    ax12.semilogy(np.arange(1, len(eig_full) + 1), eig_full, "o-",
                  color=PALETTE["K"], markersize=3, linewidth=1.5, label="K", alpha=0.8)
    ax12.semilogy(np.arange(1, len(eig_KL) + 1), eig_KL, "s-",
                  color=PALETTE["K_L"], markersize=3, linewidth=1.5, label="K_L", alpha=0.8)
    _ax_style(ax12, title="Spectrum Comparison", xlabel="Mode k", ylabel="λ_k")
    ax12.legend(fontsize=9)
    ax12.set_xlim(left=1)

    ax13 = fig.add_subplot(gs[1, 3])
    labels = ["κ(K)", "κ(K_L)", "rank(K)", "rank(K_L)"]
    values = [np.log10(max(stats_full["condition_number"], 1)),
              np.log10(max(stats_KL["condition_number"], 1)),
              np.log10(max(stats_full["effective_rank"], 1)),
              np.log10(max(stats_KL["effective_rank"], 1))]
    raw = [stats_full["condition_number"], stats_KL["condition_number"],
           stats_full["effective_rank"], stats_KL["effective_rank"]]
    colors = [PALETTE["K"], PALETTE["K_L"], PALETTE["K"], PALETTE["K_L"]]

    bars = ax13.bar(range(4), values, color=colors, alpha=0.85, edgecolor="white")
    for bar, rv in zip(bars, raw):
        ax13.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                  f"{rv:.1e}" if rv > 999 else f"{rv:.1f}", ha="center", fontsize=7, rotation=45)
    ax13.set_xticks(range(4))
    ax13.set_xticklabels(labels, fontsize=9, rotation=20, ha="right")
    _ax_style(ax13, title="Statistics (log₁₀)", ylabel="log₁₀")

    ax20 = fig.add_subplot(gs[2, 0:2])
    diag = np.diag(K_full)
    cumsum = 0
    for size, label, color in zip(block_sizes, block_labels, block_colors):
        if size > 0:
            ax20.bar(np.arange(cumsum, cumsum + size), diag[cumsum:cumsum + size],
                     color=color, alpha=0.85, edgecolor="white", linewidth=0.3, label=label)
            if cumsum > 0:
                ax20.axvline(cumsum - 0.5, color="red", linewidth=2, alpha=0.8)
        cumsum += size
    ax20.legend(fontsize=9, loc="upper right")
    _ax_style(ax20, title="Diagonal of Full K (by blocks)", xlabel="Node i", ylabel=r"$K_{ii}$")

    ax21 = fig.add_subplot(gs[2, 2])
    n_modes = min(25, len(eig_full))
    rates_K = 1.0 - np.exp(-eig_full[:n_modes].clip(0))
    ax21.bar(range(1, n_modes + 1), rates_K, color=PALETTE["K"], alpha=0.85, edgecolor="white")
    ax21.axhline(0.5, color="grey", ls="--", lw=1.5, alpha=0.7)
    ax21.axhline(0.9, color=PALETTE["good"], ls="--", lw=1.5, alpha=0.7)
    _ax_style(ax21, title=r"K Convergence: $1-e^{-\lambda_k}$", xlabel="Mode k", ylabel="Rate")
    ax21.set_xlim(0.5, n_modes + 0.5)
    ax21.set_ylim(0, 1.05)

    ax22 = fig.add_subplot(gs[2, 3])
    n_modes_L = min(25, len(eig_KL))
    rates_KL = 1.0 - np.exp(-eig_KL[:n_modes_L].clip(0))
    ax22.bar(range(1, n_modes_L + 1), rates_KL, color=PALETTE["K_L"], alpha=0.85, edgecolor="white")
    ax22.axhline(0.5, color="grey", ls="--", lw=1.5, alpha=0.7)
    ax22.axhline(0.9, color=PALETTE["good"], ls="--", lw=1.5, alpha=0.7)
    _ax_style(ax22, title=r"$K_{\mathcal{L}}$ Convergence", xlabel="Mode k", ylabel="Rate")
    ax22.set_xlim(0.5, n_modes_L + 0.5)
    ax22.set_ylim(0, 1.05)

    blocks_str = ", ".join([f"{l}({s})" for l, s in zip(block_labels, block_sizes)])
    fig.suptitle(
        f"NTK Analysis — Full Matrix | Epoch {epoch}\n"
        f"Blocks: {blocks_str} | κ(K)={stats_full['condition_number']:.1e}",
        fontsize=13, fontweight="700", color=PALETTE["text"], y=1.01
    )

    out_path = os.path.join(output_dir, f"ntk_full_epoch{epoch:04d}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[NTK Full] Saved → {out_path}")

    return {
        "K_full": K_full,
        "K_L": K_L,
        "eigenvalues_K": eig_full,
        "eigenvalues_KL": eig_KL,
        "condition_number_K": stats_full["condition_number"],
        "condition_number_KL": stats_KL["condition_number"],
        "effective_rank_K": stats_full["effective_rank"],
        "effective_rank_KL": stats_KL["effective_rank"],
        "trace_K": stats_full["trace"],
        "trace_KL": stats_KL["trace"],
        "dirichlet": result_dirichlet,
        "neumann": result_neumann,
        "n_interior": n_in,
        "n_dirichlet": n_D,
        "n_neumann": n_N,
    }

def plot_ntk_spectrum_analysis(
            model: nn.Module,
            epoch: int,
            X_interior: torch.Tensor,
            X_boundary: Optional[torch.Tensor] = None,
            normals: Optional[torch.Tensor] = None,
            bc_mask: Optional[torch.Tensor] = None,
            n_interior: int = 64,
            n_boundary: int = 32,
            output_dir: str = "data/ntk_plots",
        ) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    device = next(model.parameters()).device

    X_in = _subsample(X_interior.to(device), n_interior)
    n_in = len(X_in)

    print(f"[NTK Spectrum] Epoch {epoch}: Analyzing spectra...")

    J_in = compute_jacobian(model, X_in)
    J_L_in = compute_pde_jacobian(model, X_in)

    K_in = compute_ntk_from_jacobian(J_in).cpu().numpy()
    K_L = compute_ntk_from_jacobian(J_L_in).cpu().numpy()

    eig_K_in = np.sort(np.linalg.eigvalsh(K_in))[::-1].clip(0)
    eig_KL = np.sort(np.linalg.eigvalsh(K_L))[::-1].clip(0)

    stats_K_in = _compute_spectrum_stats(eig_K_in)
    stats_KL = _compute_spectrum_stats(eig_KL)

    J_all_list = [J_in]
    block_sizes = [n_in]

    eig_D, eig_N = None, None
    stats_D, stats_N = None, None
    n_D, n_N = 0, 0

    if X_boundary is not None and len(X_boundary) > 0 and bc_mask is not None:
        X_bd = _subsample(X_boundary.to(device), n_boundary)
        normals_sub = _subsample(normals.to(device), n_boundary) if normals is not None else None
        bc_mask_sub = _subsample(bc_mask.to(device), n_boundary)

        xy_D, xy_N, idx_D, idx_N = split_boundary_points(X_bd, bc_mask_sub)
        n_D, n_N = len(xy_D), len(xy_N)

        if n_D > 0:
            normals_D = normals_sub[idx_D] if normals_sub is not None else None
            J_D = compute_bc_jacobian(model, xy_D, normals_D, "dirichlet")
            J_all_list.append(J_D)
            block_sizes.append(n_D)

            K_D = compute_ntk_from_jacobian(J_D).cpu().numpy()
            eig_D = np.sort(np.linalg.eigvalsh(K_D))[::-1].clip(0)
            stats_D = _compute_spectrum_stats(eig_D)

        if n_N > 0:
            normals_N = normals_sub[idx_N] if normals_sub is not None else None
            J_N = compute_bc_jacobian(model, xy_N, normals_N, "neumann")
            J_all_list.append(J_N)
            block_sizes.append(n_N)

            K_N = compute_ntk_from_jacobian(J_N).cpu().numpy()
            eig_N = np.sort(np.linalg.eigvalsh(K_N))[::-1].clip(0)
            stats_N = _compute_spectrum_stats(eig_N)

    J_all = torch.cat(J_all_list, dim=0)
    K_full = (J_all @ J_all.T).cpu().numpy()
    eig_full = np.sort(np.linalg.eigvalsh(K_full))[::-1].clip(0)
    stats_full = _compute_spectrum_stats(eig_full)

    fig = plt.figure(figsize=(20, 14), facecolor="white")
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.30)

    ax00 = fig.add_subplot(gs[0, 0])
    ax00.semilogy(np.arange(1, len(eig_full) + 1), eig_full, "o-",
                  color=PALETTE["K"], markersize=4, linewidth=1.8,
                  label=f"κ={stats_full['condition_number']:.1e}")
    ax00.fill_between(np.arange(1, len(eig_full) + 1), eig_full, alpha=0.15, color=PALETTE["K"])
    _ax_style(ax00, title="Full K Spectrum", xlabel="Mode k", ylabel="λ_k")
    ax00.legend(fontsize=9)
    ax00.set_xlim(left=1)

    ax01 = fig.add_subplot(gs[0, 1])
    ax01.semilogy(np.arange(1, n_in + 1), eig_KL, "s-",
                  color=PALETTE["K_L"], markersize=4, linewidth=1.8,
                  label=f"κ={stats_KL['condition_number']:.1e}")
    ax01.fill_between(np.arange(1, n_in + 1), eig_KL, alpha=0.15, color=PALETTE["K_L"])
    _ax_style(ax01, title=r"$K_{\mathcal{L}}$ (PDE) Spectrum", xlabel="Mode k", ylabel="λ_k")
    ax01.legend(fontsize=9)
    ax01.set_xlim(left=1)

    ax02 = fig.add_subplot(gs[0, 2])
    if eig_D is not None and len(eig_D) > 0:
        ax02.semilogy(np.arange(1, len(eig_D) + 1), eig_D, "^-",
                      color=PALETTE["dirichlet"], markersize=4, linewidth=1.8,
                      label=f"κ={stats_D['condition_number']:.1e}")
        ax02.fill_between(np.arange(1, len(eig_D) + 1), eig_D, alpha=0.15, color=PALETTE["dirichlet"])
        _ax_style(ax02, title="K_D (Dirichlet)", xlabel="Mode k", ylabel="λ_k")
        ax02.legend(fontsize=9)
        ax02.set_xlim(left=1)
    else:
        ax02.text(0.5, 0.5, "No Dirichlet points", ha="center", va="center", fontsize=12)
        ax02.set_xlim(0, 1)
        ax02.set_ylim(0, 1)

    ax03 = fig.add_subplot(gs[0, 3])
    if eig_N is not None and len(eig_N) > 0:
        ax03.semilogy(np.arange(1, len(eig_N) + 1), eig_N, "v-",
                      color=PALETTE["neumann"], markersize=4, linewidth=1.8,
                      label=f"κ={stats_N['condition_number']:.1e}")
        ax03.fill_between(np.arange(1, len(eig_N) + 1), eig_N, alpha=0.15, color=PALETTE["neumann"])
        _ax_style(ax03, title="K_N (Neumann)", xlabel="Mode k", ylabel="λ_k")
        ax03.legend(fontsize=9)
        ax03.set_xlim(left=1)
    else:
        ax03.text(0.5, 0.5, "No Neumann points", ha="center", va="center", fontsize=12)
        ax03.set_xlim(0, 1)
        ax03.set_ylim(0, 1)

    ax10 = fig.add_subplot(gs[1, 0:2])
    ax10.semilogy(np.arange(1, len(eig_full) + 1), eig_full, "o-",
                  color=PALETTE["K"], markersize=3, linewidth=1.5, label="K", alpha=0.8)
    ax10.semilogy(np.arange(1, n_in + 1), eig_KL, "s-",
                  color=PALETTE["K_L"], markersize=3, linewidth=1.5, label="K_L", alpha=0.8)
    if eig_D is not None:
        ax10.semilogy(np.arange(1, len(eig_D) + 1), eig_D, "^-",
                      color=PALETTE["dirichlet"], markersize=3, linewidth=1.5, label="K_D", alpha=0.8)
    if eig_N is not None:
        ax10.semilogy(np.arange(1, len(eig_N) + 1), eig_N, "v-",
                      color=PALETTE["neumann"], markersize=3, linewidth=1.5, label="K_N", alpha=0.8)
    _ax_style(ax10, title="All Spectra Comparison", xlabel="Mode k", ylabel="λ_k")
    ax10.legend(fontsize=10, loc="upper right")
    ax10.set_xlim(left=1)

    ax12 = fig.add_subplot(gs[1, 2])
    labels_k = ["κ(K)", "κ(K_L)"]
    vals_k = [stats_full["condition_number"], stats_KL["condition_number"]]
    colors_k = [PALETTE["K"], PALETTE["K_L"]]
    if stats_D is not None:
        labels_k.append("κ(K_D)")
        vals_k.append(stats_D["condition_number"])
        colors_k.append(PALETTE["dirichlet"])
    if stats_N is not None:
        labels_k.append("κ(K_N)")
        vals_k.append(stats_N["condition_number"])
        colors_k.append(PALETTE["neumann"])

    bars = ax12.bar(range(len(labels_k)), [np.log10(max(v, 1)) for v in vals_k],
                    color=colors_k, alpha=0.85, edgecolor="white")
    for bar, rv in zip(bars, vals_k):
        ax12.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                  f"{rv:.1e}" if rv > 999 else f"{rv:.1f}", ha="center", fontsize=7, rotation=45)
    ax12.set_xticks(range(len(labels_k)))
    ax12.set_xticklabels(labels_k, fontsize=9, rotation=20, ha="right")
    _ax_style(ax12, title="Condition Numbers (log₁₀)", ylabel="log₁₀(κ)")

    ax13 = fig.add_subplot(gs[1, 3])
    labels_r = ["rank(K)", "rank(K_L)"]
    vals_r = [stats_full["effective_rank"], stats_KL["effective_rank"]]
    colors_r = [PALETTE["K"], PALETTE["K_L"]]
    if stats_D is not None:
        labels_r.append("rank(K_D)")
        vals_r.append(stats_D["effective_rank"])
        colors_r.append(PALETTE["dirichlet"])
    if stats_N is not None:
        labels_r.append("rank(K_N)")
        vals_r.append(stats_N["effective_rank"])
        colors_r.append(PALETTE["neumann"])

    bars = ax13.bar(range(len(labels_r)), vals_r, color=colors_r, alpha=0.85, edgecolor="white")
    for bar, rv in zip(bars, vals_r):
        ax13.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                  f"{rv:.1f}", ha="center", fontsize=9)
    ax13.set_xticks(range(len(labels_r)))
    ax13.set_xticklabels(labels_r, fontsize=9, rotation=20, ha="right")
    _ax_style(ax13, title="Effective Ranks", ylabel="Rank")

    ax20 = fig.add_subplot(gs[2, 0])
    n_modes = min(25, len(eig_full))
    rates_K = 1.0 - np.exp(-eig_full[:n_modes].clip(0))
    ax20.bar(range(1, n_modes + 1), rates_K, color=PALETTE["K"], alpha=0.85, edgecolor="white")
    ax20.axhline(0.5, color="grey", ls="--", lw=1.5, alpha=0.7)
    ax20.axhline(0.9, color=PALETTE["good"], ls="--", lw=1.5, alpha=0.7)
    _ax_style(ax20, title="K Convergence", xlabel="Mode k", ylabel="Rate")
    ax20.set_xlim(0.5, n_modes + 0.5)
    ax20.set_ylim(0, 1.05)

    ax21 = fig.add_subplot(gs[2, 1])
    n_modes_L = min(25, n_in)
    rates_KL = 1.0 - np.exp(-eig_KL[:n_modes_L].clip(0))
    ax21.bar(range(1, n_modes_L + 1), rates_KL, color=PALETTE["K_L"], alpha=0.85, edgecolor="white")
    ax21.axhline(0.5, color="grey", ls="--", lw=1.5, alpha=0.7)
    ax21.axhline(0.9, color=PALETTE["good"], ls="--", lw=1.5, alpha=0.7)
    _ax_style(ax21, title="K_L Convergence", xlabel="Mode k", ylabel="Rate")
    ax21.set_xlim(0.5, n_modes_L + 0.5)
    ax21.set_ylim(0, 1.05)

    ax22 = fig.add_subplot(gs[2, 2])
    if eig_D is not None and len(eig_D) > 0:
        n_modes_D = min(25, len(eig_D))
        rates_D = 1.0 - np.exp(-eig_D[:n_modes_D].clip(0))
        ax22.bar(range(1, n_modes_D + 1), rates_D, color=PALETTE["dirichlet"], alpha=0.85, edgecolor="white")
        ax22.axhline(0.5, color="grey", ls="--", lw=1.5, alpha=0.7)
        ax22.axhline(0.9, color=PALETTE["good"], ls="--", lw=1.5, alpha=0.7)
        _ax_style(ax22, title="K_D Convergence", xlabel="Mode k", ylabel="Rate")
        ax22.set_xlim(0.5, n_modes_D + 0.5)
        ax22.set_ylim(0, 1.05)
    else:
        ax22.text(0.5, 0.5, "No Dirichlet data", ha="center", va="center", fontsize=12)

    ax23 = fig.add_subplot(gs[2, 3])
    if eig_N is not None and len(eig_N) > 0:
        n_modes_N = min(25, len(eig_N))
        rates_N = 1.0 - np.exp(-eig_N[:n_modes_N].clip(0))
        ax23.bar(range(1, n_modes_N + 1), rates_N, color=PALETTE["neumann"], alpha=0.85, edgecolor="white")
        ax23.axhline(0.5, color="grey", ls="--", lw=1.5, alpha=0.7)
        ax23.axhline(0.9, color=PALETTE["good"], ls="--", lw=1.5, alpha=0.7)
        _ax_style(ax23, title="K_N Convergence", xlabel="Mode k", ylabel="Rate")
        ax23.set_xlim(0.5, n_modes_N + 0.5)
        ax23.set_ylim(0, 1.05)
    else:
        ax23.text(0.5, 0.5, "No Neumann data", ha="center", va="center", fontsize=12)

    fig.suptitle(
        f"NTK Spectrum Analysis | Epoch {epoch}\n"
        f"K: κ={stats_full['condition_number']:.1e}, rank={stats_full['effective_rank']:.1f} | "
        f"K_L: κ={stats_KL['condition_number']:.1e}, rank={stats_KL['effective_rank']:.1f}",
        fontsize=13, fontweight="700", color=PALETTE["text"], y=1.01
    )

    out_path = os.path.join(output_dir, f"ntk_spectrum_analysis_epoch{epoch:04d}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[NTK Spectrum] Saved → {out_path}")

    return {
        "full_K": {"eigenvalues": eig_full, **stats_full},
        "pde_KL": {"eigenvalues": eig_KL, **stats_KL},
        "interior_K": {"eigenvalues": eig_K_in, **stats_K_in},
        "dirichlet": {"eigenvalues": eig_D, **stats_D} if stats_D else None,
        "neumann": {"eigenvalues": eig_N, **stats_N} if stats_N else None,
    }

def plot_spectrum_evolution(
            spectra_history: List[Dict],
            epochs: List[int],
            output_dir: str = "data/ntk_plots",
        ) -> None:
    if len(spectra_history) < 2:
        return

    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(22, 12), facecolor="white")
    gs = GridSpec(2, 5, figure=fig, hspace=0.35, wspace=0.30)

    cmap_ev = plt.cm.viridis(np.linspace(0, 1, len(spectra_history)))

    ax = fig.add_subplot(gs[0, 0])
    for i, (sp, ep) in enumerate(zip(spectra_history, epochs)):
        eig = sp.get("eig_K", sp.get("full_K", {}).get("eigenvalues", []))
        if len(eig) > 0:
            ax.semilogy(np.arange(1, len(eig) + 1), eig, "o-", markersize=2,
                        color=cmap_ev[i], linewidth=1, label=f"ep {ep}")
    _ax_style(ax, title="K Spectrum Evolution", xlabel="Mode k", ylabel="λ_k")
    ax.legend(fontsize=6, ncol=2)

    ax = fig.add_subplot(gs[0, 1])
    for i, (sp, ep) in enumerate(zip(spectra_history, epochs)):
        eig = sp.get("eig_KL", sp.get("pde_KL", {}).get("eigenvalues", []))
        if len(eig) > 0:
            ax.semilogy(np.arange(1, len(eig) + 1), eig, "s-", markersize=2,
                        color=cmap_ev[i], linewidth=1, label=f"ep {ep}")
    _ax_style(ax, title=r"$K_{\mathcal{L}}$ Evolution", xlabel="Mode k", ylabel="λ_k")
    ax.legend(fontsize=6, ncol=2)

    ax = fig.add_subplot(gs[0, 2])
    has_D = any(sp.get("dirichlet") is not None for sp in spectra_history)
    if has_D:
        for i, (sp, ep) in enumerate(zip(spectra_history, epochs)):
            if sp.get("dirichlet") and sp["dirichlet"].get("eigenvalues") is not None:
                eig = sp["dirichlet"]["eigenvalues"]
                ax.semilogy(np.arange(1, len(eig) + 1), eig, "^-", markersize=2,
                            color=cmap_ev[i], linewidth=1, label=f"ep {ep}")
        _ax_style(ax, title="K_D Evolution", xlabel="Mode k", ylabel="λ_k")
        ax.legend(fontsize=6, ncol=2)
    else:
        ax.text(0.5, 0.5, "No Dirichlet data", ha="center", va="center", fontsize=11)

    ax = fig.add_subplot(gs[0, 3])
    has_N = any(sp.get("neumann") is not None for sp in spectra_history)
    if has_N:
        for i, (sp, ep) in enumerate(zip(spectra_history, epochs)):
            if sp.get("neumann") and sp["neumann"].get("eigenvalues") is not None:
                eig = sp["neumann"]["eigenvalues"]
                ax.semilogy(np.arange(1, len(eig) + 1), eig, "v-", markersize=2,
                            color=cmap_ev[i], linewidth=1, label=f"ep {ep}")
        _ax_style(ax, title="K_N Evolution", xlabel="Mode k", ylabel="λ_k")
        ax.legend(fontsize=6, ncol=2)
    else:
        ax.text(0.5, 0.5, "No Neumann data", ha="center", va="center", fontsize=11)

    ax = fig.add_subplot(gs[0, 4])
    kappas_K = [s.get("kappa_K", s.get("full_K", {}).get("condition_number", float("nan")))
                for s in spectra_history]
    ax.semilogy(epochs, kappas_K, "o-", color=PALETTE["K"],
                markersize=6, linewidth=1.8, label="κ(K)")

    kappas_KL = [s.get("kappa_KL", s.get("pde_KL", {}).get("condition_number", float("nan")))
                 for s in spectra_history]
    ax.semilogy(epochs, kappas_KL, "s-", color=PALETTE["K_L"],
                markersize=6, linewidth=1.8, label=r"κ($K_{\mathcal{L}}$)")

    if has_D:
        kappas_D = [s["dirichlet"]["condition_number"] if s.get("dirichlet") else float("nan")
                    for s in spectra_history]
        ax.semilogy(epochs, kappas_D, "^-", color=PALETTE["dirichlet"],
                    markersize=6, linewidth=1.8, label="κ(K_D)")

    if has_N:
        kappas_N = [s["neumann"]["condition_number"] if s.get("neumann") else float("nan")
                    for s in spectra_history]
        ax.semilogy(epochs, kappas_N, "v-", color=PALETTE["neumann"],
                    markersize=6, linewidth=1.8, label="κ(K_N)")

    _ax_style(ax, title="Condition Number Evolution", xlabel="Epoch", ylabel="κ")
    ax.legend(fontsize=9)

    ax = fig.add_subplot(gs[1, 0])
    ranks_K = [s.get("rank_K", s.get("full_K", {}).get("effective_rank", float("nan")))
               for s in spectra_history]
    ax.plot(epochs, ranks_K, "o-", color=PALETTE["K"], markersize=6, linewidth=1.8, label="rank(K)")

    ranks_KL = [s.get("rank_KL", s.get("pde_KL", {}).get("effective_rank", float("nan")))
                for s in spectra_history]
    ax.plot(epochs, ranks_KL, "s-", color=PALETTE["K_L"], markersize=6, linewidth=1.8, label=r"rank($K_{\mathcal{L}}$)")

    if has_D:
        ranks_D = [s["dirichlet"]["effective_rank"] if s.get("dirichlet") else float("nan")
                   for s in spectra_history]
        ax.plot(epochs, ranks_D, "^-", color=PALETTE["dirichlet"], markersize=6, linewidth=1.8, label="rank(K_D)")

    if has_N:
        ranks_N = [s["neumann"]["effective_rank"] if s.get("neumann") else float("nan")
                   for s in spectra_history]
        ax.plot(epochs, ranks_N, "v-", color=PALETTE["neumann"], markersize=6, linewidth=1.8, label="rank(K_N)")

    _ax_style(ax, title="Effective Rank Evolution", xlabel="Epoch", ylabel="Rank")
    ax.legend(fontsize=9)

    ax = fig.add_subplot(gs[1, 1])
    traces_K = [s.get("trace_K", s.get("full_K", {}).get("trace", float("nan")))
                for s in spectra_history]
    ax.plot(epochs, traces_K, "o-", color=PALETTE["K"], markersize=6, linewidth=1.8, label="tr(K)")

    traces_KL = [s.get("trace_KL", s.get("pde_KL", {}).get("trace", float("nan")))
                 for s in spectra_history]
    ax.plot(epochs, traces_KL, "s-", color=PALETTE["K_L"], markersize=6, linewidth=1.8, label=r"tr($K_{\mathcal{L}}$)")
    _ax_style(ax, title="Trace Evolution", xlabel="Epoch", ylabel="Trace")
    ax.legend(fontsize=9)

    ax = fig.add_subplot(gs[1, 2])
    for mode_idx in [0, 4, 9]:
        values = []
        for sp in spectra_history:
            eig = sp.get("eig_K", sp.get("full_K", {}).get("eigenvalues", []))
            values.append(eig[mode_idx] if len(eig) > mode_idx else float("nan"))
        ax.semilogy(epochs, values, "o-", markersize=5, linewidth=1.5, label=f"λ_{mode_idx+1}")
    _ax_style(ax, title="Top K Modes Evolution", xlabel="Epoch", ylabel="λ")
    ax.legend(fontsize=9)

    ax = fig.add_subplot(gs[1, 3])
    for mode_idx in [0, 4, 9]:
        values = []
        for sp in spectra_history:
            eig = sp.get("eig_KL", sp.get("pde_KL", {}).get("eigenvalues", []))
            values.append(eig[mode_idx] if len(eig) > mode_idx else float("nan"))
        ax.semilogy(epochs, values, "s-", markersize=5, linewidth=1.5, label=f"λ_{mode_idx+1}")
    _ax_style(ax, title="Top K_L Modes Evolution", xlabel="Epoch", ylabel="λ")
    ax.legend(fontsize=9)

    ax = fig.add_subplot(gs[1, 4])
    ratio = [k_kl / k_k if (k_k and k_k > 0 and k_kl and k_kl > 0) else float("nan")
             for k_k, k_kl in zip(kappas_K, kappas_KL)]
    ax.semilogy(epochs, ratio, "o-", color=PALETTE["accent"], markersize=6, linewidth=1.8)
    ax.axhline(1.0, color="grey", ls="--", lw=1.5, alpha=0.7)
    _ax_style(ax, title=r"$\kappa(K_{\mathcal{L}}) / \kappa(K)$", xlabel="Epoch", ylabel="Ratio")

    fig.suptitle("NTK Spectrum Evolution During Training",
                 fontsize=14, fontweight="700", color=PALETTE["text"], y=1.01)
    fig.tight_layout()

    out_path = os.path.join(output_dir, "ntk_spectrum_evolution.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[NTK] Saved spectrum evolution → {out_path}")
