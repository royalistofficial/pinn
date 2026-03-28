from __future__ import annotations

import os
from typing import Optional, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch

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

RUSSIAN_LABELS = {
    "spectrum": "Спектр NTK",
    "mode": "Мода k",
    "eigenvalue": "Собственное число λₖ",
    "condition_number": "Число обусловленности κ",
    "effective_rank": "Эффективный ранг",
    "epoch": "Эпоха",
    "pde": "ДУЧП (K_ℒ)",
    "dirichlet": "Дирихле (K_D)",
    "neumann": "Нейман (K_N)",
    "solution": "Решение (K)",
    "comparison": "Сравнение спектров",
    "statistics": "Статистика",
    "evolution": "Эволюция",
    "health_score": "Оценка здоровья",
    "balance_score": "Оценка баланса",
    "trace": "След (энергия)",
    "interior_points": "Внутренние точки",
    "boundary_points": "Граничные точки",
    "bottleneck": "Узкое место",
    "convergence": "Сходимость",
    "loss": "Функция потерь",
    "log_scale": "логарифмическая шкала",
}

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

def plot_ntk_combined(
            model: nn.Module,
            epoch: int,
            X_interior: torch.Tensor,
            X_boundary: Optional[torch.Tensor] = None,
            normals: Optional[torch.Tensor] = None,
            bc_mask: Optional[torch.Tensor] = None,
            n_interior: int = 64,
            n_boundary: int = 32,
            output_dir: str = "data/ntk_plots",
            learning_rate: float = 1e-3,
        ) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    device = next(model.parameters()).device

    X_in = _subsample(X_interior.to(device), n_interior)
    n_in = len(X_in)

    print(f"[NTK] Эпоха {epoch}: Анализ NTK (внутренних={n_interior}, граничных={n_boundary})...")

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

    fig = plt.figure(figsize=(18, 12), facecolor="white")
    gs = GridSpec(2, 3, figure=fig, hspace=0.30, wspace=0.25)

    ax = fig.add_subplot(gs[0, 0])
    ax.semilogy(np.arange(1, len(eig_full) + 1), eig_full, "o-",
                color=PALETTE["K"], markersize=3, linewidth=1.5, 
                label=f"K (κ={stats_full['condition_number']:.1e})", alpha=0.8)
    ax.semilogy(np.arange(1, n_in + 1), eig_KL, "s-",
                color=PALETTE["K_L"], markersize=3, linewidth=1.5, 
                label=f"K_ℒ (κ={stats_KL['condition_number']:.1e})", alpha=0.8)
    if eig_D is not None:
        ax.semilogy(np.arange(1, len(eig_D) + 1), eig_D, "^-",
                    color=PALETTE["dirichlet"], markersize=3, linewidth=1.5, 
                    label=f"K_D", alpha=0.8)
    if eig_N is not None:
        ax.semilogy(np.arange(1, len(eig_N) + 1), eig_N, "v-",
                    color=PALETTE["neumann"], markersize=3, linewidth=1.5, 
                    label=f"K_N", alpha=0.8)
    _ax_style(ax, title="Сравнение спектров NTK", 
              xlabel="Мода k", ylabel="Собственное число λₖ")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(left=1)

    ax = fig.add_subplot(gs[0, 1])
    ax.semilogy(np.arange(1, n_in + 1), eig_KL, "s-",
                color=PALETTE["K_L"], markersize=4, linewidth=1.8,
                label=f"K_ℒ")
    ax.fill_between(np.arange(1, n_in + 1), eig_KL, alpha=0.15, color=PALETTE["K_L"])
    _ax_style(ax, title="Спектр PDE NTK (K_ℒ)", 
              xlabel="Мода k", ylabel="Собственное число λₖ")
    ax.legend(fontsize=9)
    ax.set_xlim(left=1)

    ax = fig.add_subplot(gs[0, 2])
    labels_k = ["κ(K)", "κ(K_ℒ)"]
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

    bars = ax.bar(range(len(labels_k)), [np.log10(max(v, 1)) for v in vals_k],
                  color=colors_k, alpha=0.85, edgecolor="white")
    for bar, rv in zip(bars, vals_k):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{rv:.1e}" if rv > 999 else f"{rv:.1f}", ha="center", fontsize=7, rotation=45)
    ax.set_xticks(range(len(labels_k)))
    ax.set_xticklabels(labels_k, fontsize=9, rotation=20, ha="right")
    _ax_style(ax, title="Числа обусловленности (log₁₀)", ylabel="log₁₀(κ)")

    ax = fig.add_subplot(gs[1, 0])
    labels_r = ["rank(K)", "rank(K_ℒ)"]
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

    bars = ax.bar(range(len(labels_r)), vals_r, color=colors_r, alpha=0.85, edgecolor="white")
    for bar, rv in zip(bars, vals_r):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{rv:.1f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(labels_r)))
    ax.set_xticklabels(labels_r, fontsize=9, rotation=20, ha="right")
    _ax_style(ax, title="Эффективные ранги", ylabel="Ранг")

    ax = fig.add_subplot(gs[1, 1])
    trace_values = [stats_full["trace"], stats_KL["trace"]]
    trace_labels = ["K", "K_ℒ"]
    trace_colors = [PALETTE["K"], PALETTE["K_L"]]

    if stats_D is not None:
        trace_values.append(stats_D["trace"])
        trace_labels.append("K_D")
        trace_colors.append(PALETTE["dirichlet"])
    if stats_N is not None:
        trace_values.append(stats_N["trace"])
        trace_labels.append("K_N")
        trace_colors.append(PALETTE["neumann"])

    total_trace = sum(trace_values)
    fractions = [t / total_trace for t in trace_values]

    bars = ax.bar(range(len(fractions)), fractions, color=trace_colors, alpha=0.85, edgecolor="white")
    for bar, frac in zip(bars, fractions):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                f"{frac*100:.1f}%", ha="center", fontsize=8)
    ax.set_xticks(range(len(trace_labels)))
    ax.set_xticklabels(trace_labels, fontsize=9, rotation=20, ha="right")
    ax.set_ylim(0, 1.1)
    _ax_style(ax, title="Распределение энергии (след)", ylabel="Доля")

    ax = fig.add_subplot(gs[1, 2])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Сводка анализа", fontsize=11, fontweight="600", color=PALETTE["text"])

    rect = FancyBboxPatch((0.2, 0.2), 9.6, 9.6, boxstyle="round,pad=0.1",
                          facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"], linewidth=1)
    ax.add_patch(rect)

    y_pos = 9.0
    line_height = 0.9

    ax.text(5, y_pos, f"Эпоха: {epoch}", ha="center", fontsize=11, fontweight="bold")
    y_pos -= line_height * 1.5

    ax.text(1, y_pos, "Точки:", fontsize=9, fontweight="500")
    y_pos -= line_height
    ax.text(1.5, y_pos, f"• Внутренние: {n_in}", fontsize=8)
    y_pos -= line_height * 0.7
    ax.text(1.5, y_pos, f"• Дирихле: {n_D}", fontsize=8)
    y_pos -= line_height * 0.7
    ax.text(1.5, y_pos, f"• Нейман: {n_N}", fontsize=8)
    y_pos -= line_height * 1.2

    ax.text(1, y_pos, "K_ℒ (PDE):", fontsize=9, fontweight="500")
    y_pos -= line_height
    ax.text(1.5, y_pos, f"κ = {stats_KL['condition_number']:.2e}", fontsize=8)
    y_pos -= line_height * 0.7
    ax.text(1.5, y_pos, f"rank = {stats_KL['effective_rank']:.1f}", fontsize=8)
    y_pos -= line_height * 1.2

    if stats_D is not None:
        ax.text(1, y_pos, "K_D (Дирихле):", fontsize=9, fontweight="500")
        y_pos -= line_height
        ax.text(1.5, y_pos, f"κ = {stats_D['condition_number']:.2e}", fontsize=8)
        y_pos -= line_height * 1.2

    if stats_N is not None:
        ax.text(1, y_pos, "K_N (Нейман):", fontsize=9, fontweight="500")
        y_pos -= line_height
        ax.text(1.5, y_pos, f"κ = {stats_N['condition_number']:.2e}", fontsize=8)
        y_pos -= line_height * 1.2

    health_score = 100.0
    kappa_ratio = stats_KL["condition_number"] / stats_full["condition_number"] if stats_full["condition_number"] > 0 else float("inf")
    if kappa_ratio > 100:
        health_score -= 30
    elif kappa_ratio > 10:
        health_score -= 15
    elif kappa_ratio > 5:
        health_score -= 5

    rank_ratio = stats_KL["effective_rank"] / stats_full["effective_rank"] if stats_full["effective_rank"] > 0 else 0.0
    if rank_ratio < 0.5:
        health_score -= 25
    elif rank_ratio < 0.7:
        health_score -= 10

    health_score = max(0, min(100, health_score))
    health_color = PALETTE["good"] if health_score > 70 else (PALETTE["warning"] if health_score > 40 else PALETTE["critical"])

    ax.text(1, y_pos, "Оценка здоровья:", fontsize=9, fontweight="500")
    ax.text(6, y_pos, f"{health_score:.0f}/100", fontsize=11, fontweight="bold", color=health_color)

    fig.suptitle(
        f"NTK Анализ | Эпоха {epoch}\n"
        f"κ(K)={stats_full['condition_number']:.1e}, κ(K_ℒ)={stats_KL['condition_number']:.1e}, "
        f"rank(K_ℒ)={stats_KL['effective_rank']:.1f}",
        fontsize=13, fontweight="700", color=PALETTE["text"], y=1.01
    )

    out_path = os.path.join(output_dir, f"ntk_analysis_epoch{epoch:04d}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[NTK] Сохранено → {out_path}")

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
        "dirichlet": {"eigenvalues": eig_D, **stats_D} if stats_D else None,
        "neumann": {"eigenvalues": eig_N, **stats_N} if stats_N else None,
        "n_interior": n_in,
        "n_dirichlet": n_D,
        "n_neumann": n_N,
    }

def plot_ntk_evolution(
            spectra_history: List[Dict],
            epochs: List[int],
            output_dir: str = "data/ntk_plots",
        ) -> None:
    if len(spectra_history) < 2:
        return

    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(20, 10), facecolor="white")
    gs = GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.30)

    cmap_ev = plt.cm.viridis(np.linspace(0, 1, len(spectra_history)))

    ax = fig.add_subplot(gs[0, 0])
    for i, (sp, ep) in enumerate(zip(spectra_history, epochs)):
        eig = sp.get("eig_K", sp.get("full_K", {}).get("eigenvalues", []))
        if len(eig) > 0:
            ax.semilogy(np.arange(1, len(eig) + 1), eig, "o-", markersize=2,
                        color=cmap_ev[i], linewidth=1, label=f"эп. {ep}")
    _ax_style(ax, title="Эволюция спектра K", xlabel="Мода k", ylabel="Собственное число λₖ")
    ax.legend(fontsize=6, ncol=2)

    ax = fig.add_subplot(gs[0, 1])
    for i, (sp, ep) in enumerate(zip(spectra_history, epochs)):
        eig = sp.get("eig_KL", sp.get("pde_KL", {}).get("eigenvalues", []))
        if len(eig) > 0:
            ax.semilogy(np.arange(1, len(eig) + 1), eig, "s-", markersize=2,
                        color=cmap_ev[i], linewidth=1, label=f"эп. {ep}")
    _ax_style(ax, title="Эволюция спектра K_ℒ", xlabel="Мода k", ylabel="Собственное число λₖ")
    ax.legend(fontsize=6, ncol=2)

    ax = fig.add_subplot(gs[0, 2])
    has_D = any(sp.get("dirichlet") is not None for sp in spectra_history)
    if has_D:
        for i, (sp, ep) in enumerate(zip(spectra_history, epochs)):
            if sp.get("dirichlet") and sp["dirichlet"].get("eigenvalues") is not None:
                eig = sp["dirichlet"]["eigenvalues"]
                ax.semilogy(np.arange(1, len(eig) + 1), eig, "^-", markersize=2,
                            color=cmap_ev[i], linewidth=1, label=f"эп. {ep}")
        _ax_style(ax, title="Эволюция K_D", xlabel="Мода k", ylabel="Собственное число λₖ")
        ax.legend(fontsize=6, ncol=2)
    else:
        ax.text(0.5, 0.5, "Нет данных Дирихле", ha="center", va="center", fontsize=11)

    ax = fig.add_subplot(gs[0, 3])
    has_N = any(sp.get("neumann") is not None for sp in spectra_history)
    if has_N:
        for i, (sp, ep) in enumerate(zip(spectra_history, epochs)):
            if sp.get("neumann") and sp["neumann"].get("eigenvalues") is not None:
                eig = sp["neumann"]["eigenvalues"]
                ax.semilogy(np.arange(1, len(eig) + 1), eig, "v-", markersize=2,
                            color=cmap_ev[i], linewidth=1, label=f"эп. {ep}")
        _ax_style(ax, title="Эволюция K_N", xlabel="Мода k", ylabel="Собственное число λₖ")
        ax.legend(fontsize=6, ncol=2)
    else:
        ax.text(0.5, 0.5, "Нет данных Неймана", ha="center", va="center", fontsize=11)

    ax = fig.add_subplot(gs[1, 0])
    kappas_K = [s.get("kappa_K", s.get("full_K", {}).get("condition_number", float("nan")))
                for s in spectra_history]
    ax.semilogy(epochs, kappas_K, "o-", color=PALETTE["K"],
                markersize=6, linewidth=1.8, label="κ(K)")

    kappas_KL = [s.get("kappa_KL", s.get("pde_KL", {}).get("condition_number", float("nan")))
                 for s in spectra_history]
    ax.semilogy(epochs, kappas_KL, "s-", color=PALETTE["K_L"],
                markersize=6, linewidth=1.8, label="κ(K_ℒ)")

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

    _ax_style(ax, title="Эволюция чисел обусловленности", xlabel="Эпоха", ylabel="κ")
    ax.legend(fontsize=9)

    ax = fig.add_subplot(gs[1, 1])
    ranks_K = [s.get("rank_K", s.get("full_K", {}).get("effective_rank", float("nan")))
               for s in spectra_history]
    ax.plot(epochs, ranks_K, "o-", color=PALETTE["K"], markersize=6, linewidth=1.8, label="rank(K)")

    ranks_KL = [s.get("rank_KL", s.get("pde_KL", {}).get("effective_rank", float("nan")))
                for s in spectra_history]
    ax.plot(epochs, ranks_KL, "s-", color=PALETTE["K_L"], markersize=6, linewidth=1.8, label="rank(K_ℒ)")

    if has_D:
        ranks_D = [s["dirichlet"]["effective_rank"] if s.get("dirichlet") else float("nan")
                   for s in spectra_history]
        ax.plot(epochs, ranks_D, "^-", color=PALETTE["dirichlet"], markersize=6, linewidth=1.8, label="rank(K_D)")

    if has_N:
        ranks_N = [s["neumann"]["effective_rank"] if s.get("neumann") else float("nan")
                   for s in spectra_history]
        ax.plot(epochs, ranks_N, "v-", color=PALETTE["neumann"], markersize=6, linewidth=1.8, label="rank(K_N)")

    _ax_style(ax, title="Эволюция эффективных рангов", xlabel="Эпоха", ylabel="Ранг")
    ax.legend(fontsize=9)

    ax = fig.add_subplot(gs[1, 2])
    traces_K = [s.get("trace_K", s.get("full_K", {}).get("trace", float("nan")))
                for s in spectra_history]
    ax.semilogy(epochs, traces_K, "o-", color=PALETTE["K"], markersize=6, linewidth=1.8, label="trace(K)")

    traces_KL = [s.get("trace_KL", s.get("pde_KL", {}).get("trace", float("nan")))
                 for s in spectra_history]
    ax.semilogy(epochs, traces_KL, "s-", color=PALETTE["K_L"], markersize=6, linewidth=1.8, label="trace(K_ℒ)")

    _ax_style(ax, title="Эволюция следов (энергии)", xlabel="Эпоха", ylabel="След")
    ax.legend(fontsize=9)

    ax = fig.add_subplot(gs[1, 3])
    kappa_ratios = []
    rank_ratios = []
    for s in spectra_history:
        k_K = s.get("kappa_K", s.get("full_K", {}).get("condition_number", 1))
        k_KL = s.get("kappa_KL", s.get("pde_KL", {}).get("condition_number", 1))
        r_K = s.get("rank_K", s.get("full_K", {}).get("effective_rank", 1))
        r_KL = s.get("rank_KL", s.get("pde_KL", {}).get("effective_rank", 1))

        kappa_ratios.append(k_KL / k_K if k_K > 0 and k_K < float("inf") else float("nan"))
        rank_ratios.append(r_KL / r_K if r_K > 0 else float("nan"))

    ax.plot(epochs, kappa_ratios, "o-", color=PALETTE["critical"], markersize=6, linewidth=1.8, label="κ(K_ℒ)/κ(K)")
    ax.plot(epochs, rank_ratios, "s-", color=PALETTE["good"], markersize=6, linewidth=1.8, label="rank(K_ℒ)/rank(K)")
    _ax_style(ax, title="Отношения K_ℒ/K", xlabel="Эпоха", ylabel="Отношение")
    ax.legend(fontsize=9)

    fig.suptitle(
        f"Эволюция NTK анализа\n"
        f"Эпохи: {epochs[0]} → {epochs[-1]}",
        fontsize=14, fontweight="700", color=PALETTE["text"], y=1.01
    )

    out_path = os.path.join(output_dir, "ntk_evolution.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[NTK Evolution] Сохранено → {out_path}")

def plot_ntk_pde(*args, **kwargs):
    return plot_ntk_combined(*args, **kwargs)

def plot_ntk_full(*args, **kwargs):
    return plot_ntk_combined(*args, **kwargs)

def plot_ntk_spectrum_analysis(*args, **kwargs):
    return plot_ntk_combined(*args, **kwargs)

def plot_spectrum_evolution(*args, **kwargs):
    return plot_ntk_evolution(*args, **kwargs)
