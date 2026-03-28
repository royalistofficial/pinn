from __future__ import annotations
import os
from typing import Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.patheffects as pe

COLORS = {
    "loss": "#2563EB",
    "pde": "#3B82F6",
    "energy": "#0F172A",
    "energy_rel": "#6366F1",
    "rel_l2": "#059669",
    "dir": "#DC2626",
    "neu": "#059669",
    "lr": "#D946EF",
    "w_pde": "#3B82F6",       
    "w_dirichlet": "#DC2626", 
    "w_neumann": "#059669",   
    "l2_pred": "#8B5CF6",     
    "energy_pred": "#F59E0B", 
    "grid": "#E2E8F0",
    "bg": "#FAFBFC",
    "text": "#334155",
}

_STROKE = [pe.withStroke(linewidth=2.5, foreground="white")]

def _ax_style(ax, title: str = "", xlabel: str = "Эпоха", ylabel: str = ""):
    ax.set_facecolor(COLORS["bg"])
    ax.set_title(title, fontsize=12, fontweight="600", color=COLORS["text"], pad=10)
    ax.set_xlabel(xlabel, fontsize=9, color=COLORS["text"])
    ax.set_ylabel(ylabel, fontsize=9, color=COLORS["text"])
    ax.tick_params(colors=COLORS["text"], labelsize=8)
    ax.grid(True, alpha=0.5, color=COLORS["grid"], linewidth=0.5)
    for sp in ax.spines.values():
        sp.set_color(COLORS["grid"])
        sp.set_linewidth(0.7)

def _annotate_last(ax, xs, ys, color, fmt=".2e", dy=0):
    if ys and np.isfinite(ys[-1]):
        ax.annotate(
            f"{ys[-1]:{fmt}}",
            (xs[-1], ys[-1]),
            textcoords="offset points",
            xytext=(8, dy),
            fontsize=7,
            color=color,
            fontweight="600",
            va="center",
            path_effects=_STROKE,
        )

def _save_fig(fig, path, dpi=180):
    try:
        fig.tight_layout(pad=1.5)
    except Exception:
        pass
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)

def plot_training_metrics(history: Dict, domain: str, path: str) -> None:
    if len(history.get("epoch", [])) < 2:
        return

    epochs = history["epoch"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.patch.set_facecolor("white")

    ax = axes[0, 0]
    loss = history.get("loss", [])
    pde = history.get("pde", [])
    ax.semilogy(epochs, loss, color=COLORS["loss"], lw=2, label="Общие потери")
    _annotate_last(ax, epochs, loss, COLORS["loss"], dy=8)
    _ax_style(ax, "Функция потерь", ylabel="Потери")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    energy = history.get("energy", [])
    if energy:
        ax.semilogy(epochs, energy, color=COLORS["energy"], lw=2.2, label="Энергетическая ошибка")
        ax.fill_between(epochs, energy, alpha=0.08, color=COLORS["energy"])
        _annotate_last(ax, epochs, energy, COLORS["energy"])

    _ax_style(ax, r"Энергетическая ошибка $\|\nabla(u-v)\|^2$", ylabel="Ошибка")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    rel = history.get("rel_err", [])
    if rel:
        ax.semilogy(epochs, rel, color=COLORS["rel_l2"], lw=2.2, label=r"Относительная ошибка $L_2$")
        _annotate_last(ax, epochs, rel, COLORS["rel_l2"])

    _ax_style(ax, r"Относительная ошибка $L_2$", ylabel="Ошибка")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    w_pde = history.get("w_pde", [])
    w_dirichlet = history.get("w_dirichlet", [])
    w_neumann = history.get("w_neumann", [])

    if w_pde:
        ax.plot(epochs, w_pde, color=COLORS["w_pde"], lw=2, label=r"$w_{ДУЧП}$", marker="o", 
                markersize=3, markevery=max(1, len(epochs)//20))
        _annotate_last(ax, epochs, w_pde, COLORS["w_pde"], fmt=".2f", dy=8)
    if w_dirichlet:
        ax.plot(epochs, w_dirichlet, color=COLORS["w_dirichlet"], lw=2, label=r"$w_{Дирихле}$", 
                marker="s", markersize=3, markevery=max(1, len(epochs)//20))
        _annotate_last(ax, epochs, w_dirichlet, COLORS["w_dirichlet"], fmt=".2f", dy=-8)
    if w_neumann:
        ax.plot(epochs, w_neumann, color=COLORS["w_neumann"], lw=2, label=r"$w_{Нейман}$", 
                marker="^", markersize=3, markevery=max(1, len(epochs)//20))
        _annotate_last(ax, epochs, w_neumann, COLORS["w_neumann"], fmt=".2f", dy=8)

    if w_pde or w_dirichlet or w_neumann:
        ax.set_ylim(bottom=0, top=max(
            max(w_pde) if w_pde else 1,
            max(w_dirichlet) if w_dirichlet else 1,
            max(w_neumann) if w_neumann else 1
        ) * 1.2)

    _ax_style(ax, "Вклады весов функции потерь", ylabel="Вес")
    ax.legend(fontsize=8)

    if w_pde and w_dirichlet:
        weights_text = f"Текущие: ДУЧП={w_pde[-1]:.2f}, Дир={w_dirichlet[-1]:.2f}"
        if w_neumann:
            weights_text += f", Нейм={w_neumann[-1]:.2f}"
        ax.text(0.02, 0.98, weights_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', color=COLORS["text"],
                bbox=dict(boxstyle='round', facecolor=COLORS["bg"], alpha=0.8))

    fig.suptitle(
        f"Метрики обучения PINN ({domain})",
        fontsize=14, fontweight="700", color=COLORS["text"], y=1.01
    )
    _save_fig(fig, path)

def plot_solution_fields(
        epoch: int,
        tri: mtri.Triangulation,
        v: np.ndarray,
        u_exact: np.ndarray,
        abs_error: np.ndarray,
        energy_density: np.ndarray,
        save_path: str,
    ) -> None:
    import matplotlib.colors as mcolors

    def _tripcolor_panel(ax, tri, values, title, cmap, clip_zero=False):
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
        plt.colorbar(tc, ax=ax, shrink=0.82)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    panels = [
        (v, f"Решение v (Эпоха {epoch})", "viridis", False),
        (u_exact, "Точное решение u", "viridis", False),
        (abs_error, "Абсолютная ошибка |u - v|", "turbo", False),
        (energy_density, r"Энергетическая ошибка $|\nabla e|^2$", "turbo", True),
    ]

    for ax, (vals, title, cmap, cz) in zip(axes.flat, panels):
        _tripcolor_panel(ax, tri, vals, title, cmap, clip_zero=cz)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
