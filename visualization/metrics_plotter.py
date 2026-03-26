from __future__ import annotations
import os
from typing import Dict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

_C = {
    "pde": "#2563EB", "energy": "#0F172A", "energy_rel": "#6366F1",
    "rel_l2": "#059669", "dir": "#DC2626", "neu": "#059669",
    "grid": "#E2E8F0", "bg": "#FAFBFC", "text": "#334155",
}
_STROKE = [pe.withStroke(linewidth=2.5, foreground="white")]

def _style(ax, title="", xlabel="Эпоха", ylabel=""):
    ax.set_facecolor(_C["bg"])
    ax.set_title(title, fontsize=12, fontweight="600", color=_C["text"], pad=10)
    ax.set_xlabel(xlabel, fontsize=9, color=_C["text"])
    ax.set_ylabel(ylabel, fontsize=9, color=_C["text"])
    ax.tick_params(colors=_C["text"], labelsize=8)
    ax.grid(True, alpha=0.5, color=_C["grid"], linewidth=0.5)
    for s in ax.spines.values():
        s.set_color(_C["grid"]); s.set_linewidth(0.7)

def _ann(ax, xs, ys, color, fmt=".2e", dy=0):
    if ys and np.isfinite(ys[-1]):
        ax.annotate(f"{ys[-1]:{fmt}}", (xs[-1], ys[-1]),
                    textcoords="offset points", xytext=(8, dy),
                    fontsize=7, color=color, fontweight="600", va="center",
                    path_effects=_STROKE)

def _save(fig, path, dpi=180):
    try: fig.tight_layout(pad=1.5)
    except Exception: pass
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)

def plot_pretrain_metrics(h: Dict, domain: str, path: str) -> None:
    if len(h.get("epoch", [])) < 2:
        return
    ep = h["epoch"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.patch.set_facecolor("white")

    ax = axes[0, 0]
    loss = h.get("loss", [])
    pde = h.get("pde", [])
    ax.semilogy(ep, loss, color=_C["pde"], lw=2, label="Полная потеря")
    _ann(ax, ep, loss, _C["pde"], dy=8)
    if pde:
        ax.semilogy(ep, pde, color=_C["pde"], lw=1, alpha=0.4, ls="--", label="PDE")
        _ann(ax, ep, pde, _C["pde"], dy=-10)
    _style(ax, "Функция потерь", ylabel="Потери")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ee = h.get("energy", [])
    ax.semilogy(ep, ee, color=_C["energy"], lw=2.2)
    ax.fill_between(ep, ee, alpha=0.08, color=_C["energy"])
    _ann(ax, ep, ee, _C["energy"])
    _style(ax, r"Энергетическая ошибка $\|\nabla(u-v)\|^2$", ylabel="Ошибка")

    ax = axes[1, 0]
    rel = h.get("rel_err", [])
    if rel:
        ax.semilogy(ep, rel, color=_C["rel_l2"], lw=2.2)
        _ann(ax, ep, rel, _C["rel_l2"])
        _style(ax, r"Относительная $L_2$ ошибка", ylabel="Ошибка")

    ax = axes[1, 1]
    lr_vals = h.get("lr", [])
    if lr_vals:
        ax.semilogy(ep, lr_vals, color="#D946EF", lw=2.2, label="Learning Rate")
        _ann(ax, ep, lr_vals, "#D946EF", fmt=".2e")
    _style(ax, "Скорость обучения (LR)", ylabel="LR")
    ax.legend(fontsize=8)

    fig.suptitle(f"PINN Training Metrics ({domain})",
                 fontsize=14, fontweight="700", color=_C["text"], y=1.01)
    _save(fig, path)
