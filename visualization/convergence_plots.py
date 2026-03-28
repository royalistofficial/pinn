from __future__ import annotations

import os
from typing import Optional, Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch

from visualization.ntk_plots import PALETTE, _ax_style
from training.convergence_prediction import ConvergencePrediction

def plot_ntk_master_dashboard(
            prediction: ConvergencePrediction,
            epoch: int,
            actual_losses: Optional[Dict[str, List[float]]] = None,
            output_dir: str = "data/ntk_plots",
        ) -> str:
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(16, 10), facecolor="white")
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.25)

    components = []
    if prediction.pde: 
        components.append(("ДУЧП", prediction.pde, PALETTE["K_L"]))
    if prediction.dirichlet: 
        components.append(("Дирихле", prediction.dirichlet, PALETTE["dirichlet"]))
    if prediction.neumann: 
        components.append(("Нейман", prediction.neumann, PALETTE["neumann"]))

    ax_loss = fig.add_subplot(gs[0, :2])
    epochs_pred = prediction.predicted_epochs

    ax_loss.semilogy(epochs_pred, prediction.predicted_loss_pde, "o-", color=PALETTE["K_L"], markersize=5, linewidth=2, label="ДУЧП (прогноз)")
    ax_loss.semilogy(epochs_pred, prediction.predicted_loss_bc, "s-", color=PALETTE["dirichlet"], markersize=5, linewidth=2, label="Гран. условия (прогноз)")
    ax_loss.semilogy(epochs_pred, prediction.predicted_loss_total, "^-", color=PALETTE["K"], markersize=5, linewidth=2, label="Общий Loss (прогноз)")

    if actual_losses and "epochs" in actual_losses:
        actual_ep = actual_losses["epochs"]
        if "total" in actual_losses:
            ax_loss.semilogy(actual_ep, actual_losses["total"], "k--", alpha=0.5, label="Факт. Loss")

    _ax_style(ax_loss, title="Прогноз сходимости (Ожидаемое падение Loss)", xlabel="Эпохи", ylabel="Значение Loss")
    ax_loss.axvline(prediction.bottleneck_epochs, color=PALETTE["critical"], ls=":", lw=2, alpha=0.7, label="Ожид. сходимость 1%")
    ax_loss.legend(loc="upper right", fontsize=9)
    ax_loss.set_xlim(left=0)

    ax_text = fig.add_subplot(gs[0, 2])
    ax_text.axis("off")

    ax_text.set_title("Диагностика и Выводы", fontsize=12, fontweight="bold", color=PALETTE["text"])

    rect = FancyBboxPatch((0.05, 0.05), 0.9, 0.9, boxstyle="round,pad=0.05", facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"], linewidth=1, transform=ax_text.transAxes)
    ax_text.add_patch(rect)

    y_pos = 0.85
    health_color = PALETTE["good"] if prediction.health_score > 70 else (PALETTE["warning"] if prediction.health_score > 40 else PALETTE["critical"])

    ax_text.text(0.1, y_pos, f"Здоровье сети: {prediction.health_score:.0f}/100", fontsize=12, fontweight="bold", color=health_color, transform=ax_text.transAxes)
    y_pos -= 0.15

    ax_text.text(0.1, y_pos, "Узкое место (Bottleneck):", fontsize=10, fontweight="bold", transform=ax_text.transAxes)
    y_pos -= 0.08

    ax_text.text(0.15, y_pos, f"- {prediction.bottleneck_component} (~{prediction.bottleneck_epochs} эпох)", fontsize=10, color=PALETTE["critical"], fontweight="bold", transform=ax_text.transAxes)
    y_pos -= 0.15

    ax_text.text(0.1, y_pos, "Рекомендации:", fontsize=10, fontweight="bold", transform=ax_text.transAxes)
    y_pos -= 0.08
    for rec in prediction.recommendations[:4]:  

        clean_rec = rec.replace("⚠ ", "! ").replace("✓ ", "* ")
        ax_text.text(0.1, y_pos, clean_rec, fontsize=9, transform=ax_text.transAxes, wrap=True)
        y_pos -= 0.08

    ax_spec = fig.add_subplot(gs[1, 0])
    for name, metrics, color in components:
        eig = metrics.eigenvalues
        if len(eig) > 0:
            ax_spec.semilogy(np.arange(1, len(eig) + 1), eig, "o-", color=color, markersize=4, linewidth=1.5, label=name)
    _ax_style(ax_spec, title="Спектры NTK (Чем выше/положе - тем лучше)", xlabel="Индекс моды k", ylabel="Собственное число λ")
    ax_spec.legend(fontsize=8)
    ax_spec.set_xlim(left=1)

    ax_cond = fig.add_subplot(gs[1, 1])
    labels, vals, cols = [], [], []
    for name, metrics, color in components:
        labels.append(name)
        vals.append(np.log10(max(metrics.condition_number, 1)))
        cols.append(color)

    bars = ax_cond.bar(range(len(labels)), vals, color=cols, alpha=0.85)
    for bar, metrics in zip(bars, [m for _, m, _ in components]):
        ax_cond.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05, f"{metrics.condition_number:.1e}", ha="center", fontsize=9)

    ax_cond.set_xticks(range(len(labels)))
    ax_cond.set_xticklabels(labels, fontsize=10)
    _ax_style(ax_cond, title="Обусловленность κ (Меньше - лучше)", ylabel="log₁₀(κ)")

    ax_time = fig.add_subplot(gs[1, 2])
    t_vals = [m.t_epsilon for _, m, _ in components]
    bars_t = ax_time.bar(range(len(labels)), t_vals, color=cols, alpha=0.85)
    for bar, t in zip(bars_t, t_vals):
        ax_time.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05, f"{t:.0f}", ha="center", fontsize=9)

    ax_time.set_xticks(range(len(labels)))
    ax_time.set_xticklabels(labels, fontsize=10)
    _ax_style(ax_time, title="Эпох до 1% ошибки (Быстрее - лучше)", ylabel="Эпохи")

    fig.suptitle(f"NTK Master Dashboard | Эпоха {epoch}", fontsize=16, fontweight="bold", color="#2c3e50", y=1.02)

    out_path = os.path.join(output_dir, f"ntk_master_dashboard_epoch{epoch:04d}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return out_path

def plot_convergence_evolution(
            predictions_history: List[ConvergencePrediction],  
            output_dir: str = "data/ntk_plots",
        ) -> str:
    if len(predictions_history) < 2:
        return ""

    os.makedirs(output_dir, exist_ok=True)

    epochs = [p.epoch for p in predictions_history]

    fig = plt.figure(figsize=(18, 10), facecolor="white")
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)

    ax = fig.add_subplot(gs[0, 0])
    health_scores = [p.health_score for p in predictions_history]
    ax.plot(epochs, health_scores, "o-", color=PALETTE["good"], markersize=6, linewidth=2)
    ax.axhline(70, color=PALETTE["warning"], ls="--", lw=1.5, alpha=0.7)
    ax.axhline(40, color=PALETTE["critical"], ls="--", lw=1.5, alpha=0.7)
    ax.fill_between(epochs, health_scores, alpha=0.2, color=PALETTE["good"])
    _ax_style(ax, title="Эволюция оценки здоровья", xlabel="Эпоха", ylabel="Оценка")
    ax.set_ylim(0, 100)

    ax = fig.add_subplot(gs[0, 1])
    balance_scores = [p.balance_score for p in predictions_history]
    ax.plot(epochs, balance_scores, "s-", color=PALETTE["accent"], markersize=6, linewidth=2)
    ax.axhline(70, color=PALETTE["warning"], ls="--", lw=1.5, alpha=0.7)
    ax.fill_between(epochs, balance_scores, alpha=0.2, color=PALETTE["accent"])
    _ax_style(ax, title="Эволюция оценки баланса", xlabel="Эпоха", ylabel="Оценка")
    ax.set_ylim(0, 100)

    ax = fig.add_subplot(gs[0, 2])
    bottleneck_epochs = [p.bottleneck_epochs for p in predictions_history]
    ax.semilogy(epochs, bottleneck_epochs, "^-", color=PALETTE["critical"], markersize=6, linewidth=2)
    ax.fill_between(epochs, bottleneck_epochs, alpha=0.2, color=PALETTE["critical"])
    _ax_style(ax, title="Эволюция времени до сходимости", xlabel="Эпоха", ylabel="t_ε (эпохи)")

    ax = fig.add_subplot(gs[1, 0])
    kappa_pde = [p.pde.condition_number if p.pde else 1e10 for p in predictions_history]
    kappa_dir = [p.dirichlet.condition_number if p.dirichlet else 1e10 for p in predictions_history]
    kappa_neu = [p.neumann.condition_number if p.neumann else 1e10 for p in predictions_history]

    ax.semilogy(epochs, kappa_pde, "o-", color=PALETTE["K_L"], markersize=4, linewidth=1.5, label="ДУЧП")
    ax.semilogy(epochs, kappa_dir, "s-", color=PALETTE["dirichlet"], markersize=4, linewidth=1.5, label="Дирихле")
    ax.semilogy(epochs, kappa_neu, "v-", color=PALETTE["neumann"], markersize=4, linewidth=1.5, label="Нейман")
    _ax_style(ax, title="Эволюция чисел обусловленности", xlabel="Эпоха", ylabel="κ")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    t_pde = [p.pde.t_epsilon if p.pde else 0 for p in predictions_history]
    t_dir = [p.dirichlet.t_epsilon if p.dirichlet else 0 for p in predictions_history]
    t_neu = [p.neumann.t_epsilon if p.neumann else 0 for p in predictions_history]

    ax.semilogy(epochs, t_pde, "o-", color=PALETTE["K_L"], markersize=5, linewidth=2, label="ДУЧП")
    if any(t > 0 for t in t_dir):
        ax.semilogy(epochs, t_dir, "s-", color=PALETTE["dirichlet"], markersize=5, linewidth=2, label="Дирихле")
    if any(t > 0 for t in t_neu):
        ax.semilogy(epochs, t_neu, "v-", color=PALETTE["neumann"], markersize=5, linewidth=2, label="Нейман")
    _ax_style(ax, title="Эволюция t_ε по компонентам", xlabel="Эпоха", ylabel="t_ε (эпохи)")
    ax.legend(fontsize=9)

    ax = fig.add_subplot(gs[1, 2])
    rank_pde = [p.pde.effective_rank if p.pde else 0 for p in predictions_history]
    ax.plot(epochs, rank_pde, "o-", color=PALETTE["K_L"], markersize=5, linewidth=2)
    _ax_style(ax, title="Эволюция эффективного ранга ДУЧП", xlabel="Эпоха", ylabel="Ранг")

    fig.suptitle(
        f"Эволюция прогнозов сходимости NTK\n"
        f"Эпохи: {epochs[0]} → {epochs[-1]} | "
        f"Здоровье: {health_scores[0]:.0f} → {health_scores[-1]:.0f}",
        fontsize=14, fontweight="700", color=PALETTE["text"], y=1.01
    )

    out_path = os.path.join(output_dir, "ntk_convergence_evolution.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return out_path