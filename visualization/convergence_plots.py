from __future__ import annotations

import os
from typing import Optional, Dict, Any, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, Rectangle

from visualization.ntk_plots import PALETTE, _ax_style

def plot_component_comparison(
        pde_metrics: Dict[str, Any],
        dirichlet_metrics: Optional[Dict[str, Any]] = None,
        neumann_metrics: Optional[Dict[str, Any]] = None,
        solution_metrics: Optional[Dict[str, Any]] = None,
        epoch: int = 0,
        learning_rate: float = 1e-3,
        output_dir: str = "data/ntk_plots",
    ) -> str:
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(16, 10), facecolor="white")
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)

    components = []
    labels = []
    colors = []

    if pde_metrics is not None:
        components.append(("ДУЧП", pde_metrics, PALETTE["K_L"]))
        labels.append("K_ℒ (ДУЧП)")
        colors.append(PALETTE["K_L"])

    if dirichlet_metrics is not None:
        components.append(("Дирихле", dirichlet_metrics, PALETTE["dirichlet"]))
        labels.append("K_D (Дирихле)")
        colors.append(PALETTE["dirichlet"])

    if neumann_metrics is not None:
        components.append(("Нейман", neumann_metrics, PALETTE["neumann"]))
        labels.append("K_N (Нейман)")
        colors.append(PALETTE["neumann"])

    if solution_metrics is not None:
        components.append(("Решение", solution_metrics, PALETTE["K"]))
        labels.append("K (Решение)")
        colors.append(PALETTE["K"])

    ax = fig.add_subplot(gs[0, 0])
    for name, metrics, color in components:
        eig = metrics.get("eigenvalues", metrics.get("eigenvalues_KL", []))
        if len(eig) > 0:
            ax.semilogy(np.arange(1, len(eig) + 1), eig, "o-", 
                       markersize=3, linewidth=1.8, color=color, label=name, alpha=0.85)
    _ax_style(ax, title="Сравнение спектров", xlabel="Мода k", ylabel="Собственное число λₖ")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(left=1)

    ax = fig.add_subplot(gs[0, 1])
    kappa_values = []
    kappa_labels = []
    kappa_colors = []

    for name, metrics, color in components:
        kappa = metrics.get("condition_number", metrics.get("condition_number_KL", 1e10))
        kappa_values.append(np.log10(max(kappa, 1)))
        kappa_labels.append(name)
        kappa_colors.append(color)

    bars = ax.bar(range(len(kappa_values)), kappa_values, color=kappa_colors, alpha=0.85, edgecolor="white")
    for bar, kv in zip(bars, kappa_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
               f"10^{kv:.0f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(kappa_labels)))
    ax.set_xticklabels(kappa_labels, fontsize=9, rotation=20, ha="right")
    _ax_style(ax, title="Числа обусловленности (log₁₀)", ylabel="log₁₀(κ)")

    ax = fig.add_subplot(gs[0, 2])
    trace_values = []
    trace_labels = []
    trace_colors = []

    for name, metrics, color in components:
        eig = metrics.get("eigenvalues", [])
        if len(eig) > 0:
            trace_values.append(eig.sum())
            trace_labels.append(name)
            trace_colors.append(color)

    if trace_values:
        total_trace = sum(trace_values)
        fractions = [t / total_trace for t in trace_values]
        bars = ax.bar(range(len(fractions)), fractions, color=trace_colors, alpha=0.85, edgecolor="white")
        for bar, frac in zip(bars, fractions):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                   f"{frac*100:.1f}%", ha="center", fontsize=8)
        ax.set_xticks(range(len(trace_labels)))
        ax.set_xticklabels(trace_labels, fontsize=9, rotation=20, ha="right")
        ax.set_ylim(0, 1.1)
    _ax_style(ax, title="Распределение энергии", ylabel="Доля")

    ax = fig.add_subplot(gs[1, 0])
    tau_values = []
    tau_labels = []
    tau_colors = []

    for name, metrics, color in components:
        eig = metrics.get("eigenvalues", [])
        if len(eig) > 0 and eig[-1] > 1e-12:
            tau = 1.0 / (2 * eig[-1] * learning_rate)
            tau_values.append(tau)
            tau_labels.append(name)
            tau_colors.append(color)

    if tau_values:
        bars = ax.bar(range(len(tau_values)), tau_values, color=tau_colors, alpha=0.85, edgecolor="white")
        for bar, tv in zip(bars, tau_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                   f"{tv:.0f}", ha="center", fontsize=8, rotation=45)
        ax.set_xticks(range(len(tau_labels)))
        ax.set_xticklabels(tau_labels, fontsize=9, rotation=20, ha="right")
    _ax_style(ax, title="Характерное время τ (эпохи)", ylabel="τ")

    ax = fig.add_subplot(gs[1, 1])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Анализ узких мест", fontsize=11, fontweight="600", color=PALETTE["text"])

    rect = FancyBboxPatch((0.2, 0.2), 9.6, 9.6, boxstyle="round,pad=0.1",
                          facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"], linewidth=1)
    ax.add_patch(rect)

    y_pos = 9.0
    max_time = 0
    bottleneck_name = "Неизвестно"

    for name, metrics, color in components:
        eig = metrics.get("eigenvalues", [])
        if len(eig) > 0 and eig[-1] > 1e-12:
            t_eps = -np.log(0.01) / (2 * eig[-1] * learning_rate)
            if t_eps > max_time:
                max_time = t_eps
                bottleneck_name = name

    ax.text(5, y_pos, "УЗКОЕ МЕСТО СХОДИМОСТИ", ha="center", fontsize=10, fontweight="bold")
    y_pos -= 1.5

    if max_time > 0:
        status_color = PALETTE["critical"] if max_time > 10000 else (PALETTE["warning"] if max_time > 1000 else PALETTE["good"])

        ax.text(1, y_pos, "Компонент:", fontsize=9, fontweight="500")
        ax.text(6, y_pos, bottleneck_name, fontsize=9, color=status_color, fontweight="bold")
        y_pos -= 1.2

        ax.text(1, y_pos, "t_ε=1%:", fontsize=9, fontweight="500")
        ax.text(6, y_pos, f"{max_time:.0f} эпох", fontsize=9, color=status_color, fontweight="bold")
        y_pos -= 1.5

        ax.text(5, y_pos, "─" * 20, ha="center", fontsize=8, color=PALETTE["text"])
        y_pos -= 1.0

        ax.text(1, y_pos, "Рекомендация:", fontsize=9, fontweight="bold")
        y_pos -= 1.2

        if max_time > 10000:
            ax.text(1, y_pos, "⚠ Очень медленная сходимость", fontsize=8, color=PALETTE["critical"])
            y_pos -= 1.0
            ax.text(1, y_pos, "  → Увеличить вес компонента", fontsize=8)
            y_pos -= 0.8
            ax.text(1, y_pos, "  → Использовать curriculum", fontsize=8)
        elif max_time > 1000:
            ax.text(1, y_pos, "⚠ Умеренное замедление", fontsize=8, color=PALETTE["warning"])
            y_pos -= 1.0
            ax.text(1, y_pos, "  → Проверить скорость обучения", fontsize=8)
        else:
            ax.text(1, y_pos, "✓ Быстрая сходимость", fontsize=8, color=PALETTE["good"])

    ax = fig.add_subplot(gs[1, 2])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Сводка метрик", fontsize=11, fontweight="600", color=PALETTE["text"])

    rect = FancyBboxPatch((0.2, 0.2), 9.6, 9.6, boxstyle="round,pad=0.1",
                          facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"], linewidth=1)
    ax.add_patch(rect)

    y_pos = 9.0
    line_h = 0.9

    ax.text(1, y_pos, "Компонент", fontsize=8, fontweight="bold")
    ax.text(5, y_pos, "κ", fontsize=8, fontweight="bold")
    ax.text(7.5, y_pos, "τ", fontsize=8, fontweight="bold")
    y_pos -= 0.3
    ax.text(0.5, y_pos, "─" * 25, fontsize=7, color=PALETTE["grid"])
    y_pos -= line_h

    for name, metrics, color in components:
        eig = metrics.get("eigenvalues", [])
        kappa = metrics.get("condition_number", 1e10)
        tau = 0
        if len(eig) > 0 and eig[-1] > 1e-12:
            tau = 1.0 / (2 * eig[-1] * learning_rate)

        ax.text(1, y_pos, name[:8], fontsize=8, color=color, fontweight="500")
        ax.text(5, y_pos, f"{kappa:.1e}"[:7], fontsize=7)
        ax.text(7.5, y_pos, f"{tau:.0f}"[:6], fontsize=7)
        y_pos -= line_h

    fig.suptitle(
        f"Сравнение компонентов NTK | Эпоха {epoch}\n"
        f"Скорость обучения: η = {learning_rate:.1e} | Узкое место: {bottleneck_name}",
        fontsize=14, fontweight="700", color=PALETTE["text"], y=1.01
    )

    out_path = os.path.join(output_dir, f"ntk_component_comparison_epoch{epoch:04d}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return out_path

def plot_convergence_prediction(
        prediction,  
        actual_losses: Optional[Dict[str, List[float]]] = None,
        output_dir: str = "data/ntk_plots",
    ) -> str:
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(18, 12), facecolor="white")
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)

    epoch = prediction.epoch

    ax = fig.add_subplot(gs[0, 0])
    components = []
    t_eps_values = []
    colors = []

    if prediction.pde:
        components.append("ДУЧП")
        t_eps_values.append(prediction.pde.t_epsilon)
        colors.append(PALETTE["K_L"])

    if prediction.dirichlet:
        components.append("Дирихле")
        t_eps_values.append(prediction.dirichlet.t_epsilon)
        colors.append(PALETTE["dirichlet"])

    if prediction.neumann:
        components.append("Нейман")
        t_eps_values.append(prediction.neumann.t_epsilon)
        colors.append(PALETTE["neumann"])

    if components:
        bars = ax.bar(range(len(components)), t_eps_values, color=colors, alpha=0.85, edgecolor="white")
        for bar, tv in zip(bars, t_eps_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                   f"{tv:.0f}", ha="center", fontsize=8, rotation=45)
        ax.set_xticks(range(len(components)))
        ax.set_xticklabels(components, fontsize=9, rotation=20, ha="right")
    _ax_style(ax, title="Время до ε=1% сходимости", ylabel="Эпохи")

    ax = fig.add_subplot(gs[0, 1])
    kappa_values = []
    if prediction.pde:
        kappa_values.append(("ДУЧП", prediction.pde.condition_number, PALETTE["K_L"]))
    if prediction.dirichlet:
        kappa_values.append(("Дирихле", prediction.dirichlet.condition_number, PALETTE["dirichlet"]))
    if prediction.neumann:
        kappa_values.append(("Нейман", prediction.neumann.condition_number, PALETTE["neumann"]))

    if kappa_values:
        labels = [kv[0] for kv in kappa_values]
        vals = [np.log10(max(kv[1], 1)) for kv in kappa_values]
        cols = [kv[2] for kv in kappa_values]

        bars = ax.bar(range(len(labels)), vals, color=cols, alpha=0.85, edgecolor="white")
        for bar, kv in zip(bars, kappa_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                   f"{kv[1]:.1e}"[:7], ha="center", fontsize=7, rotation=45)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=9, rotation=20, ha="right")
    _ax_style(ax, title="Числа обусловленности κ (log₁₀)", ylabel="log₁₀(κ)")

    ax = fig.add_subplot(gs[0, 2])
    rank_values = []
    if prediction.pde:
        rank_values.append(("ДУЧП", prediction.pde.effective_rank, PALETTE["K_L"]))
    if prediction.dirichlet:
        rank_values.append(("Дирихле", prediction.dirichlet.effective_rank, PALETTE["dirichlet"]))
    if prediction.neumann:
        rank_values.append(("Нейман", prediction.neumann.effective_rank, PALETTE["neumann"]))

    if rank_values:
        labels = [rv[0] for rv in rank_values]
        vals = [rv[1] for rv in rank_values]
        cols = [rv[2] for rv in rank_values]

        bars = ax.bar(range(len(labels)), vals, color=cols, alpha=0.85, edgecolor="white")
        for bar, rv in zip(bars, rank_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                   f"{rv[1]:.1f}", ha="center", fontsize=8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=9, rotation=20, ha="right")
    _ax_style(ax, title="Эффективные ранги", ylabel="Ранг")

    ax = fig.add_subplot(gs[1, 0])
    epochs_pred = prediction.predicted_epochs
    ax.semilogy(epochs_pred, prediction.predicted_loss_pde, "o-", 
               color=PALETTE["K_L"], markersize=5, linewidth=2, label="L_ДУЧП (прогноз)")
    ax.semilogy(epochs_pred, prediction.predicted_loss_bc, "s-",
               color=PALETTE["dirichlet"], markersize=5, linewidth=2, label="L_ГУ (прогноз)")
    ax.semilogy(epochs_pred, prediction.predicted_loss_total, "^-",
               color=PALETTE["K"], markersize=5, linewidth=2, label="L_общ (прогноз)")

    if actual_losses and "epochs" in actual_losses:
        actual_ep = actual_losses["epochs"]
        if "pde" in actual_losses:
            ax.semilogy(actual_ep, actual_losses["pde"], "o",
                       color=PALETTE["K_L"], markersize=4, alpha=0.5, label="L_ДУЧП (факт)")
        if "total" in actual_losses:
            ax.semilogy(actual_ep, actual_losses["total"], "o",
                       color=PALETTE["K"], markersize=4, alpha=0.5, label="L_общ (факт)")

    _ax_style(ax, title="Прогноз динамики потерь", xlabel="Эпоха", ylabel="Потери")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(left=0)
    ax.axvline(prediction.bottleneck_epochs, color=PALETTE["critical"], ls="--", lw=1.5, alpha=0.7)

    ax = fig.add_subplot(gs[1, 1])
    if prediction.pde:
        eig = prediction.pde.eigenvalues
        ax.semilogy(np.arange(1, len(eig) + 1), eig, "o-",
                   color=PALETTE["K_L"], markersize=3, linewidth=1.5, label="ДУЧП (K_ℒ)")

    if prediction.dirichlet:
        eig = prediction.dirichlet.eigenvalues
        ax.semilogy(np.arange(1, len(eig) + 1), eig, "s-",
                   color=PALETTE["dirichlet"], markersize=3, linewidth=1.5, label="Дирихле (K_D)")

    if prediction.neumann:
        eig = prediction.neumann.eigenvalues
        ax.semilogy(np.arange(1, len(eig) + 1), eig, "v-",
                   color=PALETTE["neumann"], markersize=3, linewidth=1.5, label="Нейман (K_N)")

    _ax_style(ax, title="Спектры NTK по компонентам", xlabel="Мода k", ylabel="Собственное число λₖ")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(left=1)

    ax = fig.add_subplot(gs[1, 2])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Показатели здоровья", fontsize=11, fontweight="600", color=PALETTE["text"])

    rect = FancyBboxPatch((0.2, 0.2), 9.6, 9.6, boxstyle="round,pad=0.1",
                          facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"], linewidth=1)
    ax.add_patch(rect)

    y_pos = 8.5

    health_color = PALETTE["good"] if prediction.health_score > 70 else (PALETTE["warning"] if prediction.health_score > 40 else PALETTE["critical"])
    ax.text(1, y_pos, "Оценка здоровья:", fontsize=10, fontweight="500")
    ax.text(7, y_pos, f"{prediction.health_score:.0f}/100", fontsize=12, fontweight="bold", color=health_color)
    y_pos -= 1.5

    balance_color = PALETTE["good"] if prediction.balance_score > 70 else (PALETTE["warning"] if prediction.balance_score > 40 else PALETTE["critical"])
    ax.text(1, y_pos, "Оценка баланса:", fontsize=10, fontweight="500")
    ax.text(7, y_pos, f"{prediction.balance_score:.0f}/100", fontsize=12, fontweight="bold", color=balance_color)
    y_pos -= 1.5

    bn_color = PALETTE["critical"] if prediction.bottleneck_epochs > 10000 else (PALETTE["warning"] if prediction.bottleneck_epochs > 1000 else PALETTE["good"])
    ax.text(1, y_pos, "Узкое место:", fontsize=10, fontweight="500")
    ax.text(7, y_pos, prediction.bottleneck_component, fontsize=10, fontweight="bold", color=bn_color)
    y_pos -= 1.2
    ax.text(1, y_pos, "Оценка эпох:", fontsize=10, fontweight="500")
    ax.text(7, y_pos, f"{prediction.bottleneck_epochs:,}", fontsize=10, fontweight="bold", color=bn_color)

    fig.suptitle(
        f"Прогноз сходимости NTK | Эпоха {epoch}\n"
        f"Узкое место: {prediction.bottleneck_component} | "
        f"Оценка эпох: {prediction.total_epochs_estimate:,} | "
        f"Здоровье: {prediction.health_score:.0f}/100",
        fontsize=14, fontweight="700", color=PALETTE["text"], y=1.01
    )

    out_path = os.path.join(output_dir, f"ntk_convergence_prediction_epoch{epoch:04d}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return out_path

def plot_convergence_evolution(
        predictions_history: List,  
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

def create_component_metrics_from_ntk_result(
        ntk_result: Dict[str, Any],
        learning_rate: float = 1e-3,
    ) -> Dict[str, Dict[str, Any]]:
    from training.convergence_prediction import compute_component_metrics

    result = {}

    if "interior" in ntk_result:
        interior = ntk_result["interior"]
        if "eigenvalues_KL" in interior:
            result["pde"] = compute_component_metrics(
                interior["eigenvalues_KL"], "PDE", learning_rate
            ).__dict__

    if "boundary" in ntk_result and ntk_result["boundary"].get("dirichlet"):
        d_data = ntk_result["boundary"]["dirichlet"]
        if "eigenvalues" in d_data:
            result["dirichlet"] = compute_component_metrics(
                d_data["eigenvalues"], "Dirichlet", learning_rate
            ).__dict__

    if "boundary" in ntk_result and ntk_result["boundary"].get("neumann"):
        n_data = ntk_result["boundary"]["neumann"]
        if "eigenvalues" in n_data:
            result["neumann"] = compute_component_metrics(
                n_data["eigenvalues"], "Neumann", learning_rate
            ).__dict__

    return result

def plot_error_prediction(
        prediction,  
        actual_l2_history: Optional[List[float]] = None,
        actual_energy_history: Optional[List[float]] = None,
        actual_epochs: Optional[List[int]] = None,
        output_dir: str = "data/ntk_plots",
    ) -> str:
    os.makedirs(output_dir, exist_ok=True)

    if not hasattr(prediction, 'error_bounds') or prediction.error_bounds is None:
        return ""

    eb = prediction.error_bounds
    epochs_pred = prediction.predicted_epochs

    fig = plt.figure(figsize=(14, 10), facecolor="white")
    gs = GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.25)

    ax = fig.add_subplot(gs[0, 0])
    if len(eb.l2_error_predicted) > 0:
        ax.semilogy(epochs_pred, eb.l2_error_predicted, "o-", 
                   color=PALETTE["K"], markersize=5, linewidth=2, 
                   label="Прогноз L² ошибки")

        ax.axhline(eb.l2_error_upper_bound,
                  color=PALETTE["critical"], ls="--", lw=1.5, 
                  alpha=0.7, label=f"Верхняя граница (C_P={eb.poincare_constant:.2f})")

    if actual_l2_history and actual_epochs:
        ax.semilogy(actual_epochs, actual_l2_history, "s",
                   color=PALETTE["good"], markersize=4, alpha=0.6,
                   label="Фактическая L² ошибка")

    _ax_style(ax, title=r"Прогноз L² ошибки $\|u - v\|_{L^2}$", 
             xlabel="Эпоха", ylabel="Ошибка")
    ax.legend(fontsize=8)
    ax.set_xlim(left=0)

    ax = fig.add_subplot(gs[0, 1])
    if len(eb.energy_error_predicted) > 0:
        ax.semilogy(epochs_pred, eb.energy_error_predicted, "o-", 
                   color=PALETTE["K_L"], markersize=5, linewidth=2,
                   label="Прогноз энергетической ошибки")

        ax.axhline(eb.energy_error_upper_bound,
                  color=PALETTE["critical"], ls="--", lw=1.5,
                  alpha=0.7, label=f"Верхняя граница (C_S={eb.stability_constant:.2f})")

    if actual_energy_history and actual_epochs:
        ax.semilogy(actual_epochs, actual_energy_history, "s",
                   color=PALETTE["good"], markersize=4, alpha=0.6,
                   label="Фактическая энергет. ошибка")

    _ax_style(ax, title=r"Прогноз энергетической ошибки $\|\nabla(u-v)\|_{L^2}$", 
             xlabel="Эпоха", ylabel="Ошибка")
    ax.legend(fontsize=8)
    ax.set_xlim(left=0)

    ax = fig.add_subplot(gs[1, 0])
    if len(eb.l2_error_predicted) > 0 and len(eb.energy_error_predicted) > 0:
        ax.semilogy(epochs_pred, eb.l2_error_predicted,
                   "o-", color=PALETTE["K"], markersize=4, linewidth=1.5,
                   label=r"L² ошибка", alpha=0.8)
        ax.semilogy(epochs_pred, eb.energy_error_predicted,
                   "s-", color=PALETTE["K_L"], markersize=4, linewidth=1.5,
                   label=r"Энергет. ошибка", alpha=0.8)
        ax.semilogy(epochs_pred, prediction.predicted_loss_pde,
                   "^-", color=PALETTE["warning"], markersize=4, linewidth=1.5,
                   label=r"PDE невязка", alpha=0.8)

    _ax_style(ax, title="Прогноз эволюции ошибок", xlabel="Эпоха", ylabel="Ошибка")
    ax.legend(fontsize=8)
    ax.set_xlim(left=0)

    ax = fig.add_subplot(gs[1, 1])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Сводка границ ошибок", fontsize=11, fontweight="600")

    rect = FancyBboxPatch((0.2, 0.2), 9.6, 9.6, boxstyle="round,pad=0.1",
                          facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"], linewidth=1)
    ax.add_patch(rect)

    y_pos = 9.0

    ax.text(1, y_pos, "Константа Пуанкаре:", fontsize=10, fontweight="500")
    ax.text(7, y_pos, f"C_P = {eb.poincare_constant:.3f}",
           fontsize=10, fontweight="bold")
    y_pos -= 1.2

    ax.text(1, y_pos, "Константа устойчивости:", fontsize=10, fontweight="500")
    ax.text(7, y_pos, f"C_S = {eb.stability_constant:.3f}",
           fontsize=10, fontweight="bold")
    y_pos -= 1.5

    ax.text(1, y_pos, "─" * 20, fontsize=8, color=PALETTE["grid"])
    y_pos -= 1.0

    l2_color = PALETTE["good"] if eb.l2_error_upper_bound < 0.1 else (PALETTE["warning"] if eb.l2_error_upper_bound < 1.0 else PALETTE["critical"])
    ax.text(1, y_pos, "Верхняя граница L²:", fontsize=10, fontweight="500")
    ax.text(7, y_pos, f"{eb.l2_error_upper_bound:.2e}",
           fontsize=10, fontweight="bold", color=l2_color)
    y_pos -= 1.2

    energy_color = PALETTE["good"] if eb.energy_error_upper_bound < 0.1 else (PALETTE["warning"] if eb.energy_error_upper_bound < 1.0 else PALETTE["critical"])
    ax.text(1, y_pos, "Верхняя граница энерг.:", fontsize=10, fontweight="500")
    ax.text(7, y_pos, f"{eb.energy_error_upper_bound:.2e}",
           fontsize=10, fontweight="bold", color=energy_color)
    y_pos -= 1.5

    ax.text(1, y_pos, "Начальная невязка:", fontsize=10, fontweight="500")
    ax.text(7, y_pos, f"{eb.residual_norm_init:.2e}",
           fontsize=10, fontweight="bold")

    fig.suptitle(
        f"Границы ошибок из NTK анализа | Эпоха {prediction.epoch}",
        fontsize=13, fontweight="700", y=1.01
    )

    out_path = os.path.join(output_dir, f"error_prediction_epoch{prediction.epoch:04d}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return out_path
