import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def plot_ntk_master_dashboard(
    epoch: int,
    components: dict,
    output_dir: str = "data",
    prefix: str = "ntk",
    title_prefix: str = "Спектральный анализ NTK"
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    fig = plt.figure(figsize=(20, 16), facecolor="white")
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.25)

    comps = [(name, data) for name, data in components.items() if len(data["eigenvalues"]) > 0]
    names = [name for name, _ in comps]
    colors = [data["color"] for _, data in comps]

    ax1 = fig.add_subplot(gs[0, 0])
    for name, data in comps:
        eig = data["eigenvalues"]
        k_idx = np.arange(1, len(eig) + 1)
        ax1.plot(k_idx, np.log10(eig), marker=data["marker"], color=data["color"], 
                 linestyle='-', markersize=4, linewidth=1.5, label=name)
    ax1.set_title("1. Собственные значения $\\log_{10}(\\lambda_k)$", fontweight='bold')
    ax1.set_xlabel("Индекс $k$")
    ax1.set_ylabel("$\\log_{10}(\\lambda_k)$")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    for name, data in comps:
        eig = data["eigenvalues"]
        ax2.hist(np.log10(eig), bins=10, color=data["color"], alpha=0.4, 
                 histtype='stepfilled', label=name)
        ax2.hist(np.log10(eig), bins=10, color=data["color"], histtype='step', linewidth=1.5)
    ax2.set_title("2. Спектральная плотность", fontweight='bold')
    ax2.set_xlabel("$\\log_{10}(\\lambda_k)$")
    ax2.set_ylabel("Количество (частота)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])
    for name, data in comps:
        eig = data["eigenvalues"]
        k_idx = np.arange(1, len(eig) + 1)
        cum = np.cumsum(eig) / np.sum(eig)
        ax3.plot(k_idx, cum, color=data["color"], linewidth=2, label=name)
    ax3.axhline(0.9, color='k', linestyle='--', alpha=0.5, label='90% Энергии')
    ax3.set_title("3. Кумулятивная доля энергии", fontweight='bold')
    ax3.set_xlabel("Индекс $k$")
    ax3.set_ylabel("Доля от общего следа")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])

    for name, data in comps:
        eig = data["eigenvalues"]
        k_idx = np.arange(1, len(eig) + 1)
        log_k = np.log10(k_idx)
        log_eig = np.log10(eig)

        alpha = data["metrics"]["decay"]["power_law_alpha"]
        c = np.mean(log_eig + alpha * log_k)

        if "+" in name or "Полная" in name:
            target_ax = ax5
        else:
            target_ax = ax4

        target_ax.plot(log_k, log_eig, marker=data["marker"], color=data["color"], 
                 linestyle='', markersize=3, alpha=0.6, label=name)
        target_ax.plot(log_k, -alpha * log_k + c, color=data["color"], linestyle='--', 
                 linewidth=2, label=f'Степ. $\\alpha={alpha:.2f}$')

    ax4.set_title("4. Затухание (Базовые компоненты)", fontweight='bold')
    ax4.set_xlabel("$\\log_{10}(k)$")
    ax4.set_ylabel("$\\log_{10}(\\lambda_k)$")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    ax5.set_title("5. Затухание (Пары и Полная матрица)", fontweight='bold')
    ax5.set_xlabel("$\\log_{10}(k)$")
    ax5.set_ylabel("$\\log_{10}(\\lambda_k)$")
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(gs[1, 2])
    trace_vals = [data["metrics"]["trace"] for _, data in comps]
    bars6 = ax6.bar(names, trace_vals, color=colors, alpha=0.8)
    ax6.set_title("6. След матрицы $\\text{Tr}(K)$", fontweight='bold')
    ax6.set_ylabel("След (Энергия)")
    ax6.tick_params(axis='x', rotation=25, labelsize=8)
    for bar, val in zip(bars6, trace_vals):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(trace_vals)*0.02), 
                 f"{val:.1e}", ha='center', va='bottom', fontsize=8)
    ax6.grid(axis='y', alpha=0.3)

    ax7 = fig.add_subplot(gs[2, 0])
    cond_vals = [np.log10(max(data["metrics"]["condition_number"], 1)) for _, data in comps]
    bars1 = ax7.bar(names, cond_vals, color=colors, alpha=0.8)
    ax7.set_title("7. Число обусловленности (log10 κ)", fontweight='bold')
    ax7.set_ylabel("$\\log_{10}(\\kappa)$")
    ax7.tick_params(axis='x', rotation=25, labelsize=8)
    for bar, val in zip(bars1, cond_vals):
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                 f"{val:.1f}", ha='center', va='bottom', fontsize=8)
    ax7.grid(axis='y', alpha=0.3)

    ax8 = fig.add_subplot(gs[2, 1])
    rank_vals = [data["metrics"]["effective_rank"] for _, data in comps]
    bars2 = ax8.bar(names, rank_vals, color=colors, alpha=0.8)
    ax8.set_title("8. Эффективный ранг", fontweight='bold')
    ax8.set_ylabel("Ранг")
    ax8.tick_params(axis='x', rotation=25, labelsize=8)
    for bar, val in zip(bars2, rank_vals):
        ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                 f"{val:.1f}", ha='center', va='bottom', fontsize=8)
    ax8.grid(axis='y', alpha=0.3)

    ax9 = fig.add_subplot(gs[2, 2])
    frob_vals = [data.get("metrics", {}).get("frobenius_norm", 0.0) for _, data in comps]
    bars3 = ax9.bar(names, frob_vals, color=colors, alpha=0.8)
    ax9.set_title("9. Норма Фробениуса $||K||_F$", fontweight='bold')
    ax9.set_ylabel("Норма")
    ax9.tick_params(axis='x', rotation=25, labelsize=8)
    for bar, val in zip(bars3, frob_vals):
        ax9.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(frob_vals)*0.02), 
                 f"{val:.1e}", ha='center', va='bottom', fontsize=8)
    ax9.grid(axis='y', alpha=0.3)

    fig.suptitle(f"{title_prefix} | Эпоха {epoch}", fontsize=18, fontweight='bold', y=0.98)

    out_path = os.path.join(output_dir, f"{prefix}_dashboard_epoch{epoch:04d}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path