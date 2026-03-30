import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def plot_ntk_evolution(history: list, output_dir: str = "data") -> str:
    if len(history) < 2:
        return ""

    os.makedirs(output_dir, exist_ok=True)

    epochs = [res.epoch for res in history]

    kappas_K = [res.metrics_K.get("condition_number", np.nan) for res in history]
    kappas_KL = [res.metrics_KL.get("condition_number", np.nan) for res in history]
    kappas_D = [res.metrics_D.get("condition_number", np.nan) if res.metrics_D else np.nan for res in history]
    kappas_N = [res.metrics_N.get("condition_number", np.nan) if res.metrics_N else np.nan for res in history]

    ranks_K = [res.metrics_K.get("effective_rank", np.nan) for res in history]
    ranks_KL = [res.metrics_KL.get("effective_rank", np.nan) for res in history]
    ranks_D = [res.metrics_D.get("effective_rank", np.nan) if res.metrics_D else np.nan for res in history]
    ranks_N = [res.metrics_N.get("effective_rank", np.nan) if res.metrics_N else np.nan for res in history]

    traces_K = [res.metrics_K.get("trace", np.nan) for res in history]
    traces_KL = [res.metrics_KL.get("trace", np.nan) for res in history]

    fig = plt.figure(figsize=(18, 10), facecolor="white")
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.2)

    c_K, c_KL, c_D, c_N = "#2563EB", "#DC2626", "#059669", "#D97706"

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.semilogy(epochs, kappas_K, 'o-', color=c_K, label="K (Все внутренние)", markersize=5)
    ax1.semilogy(epochs, kappas_KL, 's-', color=c_KL, label="K_L (PDE)", markersize=5)
    if not np.isnan(kappas_D).all():
        ax1.semilogy(epochs, kappas_D, '^-', color=c_D, label="K_D (Дирихле)", markersize=5)
    if not np.isnan(kappas_N).all():
        ax1.semilogy(epochs, kappas_N, 'v-', color=c_N, label="K_N (Нейман)", markersize=5)
    ax1.set_title("Эволюция числа обусловленности (κ)", fontweight="bold")
    ax1.set_xlabel("Эпоха")
    ax1.set_ylabel("κ (log-scale)")
    ax1.grid(True, alpha=0.4)
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, ranks_K, 'o-', color=c_K, label="K", markersize=5)
    ax2.plot(epochs, ranks_KL, 's-', color=c_KL, label="K_L", markersize=5)
    if not np.isnan(ranks_D).all():
        ax2.plot(epochs, ranks_D, '^-', color=c_D, label="K_D", markersize=5)
    if not np.isnan(ranks_N).all():
        ax2.plot(epochs, ranks_N, 'v-', color=c_N, label="K_N", markersize=5)
    ax2.set_title("Эволюция эффективного ранга", fontweight="bold")
    ax2.set_xlabel("Эпоха")
    ax2.set_ylabel("Ранг")
    ax2.grid(True, alpha=0.4)
    ax2.legend()

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.semilogy(epochs, traces_K, 'o-', color=c_K, label="K", markersize=5)
    ax3.semilogy(epochs, traces_KL, 's-', color=c_KL, label="K_L", markersize=5)
    ax3.set_title("Эволюция следа (Суммарная энергия)", fontweight="bold")
    ax3.set_xlabel("Эпоха")
    ax3.set_ylabel("Trace(K) (log-scale)")
    ax3.grid(True, alpha=0.4)
    ax3.legend()

    ax4 = fig.add_subplot(gs[1, 1])
    cmap = plt.cm.plasma(np.linspace(0, 1, len(history)))
    for i, res in enumerate(history):
        eig = res.eigenvalues_KL
        if len(eig) > 0:
            k_idx = np.arange(1, len(eig) + 1)
            ax4.plot(k_idx, np.log10(eig), color=cmap[i], alpha=0.7, linewidth=1.5)

    sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(vmin=epochs[0], vmax=epochs[-1]))
    cbar = fig.colorbar(sm, ax=ax4)
    cbar.set_label("Эпоха")
    ax4.set_title("Эволюция спектра PDE (K_L)", fontweight="bold")
    ax4.set_xlabel("Индекс k")
    ax4.set_ylabel("$\\log_{10}(\\lambda_k)$")
    ax4.grid(True, alpha=0.4)

    fig.suptitle("Эволюция характеристик NTK в процессе обучения", fontsize=16, fontweight="bold", y=0.98)

    out_path = os.path.join(output_dir, "ntk_evolution_dashboard.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return out_path