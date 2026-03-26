import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyBboxPatch

def analyze_and_plot_weights(pth_path: str, save_dir: str = "data"):
    print(f"Загрузка весов из: {pth_path}")
    if not os.path.exists(pth_path):
        print("Файл не найден!")
        return

    state_dict = torch.load(pth_path, map_location="cpu", weights_only=True)
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "="*70)
    print(f"{'Слой':<30} | {'Форма':<15} | {'Min':<8} | {'Max':<8} | {'Mean':<8}")
    print("="*70)
    for name, tensor in state_dict.items():
        val = tensor.numpy()
        print(f"{name:<30} | {str(list(tensor.shape)):<15} | "
              f"{val.min():>8.4f} | {val.max():>8.4f} | {val.mean():>8.4f}")
    print("="*70 + "\n")

    weight_tensors = {n: t.numpy().flatten() for n, t in state_dict.items() if 'weight' in n}
    if weight_tensors:
        n = len(weight_tensors)
        cols = 3
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
        axes = np.atleast_1d(axes).flatten()
        for i, (name, vals) in enumerate(weight_tensors.items()):
            sns.histplot(vals, bins=50, ax=axes[i], kde=True, color="steelblue")
            axes[i].set_title(f"Гистограмма: {name}", fontsize=10)
        for j in range(len(weight_tensors), len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "weights_histograms.png"), dpi=150)
        plt.close(fig)

    matrix_tensors = {n: t.numpy() for n, t in state_dict.items() if len(t.shape) == 2}
    if matrix_tensors:
        for name, matrix in matrix_tensors.items():
            fig, ax = plt.subplots(figsize=(8, 6))
            vmax = np.max(np.abs(matrix))
            if vmax == 0: vmax = 1e-8
            sns.heatmap(matrix, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax, ax=ax)
            ax.set_title(f"Тепловая карта: {name}")
            plt.tight_layout()
            safe_name = name.replace(".", "_")
            fig.savefig(os.path.join(save_dir, f"heatmap_{safe_name}.png"), dpi=150)
            plt.close(fig)

if __name__ == "__main__":
    analyze_and_plot_weights("data/l_shape_best_pinn.pth")
