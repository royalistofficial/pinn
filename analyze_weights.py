import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyBboxPatch

def analyze_and_plot_weights(pth_path: str, save_dir: str = "data"):
    print(f"Загрузка весов из: {pth_path}")
    if not os.path.exists(pth_path):
        print("Файл не найден! Убедитесь, что вы запустили обучение и модель сохранилась.")
        return

    # Загружаем словарь весов на CPU
    state_dict = torch.load(pth_path, map_location="cpu", weights_only=True)
    os.makedirs(save_dir, exist_ok=True)
    
    # ==========================================
    # 1. Текстовый анализ (Статистика)
    # ==========================================
    print("\n" + "="*70)
    print(f"{'Название слоя':<30} | {'Форма':<15} | {'Минимум':<8} | {'Максимум':<8} | {'Среднее':<8}")
    print("="*70)
    
    for name, tensor in state_dict.items():
        shape_str = str(list(tensor.shape))
        val = tensor.numpy()
        print(f"{name:<30} | {shape_str:<15} | {val.min():>8.4f} | {val.max():>8.4f} | {val.mean():>8.4f}")
    
    print("="*70 + "\n")

    # ==========================================
    # 2. Гистограммы распределения весов
    # ==========================================
    weight_tensors = {name: tensor.numpy().flatten() for name, tensor in state_dict.items() if 'weight' in name}
    if weight_tensors:
        n_layers = len(weight_tensors)
        cols = 3
        rows = (n_layers + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
        axes = np.atleast_1d(axes).flatten()
        
        for i, (name, vals) in enumerate(weight_tensors.items()):
            sns.histplot(vals, bins=50, ax=axes[i], kde=True, color="steelblue")
            axes[i].set_title(f"Гистограмма: {name}", fontsize=10)
            
        for j in range(len(weight_tensors), len(axes)):
            fig.delaxes(axes[j])
            
        plt.tight_layout()
        hist_path = os.path.join(save_dir, "weights_histograms.png")
        fig.savefig(hist_path, dpi=150, facecolor="white")
        plt.close(fig)
        print(f"Гистограммы сохранены в: {hist_path}")

    # ==========================================
    # 3 & 4. Тепловые карты и Детальные Графы
    # ==========================================
    matrix_tensors = {name: tensor.numpy() for name, tensor in state_dict.items() if len(tensor.shape) == 2}
    
    if matrix_tensors:
        n_matrices = len(matrix_tensors)
        fig, axes = plt.subplots(n_matrices, 2, figsize=(14, 6 * n_matrices))
        if n_matrices == 1: axes = [axes]
            
        for i, (name, matrix) in enumerate(matrix_tensors.items()):
            ax_heat, ax_graph = axes[i][0], axes[i][1]
            out_features, in_features = matrix.shape
            vmax = np.max(np.abs(matrix))
            if vmax == 0: vmax = 1e-8

            # Тепловая карта
            sns.heatmap(matrix, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax, ax=ax_heat)
            ax_heat.set_title(f"Тепловая карта: {name}\n({out_features}x{in_features})", fontsize=11)
            
            # Детальный двудольный граф
            if in_features > 64 or out_features > 64:
                ax_graph.text(0.5, 0.5, f"Слой слишком большой для графа\n({out_features}x{in_features})", 
                              ha='center', va='center', fontsize=12)
                ax_graph.axis("off")
                continue

            y_in = np.linspace(1, 0, in_features)
            y_out = np.linspace(1, 0, out_features)
            lines, colors, linewidths = [], [], []
            
            for j in range(out_features):
                for k in range(in_features):
                    weight = matrix[j, k]
                    lines.append([(0, y_in[k]), (1, y_out[j])])
                    intensity = min(1.0, abs(weight) / vmax)
                    if weight > 0:
                        colors.append((0.8, 0.1, 0.1, intensity)) # Красный
                    else:
                        colors.append((0.1, 0.1, 0.8, intensity)) # Синий
                    linewidths.append(0.5 + 2.0 * intensity)

            lc = LineCollection(lines, colors=colors, linewidths=linewidths)
            ax_graph.add_collection(lc)
            ax_graph.scatter(np.zeros(in_features), y_in, color="black", s=30, zorder=5)
            ax_graph.scatter(np.ones(out_features), y_out, color="black", s=30, zorder=5)
            
            ax_graph.set_xlim(-0.1, 1.1)
            ax_graph.set_ylim(-0.05, 1.05)
            ax_graph.axis("off")
            ax_graph.set_title(f"Детальный граф связей: {name}", fontsize=11)

        plt.tight_layout()
        combined_path = os.path.join(save_dir, "weights_heatmaps_and_graphs.png")
        fig.savefig(combined_path, dpi=150, facecolor="white")
        plt.close(fig)
        print(f"Тепловые карты и Графы слоев сохранены в: {combined_path}")

    # ==========================================
    # 5. Макро-граф архитектуры (Блок-схема)
    # ==========================================
    if matrix_tensors:
        n_matrices = len(matrix_tensors)
        # Высота картинки динамически подстраивается под глубину сети
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(6, 1.5 * n_matrices)))
        
        y_pos = np.linspace(0.95, 0.05, n_matrices)
        box_width = 0.5
        box_height = min(0.08, 0.8 / n_matrices)
        
        means = [np.mean(np.abs(m)) for m in matrix_tensors.values()]
        max_mean = max(means) if means else 1.0
        
        for ax, show_w, title in zip([ax1, ax2], [False, True], 
                                     ["Классическая архитектура", "Архитектура с силой связей (Mean |W|)"]):
            ax.set_title(title, fontsize=14, pad=20, fontweight="bold")
            ax.axis('off')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            
            for i, ((name, matrix), y) in enumerate(zip(matrix_tensors.items(), y_pos)):
                out_f, in_f = matrix.shape
                mean_w = means[i]
                
                # Настройка цветов
                box_color = "#F1F5F9"  # Светло-серый по умолчанию
                edge_color = "#475569"
                text_color = "black"
                
                if show_w:
                    intensity = min(1.0, mean_w / max_mean)
                    box_color = plt.cm.Reds(intensity * 0.8) # Красный оттенок в зависимости от силы
                    if intensity > 0.6: text_color = "white" # Белый текст на темном фоне
                
                # Рисуем красивый блок с закругленными углами
                box = FancyBboxPatch(
                    (0.5 - box_width/2, y - box_height/2), box_width, box_height,
                    boxstyle="round,pad=0.03", facecolor=box_color, edgecolor=edge_color, lw=2, zorder=3
                )
                ax.add_patch(box)
                
                # Текст внутри блока
                text = f"{name}\n[{in_f} → {out_f}]"
                if show_w:
                    text += f"\nСила: {mean_w:.4f}"
                    
                ax.text(0.5, y, text, ha='center', va='center', color=text_color, fontsize=10, zorder=4)
                
                # Рисуем стрелку от предыдущего блока к текущему
                if i > 0:
                    prev_y = y_pos[i-1]
                    arrow_y_start = prev_y - box_height/2 - 0.03
                    arrow_y_end = y + box_height/2 + 0.03
                    
                    arrow_lw = 1.5
                    arrow_color = "#94A3B8" # Серая стрелка
                    
                    if show_w:
                        intensity = min(1.0, mean_w / max_mean)
                        arrow_lw = 1.5 + 4 * intensity # Чем сильнее связь, тем толще стрелка
                        arrow_color = plt.cm.Reds(intensity)
                        
                    ax.annotate('', xy=(0.5, arrow_y_end), xytext=(0.5, arrow_y_start),
                                arrowprops=dict(arrowstyle="->", lw=arrow_lw, color=arrow_color), zorder=2)
                                
        plt.tight_layout()
        block_path = os.path.join(save_dir, "weights_block_graph.png")
        fig.savefig(block_path, dpi=150, facecolor="white")
        plt.close(fig)
        print(f"Макро-графы архитектуры сохранены в: {block_path}")

if __name__ == "__main__":
    # Укажите путь к вашему файлу с весами
    model_file = "data/l_shape_best_pinn.pth" 
    analyze_and_plot_weights(model_file)