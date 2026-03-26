from __future__ import annotations
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from networks.ntk_utils import (
    ntk_spectrum_analysis, ntk_predict,
    extract_learned_frequencies,
)

def plot_ntk_analysis(
        model: nn.Module,
        epoch: int,
        output_dir: str = "data/ntk_plots",
    ) -> None:
    os.makedirs(output_dir, exist_ok=True)
    device = next(model.parameters()).device

    n_pts = 64
    xy = torch.rand(n_pts, 2, device=device) * 2 - 1

    analysis = ntk_spectrum_analysis(model, xy)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.patch.set_facecolor("white")

    ax = axes[0, 0]
    im = ax.imshow(analysis["ntk_matrix"], cmap='viridis', aspect='auto')
    ax.set_title(f'NTK матрица K (эпоха {epoch})', fontsize=12, fontweight='600')
    ax.set_xlabel('Точка j')
    ax.set_ylabel('Точка i')
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[0, 1]
    eigs = analysis["eigenvalues"]
    ax.semilogy(eigs, 'o-', markersize=4, color='#2563EB', linewidth=1.5)
    ax.fill_between(range(len(eigs)), eigs, alpha=0.1, color='#2563EB')
    ax.set_title(
        f'Спектр NTK: κ={analysis["condition_number"]:.1e}, '
        f'ранг_эфф={analysis["effective_rank"]:.1f}',
        fontsize=11, fontweight='600',
    )
    ax.set_xlabel('Индекс моды k')
    ax.set_ylabel('λₖ (лог. шкала)')
    ax.axhline(y=eigs.mean(), color='red', ls='--', alpha=0.5,
               label=f'⟨λ⟩ = {eigs.mean():.2e}')
    if len(eigs) > 1:
        ax.axhline(y=eigs[0], color='green', ls=':', alpha=0.4,
                   label=f'λ_max = {eigs[0]:.2e}')
        ax.axhline(y=eigs[eigs > 1e-10][-1] if np.any(eigs > 1e-10) else eigs[-1],
                   color='orange', ls=':', alpha=0.4,
                   label=f'λ_min = {eigs[eigs > 1e-10][-1]:.2e}' if np.any(eigs > 1e-10) else '')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    rates = analysis["top_convergence_rates"]
    colors = plt.cm.RdYlGn(rates / max(rates.max(), 1e-10))
    ax.bar(range(len(rates)), rates, color=colors, edgecolor='gray', linewidth=0.5)
    ax.set_title(
        f'Скорость сходимости мод: 1 - exp(−λₖ)',
        fontsize=11, fontweight='600',
    )
    ax.set_xlabel('Мода k')
    ax.set_ylabel('Скорость')
    ax.axhline(y=0.5, color='gray', ls='--', alpha=0.3, label='50% за эпоху')
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    freqs = extract_learned_frequencies(model)
    if len(freqs) > 0:
        ax.hist(freqs, bins=max(10, len(freqs) // 2), alpha=0.8,
                color='#059669', edgecolor='black', linewidth=0.5)
        ax.axvline(x=np.median(freqs), color='red', ls='--',
                   label=f'медиана = {np.median(freqs):.2f}')
        ax.set_title(f'Частоты Фурье |B| (эпоха {epoch})', fontsize=11, fontweight='600')
        ax.set_xlabel('Частота')
        ax.set_ylabel('Количество')
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'Нет Фурье-частот', ha='center', va='center')
        ax.set_title('Частоты Фурье')

    plt.tight_layout()
    fig.savefig(f'{output_dir}/ntk_analysis_{epoch:04d}.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[NTK] Сохранены графики для эпохи {epoch}")

def plot_ntk_prediction_comparison(
        model: nn.Module,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        lr: float,
        epoch: int,
        output_dir: str = "data/ntk_plots",
    ) -> None:
    os.makedirs(output_dir, exist_ok=True)

    y_ntk = ntk_predict(model, X_train, y_train, X_test, lr, epoch)

    model.eval()
    with torch.no_grad():
        y_model = model(X_test)
    model.train()

    y_test_np = y_test.cpu().numpy().flatten()
    y_ntk_np = y_ntk.cpu().numpy().flatten()
    y_model_np = y_model.cpu().numpy().flatten()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("white")

    ax = axes[0]
    ax.scatter(y_test_np, y_model_np, s=5, alpha=0.5, label='PINN', color='#2563EB')
    ax.scatter(y_test_np, y_ntk_np, s=5, alpha=0.5, label='NTK', color='#DC2626')
    lims = [min(y_test_np.min(), y_model_np.min()),
            max(y_test_np.max(), y_model_np.max())]
    ax.plot(lims, lims, 'k--', alpha=0.3)
    ax.set_xlabel('Точное решение')
    ax.set_ylabel('Предсказание')
    ax.set_title('PINN vs NTK предсказания')
    ax.legend()

    ax = axes[1]
    ax.hist(np.abs(y_model_np - y_test_np), bins=30, alpha=0.7,
            label='PINN', color='#2563EB')
    ax.hist(np.abs(y_ntk_np - y_test_np), bins=30, alpha=0.7,
            label='NTK', color='#DC2626')
    ax.set_title('Распределение |ошибок|')
    ax.legend()

    ax = axes[2]
    err_pinn = np.sort(np.abs(y_model_np - y_test_np))[::-1]
    err_ntk = np.sort(np.abs(y_ntk_np - y_test_np))[::-1]
    rms_p = np.sqrt(np.mean(err_pinn ** 2))
    rms_n = np.sqrt(np.mean(err_ntk ** 2))
    ax.semilogy(err_pinn, label=f'PINN (RMS={rms_p:.4e})', color='#2563EB')
    ax.semilogy(err_ntk, label=f'NTK (RMS={rms_n:.4e})', color='#DC2626')
    ax.set_title('Упорядоченные ошибки')
    ax.set_xlabel('Индекс')
    ax.legend()

    plt.tight_layout()
    fig.savefig(f'{output_dir}/ntk_vs_pinn_{epoch:04d}.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)

def plot_spectrum_evolution(
        spectra_history: list[dict],
        epochs: list[int],
        output_dir: str = "data/ntk_plots",
    ) -> None:
    if len(spectra_history) < 2:
        return

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("white")

    cmap = plt.cm.viridis(np.linspace(0, 1, len(spectra_history)))

    ax = axes[0]
    for i, (spec, ep) in enumerate(zip(spectra_history, epochs)):
        eigs = spec["eigenvalues"]
        ax.semilogy(eigs, 'o-', markersize=2, color=cmap[i],
                    label=f'ep {ep}', linewidth=1)
    ax.set_title('Эволюция спектра NTK', fontweight='600')
    ax.set_xlabel('Мода k')
    ax.set_ylabel('λₖ')
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    kappas = [s["condition_number"] for s in spectra_history]
    ax.semilogy(epochs, kappas, 's-', color='#DC2626', markersize=6)
    ax.set_title('Число обусловленности κ(K)', fontweight='600')
    ax.set_xlabel('Эпоха')
    ax.set_ylabel('κ')
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ranks = [s["effective_rank"] for s in spectra_history]
    ax.plot(epochs, ranks, 'o-', color='#059669', markersize=6)
    ax.set_title('Эффективный ранг', fontweight='600')
    ax.set_xlabel('Эпоха')
    ax.set_ylabel('ранг')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(f'{output_dir}/ntk_spectrum_evolution.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)

def plot_adaptive_frequencies(
        init_freqs: np.ndarray,
        learned_freqs: np.ndarray,
        spectrum_info: Optional[dict] = None,
        output_dir: str = "data/ntk_plots",
    ) -> None:
    os.makedirs(output_dir, exist_ok=True)

    n_panels = 3 if spectrum_info is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    fig.patch.set_facecolor("white")

    ax = axes[0]
    if len(init_freqs) > 0:
        ax.stem(range(len(init_freqs)), np.exp(init_freqs),
                linefmt='b-', markerfmt='bo', basefmt='gray',
                label='Начальные')
    if len(learned_freqs) > 0:
        ax.stem(range(len(learned_freqs)), learned_freqs,
                linefmt='r-', markerfmt='r^', basefmt='gray',
                label='Выученные')
    ax.set_title('Частоты: начальные vs выученные', fontweight='600')
    ax.set_xlabel('Индекс')
    ax.set_ylabel('Частота')
    ax.legend()

    ax = axes[1]
    if len(init_freqs) > 0:
        ax.hist(np.exp(init_freqs), bins=15, alpha=0.6, label='Начальные',
                color='#2563EB', edgecolor='black')
    if len(learned_freqs) > 0:
        ax.hist(learned_freqs, bins=15, alpha=0.6, label='Выученные',
                color='#DC2626', edgecolor='black')
    ax.set_title('Распределение частот', fontweight='600')
    ax.set_xlabel('Частота')
    ax.legend()

    if spectrum_info is not None and n_panels > 2:
        ax = axes[2]
        rf = spectrum_info["radial_freqs"]
        rp = spectrum_info["radial_power"]
        mask = rp > 1e-10
        if mask.any():
            ax.loglog(rf[mask], rp[mask], 'k.-', linewidth=1, markersize=4)

            for f in np.exp(init_freqs):
                ax.axvline(x=f, color='blue', alpha=0.2, linewidth=0.8)
        ax.set_title('Спектр мощности f(x,y)', fontweight='600')
        ax.set_xlabel('Частота')
        ax.set_ylabel('Мощность')

    plt.tight_layout()
    fig.savefig(f'{output_dir}/adaptive_frequencies.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("[NTK] Сохранён график адаптивных частот")
