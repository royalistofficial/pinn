from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

from networks.ntk_utils import (
    compute_jacobian,
    compute_pde_jacobian,
    compute_ntk_from_jacobian,
    ntk_spectrum_analysis,
    extract_learned_frequencies,
)

_PALETTE = {
    "K":      "#2563EB",   
    "K_L":    "#DC2626",   
    "diag_K": "#0EA5E9",
    "diag_L": "#F97316",
    "grid":   "#CBD5E1",
    "bg":     "#F8FAFC",
    "text":   "#1E293B",
    "accent": "#7C3AED",
}

_CMAP_K  = "viridis"
_CMAP_KL = "plasma"

def _ax_style(ax: plt.Axes, title: str = "", xlabel: str = "",
              ylabel: str = "", fontsize: int = 11) -> None:
    ax.set_facecolor(_PALETTE["bg"])
    ax.set_title(title, fontsize=fontsize, fontweight="600",
                 color=_PALETTE["text"], pad=8)
    ax.set_xlabel(xlabel, fontsize=9, color=_PALETTE["text"])
    ax.set_ylabel(ylabel, fontsize=9, color=_PALETTE["text"])
    ax.tick_params(colors=_PALETTE["text"], labelsize=8)
    ax.grid(True, alpha=0.4, color=_PALETTE["grid"], linewidth=0.5, zorder=0)
    for sp in ax.spines.values():
        sp.set_color(_PALETTE["grid"])
        sp.set_linewidth(0.7)

def _matrix_panel(
        fig: plt.Figure,
        ax: plt.Axes,
        M: np.ndarray,
        title: str,
        cmap: str,
        node_labels: bool = True,
        label_step: int = 1,
    ) -> None:
    N = M.shape[0]
    vmax = np.abs(M).max()
    if vmax < 1e-30:
        vmax = 1e-8

    if M.min() < -1e-10 * vmax:
        norm = mcolors.TwoSlopeNorm(vmin=M.min(), vcenter=0, vmax=vmax)
        cmap_use = "RdBu_r"
    else:
        norm = mcolors.Normalize(vmin=0, vmax=vmax)
        cmap_use = cmap

    im = ax.imshow(M, cmap=cmap_use, norm=norm, aspect="auto", interpolation="nearest")

    for k in range(N + 1):
        ax.axhline(k - 0.5, color="white", linewidth=0.4, alpha=0.6)
        ax.axvline(k - 0.5, color="white", linewidth=0.4, alpha=0.6)

    ticks = np.arange(0, N, label_step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    if node_labels:
        ax.set_xticklabels([str(t) for t in ticks],
                           rotation=90, fontsize=max(5, 8 - N // 8))
        ax.set_yticklabels([str(t) for t in ticks],
                           fontsize=max(5, 8 - N // 8))
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    ax.set_xlabel("Узел j", fontsize=9, color=_PALETTE["text"])
    ax.set_ylabel("Узел i", fontsize=9, color=_PALETTE["text"])
    ax.set_title(title, fontsize=10, fontweight="600",
                 color=_PALETTE["text"], pad=8)

    cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cb.ax.tick_params(labelsize=7)
    cb.set_label("Значение", fontsize=8)

def _subsample(X: torch.Tensor, n: int) -> torch.Tensor:
    N = len(X)
    if N <= n:
        return X
    idx = torch.linspace(0, N - 1, n, device=X.device).long()
    return X[idx]

def _order_xy(X_np: np.ndarray) -> np.ndarray:
    n_bins = max(1, int(np.sqrt(len(X_np))))
    x_bin  = np.round(X_np[:, 0] * n_bins) / n_bins
    return np.lexsort((X_np[:, 1], x_bin))    

def _order_hilbert(X_np: np.ndarray) -> np.ndarray:
    def _to_hilbert(x: np.ndarray, y: np.ndarray, order: int = 8) -> np.ndarray:
        n  = 1 << order
        ix = np.clip((x * n).astype(int), 0, n - 1).copy()
        iy = np.clip((y * n).astype(int), 0, n - 1).copy()
        d  = np.zeros(len(x), dtype=np.int64)
        s  = n >> 1
        while s > 0:
            rx = ((ix & s) > 0).astype(int)
            ry = ((iy & s) > 0).astype(int)
            d += s * s * ((3 * rx) ^ ry)
            flip = (rx == 1) & (ry == 0)
            ix_new = np.where(flip, n - 1 - iy,
                     np.where(rx == 0, iy, ix))
            iy_new = np.where(flip, n - 1 - ix,
                     np.where(rx == 0, ix, iy))
            ix, iy = ix_new, iy_new
            s >>= 1
        return d

    lo  = X_np.min(axis=0);  hi = X_np.max(axis=0)
    rng = np.where(hi - lo > 1e-10, hi - lo, 1.0)
    xn  = (X_np[:, 0] - lo[0]) / rng[0]
    yn  = (X_np[:, 1] - lo[1]) / rng[1]
    return np.argsort(_to_hilbert(xn, yn))

def _order_spectral_K(K: np.ndarray) -> np.ndarray:
    K_pos     = K - K.min()
    deg       = K_pos.sum(axis=1)
    D_si      = np.where(deg > 1e-12, 1.0 / np.sqrt(deg), 0.0)
    L         = np.eye(len(K)) - D_si[:, None] * K_pos * D_si[None, :]
    try:
        _, vecs = np.linalg.eigh(L)
        return np.argsort(vecs[:, 1])   
    except np.linalg.LinAlgError:
        return np.arange(len(K))

def _order_spectral_KL(K_L: np.ndarray) -> np.ndarray:
    return _order_spectral_K(K_L)

_ORDER_FUNCS = {
    "original":   None,                  
    "xy":         _order_xy,
    "hilbert":    _order_hilbert,
    "spectral_K": _order_spectral_K,     
    "spectral_KL": _order_spectral_KL,   
}

_ORDER_LABELS = {
    "original":    "исходный порядок",
    "xy":          "сортировка x→y",
    "hilbert":     "кривая Гильберта",
    "spectral_K":  "спектральный по K",
    "spectral_KL": "спектральный по K_L",
}

def plot_ntk_full_analysis(
        model: nn.Module,
        epoch: int,
        X_train: torch.Tensor,
        n_pts: int = 64,
        output_dir: str = "data/ntk_plots",
        node_order: str = "hilbert",

    ) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    device = next(model.parameters()).device

    X = _subsample(X_train.to(device), n_pts)   
    X_np = X.detach().cpu().numpy()
    N    = len(X)

    print(f"[NTK] Epoch {epoch}: J   (N={N}) …")
    J   = compute_jacobian(model, X)        
    print(f"[NTK] Epoch {epoch}: J_L (N={N}) …")
    J_L = compute_pde_jacobian(model, X)    

    K   = compute_ntk_from_jacobian(J).cpu().numpy()    
    K_L = compute_ntk_from_jacobian(J_L).cpu().numpy()  

    eig_K  = np.sort(np.linalg.eigvalsh(K))[::-1].clip(0)
    eig_KL = np.sort(np.linalg.eigvalsh(K_L))[::-1].clip(0)

    def _cond(e):
        ep = e[e > 1e-10]
        return float(ep[0] / ep[-1]) if len(ep) > 1 else float("inf")

    def _eff_rank(e):
        ep = e[e > 1e-10]
        if len(ep) == 0:
            return 0.0
        p = ep / ep.sum()
        return float(np.exp(-np.sum(p * np.log(p + 1e-30))))

    kappa_K  = _cond(eig_K)
    kappa_KL = _cond(eig_KL)
    rank_K   = _eff_rank(eig_K)
    rank_KL  = _eff_rank(eig_KL)

    order_key = node_order if node_order in _ORDER_FUNCS else "hilbert"
    order_fn  = _ORDER_FUNCS[order_key]

    if order_fn is None:
        order_idx = np.arange(N)               
    elif order_key in ("spectral_K", "spectral_KL"):

        order_idx = order_fn(K if order_key == "spectral_K" else K_L)
    else:
        order_idx = order_fn(X_np)             

    order_label = _ORDER_LABELS.get(order_key, order_key)

    K_sorted   = K[np.ix_(order_idx, order_idx)]
    K_L_sorted = K_L[np.ix_(order_idx, order_idx)]
    X_sorted   = X_np[order_idx]

    orig_labels = order_idx   

    tick_fs    = max(4, 8 - N // 10)
    label_step = max(1, N // 12)

    fig = plt.figure(figsize=(20, 17), facecolor="white")
    gs  = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

    ax00 = fig.add_subplot(gs[0, 0])
    display_idx = np.arange(N)           
    sc = ax00.scatter(X_sorted[:, 0], X_sorted[:, 1],
                      c=display_idx, cmap="viridis", s=60,
                      edgecolors="k", linewidths=0.4, zorder=5)
    ann_fs = max(4, 7 - N // 15)
    for i in range(N):

        label = f"{i}" if order_key == "original" else f"{i}\n({orig_labels[i]})"
        ax00.annotate(label, (X_sorted[i, 0], X_sorted[i, 1]),
                      textcoords="offset points", xytext=(3, 3),
                      fontsize=ann_fs, color=_PALETTE["text"],
                      fontweight="bold", zorder=6)
    cb00 = fig.colorbar(sc, ax=ax00, shrink=0.8, pad=0.02)
    cb00.set_label("Порядок в матрице", fontsize=8)
    cb00.ax.tick_params(labelsize=7)
    ax00.set_aspect("equal")
    _ax_style(ax00,
              title=f"Узлы NTK-анализа (эпоха {epoch})\n"
                    f"Сортировка: {order_label}  |  N={N}",
              xlabel="x", ylabel="y")

    ax01 = fig.add_subplot(gs[0, 1])
    _matrix_panel(
        fig, ax01, K_sorted,
        title=r"$K(x_i,\,x_j)=\nabla_\theta u_\theta(x_i)^\top"
              r"\nabla_\theta u_\theta(x_j)$"
              f"\nСтандартный NTK  (эпоха {epoch})  [{order_label}]",
        cmap=_CMAP_K, node_labels=True, label_step=label_step,
    )

    ax02 = fig.add_subplot(gs[0, 2])
    _matrix_panel(
        fig, ax02, K_L_sorted,
        title=(r"$K_{\mathcal{L}}(x_i,\,x_j)="
               r"\partial_\theta(\mathcal{L}u_\theta)(x_i)^\top"
               r"\partial_\theta(\mathcal{L}u_\theta)(x_j)$"
               r"$\,$PDE-NTK $\mathcal{L}=-\Delta$"
               f"  (эпоха {epoch})  [{order_label}]"),
        cmap=_CMAP_KL, node_labels=True, label_step=label_step,
    )

    diag_K  = np.diag(K_sorted)
    diag_KL = np.diag(K_L_sorted)
    node_idx = np.arange(N)   

    ax10 = fig.add_subplot(gs[1, 0])
    ax10.bar(node_idx, diag_K, color=_PALETTE["K"], alpha=0.85,
             edgecolor="white", linewidth=0.4, label="diag K")
    from matplotlib.ticker import FixedLocator, FixedFormatter
    step10 = max(1, N // 12)
    ticks10 = node_idx[::step10]
    ax10.xaxis.set_major_locator(FixedLocator(ticks10))
    ax10.xaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks10]))
    ax10.tick_params(axis="x", rotation=90, labelsize=tick_fs)
    _ax_style(ax10,
              title=r"Диагональ $K(x_i,x_i)$ — чувствительность узлов",
              xlabel="Узел i", ylabel=r"$K(x_i,x_i)$")
    ax10.legend(fontsize=8)

    ax11 = fig.add_subplot(gs[1, 1])
    ax11.bar(node_idx, diag_KL, color=_PALETTE["K_L"], alpha=0.85,
             edgecolor="white", linewidth=0.4, label=r"diag $K_{\mathcal{L}}$")
    ax11.xaxis.set_major_locator(FixedLocator(ticks10))
    ax11.xaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks10]))
    ax11.tick_params(axis="x", rotation=90, labelsize=tick_fs)
    _ax_style(ax11,
              title=r"Диагональ $K_{\mathcal{L}}(x_i,x_i)$ — вклад в residual",
              xlabel="Узел i", ylabel=r"$K_{\mathcal{L}}(x_i,x_i)$")
    ax11.legend(fontsize=8)

    ax12 = fig.add_subplot(gs[1, 2])
    sc12 = ax12.scatter(diag_K, diag_KL,
                        c=node_idx, cmap="tab20", s=55,
                        edgecolors="k", linewidths=0.4, zorder=5)
    for i in range(N):
        lbl = str(i) if order_key == "original" else f"{i}({orig_labels[i]})"
        ax12.annotate(lbl, (diag_K[i], diag_KL[i]),
                      textcoords="offset points", xytext=(3, 3),
                      fontsize=max(4, 7 - N // 15),
                      color=_PALETTE["text"], zorder=6)
    cb12 = fig.colorbar(sc12, ax=ax12, shrink=0.8)
    cb12.set_label("Узел", fontsize=8)
    cb12.ax.tick_params(labelsize=7)
    _ax_style(ax12,
              title=r"Диагональ: $K$ vs $K_{\mathcal{L}}$ по узлам",
              xlabel=r"$K(x_i,x_i)$", ylabel=r"$K_{\mathcal{L}}(x_i,x_i)$")

    ax20 = fig.add_subplot(gs[2, 0])
    modes = np.arange(1, N + 1)
    ax20.semilogy(modes, eig_K,  "o-", markersize=4,
                  color=_PALETTE["K"],  linewidth=1.8,
                  label=f"K   κ={kappa_K:.1e}")
    ax20.semilogy(modes, eig_KL, "s-", markersize=4,
                  color=_PALETTE["K_L"], linewidth=1.8,
                  label=rf"$K_{{\mathcal{{L}}}}$  κ={kappa_KL:.1e}")
    ax20.fill_between(modes, eig_K,  alpha=0.10, color=_PALETTE["K"])
    ax20.fill_between(modes, eig_KL, alpha=0.10, color=_PALETTE["K_L"])
    _ax_style(ax20,
              title=f"Спектры собственных значений  (N={N})",
              xlabel="Мода k (убывание)", ylabel="λ_k  (лог. шкала)")
    ax20.legend(fontsize=9)
    ax20.set_xlim(left=1)

    ax21 = fig.add_subplot(gs[2, 1])
    bar_labels  = [r"$\kappa(K)$", r"$\kappa(K_{\mathcal{L}})$",
                   r"$\mathrm{rank}_{eff}(K)$",
                   r"$\mathrm{rank}_{eff}(K_{\mathcal{L}})$"]
    bar_vals = [
        np.log10(max(kappa_K,  1.0)),
        np.log10(max(kappa_KL, 1.0)),
        np.log10(max(rank_K,   1.0)),
        np.log10(max(rank_KL,  1.0)),
    ]
    raw_vals = [kappa_K, kappa_KL, rank_K, rank_KL]
    bar_colors = [_PALETTE["K"], _PALETTE["K_L"],
                  _PALETTE["diag_K"], _PALETTE["diag_L"]]
    x_pos = np.arange(len(bar_labels))
    bars = ax21.bar(x_pos, bar_vals, color=bar_colors,
                    alpha=0.85, edgecolor="white", linewidth=0.5)
    for bar, rv in zip(bars, raw_vals):
        ht = bar.get_height()
        lbl = f"{rv:.2e}" if rv > 999 else f"{rv:.2f}"
        ax21.text(bar.get_x() + bar.get_width() / 2, ht + 0.04,
                  lbl, ha="center", va="bottom",
                  fontsize=8, color=_PALETTE["text"])

    ax21.xaxis.set_major_locator(FixedLocator(x_pos))
    ax21.xaxis.set_major_formatter(FixedFormatter(bar_labels))
    ax21.tick_params(axis="x", rotation=10, labelsize=9)
    _ax_style(ax21,
              title="Обусловленность и эффективный ранг\n(log₁₀ масштаб)",
              xlabel="", ylabel="log₁₀(значение)")

    ax22 = fig.add_subplot(gs[2, 2])
    n_modes  = min(32, N)
    m_idx    = np.arange(1, n_modes + 1)
    rates_K  = 1.0 - np.exp(-eig_K[:n_modes].clip(0))
    rates_KL = 1.0 - np.exp(-eig_KL[:n_modes].clip(0))
    ax22.bar(m_idx - 0.2, rates_K,  0.35, color=_PALETTE["K"],
             alpha=0.85, edgecolor="white", linewidth=0.4, label="K")
    ax22.bar(m_idx + 0.2, rates_KL, 0.35, color=_PALETTE["K_L"],
             alpha=0.85, edgecolor="white", linewidth=0.4,
             label=r"$K_{\mathcal{L}}$")
    ax22.axhline(0.5, color="grey", ls="--", lw=1, alpha=0.6, label="50% порог")
    _ax_style(ax22,
              title=r"Скорость сходимости мод: $1-e^{-\lambda_k}$",
              xlabel="Мода k", ylabel="Скорость")
    ax22.set_xlim(0.5, n_modes + 0.5)
    ax22.set_ylim(0, 1.05)
    ax22.legend(fontsize=9)

    fig.suptitle(
        f"NTK Анализ PINN — Эпоха {epoch}  |  N={N} точек\n"
        r"$K=\nabla_\theta u^\top\nabla_\theta u$"
        r"  |  "
        r"$K_{\mathcal{L}}=\partial_\theta(\mathcal{L}u)^\top"
        r"\partial_\theta(\mathcal{L}u)$,  $\mathcal{L}=-\Delta$",
        fontsize=12, fontweight="700", color=_PALETTE["text"], y=1.01,
    )

    out_path = os.path.join(output_dir, f"ntk_full_analysis_{epoch:04d}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[NTK] Saved → {out_path}")

    return {
        "K":        K,
        "K_L":      K_L,
        "J":        J.cpu().numpy(),
        "J_L":      J_L.cpu().numpy(),
        "X":        X_np,
        "eig_K":    eig_K,
        "eig_KL":   eig_KL,
        "kappa_K":  kappa_K,
        "kappa_KL": kappa_KL,
        "order_idx": order_idx,      
        "order_key": order_key,
    }

def plot_spectrum_evolution(
        spectra_history: list[dict],
        epochs: list[int],
        output_dir: str = "data/ntk_plots",
    ) -> None:
    if len(spectra_history) < 2:
        return

    os.makedirs(output_dir, exist_ok=True)

    has_KL = all("eig_KL" in s for s in spectra_history)

    ncols = 4 if has_KL else 3
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5), facecolor="white")
    cmap_ev = plt.cm.viridis(np.linspace(0, 1, len(spectra_history)))

    ax = axes[0]
    for i, (sp, ep) in enumerate(zip(spectra_history, epochs)):
        eig = sp.get("eig_K", sp.get("eigenvalues", []))
        if len(eig):
            ax.semilogy(np.arange(1, len(eig) + 1), eig,
                        "o-", markersize=2, color=cmap_ev[i],
                        linewidth=1, label=f"ep {ep}")
    _ax_style(ax, title="Эволюция спектра K",
              xlabel="Мода k", ylabel="λ_k")
    ax.legend(fontsize=6, ncol=2)

    if has_KL:
        ax = axes[1]
        for i, (sp, ep) in enumerate(zip(spectra_history, epochs)):
            eig = sp.get("eig_KL", [])
            if len(eig):
                ax.semilogy(np.arange(1, len(eig) + 1), eig,
                            "s-", markersize=2, color=cmap_ev[i],
                            linewidth=1, label=f"ep {ep}")
        _ax_style(ax, title=r"Эволюция спектра $K_\mathcal{L}$",
                  xlabel="Мода k", ylabel="λ_k")
        ax.legend(fontsize=6, ncol=2)
        start_cond = 2
    else:
        start_cond = 1

    ax = axes[start_cond]
    kappas_K  = [s.get("kappa_K",  s.get("condition_number", float("nan")))
                 for s in spectra_history]
    ax.semilogy(epochs, kappas_K, "o-", color=_PALETTE["K"],
                markersize=6, linewidth=1.8, label="κ(K)")
    if has_KL:
        kappas_KL = [s.get("kappa_KL", float("nan")) for s in spectra_history]
        ax.semilogy(epochs, kappas_KL, "s-", color=_PALETTE["K_L"],
                    markersize=6, linewidth=1.8, label=r"κ($K_\mathcal{L}$)")
    _ax_style(ax, title="Число обусловленности κ",
              xlabel="Эпоха", ylabel="κ")
    ax.legend(fontsize=9)

    ax = axes[start_cond + 1]
    ranks_K = [s.get("rank_K", float("nan")) for s in spectra_history]
    ax.plot(epochs, ranks_K, "o-", color=_PALETTE["K"],
            markersize=6, linewidth=1.8, label=r"rank$_{eff}(K)$")
    if has_KL:
        ranks_KL = [s.get("rank_KL", float("nan")) for s in spectra_history]
        ax.plot(epochs, ranks_KL, "s-", color=_PALETTE["K_L"],
                markersize=6, linewidth=1.8, label=r"rank$_{eff}(K_\mathcal{L})$")
    _ax_style(ax, title="Эффективный ранг",
              xlabel="Эпоха", ylabel="ранг")
    ax.legend(fontsize=9)

    fig.suptitle("Эволюция NTK-спектров в процессе обучения",
                 fontsize=13, fontweight="700", color=_PALETTE["text"])
    fig.tight_layout()

    out_path = os.path.join(output_dir, "ntk_spectrum_evolution.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[NTK] Saved spectrum evolution → {out_path}")

def plot_adaptive_frequencies(
    init_freqs: np.ndarray,
    learned_freqs: np.ndarray,
    spectrum_info: Optional[dict] = None,
    output_dir: str = "data/ntk_plots",
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    n_panels = 3 if spectrum_info is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5),
                             facecolor="white")

    ax = axes[0]
    if len(init_freqs) > 0:
        ax.stem(np.arange(len(init_freqs)), np.exp(init_freqs),
                linefmt="b-", markerfmt="bo", basefmt="gray",
                label="Начальные")
    if len(learned_freqs) > 0:
        ax.stem(np.arange(len(learned_freqs)), learned_freqs,
                linefmt="r-", markerfmt="r^", basefmt="gray",
                label="Выученные")
    _ax_style(ax, title="Частоты Фурье: начальные vs выученные",
              xlabel="Индекс", ylabel="Частота")
    ax.legend(fontsize=9)

    ax = axes[1]
    if len(init_freqs) > 0:
        ax.hist(np.exp(init_freqs), bins=15, alpha=0.65,
                label="Начальные", color=_PALETTE["K"], edgecolor="white")
    if len(learned_freqs) > 0:
        ax.hist(learned_freqs, bins=15, alpha=0.65,
                label="Выученные", color=_PALETTE["K_L"], edgecolor="white")
    _ax_style(ax, title="Распределение частот",
              xlabel="Частота", ylabel="Количество")
    ax.legend(fontsize=9)

    if spectrum_info is not None and n_panels > 2:
        ax = axes[2]
        rf = spectrum_info["radial_freqs"]
        rp = spectrum_info["radial_power"]
        mask = rp > 1e-10
        if mask.any():
            ax.loglog(rf[mask], rp[mask], "k.-", linewidth=1, markersize=4,
                      label="Спектр мощности f(x)")
        for f in np.exp(init_freqs):
            ax.axvline(x=f, color=_PALETTE["K"], alpha=0.25, linewidth=1)
        _ax_style(ax, title="Спектр мощности правой части f(x,y)",
                  xlabel="Частота", ylabel="Мощность")
        ax.legend(fontsize=9)

    fig.suptitle("Адаптивная инициализация частот Фурье",
                 fontsize=12, fontweight="700", color=_PALETTE["text"])
    fig.tight_layout()

    out_path = os.path.join(output_dir, "adaptive_frequencies.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[NTK] Saved adaptive frequencies → {out_path}")

def plot_ntk_analysis(
        model: nn.Module,
        epoch: int,
        X_train: Optional[torch.Tensor] = None,
        output_dir: str = "data/ntk_plots",
    ) -> None:
    if X_train is None:
        device = next(model.parameters()).device
        grid = torch.linspace(-1, 1, 8)
        X_, Y_ = torch.meshgrid(grid, grid, indexing="ij")
        X_train = torch.stack([X_.flatten(), Y_.flatten()], dim=1).to(device)

    plot_ntk_full_analysis(
        model=model,
        epoch=epoch,
        X_train=X_train,
        n_vis=16,
        n_spectrum=min(64, len(X_train)),
        output_dir=output_dir,
    )