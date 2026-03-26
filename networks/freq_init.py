from __future__ import annotations
import math
from typing import Optional
import numpy as np
import torch

from geometry.domains import BaseDomain
from geometry.mesher import get_inside_mask
from problems.solutions import AnalyticalSolution
from fem.apriori_estimates import compute_regularity_exponent

def estimate_rhs_spectrum(
        solution: AnalyticalSolution,
        domain: BaseDomain,
        n_points: int = 500,
        n_grid: int = 64,
    ) -> dict:
    bv = domain.boundary_vertices()
    bs = domain.boundary_segments()
    xmin, ymin = bv.min(axis=0)
    xmax, ymax = bv.max(axis=0)

    xs = np.linspace(xmin, xmax, n_grid)
    ys = np.linspace(ymin, ymax, n_grid)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    mask = get_inside_mask(grid, bv, bs)

    f_grid = np.zeros(len(grid))
    xy_t = torch.tensor(grid, dtype=torch.float32)
    with torch.no_grad():
        f_vals = solution.rhs(xy_t).numpy().flatten()
    f_grid = f_vals.copy()
    f_grid[~mask] = 0.0  

    f_2d = f_grid.reshape(n_grid, n_grid)
    F_2d = np.fft.fft2(f_2d)
    power_2d = np.abs(F_2d) ** 2

    center = n_grid // 2
    freq_x = np.fft.fftfreq(n_grid, d=(xmax - xmin) / n_grid)
    freq_y = np.fft.fftfreq(n_grid, d=(ymax - ymin) / n_grid)
    fx, fy = np.meshgrid(freq_x, freq_y)
    radii = np.sqrt(fx ** 2 + fy ** 2)

    max_freq = radii.max()
    n_bins = min(n_grid // 2, 32)
    bin_edges = np.linspace(0, max_freq, n_bins + 1)
    radial_power = np.zeros(n_bins)
    radial_freqs = np.zeros(n_bins)

    for b in range(n_bins):
        mask_bin = (radii >= bin_edges[b]) & (radii < bin_edges[b + 1])
        if mask_bin.any():
            radial_power[b] = power_2d[mask_bin].mean()
            radial_freqs[b] = 0.5 * (bin_edges[b] + bin_edges[b + 1])

    nonzero = radial_power > 1e-10
    if nonzero.any():
        order = np.argsort(radial_power[nonzero])[::-1]
        dominant = radial_freqs[nonzero][order]
    else:
        dominant = np.array([1.0])

    pos_mask = (radial_freqs > 1e-6) & (radial_power > 1e-10)
    if pos_mask.sum() >= 3:
        log_f = np.log(radial_freqs[pos_mask])
        log_p = np.log(radial_power[pos_mask])
        A = np.vstack([log_f, np.ones_like(log_f)]).T
        result = np.linalg.lstsq(A, log_p, rcond=None)
        decay_rate = -result[0][0]
    else:
        decay_rate = 2.0  

    return {
        "radial_freqs": radial_freqs,
        "radial_power": radial_power,
        "dominant_freqs": dominant,
        "spectral_decay_rate": decay_rate,
    }

def compute_adaptive_frequencies(
    solution: AnalyticalSolution,
    domain: BaseDomain,
    n_fourier: int,
    n_points: int = 500,
) -> torch.Tensor:

    spectrum = estimate_rhs_spectrum(solution, domain, n_points=n_points)

    alpha = compute_regularity_exponent(domain.name)

    dominant = spectrum["dominant_freqs"]
    decay = spectrum["spectral_decay_rate"]

    bv = domain.boundary_vertices()
    diam = np.linalg.norm(bv.max(axis=0) - bv.min(axis=0))
    f_min = max(0.5 / diam, 0.05)

    if alpha < 1.0:
        f_max = max(dominant[0] if len(dominant) > 0 else 3.0, 3.0 / alpha)
    else:
        f_max = max(dominant[0] if len(dominant) > 0 else 3.0, 4.0)

    f_max = min(f_max, 20.0)  

    if alpha < 1.0:

        n_low = max(1, n_fourier // 5)
        n_mid = max(1, n_fourier * 3 // 10)
        n_high = n_fourier - n_low - n_mid

        f_boundary = f_min + (f_max - f_min) / 3
        f_mid_boundary = f_min + 2 * (f_max - f_min) / 3

        freqs_low = np.linspace(f_min, f_boundary, n_low)
        freqs_mid = np.linspace(f_boundary, f_mid_boundary, n_mid)
        freqs_high = np.linspace(f_mid_boundary, f_max, n_high)

        all_freqs = np.concatenate([freqs_low, freqs_mid, freqs_high])
    else:

        all_freqs = np.exp(np.linspace(np.log(f_min), np.log(f_max), n_fourier))

    n_dominant = min(len(dominant), n_fourier // 3)
    for i in range(n_dominant):
        if dominant[i] > 0 and dominant[i] < f_max:

            idx = np.argmin(np.abs(all_freqs - dominant[i]))
            all_freqs[idx] = dominant[i]

    all_freqs = np.sort(all_freqs)
    log_freqs = np.log(np.clip(all_freqs, 1e-3, None))

    print(f"[AdaptiveFreq] α={alpha:.3f}, decay={decay:.2f}, "
          f"f_range=[{f_min:.3f}, {f_max:.3f}], "
          f"dominant_top3={dominant[:3].tolist()}")

    return torch.tensor(log_freqs, dtype=torch.float32)

def compute_adaptive_frequencies_simple(
        domain: BaseDomain,
        n_fourier: int,
        freq_min: float = 0.1,
        freq_max: float = 3.0,
    ) -> torch.Tensor:
    alpha = compute_regularity_exponent(domain.name)

    if alpha < 1.0:

        t = np.linspace(0, 1, n_fourier) ** (1.0 / (alpha + 0.1))
        freqs = freq_min + (freq_max / alpha - freq_min) * t
    else:

        freqs = np.exp(np.linspace(np.log(freq_min), np.log(freq_max), n_fourier))

    log_freqs = np.log(np.clip(freqs, 1e-3, None))
    return torch.tensor(log_freqs, dtype=torch.float32)
