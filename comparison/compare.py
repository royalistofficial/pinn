from __future__ import annotations
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from geometry.domains import BaseDomain, make_domain
from geometry.mesher import Mesher
from geometry.quadrature import QuadratureBuilder
from problems.solutions import AnalyticalSolution
from fem.solver import FEMSolver, FEMMesh, build_fem_mesh, FEMResult
from fem.apriori_estimates import (
    APrioriEstimate, ConvergenceStudy,
    theoretical_convergence_rates, compute_convergence_rate,
    analyze_convergence, format_apriori_report, compute_regularity_exponent,
)
from config import DEVICE

@dataclass
class ComparisonResult:
    domain_name: str

    fem_errors: Dict[str, float]
    fem_time: float
    fem_n_dof: int
    fem_h_max: float

    pinn_errors: Dict[str, float]
    pinn_time: float
    pinn_n_params: int

    convergence_study: Optional[ConvergenceStudy] = None

def run_fem_solution(
        domain: BaseDomain,
        solution: AnalyticalSolution,
        max_area: float = 0.01,
    ) -> Tuple[FEMResult, Dict[str, float], float]:
    t0 = time.time()

    mesh = build_fem_mesh(domain, max_area=max_area)

    def f_func(x, y):
        xy = torch.tensor([[x, y]], dtype=torch.float32)
        return solution.rhs(xy).item()

    def u_func(x, y):
        xy = torch.tensor([[x, y]], dtype=torch.float32)
        return solution.eval(xy).item()

    def grad_func(x, y):
        xy = torch.tensor([[x, y]], dtype=torch.float32)
        gx, gy = solution.grad(xy)
        return gx.item(), gy.item()

    solver = FEMSolver(
        mesh=mesh,
        f_func=f_func,
        u_exact_func=u_func,
        grad_exact_func=grad_func,
    )

    result = solver.solve()
    errors = solver.compute_errors()
    elapsed = time.time() - t0

    return result, errors, elapsed

def run_fem_convergence_study(
        domain: BaseDomain,
        solution: AnalyticalSolution,
        area_levels: List[float] = None,
    ) -> ConvergenceStudy:
    if area_levels is None:
        area_levels = [0.1, 0.05, 0.02, 0.01, 0.005]

    estimates = []
    for area in area_levels:
        result, errors, _ = run_fem_solution(domain, solution, max_area=area)

        est = APrioriEstimate(
            h=result.h_max,
            theoretical_rate_l2=0.0,  
            theoretical_rate_h1=0.0,
            actual_error_l2=errors.get("l2_error", 0.0),
            actual_error_h1=errors.get("energy_error", 0.0),
            n_dof=result.n_dof,
        )
        estimates.append(est)
        print(f"  h={result.h_max:.4e}, N={result.n_dof:6d}, "
              f"L2={errors.get('l2_error', 0):.4e}, "
              f"H1={errors.get('energy_error', 0):.4e}")

    study = analyze_convergence(estimates, domain.name)
    return study

def evaluate_pinn_errors(
        pinn: torch.nn.Module,
        solution: AnalyticalSolution,
        domain: BaseDomain,
        n_eval_pts: int = 5000,
    ) -> Dict[str, float]:
    from geometry.mesher import get_inside_mask

    bv = domain.boundary_vertices()
    bs = domain.boundary_segments()

    xmin, ymin = bv.min(axis=0)
    xmax, ymax = bv.max(axis=0)

    pts_list = []
    while len(pts_list) < n_eval_pts:
        cands = np.random.uniform(
            [xmin, ymin], [xmax, ymax],
            size=(n_eval_pts * 2, 2)
        )
        mask = get_inside_mask(cands, bv, bs)
        pts_list.extend(cands[mask].tolist())

    pts = np.array(pts_list[:n_eval_pts])
    xy = torch.tensor(pts, dtype=torch.float32, device=DEVICE)

    pinn.eval()
    xy_grad = xy.clone().requires_grad_(True)

    with torch.enable_grad():
        v = pinn(xy_grad)
        gv = torch.autograd.grad(v, xy_grad, torch.ones_like(v), create_graph=False)[0]

    with torch.no_grad():
        u_exact = solution.eval(xy)
        g_exact = solution.grad_vector(xy)

        l2_err = torch.sqrt(torch.mean((v.detach() - u_exact) ** 2)).item()
        l2_norm = torch.sqrt(torch.mean(u_exact ** 2)).item()

        h1_err = torch.sqrt(torch.mean((gv.detach() - g_exact) ** 2)).item()
        h1_norm = torch.sqrt(torch.mean(g_exact ** 2)).item()

    pinn.train()

    return {
        "l2_error": l2_err,
        "relative_l2": l2_err / max(l2_norm, 1e-30),
        "energy_error": h1_err,
        "relative_energy": h1_err / max(h1_norm, 1e-30),
    }

def plot_comparison(
        fem_result: FEMResult,
        pinn_model: torch.nn.Module,
        solution: AnalyticalSolution,
        domain: BaseDomain,
        save_path: str,
    ):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    pts = fem_result.mesh.points
    elems = fem_result.mesh.elements
    tri = mtri.Triangulation(pts[:, 0], pts[:, 1], elems)

    u_exact = np.array([
        solution.eval(torch.tensor([[x, y]], dtype=torch.float32)).item()
        for x, y in pts
    ])
    axes[0, 0].tripcolor(tri, u_exact, shading='gouraud', cmap='viridis')
    axes[0, 0].set_title('Точное решение u(x,y)')
    axes[0, 0].set_aspect('equal')

    tc = axes[0, 1].tripcolor(tri, fem_result.u, shading='gouraud', cmap='viridis')
    axes[0, 1].set_title(f'МКЭ (N={fem_result.n_dof})')
    axes[0, 1].set_aspect('equal')
    plt.colorbar(tc, ax=axes[0, 1])

    xy_torch = torch.tensor(pts, dtype=torch.float32, device=DEVICE)
    pinn_model.eval()
    with torch.no_grad():
        u_pinn = pinn_model(xy_torch).cpu().numpy().flatten()
    pinn_model.train()

    tc = axes[0, 2].tripcolor(tri, u_pinn, shading='gouraud', cmap='viridis')
    axes[0, 2].set_title('PINN')
    axes[0, 2].set_aspect('equal')
    plt.colorbar(tc, ax=axes[0, 2])

    err_fem = np.abs(fem_result.u - u_exact)
    tc = axes[1, 0].tripcolor(tri, err_fem, shading='gouraud', cmap='hot')
    axes[1, 0].set_title(f'|ошибка МКЭ| (max={err_fem.max():.2e})')
    axes[1, 0].set_aspect('equal')
    plt.colorbar(tc, ax=axes[1, 0])

    err_pinn = np.abs(u_pinn - u_exact)
    tc = axes[1, 1].tripcolor(tri, err_pinn, shading='gouraud', cmap='hot')
    axes[1, 1].set_title(f'|ошибка PINN| (max={err_pinn.max():.2e})')
    axes[1, 1].set_aspect('equal')
    plt.colorbar(tc, ax=axes[1, 1])

    ax = axes[1, 2]
    ax.scatter(err_fem, err_pinn, s=3, alpha=0.3, color='steelblue')
    max_err = max(err_fem.max(), err_pinn.max())
    ax.plot([0, max_err], [0, max_err], 'r--', alpha=0.5, label='y=x')
    ax.set_xlabel('|ошибка МКЭ|')
    ax.set_ylabel('|ошибка PINN|')
    ax.set_title('Поточечное сравнение ошибок')
    ax.legend()
    ax.set_aspect('equal')

    plt.suptitle(f'Сравнение МКЭ и PINN: {domain.name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_convergence_study(
        study: ConvergenceStudy,
        domain_name: str,
        pinn_errors: Optional[Dict[str, float]] = None,
        save_path: str = "data/convergence.png",
    ):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    h_vals = np.array([e.h for e in study.estimates])
    l2_vals = np.array([e.actual_error_l2 for e in study.estimates])
    h1_vals = np.array([e.actual_error_h1 for e in study.estimates])

    ax = axes[0]
    ax.loglog(h_vals, l2_vals, 'bo-', markersize=8, linewidth=2, label='МКЭ (факт.)')

    h_th = np.linspace(h_vals.min() * 0.5, h_vals.max() * 2, 50)
    C_l2 = l2_vals[-1] / h_vals[-1] ** study.theoretical_rate_l2
    ax.loglog(h_th, C_l2 * h_th ** study.theoretical_rate_l2,
              'b--', alpha=0.5, label=f'O(h^{{{study.theoretical_rate_l2:.2f}}}) теор.')

    if pinn_errors and "l2_error" in pinn_errors:
        ax.axhline(y=pinn_errors["l2_error"], color='r', ls='-.', linewidth=2,
                   label=f'PINN L2={pinn_errors["l2_error"]:.2e}')

    ax.set_xlabel('h (размер элемента)', fontsize=11)
    ax.set_ylabel('||u - u_h||_{L2}', fontsize=11)
    ax.set_title(f'L2 сходимость (p={study.computed_rate_l2:.2f})', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.loglog(h_vals, h1_vals, 'rs-', markersize=8, linewidth=2, label='МКЭ (факт.)')

    C_h1 = h1_vals[-1] / h_vals[-1] ** study.theoretical_rate_h1
    ax.loglog(h_th, C_h1 * h_th ** study.theoretical_rate_h1,
              'r--', alpha=0.5, label=f'O(h^{{{study.theoretical_rate_h1:.2f}}}) теор.')

    if pinn_errors and "energy_error" in pinn_errors:
        ax.axhline(y=pinn_errors["energy_error"], color='b', ls='-.', linewidth=2,
                   label=f'PINN H1={pinn_errors["energy_error"]:.2e}')

    ax.set_xlabel('h (размер элемента)', fontsize=11)
    ax.set_ylabel('||∇(u - u_h)||_{L2}', fontsize=11)
    ax.set_title(f'H1 сходимость (p={study.computed_rate_h1:.2f})', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    alpha = compute_regularity_exponent(domain_name)
    plt.suptitle(
        f'Сходимость МКЭ: {domain_name} (α={alpha:.3f})',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def format_comparison_table(result: ComparisonResult) -> str:
    lines = [
        "=" * 70,
        f"  СРАВНЕНИЕ МКЭ vs PINN: {result.domain_name}",
        "=" * 70,
        "",
        f"  {'Метрика':<30s} {'МКЭ':>15s} {'PINN':>15s}",
        "  " + "-" * 60,
    ]

    metrics = [
        ("L2 ошибка", "l2_error"),
        ("Относительная L2", "relative_l2"),
        ("Энергетическая ошибка", "energy_error"),
        ("Относительная энерг.", "relative_energy"),
    ]

    for label, key in metrics:
        fem_val = result.fem_errors.get(key, float('nan'))
        pinn_val = result.pinn_errors.get(key, float('nan'))
        lines.append(f"  {label:<30s} {fem_val:15.4e} {pinn_val:15.4e}")

    lines += [
        "",
        f"  {'Время решения (с)':<30s} {result.fem_time:15.3f} {result.pinn_time:15.3f}",
        f"  {'Число ст. свободы / парам.':<30s} {result.fem_n_dof:15d} {result.pinn_n_params:15d}",
        f"  {'h_max (МКЭ)':<30s} {result.fem_h_max:15.4e} {'—':>15s}",
        "=" * 70,
    ]
    return "\n".join(lines)
