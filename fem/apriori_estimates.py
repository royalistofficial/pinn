from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import numpy as np

@dataclass
class APrioriEstimate:
    h: float                    
    theoretical_rate_l2: float  
    theoretical_rate_h1: float  
    actual_error_l2: float      
    actual_error_h1: float      
    n_dof: int                  

@dataclass
class ConvergenceStudy:
    estimates: List[APrioriEstimate]
    computed_rate_l2: float     
    computed_rate_h1: float     
    theoretical_rate_l2: float  
    theoretical_rate_h1: float  
    effectivity_l2: float       
    effectivity_h1: float       

def compute_regularity_exponent(domain_name: str) -> float:

    max_angles = {
        "square": math.pi / 2,
        "square_mixed": math.pi / 2,
        "circle": math.pi,  
        "circle_mixed": math.pi,
        "l_shape": 3 * math.pi / 2,  
        "l_shape_mixed": 3 * math.pi / 2,
        "hollow_square": math.pi / 2,
        "hollow_square_mixed": math.pi / 2,
        "p_shape": math.pi / 2,
        "p_shape_mixed": math.pi / 2,
    }

    omega = max_angles.get(domain_name, math.pi)

    if omega <= math.pi:

        return 1.0
    else:

        return math.pi / omega

def theoretical_convergence_rates(
        domain_name: str,
        polynomial_degree: int = 1,
    ) -> Tuple[float, float]:
    alpha = compute_regularity_exponent(domain_name)
    k = polynomial_degree

    rate_h1 = min(k, alpha)

    if alpha >= 1.0:
        rate_l2 = min(k + 1, 2.0)  
    else:
        rate_l2 = min(k + 1, 2 * alpha)  

    return rate_l2, rate_h1

def compute_convergence_rate(
        h_values: np.ndarray,
        error_values: np.ndarray,
    ) -> float:
    if len(h_values) < 2 or np.any(error_values <= 0) or np.any(h_values <= 0):
        return 0.0

    log_h = np.log(h_values)
    log_e = np.log(error_values)

    A = np.vstack([log_h, np.ones_like(log_h)]).T
    result = np.linalg.lstsq(A, log_e, rcond=None)
    return result[0][0]

def analyze_convergence(
        estimates: List[APrioriEstimate],
        domain_name: str,
    ) -> ConvergenceStudy:
    rate_l2_th, rate_h1_th = theoretical_convergence_rates(domain_name)

    h_vals = np.array([e.h for e in estimates])
    l2_vals = np.array([e.actual_error_l2 for e in estimates])
    h1_vals = np.array([e.actual_error_h1 for e in estimates])

    rate_l2_comp = compute_convergence_rate(h_vals, l2_vals)
    rate_h1_comp = compute_convergence_rate(h_vals, h1_vals) if np.all(h1_vals > 0) else 0.0

    eff_l2 = rate_l2_comp / rate_l2_th if rate_l2_th > 0 else 0.0
    eff_h1 = rate_h1_comp / rate_h1_th if rate_h1_th > 0 else 0.0

    return ConvergenceStudy(
        estimates=estimates,
        computed_rate_l2=rate_l2_comp,
        computed_rate_h1=rate_h1_comp,
        theoretical_rate_l2=rate_l2_th,
        theoretical_rate_h1=rate_h1_th,
        effectivity_l2=eff_l2,
        effectivity_h1=eff_h1,
    )

def format_apriori_report(study: ConvergenceStudy, domain_name: str) -> str:
    alpha = compute_regularity_exponent(domain_name)
    lines = [
        "=" * 70,
        f"  АПРИОРНЫЕ ОЦЕНКИ ОШИБКИ МКЭ: {domain_name}",
        "=" * 70,
        "",
        f"  Показатель регулярности α = π/ω = {alpha:.4f}",
        f"  u ∈ H^{{1+α}} = H^{{{1+alpha:.4f}}}",
        "",
        f"  Теоретические порядки сходимости (P1 элементы):",
        f"    ||u - u_h||_{{L2}} = O(h^{{{study.theoretical_rate_l2:.4f}}})",
        f"    ||u - u_h||_{{H1}} = O(h^{{{study.theoretical_rate_h1:.4f}}})",
        "",
        "  Результаты расчётов:",
        f"  {'h':>10s} {'N_dof':>8s} {'L2 error':>12s} {'H1 error':>12s}",
        "  " + "-" * 50,
    ]

    for e in study.estimates:
        lines.append(
            f"  {e.h:10.4e} {e.n_dof:8d} {e.actual_error_l2:12.4e} {e.actual_error_h1:12.4e}"
        )

    lines += [
        "",
        f"  Вычисленные порядки сходимости:",
        f"    L2: {study.computed_rate_l2:.4f} (теор: {study.theoretical_rate_l2:.4f}, "
        f"эфф: {study.effectivity_l2:.2f})",
        f"    H1: {study.computed_rate_h1:.4f} (теор: {study.theoretical_rate_h1:.4f}, "
        f"эфф: {study.effectivity_h1:.2f})",
        "",
    ]

    if study.effectivity_l2 > 0.8:
        lines.append("  ✓ L2 сходимость соответствует теории")
    else:
        lines.append("  ✗ L2 сходимость ниже теоретической (возможна сингулярность)")

    if study.effectivity_h1 > 0.8:
        lines.append("  ✓ H1 сходимость соответствует теории")
    else:
        lines.append("  ✗ H1 сходимость ниже теоретической (возможна сингулярность)")

    lines.append("=" * 70)
    return "\n".join(lines)
