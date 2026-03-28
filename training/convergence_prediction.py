from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import os
import numpy as np

@dataclass
class ComponentMetrics:
    name: str                           
    eigenvalues: np.ndarray             
    condition_number: float             
    effective_rank: float               
    trace: float                        
    tau_char: float                     
    t_epsilon: float                    
    convergence_rate: float             
    spectral_gap: float                 

@dataclass
class ConvergencePrediction:
    epoch: int                          
    pde: Optional[ComponentMetrics] = None
    dirichlet: Optional[ComponentMetrics] = None
    neumann: Optional[ComponentMetrics] = None
    solution: Optional[ComponentMetrics] = None  

    bottleneck_component: str = ""      
    bottleneck_epochs: int = 0          
    total_epochs_estimate: int = 0      

def compute_component_metrics(eigenvalues: np.ndarray, name: str, learning_rate: float, target_error: float = 0.01) -> ComponentMetrics:
    eig = np.sort(eigenvalues)[::-1]
    eig = np.clip(eig, 1e-12, None)

    kappa = float(eig[0] / eig[-1])
    p = eig / eig.sum()
    effective_rank = float(np.exp(-np.sum(p * np.log(p + 1e-30))))
    trace = float(eig.sum())
    lambda_min = float(eig[-1])

    tau_char = 1.0 / (2.0 * lambda_min * learning_rate)
    t_epsilon = -np.log(target_error) * tau_char

    n_top = min(10, len(eig))
    rates = 1.0 - np.exp(-eig[:n_top])
    harmonic_rate = n_top / np.sum(1.0 / (rates + 1e-10))
    spectral_gap = float(eig[1] / eig[0]) if len(eig) > 1 else 0.0

    return ComponentMetrics(name, eig, kappa, effective_rank, trace, tau_char, t_epsilon, float(harmonic_rate), spectral_gap)

def compute_convergence_prediction(
    eigenvalues_K: np.ndarray, eigenvalues_KL: np.ndarray,
    eigenvalues_KD: Optional[np.ndarray] = None, eigenvalues_KN: Optional[np.ndarray] = None,
    learning_rate: float = 1e-3, target_error: float = 0.01,
    epoch: int = 0, initial_loss: float = 1.0, batches_per_epoch: int = 10,
) -> ConvergencePrediction:

    solution = compute_component_metrics(eigenvalues_K, "Solution", learning_rate, target_error)
    pde = compute_component_metrics(eigenvalues_KL, "PDE", learning_rate, target_error)

    dirichlet = None
    if eigenvalues_KD is not None and len(eigenvalues_KD) > 0:
        dirichlet = compute_component_metrics(eigenvalues_KD, "Dirichlet", learning_rate, target_error)

    neumann = None
    if eigenvalues_KN is not None and len(eigenvalues_KN) > 0:
        neumann = compute_component_metrics(eigenvalues_KN, "Neumann", learning_rate, target_error)

    times = [("PDE", pde.t_epsilon)]
    if dirichlet is not None: times.append(("Dirichlet", dirichlet.t_epsilon))
    if neumann is not None: times.append(("Neumann", neumann.t_epsilon))

    bottleneck_name, bottleneck_time = max(times, key=lambda x: x[1])
    total_epochs = int(min(bottleneck_time * 1.2 / batches_per_epoch, 100000))  

    return ConvergencePrediction(
        epoch=epoch, pde=pde, dirichlet=dirichlet, neumann=neumann, solution=solution,
        bottleneck_component=bottleneck_name, bottleneck_epochs=int(bottleneck_time / batches_per_epoch),
        total_epochs_estimate=total_epochs
    )

def generate_convergence_report(prediction: ConvergencePrediction, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"spectral_report_epoch{prediction.epoch}.txt")
    lines = [
        "=" * 70, f"NTK SPECTRAL ANALYSIS REPORT — Epoch {prediction.epoch}", "=" * 70, "",
        "1. COMPONENT METRICS", "-" * 70, "",
        f"{'Component':<15} {'κ':>12} {'τ (epochs)':>12} {'t_ε (epochs)':>12}", "-" * 70
    ]

    if prediction.pde: lines.append(f"{'PDE (K_L)':<15} {prediction.pde.condition_number:>12.1e} {prediction.pde.tau_char:>12.1f} {prediction.pde.t_epsilon:>12.0f}")
    if prediction.dirichlet: lines.append(f"{'Dirichlet (K_D)':<15} {prediction.dirichlet.condition_number:>12.1e} {prediction.dirichlet.tau_char:>12.1f} {prediction.dirichlet.t_epsilon:>12.0f}")
    if prediction.neumann: lines.append(f"{'Neumann (K_N)':<15} {prediction.neumann.condition_number:>12.1e} {prediction.neumann.tau_char:>12.1f} {prediction.neumann.t_epsilon:>12.0f}")

    lines.extend([
        "", "2. CONVERGENCE ESTIMATES", "-" * 70,
        f"  Bottleneck component: {prediction.bottleneck_component}",
        f"  Total epochs estimate: {prediction.total_epochs_estimate}", ""
    ])

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return filepath