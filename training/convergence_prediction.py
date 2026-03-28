from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
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
    mode_50: int                        
    mode_90: int                        
    energy_fraction: float = 0.0        

    def get_status(self) -> str:
        if self.t_epsilon < 100: return "FAST"
        elif self.t_epsilon < 1000: return "MODERATE"
        elif self.t_epsilon < 10000: return "SLOW"
        else: return "CRITICAL"

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
    recommendations: List[str] = field(default_factory=list)

    predicted_epochs: np.ndarray = field(default_factory=lambda: np.array([]))
    predicted_loss_pde: np.ndarray = field(default_factory=lambda: np.array([]))
    predicted_loss_bc: np.ndarray = field(default_factory=lambda: np.array([]))
    predicted_loss_total: np.ndarray = field(default_factory=lambda: np.array([]))

    health_score: float = 0.0           
    balance_score: float = 0.0          

def compute_component_metrics(eigenvalues: np.ndarray, name: str, learning_rate: float, target_error: float = 0.01) -> ComponentMetrics:
    eig = np.sort(eigenvalues)[::-1].clip(0)
    eig_pos = eig[eig > 1e-12]
    if len(eig_pos) == 0:
        return ComponentMetrics(name, eig, float('inf'), 0.0, 0.0, float('inf'), float('inf'), 0.0, 0.0, 0, 0)

    kappa = float(eig_pos[0] / eig_pos[-1]) if len(eig_pos) > 1 else float('inf')
    p = eig_pos / eig_pos.sum()
    effective_rank = float(np.exp(-np.sum(p * np.log(p + 1e-30))))
    trace = float(eig_pos.sum())
    lambda_min = float(eig_pos[-1])
    tau_char = 1.0 / (2.0 * lambda_min * learning_rate + 1e-30)
    t_epsilon = -np.log(target_error) * tau_char
    n_top = min(10, len(eig_pos))
    rates = 1.0 - np.exp(-eig_pos[:n_top])
    harmonic_rate = n_top / np.sum(1.0 / (rates + 1e-10))
    spectral_gap = float(eig_pos[1] / eig_pos[0]) if len(eig_pos) > 1 else 0.0
    total_energy = eig_pos.sum()
    cumsum = np.cumsum(eig_pos)
    mode_50 = int(np.searchsorted(cumsum / total_energy, 0.5) + 1)
    mode_90 = int(np.searchsorted(cumsum / total_energy, 0.9) + 1)

    return ComponentMetrics(name, eig, kappa, effective_rank, trace, tau_char, t_epsilon, float(harmonic_rate), spectral_gap, mode_50, mode_90)

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

    epochs_pred = np.array([0, 100, 500, 1000, 2000, 5000, 10000, 20000, 50000])

    def predict_loss(eig: np.ndarray, epochs: np.ndarray, lr: float, L0: float, ntk_factor: float = 0.7, floor: float = 1e-10) -> np.ndarray:
        eig_pos = eig[eig > 1e-12]
        if len(eig_pos) == 0: return np.ones_like(epochs) * L0

        n_top = min(len(eig_pos), 10)
        lambda_eff = n_top / np.sum(1.0 / eig_pos[:n_top])
        n_sustained = min(len(eig_pos), 20)
        weights = np.exp(-np.arange(n_sustained) * 0.2)
        weights = weights / weights.sum()
        lambda_sustained = 1.0 / np.sum(weights / eig_pos[:n_sustained])

        spectral_gap = float(eig_pos[1] / eig_pos[0]) if len(eig_pos) > 1 else 0.1
        transition_time = max(500.0, 100.0 / (spectral_gap + 0.01))
        loss_pred = np.zeros_like(epochs, dtype=float)

        for i, t in enumerate(epochs):
            if i == 0:
                loss_pred[i] = L0
            else:
                dt_epochs = epochs[i] - epochs[i-1]
                progress = 1 - np.exp(-epochs[i] / transition_time)
                lambda_t = lambda_eff * (1 - progress) + lambda_sustained * progress
                dt_steps = dt_epochs * batches_per_epoch
                decay = np.exp(-2 * lambda_t * ntk_factor * lr * dt_steps)
                loss_pred[i] = loss_pred[i-1] * decay
        return np.maximum(loss_pred, floor)

    predicted_loss_pde = predict_loss(eigenvalues_KL, epochs_pred, learning_rate, initial_loss * 0.7)
    bc_initial = initial_loss * 0.3
    if dirichlet is not None and neumann is not None:
        eig_bc = np.concatenate([eigenvalues_KD, eigenvalues_KN])
        predicted_loss_bc = predict_loss(eig_bc, epochs_pred, learning_rate, bc_initial)
    elif dirichlet is not None: predicted_loss_bc = predict_loss(eigenvalues_KD, epochs_pred, learning_rate, bc_initial)
    elif neumann is not None: predicted_loss_bc = predict_loss(eigenvalues_KN, epochs_pred, learning_rate, bc_initial)
    else: predicted_loss_bc = np.zeros_like(epochs_pred)

    predicted_loss_total = predicted_loss_pde + predicted_loss_bc
    health_score = _compute_health_score(pde, dirichlet, neumann)
    balance_score = _compute_balance_score(pde, dirichlet, neumann)
    recommendations = _generate_recommendations(pde, dirichlet, neumann, solution)

    return ConvergencePrediction(
        epoch=epoch, pde=pde, dirichlet=dirichlet, neumann=neumann, solution=solution,
        bottleneck_component=bottleneck_name, bottleneck_epochs=int(bottleneck_time / batches_per_epoch),
        total_epochs_estimate=total_epochs, recommendations=recommendations,
        predicted_epochs=epochs_pred, predicted_loss_pde=predicted_loss_pde,
        predicted_loss_bc=predicted_loss_bc, predicted_loss_total=predicted_loss_total,
        health_score=health_score, balance_score=balance_score,
    )

def _compute_health_score(pde: ComponentMetrics, dirichlet: Optional[ComponentMetrics], neumann: Optional[ComponentMetrics]) -> float:
    score = 100.0
    if pde.condition_number > 1e6: score -= 30
    elif pde.condition_number > 1e5: score -= 20
    elif pde.condition_number > 1e4: score -= 10
    if pde.t_epsilon > 20000: score -= 25
    elif pde.t_epsilon > 10000: score -= 15
    elif pde.t_epsilon > 5000: score -= 5
    if pde.spectral_gap < 0.1: score -= 15
    elif pde.spectral_gap < 0.3: score -= 5
    if dirichlet is not None and dirichlet.t_epsilon > 5000: score -= 10
    if neumann is not None and neumann.t_epsilon > 5000: score -= 10
    return max(0, min(100, score))

def _compute_balance_score(pde: ComponentMetrics, dirichlet: Optional[ComponentMetrics], neumann: Optional[ComponentMetrics]) -> float:
    times = [pde.t_epsilon]
    if dirichlet is not None: times.append(dirichlet.t_epsilon)
    if neumann is not None: times.append(neumann.t_epsilon)
    if len(times) < 2: return 100.0
    t_max, t_min = max(times), min(times)
    if t_max < 1e-6: return 100.0
    return (t_min / t_max) * 100

def _generate_recommendations(pde: ComponentMetrics, dirichlet: Optional[ComponentMetrics], neumann: Optional[ComponentMetrics], solution: ComponentMetrics) -> List[str]:
    recommendations = []
    if pde.t_epsilon > 10000: recommendations.extend([f"⚠ CRITICAL: PDE residual требует ~{int(pde.t_epsilon)} эпох для сходимости", "  → Увеличьте вес PDE loss или learning rate"])
    if pde.condition_number > 1e5: recommendations.extend([f"⚠ Плохая обусловленность PDE: κ(K_L) = {pde.condition_number:.1e}", "  → Рассмотрите спектральную нормализацию сети"])
    if dirichlet is not None and dirichlet.t_epsilon > 5000: recommendations.append(f"⚠ Медленная сходимость Dirichlet BC: ~{int(dirichlet.t_epsilon)} эпох")
    if neumann is not None and neumann.t_epsilon > 5000: recommendations.append(f"⚠ Медленная сходимость Neumann BC: ~{int(neumann.t_epsilon)} эпох")
    if pde.spectral_gap < 0.1: recommendations.append(f"⚠ Малый spectral gap PDE: {pde.spectral_gap:.3f}")
    if len(recommendations) == 0: recommendations.append("✓ Все компоненты хорошо сбалансированы")
    return recommendations

def generate_convergence_report(prediction: ConvergencePrediction, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"convergence_report_epoch{prediction.epoch}.txt")
    lines = [
        "=" * 70, f"NTK CONVERGENCE PREDICTION REPORT — Epoch {prediction.epoch}", "=" * 70, "",
        "1. COMPONENT METRICS", "-" * 70, "",
        f"{'Component':<15} {'κ':>12} {'τ (epochs)':>12} {'t_ε (epochs)':>12} {'Status':>10}", "-" * 70
    ]

    if prediction.pde: lines.append(f"{'PDE (K_L)':<15} {prediction.pde.condition_number:>12.1e} {prediction.pde.tau_char:>12.1f} {prediction.pde.t_epsilon:>12.0f} {prediction.pde.get_status():>10}")
    if prediction.dirichlet: lines.append(f"{'Dirichlet (K_D)':<15} {prediction.dirichlet.condition_number:>12.1e} {prediction.dirichlet.tau_char:>12.1f} {prediction.dirichlet.t_epsilon:>12.0f} {prediction.dirichlet.get_status():>10}")
    if prediction.neumann: lines.append(f"{'Neumann (K_N)':<15} {prediction.neumann.condition_number:>12.1e} {prediction.neumann.tau_char:>12.1f} {prediction.neumann.t_epsilon:>12.0f} {prediction.neumann.get_status():>10}")

    lines.extend([
        "", "2. BOTTLENECK ANALYSIS", "-" * 70,
        f"  Bottleneck component: {prediction.bottleneck_component}",
        f"  Estimated epochs for 1% convergence: {prediction.bottleneck_epochs}",
        f"  Total epochs estimate: {prediction.total_epochs_estimate}",
        f"  Health score: {prediction.health_score:.1f}/100",
        f"  Balance score: {prediction.balance_score:.1f}/100", ""
    ])

    lines.extend(["5. RECOMMENDATIONS", "-" * 70])
    for rec in prediction.recommendations: lines.append(f"  {rec}")
    lines.extend(["", "=" * 70])

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return filepath

class ConvergenceTracker:
    def __init__(self):
        self.predictions: List[ConvergencePrediction] = []
        self.actual_losses: List[Dict[str, float]] = []

    def add_prediction(self, prediction: ConvergencePrediction, actual_loss_pde: Optional[float] = None, actual_loss_bc: Optional[float] = None, actual_loss_total: Optional[float] = None) -> None:
        self.predictions.append(prediction)
        self.actual_losses.append({"pde": actual_loss_pde, "bc": actual_loss_bc, "total": actual_loss_total})