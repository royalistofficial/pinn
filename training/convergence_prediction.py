from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import os
import numpy as np

@dataclass
class ErrorBounds:
    l2_error_predicted: np.ndarray = field(default_factory=lambda: np.array([]))
    l2_error_upper_bound: float = float('inf')
    energy_error_predicted: np.ndarray = field(default_factory=lambda: np.array([]))
    energy_error_upper_bound: float = float('inf')
    poincare_constant: float = 1.0
    stability_constant: float = 1.0
    residual_norm_init: float = 1.0

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
        if self.t_epsilon < 100:
            return "FAST"
        elif self.t_epsilon < 1000:
            return "MODERATE"
        elif self.t_epsilon < 10000:
            return "SLOW"
        else:
            return "CRITICAL"

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

    error_bounds: Optional[ErrorBounds] = None

    health_score: float = 0.0           
    balance_score: float = 0.0          

def compute_component_metrics(
        eigenvalues: np.ndarray,
        name: str,
        learning_rate: float,
        target_error: float = 0.01,
    ) -> ComponentMetrics:
    eig = np.sort(eigenvalues)[::-1].clip(0)
    eig_pos = eig[eig > 1e-12]

    if len(eig_pos) == 0:

        return ComponentMetrics(
            name=name,
            eigenvalues=eig,
            condition_number=float('inf'),
            effective_rank=0.0,
            trace=0.0,
            tau_char=float('inf'),
            t_epsilon=float('inf'),
            convergence_rate=0.0,
            spectral_gap=0.0,
            mode_50=0,
            mode_90=0,
        )

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

    return ComponentMetrics(
        name=name,
        eigenvalues=eig,
        condition_number=kappa,
        effective_rank=effective_rank,
        trace=trace,
        tau_char=tau_char,
        t_epsilon=t_epsilon,
        convergence_rate=float(harmonic_rate),
        spectral_gap=spectral_gap,
        mode_50=mode_50,
        mode_90=mode_90,
    )

def compute_convergence_prediction(
    eigenvalues_K: np.ndarray,
    eigenvalues_KL: np.ndarray,
    eigenvalues_KD: Optional[np.ndarray] = None,
    eigenvalues_KN: Optional[np.ndarray] = None,
    learning_rate: float = 1e-3,
    target_error: float = 0.01,
    epoch: int = 0,
    initial_loss: float = 1.0,
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
    if dirichlet is not None:
        times.append(("Dirichlet", dirichlet.t_epsilon))
    if neumann is not None:
        times.append(("Neumann", neumann.t_epsilon))

    bottleneck_name, bottleneck_time = max(times, key=lambda x: x[1])

    total_epochs = int(min(bottleneck_time * 1.2, 100000))  

    total_energy = solution.trace + (dirichlet.trace if dirichlet else 0) + (neumann.trace if neumann else 0)
    if total_energy > 0:
        solution.energy_fraction = solution.trace / total_energy
        pde.energy_fraction = pde.trace / total_energy
        if dirichlet:
            dirichlet.energy_fraction = dirichlet.trace / total_energy
        if neumann:
            neumann.energy_fraction = neumann.trace / total_energy

    epochs_pred = np.array([0, 100, 500, 1000, 2000, 5000, 10000, 20000, 50000])

    def predict_loss(eig: np.ndarray, epochs: np.ndarray, lr: float, L0: float) -> np.ndarray:
        eig_pos = eig[eig > 1e-12]
        if len(eig_pos) == 0:
            return np.ones_like(epochs) * L0

        n = min(len(eig_pos), 10)
        lambda_eff = n / np.sum(1.0 / eig_pos[:n])
        return L0 * np.exp(-2 * lambda_eff * lr * epochs)

    predicted_loss_pde = predict_loss(eigenvalues_KL, epochs_pred, learning_rate, initial_loss * 0.7)

    bc_initial = initial_loss * 0.3
    if dirichlet is not None and neumann is not None:
        eig_bc = np.concatenate([eigenvalues_KD, eigenvalues_KN])
        predicted_loss_bc = predict_loss(eig_bc, epochs_pred, learning_rate, bc_initial)
    elif dirichlet is not None:
        predicted_loss_bc = predict_loss(eigenvalues_KD, epochs_pred, learning_rate, bc_initial)
    elif neumann is not None:
        predicted_loss_bc = predict_loss(eigenvalues_KN, epochs_pred, learning_rate, bc_initial)
    else:
        predicted_loss_bc = np.zeros_like(epochs_pred)

    predicted_loss_total = predicted_loss_pde + predicted_loss_bc

    health_score = _compute_health_score(pde, dirichlet, neumann)

    balance_score = _compute_balance_score(pde, dirichlet, neumann)

    recommendations = _generate_recommendations(pde, dirichlet, neumann, solution)

    return ConvergencePrediction(
        epoch=epoch,
        pde=pde,
        dirichlet=dirichlet,
        neumann=neumann,
        solution=solution,
        bottleneck_component=bottleneck_name,
        bottleneck_epochs=int(bottleneck_time),
        total_epochs_estimate=total_epochs,
        recommendations=recommendations,
        predicted_epochs=epochs_pred,
        predicted_loss_pde=predicted_loss_pde,
        predicted_loss_bc=predicted_loss_bc,
        predicted_loss_total=predicted_loss_total,
        health_score=health_score,
        balance_score=balance_score,
    )

def _compute_health_score(
        pde: ComponentMetrics,
        dirichlet: Optional[ComponentMetrics],
        neumann: Optional[ComponentMetrics],
    ) -> float:
    score = 100.0

    if pde.condition_number > 1e6:
        score -= 30
    elif pde.condition_number > 1e5:
        score -= 20
    elif pde.condition_number > 1e4:
        score -= 10

    if pde.t_epsilon > 20000:
        score -= 25
    elif pde.t_epsilon > 10000:
        score -= 15
    elif pde.t_epsilon > 5000:
        score -= 5

    if pde.spectral_gap < 0.1:
        score -= 15
    elif pde.spectral_gap < 0.3:
        score -= 5

    if dirichlet is not None:
        if dirichlet.t_epsilon > 5000:
            score -= 10

    if neumann is not None:
        if neumann.t_epsilon > 5000:
            score -= 10

    return max(0, min(100, score))

def _compute_balance_score(
        pde: ComponentMetrics,
        dirichlet: Optional[ComponentMetrics],
        neumann: Optional[ComponentMetrics],
    ) -> float:
    times = [pde.t_epsilon]
    if dirichlet is not None:
        times.append(dirichlet.t_epsilon)
    if neumann is not None:
        times.append(neumann.t_epsilon)

    if len(times) < 2:
        return 100.0

    t_max = max(times)
    t_min = min(times)

    if t_max < 1e-6:
        return 100.0

    ratio = t_min / t_max

    return ratio * 100

def _generate_recommendations(
        pde: ComponentMetrics,
        dirichlet: Optional[ComponentMetrics],
        neumann: Optional[ComponentMetrics],
        solution: ComponentMetrics,
    ) -> List[str]:
    recommendations = []

    if pde.t_epsilon > 10000:
        recommendations.append(
            f"⚠ CRITICAL: PDE residual требует ~{int(pde.t_epsilon)} эпох для сходимости"
        )
        recommendations.append("  → Увеличьте вес PDE loss или learning rate")
        recommendations.append("  → Рассмотрите NTK-based preconditioning")

    if pde.condition_number > 1e5:
        recommendations.append(
            f"⚠ Плохая обусловленность PDE: κ(K_L) = {pde.condition_number:.1e}"
        )
        recommendations.append("  → Рассмотрите спектральную нормализацию сети")

    if dirichlet is not None and dirichlet.t_epsilon > 5000:
        recommendations.append(
            f"⚠ Медленная сходимость Dirichlet BC: ~{int(dirichlet.t_epsilon)} эпох"
        )

    if neumann is not None and neumann.t_epsilon > 5000:
        recommendations.append(
            f"⚠ Медленная сходимость Neumann BC: ~{int(neumann.t_epsilon)} эпох"
        )

    if dirichlet is not None and neumann is not None:
        t_bc = max(dirichlet.t_epsilon, neumann.t_epsilon)
        if pde.t_epsilon > 3 * t_bc:
            recommendations.append("📊 PDE сходится значительно медленнее BC")
            recommendations.append("  → Рекомендуется curriculum learning: BC → PDE")
        elif t_bc > 3 * pde.t_epsilon:
            recommendations.append("📊 BC сходится медленнее PDE")
            recommendations.append("  → Увеличьте веса граничных условий")

    if pde.spectral_gap < 0.1:
        recommendations.append(
            f"⚠ Малый spectral gap PDE: {pde.spectral_gap:.3f}"
        )
        recommendations.append("  → Возможно, сеть в плохом локальном минимуме")

    if pde.t_epsilon < 1000 and pde.condition_number < 1e4:
        recommendations.append("✓ Хорошая обусловленность PDE NTK")

    if len(recommendations) == 0:
        recommendations.append("✓ Все компоненты хорошо сбалансированы")

    return recommendations

def generate_convergence_report(
        prediction: ConvergencePrediction,
        output_dir: str,
    ) -> str:
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"convergence_report_epoch{prediction.epoch}.txt")

    lines = []
    lines.append("=" * 70)
    lines.append("NTK CONVERGENCE PREDICTION REPORT — Epoch {}".format(prediction.epoch))
    lines.append("=" * 70)
    lines.append("")

    lines.append("1. COMPONENT METRICS")
    lines.append("-" * 70)
    lines.append("")

    lines.append(f"{'Component':<15} {'κ':>12} {'τ (epochs)':>12} {'t_ε (epochs)':>12} {'Status':>10}")
    lines.append("-" * 70)

    pde = prediction.pde
    lines.append(f"{'PDE (K_L)':<15} {pde.condition_number:>12.1e} {pde.tau_char:>12.1f} {pde.t_epsilon:>12.0f} {pde.get_status():>10}")

    if prediction.dirichlet:
        d = prediction.dirichlet
        lines.append(f"{'Dirichlet (K_D)':<15} {d.condition_number:>12.1e} {d.tau_char:>12.1f} {d.t_epsilon:>12.0f} {d.get_status():>10}")

    if prediction.neumann:
        n = prediction.neumann
        lines.append(f"{'Neumann (K_N)':<15} {n.condition_number:>12.1e} {n.tau_char:>12.1f} {n.t_epsilon:>12.0f} {n.get_status():>10}")

    if prediction.solution:
        s = prediction.solution
        lines.append(f"{'Solution (K)':<15} {s.condition_number:>12.1e} {s.tau_char:>12.1f} {s.t_epsilon:>12.0f} {s.get_status():>10}")

    lines.append("")

    lines.append("2. BOTTLENECK ANALYSIS")
    lines.append("-" * 70)
    lines.append(f"  Bottleneck component: {prediction.bottleneck_component}")
    lines.append(f"  Estimated epochs for 1% convergence: {prediction.bottleneck_epochs}")
    lines.append(f"  Total epochs estimate: {prediction.total_epochs_estimate}")
    lines.append(f"  Health score: {prediction.health_score:.1f}/100")
    lines.append(f"  Balance score: {prediction.balance_score:.1f}/100")
    lines.append("")

    lines.append("3. TIME-TO-CONVERGENCE COMPARISON")
    lines.append("-" * 70)

    if prediction.dirichlet and prediction.neumann:
        t_pde = prediction.pde.t_epsilon
        t_d = prediction.dirichlet.t_epsilon
        t_n = prediction.neumann.t_epsilon

        lines.append(f"  PDE / Dirichlet ratio: {t_pde/t_d:.1f}x")
        lines.append(f"  PDE / Neumann ratio: {t_pde/t_n:.1f}x")
        lines.append(f"  Dirichlet / Neumann ratio: {t_d/t_n:.1f}x")

    lines.append("")

    lines.append("4. SPECTRAL ANALYSIS")
    lines.append("-" * 70)

    lines.append(f"  PDE spectral gap (λ₂/λ₁): {prediction.pde.spectral_gap:.4f}")
    lines.append(f"  PDE modes for 50% energy: {prediction.pde.mode_50}")
    lines.append(f"  PDE modes for 90% energy: {prediction.pde.mode_90}")

    if prediction.dirichlet:
        lines.append(f"  Dirichlet spectral gap: {prediction.dirichlet.spectral_gap:.4f}")
        lines.append(f"  Dirichlet modes for 90% energy: {prediction.dirichlet.mode_90}")

    if prediction.neumann:
        lines.append(f"  Neumann spectral gap: {prediction.neumann.spectral_gap:.4f}")
        lines.append(f"  Neumann modes for 90% energy: {prediction.neumann.mode_90}")

    lines.append("")

    lines.append("5. RECOMMENDATIONS")
    lines.append("-" * 70)
    for rec in prediction.recommendations:
        lines.append(f"  {rec}")

    lines.append("")

    lines.append("6. PREDICTED LOSS DYNAMICS")
    lines.append("-" * 70)
    lines.append(f"  {'Epoch':<10} {'L_PDE':<12} {'L_BC':<12} {'L_total':<12}")
    lines.append("  " + "-" * 46)

    for i, ep in enumerate(prediction.predicted_epochs):
        if ep <= prediction.total_epochs_estimate * 1.5:
            lines.append(f"  {ep:<10} {prediction.predicted_loss_pde[i]:<12.4f} "
                        f"{prediction.predicted_loss_bc[i]:<12.4f} "
                        f"{prediction.predicted_loss_total[i]:<12.4f}")

    lines.append("")
    lines.append("=" * 70)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    return filepath

def compute_adaptive_weights(
        prediction: ConvergencePrediction,
        current_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
    if current_weights is None:
        current_weights = {"pde": 1.0, "dirichlet": 1.0, "neumann": 1.0}

    weights = {}

    trace_pde = prediction.pde.trace if prediction.pde else 1.0
    trace_d = prediction.dirichlet.trace if prediction.dirichlet else 1.0
    trace_n = prediction.neumann.trace if prediction.neumann else 1.0

    w_pde = 1.0 / np.sqrt(trace_pde + 1e-10)
    w_d = 1.0 / np.sqrt(trace_d + 1e-10)
    w_n = 1.0 / np.sqrt(trace_n + 1e-10)

    w_sum = w_pde + w_d + w_n
    weights["pde"] = w_pde / w_sum
    weights["dirichlet"] = w_d / w_sum
    weights["neumann"] = w_n / w_sum

    t_max = max(
        prediction.pde.t_epsilon,
        prediction.dirichlet.t_epsilon if prediction.dirichlet else 0,
        prediction.neumann.t_epsilon if prediction.neumann else 0,
    )

    if t_max > 0:

        bottleneck_factor = 2.0

        if prediction.bottleneck_component == "PDE":
            weights["pde"] *= bottleneck_factor
        elif prediction.bottleneck_component == "Dirichlet":
            weights["dirichlet"] *= bottleneck_factor
        elif prediction.bottleneck_component == "Neumann":
            weights["neumann"] *= bottleneck_factor

    w_sum = sum(weights.values())
    for k in weights:
        weights[k] /= w_sum

    return weights

def estimate_learning_rates(
        prediction: ConvergencePrediction,
        base_lr: float = 1e-3,
    ) -> Dict[str, float]:
    lrs = {"pde": base_lr, "dirichlet": base_lr, "neumann": base_lr}

    t_pde = prediction.pde.t_epsilon
    t_d = prediction.dirichlet.t_epsilon if prediction.dirichlet else t_pde
    t_n = prediction.neumann.t_epsilon if prediction.neumann else t_pde

    t_max = max(t_pde, t_d, t_n)

    if t_max > 0:

        lrs["pde"] = base_lr * (t_pde / t_max)
        lrs["dirichlet"] = base_lr * (t_d / t_max)
        lrs["neumann"] = base_lr * (t_n / t_max)

    return lrs

class ConvergenceTracker:
    def __init__(self):
        self.predictions: List[ConvergencePrediction] = []
        self.actual_losses: List[Dict[str, float]] = []

    def add_prediction(
                self,
                prediction: ConvergencePrediction,
                actual_loss_pde: Optional[float] = None,
                actual_loss_bc: Optional[float] = None,
                actual_loss_total: Optional[float] = None,
            ) -> None:
        self.predictions.append(prediction)

        loss_dict = {
            "pde": actual_loss_pde,
            "bc": actual_loss_bc,
            "total": actual_loss_total,
        }
        self.actual_losses.append(loss_dict)

    def get_prediction_accuracy(self) -> Dict[str, float]:
        if len(self.predictions) < 2:
            return {"accuracy": 0.0, "samples": 0}

        errors = []

        for i, pred in enumerate(self.predictions[1:], 1):
            actual = self.actual_losses[i]

            if actual["total"] is not None:

                epoch_idx = np.searchsorted(pred.predicted_epochs, pred.epoch)
                if epoch_idx < len(pred.predicted_loss_total):
                    predicted = pred.predicted_loss_total[epoch_idx]
                    actual_val = actual["total"]

                    if actual_val > 1e-10:
                        error = abs(predicted - actual_val) / actual_val
                        errors.append(error)

        if len(errors) == 0:
            return {"accuracy": 0.0, "samples": 0}

        mean_error = np.mean(errors)
        accuracy = max(0, 1 - mean_error) * 100

        return {
            "accuracy": accuracy,
            "mean_error": mean_error,
            "samples": len(errors),
        }

    def get_trend(self) -> Dict[str, str]:
        if len(self.predictions) < 2:
            return {"health": "unknown", "balance": "unknown"}

        health_scores = [p.health_score for p in self.predictions]
        if health_scores[-1] > health_scores[0] + 5:
            health_trend = "improving"
        elif health_scores[-1] < health_scores[0] - 5:
            health_trend = "degrading"
        else:
            health_trend = "stable"

        balance_scores = [p.balance_score for p in self.predictions]
        if balance_scores[-1] > balance_scores[0] + 5:
            balance_trend = "improving"
        elif balance_scores[-1] < balance_scores[0] - 5:
            balance_trend = "degrading"
        else:
            balance_trend = "stable"

        return {
            "health": health_trend,
            "balance": balance_trend,
        }

def estimate_pde_constants(domain_diameter: float) -> Tuple[float, float]:

    poincare = domain_diameter / np.pi

    stability = 1.0

    return float(poincare), float(stability)

def predict_error_evolution(
        eigenvalues_KL: np.ndarray,
        residual_norm_init: float,
        learning_rate: float,
        epochs: np.ndarray,
        poincare_constant: float,
        stability_constant: float,
    ) -> ErrorBounds:
    eig_pos = eigenvalues_KL[eigenvalues_KL > 1e-12]

    if len(eig_pos) == 0:
        return ErrorBounds(
            l2_error_predicted=np.zeros_like(epochs, dtype=float),
            l2_error_upper_bound=float('inf'),
            energy_error_predicted=np.zeros_like(epochs, dtype=float),
            energy_error_upper_bound=float('inf'),
            poincare_constant=poincare_constant,
            stability_constant=stability_constant,
            residual_norm_init=residual_norm_init,
        )

    n = min(len(eig_pos), 10)
    lambda_eff = n / np.sum(1.0 / eig_pos[:n])

    residual_pred = residual_norm_init * np.exp(-2 * lambda_eff * learning_rate * epochs)

    l2_error_pred = poincare_constant * residual_pred

    energy_error_pred = stability_constant * residual_pred

    return ErrorBounds(
        l2_error_predicted=l2_error_pred,
        l2_error_upper_bound=poincare_constant * residual_norm_init,
        energy_error_predicted=energy_error_pred,
        energy_error_upper_bound=stability_constant * residual_norm_init,
        poincare_constant=poincare_constant,
        stability_constant=stability_constant,
        residual_norm_init=residual_norm_init,
    )

def compute_convergence_prediction_with_errors(
    eigenvalues_K: np.ndarray,
    eigenvalues_KL: np.ndarray,
    eigenvalues_KD: Optional[np.ndarray] = None,
    eigenvalues_KN: Optional[np.ndarray] = None,
    learning_rate: float = 1e-3,
    target_error: float = 0.01,
    epoch: int = 0,
    initial_loss: float = 1.0,
    initial_residual_norm: float = 1.0,
    domain_diameter: float = 1.0,
) -> ConvergencePrediction:

    prediction = compute_convergence_prediction(
        eigenvalues_K=eigenvalues_K,
        eigenvalues_KL=eigenvalues_KL,
        eigenvalues_KD=eigenvalues_KD,
        eigenvalues_KN=eigenvalues_KN,
        learning_rate=learning_rate,
        target_error=target_error,
        epoch=epoch,
        initial_loss=initial_loss,
    )

    poincare_const, stability_const = estimate_pde_constants(domain_diameter)

    error_bounds = predict_error_evolution(
        eigenvalues_KL=eigenvalues_KL,
        residual_norm_init=initial_residual_norm,
        learning_rate=learning_rate,
        epochs=prediction.predicted_epochs,
        poincare_constant=poincare_const,
        stability_constant=stability_const,
    )

    prediction.error_bounds = error_bounds

    if error_bounds.l2_error_upper_bound < 0.1:
        prediction.recommendations.append(f"✓ Ожидаемая L² ошибка < {error_bounds.l2_error_upper_bound:.2e}")
    elif error_bounds.l2_error_upper_bound > 1.0:
        prediction.recommendations.append(
            f"⚠ Ожидаемая L² ошибка может быть большой: ~{error_bounds.l2_error_upper_bound:.2e}"
        )
        prediction.recommendations.append("  → Возможно, нужно больше эпох или другая архитектура")

    return prediction
