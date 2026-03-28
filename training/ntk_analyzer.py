from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable

import numpy as np
import torch
import torch.nn as nn

from networks.ntk_utils import ntk_comprehensive_analysis
from visualization.ntk_plots import plot_ntk_evolution
from training.convergence_prediction import (
    ConvergencePrediction,
    compute_convergence_prediction,
    generate_convergence_report,
)
from visualization.convergence_plots import (
    plot_ntk_master_dashboard,
    plot_convergence_evolution,
)
from config import OUTPUT_DIR

@dataclass
class NTKResult:
    epoch: int

    eigenvalues_KL: np.ndarray
    condition_number_KL: float
    effective_rank_KL: float
    trace_KL: float

    eigenvalues_K: np.ndarray
    condition_number_K: float
    effective_rank_K: float
    trace_K: float

    dirichlet: Optional[Dict[str, Any]] = None
    neumann: Optional[Dict[str, Any]] = None

    n_interior: int = 0
    n_dirichlet: int = 0
    n_neumann: int = 0

    kappa_ratio: float = 0.0
    rank_ratio: float = 0.0
    health_score: float = 0.0
    energy_balance: Dict[str, float] = field(default_factory=dict)

    convergence_prediction: Optional[ConvergencePrediction] = None

class NTKAnalyzer:
    def __init__(
                self,
                model: nn.Module,
                output_dir: str = OUTPUT_DIR,
                n_interior: int = 64,
                n_boundary: int = 32,
                node_order: str = "xy",
                learning_rate: float = 1e-3,
                logger: Optional[Callable[[str], None]] = None,
                generate_convergence_reports: bool = True,
            ):
        self.model = model
        self.output_dir = output_dir
        self.n_interior = n_interior
        self.n_boundary = n_boundary
        self.node_order = node_order
        self.learning_rate = learning_rate
        self.logger = logger or print
        self.generate_convergence_reports = generate_convergence_reports

        self.history: List[NTKResult] = []
        self.epochs: List[int] = []
        self.convergence_history: List[ConvergencePrediction] = []

        self.loss_history: Dict[str, List[float]] = {
            "loss": [],
            "pde": [],
            "dirichlet": [],
            "neumann": [],
        }

        os.makedirs(output_dir, exist_ok=True)

    def analyze(
                self,
                epoch: int,
                X_interior: torch.Tensor,
                X_boundary: Optional[torch.Tensor] = None,
                normals: Optional[torch.Tensor] = None,
                bc_mask: Optional[torch.Tensor] = None,
            ) -> NTKResult:
        device = next(self.model.parameters()).device
        self.logger(f"[NTK] Starting analysis for epoch {epoch}...")

        X_in = self._subsample(X_interior.to(device), self.n_interior)
        n_in = len(X_in)
        n_bd = 0

        if X_boundary is not None and len(X_boundary) > 0:
            X_bd = self._subsample(X_boundary.to(device), self.n_boundary)
            n_bd = len(X_bd)

        self.logger(f"[NTK] Points: interior={n_in}, boundary={n_bd}")
        self.logger("[NTK] Вычисляем спектры NTK напрямую (без отрисовки матриц)...")

        result_combined = ntk_comprehensive_analysis(
            model=self.model,
            X_interior=X_interior,
            X_boundary=X_boundary,
            normals=normals,
            bc_mask=bc_mask,
            n_interior=self.n_interior,
            n_boundary=self.n_boundary,
        )

        interior = result_combined["interior"]
        boundary = result_combined.get("boundary", {})

        eig_K = interior["eigenvalues_K"]
        eig_KL = interior["eigenvalues_KL"]

        dir_data = boundary.get("dirichlet")
        neu_data = boundary.get("neumann")

        eig_D = dir_data["eigenvalues"] if dir_data else None
        eig_N = neu_data["eigenvalues"] if neu_data else None

        kappa_K = interior["condition_number_K"]
        kappa_KL = interior["condition_number_KL"]
        rank_K = interior["effective_rank_K"]
        rank_KL = interior["effective_rank_KL"]
        trace_K = interior["trace_K"]
        trace_KL = interior["trace_KL"]

        n_dir = dir_data["n_points"] if dir_data else 0
        n_neu = neu_data["n_points"] if neu_data else 0

        kappa_ratio = kappa_KL / kappa_K if kappa_K > 0 and kappa_K < float("inf") else float("inf")
        rank_ratio = rank_KL / rank_K if rank_K > 0 else 0.0

        health_score = 100.0
        if kappa_ratio > 100:
            health_score -= 30
        elif kappa_ratio > 10:
            health_score -= 15
        elif kappa_ratio > 5:
            health_score -= 5

        if rank_ratio < 0.5:
            health_score -= 25
        elif rank_ratio < 0.7:
            health_score -= 10

        health_score = max(0, min(100, health_score))

        energy_K = float(eig_K.sum())
        energy_KL = float(eig_KL.sum())
        energy_D = float(eig_D.sum()) if eig_D is not None else 0.0
        energy_N = float(eig_N.sum()) if eig_N is not None else 0.0
        total_energy = energy_K + energy_D + energy_N

        energy_balance = {
            "K": energy_K / total_energy if total_energy > 0 else 0.0,
            "KL": energy_KL / total_energy if total_energy > 0 else 0.0,
            "KD": energy_D / total_energy if total_energy > 0 else 0.0,
            "KN": energy_N / total_energy if total_energy > 0 else 0.0,
        }

        convergence_pred = None
        if self.generate_convergence_reports:
            self.logger("[NTK] Computing convergence prediction...")

            convergence_pred = compute_convergence_prediction(
                eigenvalues_K=eig_K,
                eigenvalues_KL=eig_KL,
                eigenvalues_KD=eig_D,
                eigenvalues_KN=eig_N,
                learning_rate=self.learning_rate,
                target_error=0.01,
                epoch=epoch,
            )

            self.convergence_history.append(convergence_pred)

        result = NTKResult(
            epoch=epoch,
            eigenvalues_KL=eig_KL,
            condition_number_KL=kappa_KL,
            effective_rank_KL=rank_KL,
            trace_KL=trace_KL,
            eigenvalues_K=eig_K,
            condition_number_K=kappa_K,
            effective_rank_K=rank_K,
            trace_K=trace_K,
            dirichlet=dir_data,
            neumann=neu_data,
            n_interior=n_in,
            n_dirichlet=n_dir,
            n_neumann=n_neu,
            kappa_ratio=kappa_ratio,
            rank_ratio=rank_ratio,
            health_score=health_score,
            energy_balance=energy_balance,
            convergence_prediction=convergence_pred,
        )

        self.history.append(result)
        self.epochs.append(epoch)

        self._log_result(result)

        if self.generate_convergence_reports and (epoch == 0 or len(self.history) % 5 == 0):
            self._generate_convergence_report(result, epoch)

        return result

    def update_loss_history(
                self,
                loss: float,
                pde_loss: float = 0.0,
                dirichlet_loss: float = 0.0,
                neumann_loss: float = 0.0,
            ) -> None:
        self.loss_history["loss"].append(loss)
        self.loss_history["pde"].append(pde_loss)
        self.loss_history["dirichlet"].append(dirichlet_loss)
        self.loss_history["neumann"].append(neumann_loss)

    def plot_evolution(self) -> None:
        if len(self.history) < 2:
            self.logger("[NTK] Not enough data for evolution plot")
            return

        self.logger("[NTK] Generating evolution plots...")

        spectra_history = []
        for r in self.history:
            sp = {
                "eig_K": r.eigenvalues_K,
                "eig_KL": r.eigenvalues_KL,
                "kappa_K": r.condition_number_K,
                "kappa_KL": r.condition_number_KL,
                "rank_K": r.effective_rank_K,
                "rank_KL": r.effective_rank_KL,
                "trace_K": r.trace_K,
                "trace_KL": r.trace_KL,
                "full_K": {
                    "eigenvalues": r.eigenvalues_K,
                    "condition_number": r.condition_number_K,
                    "effective_rank": r.effective_rank_K,
                },
                "pde_KL": {
                    "eigenvalues": r.eigenvalues_KL,
                    "condition_number": r.condition_number_KL,
                    "effective_rank": r.effective_rank_KL,
                },
                "dirichlet": r.dirichlet,
                "neumann": r.neumann,
            }
            spectra_history.append(sp)

        plot_ntk_evolution(
            spectra_history=spectra_history,
            epochs=self.epochs,
            output_dir=self.output_dir,
        )

        self.logger("[NTK] Evolution plots saved")

        self._save_summary()

        if len(self.convergence_history) >= 2:
            self._plot_convergence_evolution()

    def _log_result(self, result: NTKResult) -> None:
        self.logger(
            f"[NTK] Results epoch {result.epoch}:\n"
            f"       K:     κ={result.condition_number_K:.2e}, "
            f"rank={result.effective_rank_K:.1f}\n"
            f"       K_L:   κ={result.condition_number_KL:.2e}, "
            f"rank={result.effective_rank_KL:.1f}\n"
            f"       Health: {result.health_score:.1f}/100"
        )

        if result.dirichlet is not None:
            self.logger(
                f"       K_D:   κ={result.dirichlet['condition_number']:.2e}, "
                f"rank={result.dirichlet['effective_rank']:.1f}"
            )

        if result.neumann is not None:
            self.logger(
                f"       K_N:   κ={result.neumann['condition_number']:.2e}, "
                f"rank={result.neumann['effective_rank']:.1f}"
            )

        if result.convergence_prediction is not None:
            self.logger(
                f"       Convergence: ~{result.convergence_prediction.total_epochs_estimate} epochs "
                f"(bottleneck: {result.convergence_prediction.bottleneck_component})"
            )

    def _generate_convergence_report(
                self,
                result: NTKResult,
                epoch: int,
            ) -> None:
        if result.convergence_prediction is None:
            return

        pred = result.convergence_prediction

        report_path = generate_convergence_report(
            prediction=pred,
            output_dir=self.output_dir,
        )
        self.logger(f"[NTK] Convergence report saved: {report_path}")

        actual_losses = None
        if len(self.loss_history["loss"]) > 0:
            actual_losses = {
                "epochs": self.epochs[:len(self.loss_history["loss"])],
                "pde": self.loss_history["pde"],
                "total": self.loss_history["loss"],
            }

        dash_path = plot_ntk_master_dashboard(
            prediction=pred,
            epoch=epoch,
            actual_losses=actual_losses,
            output_dir=self.output_dir,
        )
        self.logger(f"[NTK] Master Dashboard saved: {dash_path}")

    def _plot_convergence_evolution(self) -> None:
        plot_path = plot_convergence_evolution(
            predictions_history=self.convergence_history,
            output_dir=self.output_dir,
        )
        if plot_path:
            self.logger(f"[NTK] Convergence evolution plot saved: {plot_path}")

    def _save_summary(self) -> None:
        summary_path = os.path.join(self.output_dir, "ntk_analysis_summary.txt")

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("NTK ANALYSIS SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Total analysis epochs: {len(self.epochs)}\n")
            f.write(f"Epochs: {self.epochs}\n")
            f.write(f"Learning rate: {self.learning_rate:.2e}\n\n")

            f.write("-" * 80 + "\n")
            f.write(f"{'Epoch':>8} | {'κ(K)':>12} | {'κ(K_L)':>12} | ")
            f.write(f"{'rank(K)':>10} | {'rank(K_L)':>10} | {'Health':>8}\n")
            f.write("-" * 80 + "\n")

            for r in self.history:
                f.write(f"{r.epoch:>8} | {r.condition_number_K:>12.2e} | ")
                f.write(f"{r.condition_number_KL:>12.2e} | ")
                f.write(f"{r.effective_rank_K:>10.2f} | {r.effective_rank_KL:>10.2f} | ")
                f.write(f"{r.health_score:>7.1f}\n")

            f.write("-" * 80 + "\n\n")

            has_dir = any(r.dirichlet is not None for r in self.history)
            if has_dir:
                f.write("DIRICHLET BOUNDARY CONDITIONS:\n")
                f.write("-" * 50 + "\n")
                f.write(f"{'Epoch':>8} | {'κ(K_D)':>12} | {'rank(K_D)':>10}\n")
                f.write("-" * 50 + "\n")

                for r in self.history:
                    if r.dirichlet is not None:
                        f.write(f"{r.epoch:>8} | {r.dirichlet['condition_number']:>12.2e} | ")
                        f.write(f"{r.dirichlet['effective_rank']:>10.2f}\n")

                f.write("-" * 50 + "\n\n")

            has_neu = any(r.neumann is not None for r in self.history)
            if has_neu:
                f.write("NEUMANN BOUNDARY CONDITIONS:\n")
                f.write("-" * 50 + "\n")
                f.write(f"{'Epoch':>8} | {'κ(K_N)':>12} | {'rank(K_N)':>10}\n")
                f.write("-" * 50 + "\n")

                for r in self.history:
                    if r.neumann is not None:
                        f.write(f"{r.epoch:>8} | {r.neumann['condition_number']:>12.2e} | ")
                        f.write(f"{r.neumann['effective_rank']:>10.2f}\n")

                f.write("-" * 50 + "\n\n")

            if len(self.convergence_history) > 0:
                f.write("CONVERGENCE PREDICTIONS:\n")
                f.write("-" * 60 + "\n")
                f.write(f"{'Epoch':>8} | {'Est. Epochs':>12} | {'Bottleneck':>15}\n")
                f.write("-" * 60 + "\n")

                for r in self.history:
                    if r.convergence_prediction is not None:
                        pred = r.convergence_prediction
                        f.write(f"{r.epoch:>8} | {pred.total_epochs_estimate:>12} | ")
                        f.write(f"{pred.bottleneck_component:>15}\n")

                f.write("-" * 60 + "\n\n")

            f.write("=" * 80 + "\n")
            f.write("RECOMMENDATIONS:\n")
            f.write("=" * 80 + "\n\n")

            if len(self.history) >= 2:
                first = self.history[0]
                last = self.history[-1]

                if last.condition_number_KL > first.condition_number_KL * 2:
                    f.write("⚠ WARNING: K_L condition number increased > 2x.\n")
                    f.write("  This may indicate deteriorating PDE residual training.\n")
                    f.write("  Recommend: decrease learning rate or add regularization.\n\n")

                if last.effective_rank_KL < first.effective_rank_KL * 0.5:
                    f.write("⚠ WARNING: K_L effective rank decreased significantly.\n")
                    f.write("  This may indicate subspace collapse.\n")
                    f.write("  Recommend: check network architecture and initialization.\n\n")

                if last.dirichlet is not None and last.dirichlet["condition_number"] > 1e6:
                    f.write("⚠ WARNING: High K_D condition number.\n")
                    f.write("  Dirichlet BC training may be unstable.\n\n")

                if last.neumann is not None and last.neumann["condition_number"] > 1e6:
                    f.write("⚠ WARNING: High K_N condition number.\n")
                    f.write("  Neumann BC training may be unstable.\n\n")

                if last.condition_number_KL < first.condition_number_KL:
                    f.write("✓ K_L condition number decreased — good training dynamics.\n")

                if last.effective_rank_KL > first.effective_rank_KL:
                    f.write("✓ K_L effective rank increased — network using more modes.\n")

                if last.health_score > 80:
                    f.write("✓ High health score — training is progressing well.\n")

            if self.history and self.history[-1].convergence_prediction:
                f.write("\n")
                for rec in self.history[-1].convergence_prediction.recommendations:
                    f.write(rec + "\n")

            f.write("\n" + "=" * 80 + "\n")

        self.logger(f"[NTK] Summary saved: {summary_path}")

    @staticmethod
    def _subsample(X: torch.Tensor, n: int) -> torch.Tensor:
        if len(X) <= n:
            return X
        return X[torch.linspace(0, len(X) - 1, n, device=X.device).long()]

    def get_history_dict(self) -> List[Dict[str, Any]]:
        return [
            {
                "epoch": r.epoch,
                "eigenvalues_K": r.eigenvalues_K,
                "eigenvalues_KL": r.eigenvalues_KL,
                "condition_number_K": r.condition_number_K,
                "condition_number_KL": r.condition_number_KL,
                "effective_rank_K": r.effective_rank_K,
                "effective_rank_KL": r.effective_rank_KL,
                "trace_K": r.trace_K,
                "trace_KL": r.trace_KL,
                "dirichlet": r.dirichlet,
                "neumann": r.neumann,
                "kappa_ratio": r.kappa_ratio,
                "rank_ratio": r.rank_ratio,
                "health_score": r.health_score,
            }
            for r in self.history
        ]