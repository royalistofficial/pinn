from __future__ import annotations
import os
import torch

from config import OUTPUT_DIR
from problems.solutions import AnalyticalSolution
from functionals.operators import gradient
from functionals.errors import energy_error, relative_l2_error, relative_energy_error
from evaluation.metrics_history import MetricsHistory
from visualization.metrics_plotter import plot_pretrain_metrics
from visualization.field_plotter import plot_training_fields
from visualization.field_evaluator import evaluate_fields, refine_mesh
from file_io.logger import FileLogger

class Evaluator:
    def __init__(self, pinn, data, solution, logger, domain_name="domain", has_neumann=True):
        self.pinn = pinn
        self.data = data
        self.solution = solution
        self.logger = logger
        self.domain_name = domain_name
        self.has_neumann = has_neumann
        self.history = MetricsHistory()

    def _eval_training(self, epoch, loss, pde, dir_loss, neu_loss, lr):
        s = self.data.sample
        xq = s.quad.xy_in.clone().requires_grad_(True)
        with torch.enable_grad():
            v = self.pinn(xq)
            gv = gradient(v, xq, create_graph=False)
        with torch.no_grad():
            ee = energy_error(gv.detach(), xq, s.quad.vol_w, self.solution).item()
            rel = relative_l2_error(v.detach(), xq, s.quad.vol_w, self.solution).item()
            rel_en = relative_energy_error(gv.detach(), xq, s.quad.vol_w, self.solution).item()
            self.history.pretrain.log(
                epoch, loss=loss, pde=pde, dir_loss=dir_loss, neu_loss=neu_loss,
                energy=ee, rel_err=rel, rel_energy=rel_en, lr=lr)
            self.logger(
                f"  Epoch {epoch:04d}: loss={loss:.4e} | pde={pde:.2e} "
                f"dir={dir_loss:.2e} neu={neu_loss:.2e} | E={ee:.4e} | "
                f"RelL2={rel:.4e} RelE={rel_en:.4e} | LR={lr:.2e}")
            plot_pretrain_metrics(
                self.history.pretrain.as_dict(), self.domain_name,
                os.path.join(OUTPUT_DIR, f"{self.domain_name}_metrics.png"))
        return ee

    def _plot_training_fields(self, epoch):
        mesh = self.data.sample.quad.mesh
        tri_refi, pts_refi = refine_mesh(mesh)
        xy_t = torch.tensor(pts_refi, dtype=torch.float32,
                            device=self.data.sample.quad.xy_in.device)
        self.pinn.eval()
        fv = evaluate_fields(xy_t, self.pinn, self.solution)
        self.pinn.train()
        path = os.path.join(OUTPUT_DIR, f"{self.domain_name}_fields_{epoch:05d}.png")
        plot_training_fields(epoch, tri_refi, fv.v, fv.u_exact,
                             fv.abs_error, fv.energy_error_density, path)

    def pretrain_callback(self):
        return _TrainingCallback(self)

class _TrainingCallback:
    def __init__(self, ev):
        self.ev = ev

    def on_epoch_end(self, epoch, **metrics):
        ee = self.ev._eval_training(
            epoch, metrics.get("loss", 0), metrics.get("pde", 0),
            metrics.get("dir_loss", 0), metrics.get("neu_loss", 0),
            metrics.get("lr", 0))
        if epoch == 1 or epoch % 100 == 0:
            self.ev._plot_training_fields(epoch)
        return ee

    def on_phase_end(self):
        best = min(self.ev.history.pretrain.data.get("energy", [float("inf")]))
        self.ev.logger(f"Обучение завершено. Лучшая энергетическая ошибка E={best:.4e}")
