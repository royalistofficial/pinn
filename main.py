from __future__ import annotations
import os
import shutil
import time

import torch

from config import (DEVICE, OUTPUT_DIR, DEFAULT_TRI_AREA,
                    GAUSS_TRI_ORDER, GAUSS_LINE_ORDER, BOUNDARY_DENSITY_PTS,
                    FEM_TRI_AREA, FEM_REFINE_LEVELS)
from geometry.domains import make_domain
from geometry.mesher import Mesher
from geometry.quadrature import QuadratureBuilder
from file_io.logger import FileLogger
from problems.solutions import PolynomialSolution, SineSolution
from training.trainer import Trainer
from visualization.field_plotter import plot_mesh

from fem.solver import build_fem_mesh, FEMSolver
from fem.apriori_estimates import format_apriori_report
from comparison.compare import (
    run_fem_solution, run_fem_convergence_study,
    evaluate_pinn_errors, plot_comparison, plot_convergence_study,
    format_comparison_table, ComparisonResult,
)

def run(domain_name: str = "l_shape") -> None:
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log_path = os.path.join(OUTPUT_DIR, f"training_log_{domain_name}.txt")
    with FileLogger(log_path) as logger:
        logger.section(f"PINN + FEM: '{domain_name}'")

        domain = make_domain(domain_name)
        solution = PolynomialSolution()

        mesher = Mesher(max_area=DEFAULT_TRI_AREA, lloyd_iters=3,
                        boundary_density=BOUNDARY_DENSITY_PTS)
        mesh = mesher.build(domain)
        logger(f"PINN Mesh: {len(mesh['points'])} vertices, "
               f"{len(mesh['triangles'])} triangles")

        quad_builder = QuadratureBuilder(
            tri_order=GAUSS_TRI_ORDER, line_order=GAUSS_LINE_ORDER,
            device=DEVICE)
        quad = quad_builder.build(mesh, domain)
        logger(f"Quadrature: {len(quad.xy_in)} interior, "
               f"{len(quad.xy_bd)} boundary points")

        mesh_img = os.path.join(OUTPUT_DIR, f"{domain_name}_00_mesh.png")
        plot_mesh(mesh, domain_name, domain.bc_type, mesh_img)

        t_pinn_start = time.time()
        trainer = Trainer(domain=domain, quad=quad, solution=solution, logger=logger)
        trainer.train()
        t_pinn = time.time() - t_pinn_start
        logger(f"PINN training time: {t_pinn:.1f}s")

        logger.section("FEM Solution")

        fem_result, fem_errors, fem_time = run_fem_solution(
            domain, solution, max_area=FEM_TRI_AREA)
        logger(f"FEM: N={fem_result.n_dof}, h_max={fem_result.h_max:.4e}, "
               f"time={fem_time:.2f}s")
        for k, v in fem_errors.items():
            logger(f"  {k}: {v:.6e}")

        logger.section("A Priori Estimates & Convergence Study")

        study = run_fem_convergence_study(domain, solution, area_levels=FEM_REFINE_LEVELS)
        report = format_apriori_report(study, domain_name)
        logger(report)

        logger.section("PINN Errors")
        pinn_errors = evaluate_pinn_errors(trainer.pinn, solution, domain)
        for k, v in pinn_errors.items():
            logger(f"  {k}: {v:.6e}")

        logger.section("Comparison PINN vs FEM")

        n_params = sum(p.numel() for p in trainer.pinn.parameters() if p.requires_grad)
        comp_result = ComparisonResult(
            domain_name=domain_name,
            fem_errors=fem_errors,
            fem_time=fem_time,
            fem_n_dof=fem_result.n_dof,
            fem_h_max=fem_result.h_max,
            pinn_errors=pinn_errors,
            pinn_time=t_pinn,
            pinn_n_params=n_params,
            convergence_study=study,
        )
        table = format_comparison_table(comp_result)
        logger(table)

        plot_comparison(
            fem_result, trainer.pinn, solution, domain,
            os.path.join(OUTPUT_DIR, f"{domain_name}_comparison.png"))

        plot_convergence_study(
            study, domain_name, pinn_errors=pinn_errors,
            save_path=os.path.join(OUTPUT_DIR, f"{domain_name}_convergence.png"))

        logger(f"Finished '{domain_name}'.")

if __name__ == "__main__":
    run(domain_name="l_shape")
