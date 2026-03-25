from __future__ import annotations
import os
import shutil

from config import (DEVICE, OUTPUT_DIR, DEFAULT_TRI_AREA,
                    GAUSS_TRI_ORDER, GAUSS_LINE_ORDER, BOUNDARY_DENSITY_PTS)
from geometry.domains import make_domain
from geometry.mesher import Mesher
from geometry.quadrature import QuadratureBuilder
from file_io.logger import FileLogger
from problems.solutions import PolynomialSolution, SineSolution
from training.trainer import Trainer
from visualization.field_plotter import plot_mesh

def run(domain_name: str = "square_mixed") -> None:
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


    log_path = os.path.join(OUTPUT_DIR, f"training_log_{domain_name}.txt")
    with FileLogger(log_path) as logger:
        logger.section(f"TRAIN: '{domain_name}'")

        domain = make_domain(domain_name)

        mesher = Mesher(max_area=DEFAULT_TRI_AREA, lloyd_iters=3,
                        boundary_density=BOUNDARY_DENSITY_PTS)
        mesh = mesher.build(domain)
        logger(f"Mesh: {len(mesh['points'])} vertices, "
               f"{len(mesh['triangles'])} triangles")

        quad_builder = QuadratureBuilder(
            tri_order=GAUSS_TRI_ORDER, line_order=GAUSS_LINE_ORDER,
            device=DEVICE)
        quad = quad_builder.build(mesh, domain)
        logger(f"Quadrature: {len(quad.xy_in)} interior, "
               f"{len(quad.xy_bd)} boundary points")

        mesh_img = os.path.join(OUTPUT_DIR, f"{domain_name}_00_mesh.png")
        plot_mesh(mesh, domain_name, domain.bc_type, mesh_img)

        solution = PolynomialSolution()
        trainer = Trainer(
            domain=domain, quad=quad, solution=solution, logger=logger)
        trainer.train()

        logger(f"Finished '{domain_name}'.")

if __name__ == "__main__":
    run(domain_name="l_shape")