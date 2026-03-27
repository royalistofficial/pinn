from __future__ import annotations
import os
import shutil
import time

import torch

from config import (
    DEVICE, OUTPUT_DIR, DEFAULT_TRI_AREA,
    GAUSS_TRI_ORDER, GAUSS_LINE_ORDER, BOUNDARY_DENSITY_PTS,
)
from geometry.domains import make_domain
from geometry.mesher import Mesher
from geometry.quadrature import QuadratureBuilder
from file_io.logger import FileLogger
from problems.solutions import SineSolution
from training.trainer import Trainer
from visualization.mesh_plots import plot_mesh

def run(domain_name: str = "square") -> None:

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log_path = os.path.join(OUTPUT_DIR, f"training_log_{domain_name}.txt")

    with FileLogger(log_path) as logger:
        logger.section(f"PINN Training: '{domain_name}'")

        domain = make_domain(domain_name)
        solution = SineSolution()

        mesher = Mesher(
            max_area=DEFAULT_TRI_AREA,
            lloyd_iters=3,
            boundary_density=BOUNDARY_DENSITY_PTS,
        )
        mesh = mesher.build(domain)

        logger(
            f"Mesh: {len(mesh['points'])} vertices, "
            f"{len(mesh['triangles'])} triangles"
        )

        quad_builder = QuadratureBuilder(
            tri_order=GAUSS_TRI_ORDER,
            line_order=GAUSS_LINE_ORDER,
            device=DEVICE,
        )
        quad = quad_builder.build(mesh, domain)

        logger(
            f"Quadrature: {len(quad.xy_in)} interior, "
            f"{len(quad.xy_bd)} boundary points"
        )

        mesh_img = os.path.join(OUTPUT_DIR, f"{domain_name}_mesh.png")
        plot_mesh(mesh, domain_name, domain.bc_type, mesh_img)
        logger(f"Mesh visualization saved: {mesh_img}")

        t_start = time.time()
        trainer = Trainer(
            domain=domain,
            quad=quad,
            solution=solution,
            logger=logger,
        )
        trainer.train()
        elapsed = time.time() - t_start

        logger.section("Summary")
        logger(f"Total training time: {elapsed:.1f}s")
        logger(f"Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PINN Training")
    parser.add_argument(
        "--domain",
        type=str,
        default="square",
        choices=[
            "square", "square_mixed",
            "circle", "circle_mixed",
            "l_shape", "l_shape_mixed",
            "hollow_square", "hollow_square_mixed",
            "p_shape", "p_shape_mixed",
        ],
        help="Domain name",
    )

    args = parser.parse_args()
    run(domain_name=args.domain)
