import unittest
import math
import numpy as np
import torch

from geometry.domains import make_domain
from problems.solutions import PolynomialSolution, SineSolution
from fem.solver import build_fem_mesh, FEMSolver, FEMResult
from fem.apriori_estimates import (
    compute_regularity_exponent, theoretical_convergence_rates,
    compute_convergence_rate, APrioriEstimate, analyze_convergence,
)

class TestFEMMesh(unittest.TestCase):
    def test_square_mesh(self):
        domain = make_domain("square")
        mesh = build_fem_mesh(domain, max_area=0.1, boundary_density=30)
        self.assertGreater(len(mesh.points), 10)
        self.assertGreater(len(mesh.elements), 5)
        self.assertGreater(len(mesh.boundary_nodes), 4)

    def test_l_shape_mesh(self):
        domain = make_domain("l_shape")
        mesh = build_fem_mesh(domain, max_area=0.1, boundary_density=30)
        self.assertGreater(len(mesh.elements), 5)

    def test_boundary_edge_types(self):
        domain = make_domain("square")
        mesh = build_fem_mesh(domain, max_area=0.1, boundary_density=30)

        self.assertTrue(np.all(mesh.boundary_edge_types > 0.5))

    def test_mixed_boundary(self):
        domain = make_domain("square_mixed")
        mesh = build_fem_mesh(domain, max_area=0.1, boundary_density=30)

        has_dir = np.any(mesh.boundary_edge_types > 0.5)
        has_neu = np.any(mesh.boundary_edge_types < 0.5)
        self.assertTrue(has_dir)
        self.assertTrue(has_neu)

class TestFEMSolver(unittest.TestCase):
    def _make_solver(self, domain_name, max_area=0.05):
        domain = make_domain(domain_name)
        mesh = build_fem_mesh(domain, max_area=max_area, boundary_density=40)
        solution = PolynomialSolution()

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

        return FEMSolver(mesh, f_func, u_func, grad_func)

    def test_solve_square(self):
        solver = self._make_solver("square")
        result = solver.solve()
        self.assertEqual(len(result.u), len(solver.mesh.points))
        self.assertGreater(result.h_max, 0)

    def test_solve_l_shape(self):
        solver = self._make_solver("l_shape")
        result = solver.solve()
        self.assertGreater(result.n_dof, 10)

    def test_errors_decrease(self):
        solver_coarse = self._make_solver("square", max_area=0.1)
        solver_coarse.solve()
        err_coarse = solver_coarse.compute_errors()

        solver_fine = self._make_solver("square", max_area=0.02)
        solver_fine.solve()
        err_fine = solver_fine.compute_errors()

        self.assertLess(err_fine["l2_error"], err_coarse["l2_error"])

    def test_compute_errors(self):
        solver = self._make_solver("square")
        solver.solve()
        errors = solver.compute_errors()
        self.assertIn("l2_error", errors)
        self.assertIn("energy_error", errors)
        self.assertGreater(errors["l2_error"], 0)

class TestAPrioriEstimates(unittest.TestCase):
    def test_square_regularity(self):
        alpha = compute_regularity_exponent("square")
        self.assertAlmostEqual(alpha, 1.0)

    def test_l_shape_regularity(self):
        alpha = compute_regularity_exponent("l_shape")
        expected = math.pi / (3 * math.pi / 2)  
        self.assertAlmostEqual(alpha, expected, places=5)

    def test_circle_regularity(self):
        alpha = compute_regularity_exponent("circle")
        self.assertAlmostEqual(alpha, 1.0)

    def test_convergence_rates_square(self):
        rate_l2, rate_h1 = theoretical_convergence_rates("square")
        self.assertAlmostEqual(rate_h1, 1.0)
        self.assertAlmostEqual(rate_l2, 2.0)

    def test_convergence_rates_l_shape(self):
        rate_l2, rate_h1 = theoretical_convergence_rates("l_shape")
        alpha = 2.0 / 3.0
        self.assertAlmostEqual(rate_h1, alpha, places=5)

    def test_compute_convergence_rate(self):
        h = np.array([0.1, 0.05, 0.025])
        err = 0.5 * h**2  
        rate = compute_convergence_rate(h, err)
        self.assertAlmostEqual(rate, 2.0, places=1)

    def test_analyze_convergence(self):
        estimates = [
            APrioriEstimate(h=0.1, theoretical_rate_l2=2.0, theoretical_rate_h1=1.0,
                           actual_error_l2=0.01, actual_error_h1=0.1, n_dof=100),
            APrioriEstimate(h=0.05, theoretical_rate_l2=2.0, theoretical_rate_h1=1.0,
                           actual_error_l2=0.0025, actual_error_h1=0.05, n_dof=400),
        ]
        study = analyze_convergence(estimates, "square")
        self.assertGreater(study.computed_rate_l2, 1.5)
        self.assertGreater(study.computed_rate_h1, 0.5)

if __name__ == "__main__":
    unittest.main()
