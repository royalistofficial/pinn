import unittest
import torch

from geometry.domains import make_domain
from problems.solutions import PolynomialSolution
from comparison.compare import run_fem_solution, evaluate_pinn_errors
from networks.pinn import PINN

class TestFEMSolution(unittest.TestCase):
    def test_run_fem_square(self):
        domain = make_domain("square")
        solution = PolynomialSolution()
        result, errors, elapsed = run_fem_solution(domain, solution, max_area=0.1)
        self.assertGreater(result.n_dof, 10)
        self.assertIn("l2_error", errors)
        self.assertGreater(elapsed, 0)

    def test_run_fem_l_shape(self):
        domain = make_domain("l_shape")
        solution = PolynomialSolution()
        result, errors, _ = run_fem_solution(domain, solution, max_area=0.1)
        self.assertGreater(result.n_dof, 10)

class TestPINNErrors(unittest.TestCase):
    def test_evaluate_pinn_errors(self):
        domain = make_domain("square")
        solution = PolynomialSolution()
        pinn = PINN()
        errors = evaluate_pinn_errors(pinn, solution, domain, n_eval_pts=100)
        self.assertIn("l2_error", errors)
        self.assertIn("energy_error", errors)
        self.assertGreater(errors["l2_error"], 0)

if __name__ == "__main__":
    unittest.main()
