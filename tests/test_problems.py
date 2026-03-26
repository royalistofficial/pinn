import unittest
import math
import torch
from problems.solutions import SineSolution, ExponentialSolution, PolynomialSolution

class _SolutionTestMixin:
    solution = None

    def test_eval_shape(self):
        self.assertEqual(self.solution.eval(torch.randn(10, 2)).shape, (10, 1))

    def test_grad_shape(self):
        ux, uy = self.solution.grad(torch.randn(10, 2))
        self.assertEqual(ux.shape, (10, 1))
        self.assertEqual(uy.shape, (10, 1))

    def test_rhs_shape(self):
        self.assertEqual(self.solution.rhs(torch.randn(10, 2)).shape, (10, 1))

    def test_grad_vector_shape(self):
        self.assertEqual(self.solution.grad_vector(torch.randn(10, 2)).shape, (10, 2))

    def test_grad_matches_autograd(self):
        xy = torch.randn(20, 2, requires_grad=True)
        u = self.solution.eval(xy)
        grad_auto = torch.autograd.grad(u.sum(), xy)[0]
        ux, uy = self.solution.grad(xy)
        torch.testing.assert_close(grad_auto, torch.cat([ux, uy], dim=1), atol=1e-4, rtol=1e-4)

    def test_rhs_is_minus_laplacian(self):
        xy = torch.randn(20, 2, requires_grad=True)
        u = self.solution.eval(xy)
        gu = torch.autograd.grad(u.sum(), xy, create_graph=True)[0]
        d2x = torch.autograd.grad(gu[:, 0].sum(), xy, create_graph=True)[0][:, 0:1]
        d2y = torch.autograd.grad(gu[:, 1].sum(), xy, create_graph=True)[0][:, 1:2]
        torch.testing.assert_close(-(d2x + d2y), self.solution.rhs(xy), atol=1e-3, rtol=1e-3)

    def test_neumann_data(self):
        xy = torch.randn(10, 2)
        normals = torch.randn(10, 2)
        normals = normals / normals.norm(dim=1, keepdim=True)
        gn = self.solution.neumann_data(xy, normals)
        expected = (self.solution.grad_vector(xy) * normals).sum(dim=1, keepdim=True)
        torch.testing.assert_close(gn, expected, atol=1e-6, rtol=1e-6)

class TestSineSolution(_SolutionTestMixin, unittest.TestCase):
    solution = SineSolution()

    def test_zero_on_integer_grid(self):
        xy = torch.tensor([[1.0, 0.5], [0.0, 0.3], [0.5, 0.0]])
        torch.testing.assert_close(
            self.solution.eval(xy),
            torch.zeros_like(self.solution.eval(xy)),
            atol=1e-6, rtol=0,
        )

class TestExponentialSolution(_SolutionTestMixin, unittest.TestCase):
    solution = ExponentialSolution()

    def test_positive(self):
        self.assertTrue(torch.all(self.solution.eval(torch.randn(20, 2)) > 0))

class TestPolynomialSolution(_SolutionTestMixin, unittest.TestCase):
    solution = PolynomialSolution()

    def test_zero_at_origin(self):
        self.assertAlmostEqual(
            self.solution.eval(torch.tensor([[0.0, 0.0]])).item(), 0.0, places=10
        )

if __name__ == "__main__":
    unittest.main()
