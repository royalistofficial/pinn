import unittest
import torch
from functionals.integrals import domain_integral, boundary_integral
from functionals.operators import gradient, laplacian, normal_derivative
from functionals.errors import energy_error, relative_l2_error
from functionals.losses import pde_residual_loss, dirichlet_loss, neumann_loss
from problems.solutions import SineSolution

class TestDomainIntegral(unittest.TestCase):
    def test_constant(self):
        vals = torch.ones(4, 1); w = torch.full((4, 1), 0.5)
        self.assertAlmostEqual(domain_integral(vals, w).item(), 2.0, places=10)
        
    def test_weighted(self):
        vals = torch.tensor([[2.0], [3.0]]); w = torch.tensor([[0.5], [1.0]])
        self.assertAlmostEqual(domain_integral(vals, w).item(), 4.0, places=10)

class TestBoundaryIntegral(unittest.TestCase):
    def test_without_mask(self):
        vals = torch.ones(5, 1) * 2.0; w = torch.ones(5, 1) * 0.3
        self.assertAlmostEqual(boundary_integral(vals, w).item(), 3.0, places=10)
        
    def test_with_mask(self):
        vals = torch.ones(4, 1); w = torch.ones(4, 1)
        mask = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
        self.assertAlmostEqual(boundary_integral(vals, w, mask=mask).item(), 2.0, places=10)

class TestOperators(unittest.TestCase):
    def test_gradient_linear(self):
        xy = torch.randn(10, 2, requires_grad=True)
        v = 3 * xy[:, 0:1] + 2 * xy[:, 1:2]
        gv = gradient(v, xy)
        torch.testing.assert_close(gv[:, 0], torch.full((10,), 3.0), atol=1e-5, rtol=0)
        torch.testing.assert_close(gv[:, 1], torch.full((10,), 2.0), atol=1e-5, rtol=0)
        
    def test_laplacian_quadratic(self):
        xy = torch.randn(10, 2, requires_grad=True)
        v = xy[:, 0:1]**2 + xy[:, 1:2]**2
        _, lap = laplacian(v, xy)
        torch.testing.assert_close(lap, torch.full((10, 1), 4.0), atol=1e-4, rtol=0)
        
    def test_laplacian_harmonic(self):
        xy = torch.randn(10, 2, requires_grad=True)
        v = xy[:, 0:1] * xy[:, 1:2]
        _, lap = laplacian(v, xy)
        torch.testing.assert_close(lap, torch.zeros(10, 1), atol=1e-5, rtol=0)
        
    def test_normal_derivative(self):
        xy = torch.randn(5, 2, requires_grad=True); v = xy[:, 0:1]
        normals = torch.tensor([[1, 0]] * 5, dtype=torch.float32)
        dvdn = normal_derivative(v, xy, normals)
        torch.testing.assert_close(dvdn, torch.ones(5, 1), atol=1e-5, rtol=0)

class TestLosses(unittest.TestCase):
    def test_pde_zero_residual(self):
        lap_v = torch.tensor([[-2.0], [-2.0]]); f = torch.tensor([[2.0], [2.0]])
        self.assertAlmostEqual(pde_residual_loss(lap_v, f, torch.ones(2, 1)).item(), 0.0, places=8)
        
    def test_dirichlet_zero(self):
        v = torch.tensor([[1.0], [2.0]]); g = torch.tensor([[1.0], [2.0]])
        self.assertAlmostEqual(dirichlet_loss(v, g, torch.ones(2, 1), torch.ones(2, 1)).item(), 0.0, places=8)
        
    def test_dirichlet_nonzero(self):
        v = torch.tensor([[1.0], [2.0]]); g = torch.zeros(2, 1)
        self.assertAlmostEqual(dirichlet_loss(v, g, torch.ones(2, 1), torch.ones(2, 1)).item(), 5.0, places=5)
        
    def test_neumann_mask(self):
        dvdn = torch.randn(5, 1); g_N = torch.zeros(5, 1)
        bc_mask = torch.ones(5, 1)
        self.assertAlmostEqual(neumann_loss(dvdn, g_N, torch.ones(5, 1), bc_mask).item(), 0.0, places=8)

class TestErrors(unittest.TestCase):
    def test_energy_error_zero_for_exact(self):
        sol = SineSolution(); xy = torch.rand(30, 2) * 2 - 1
        self.assertAlmostEqual(energy_error(sol.grad_vector(xy), xy, torch.ones(30,1)*0.1, sol).item(), 0.0, places=6)
        
    def test_relative_l2_zero_for_exact(self):
        sol = SineSolution(); xy = torch.rand(30, 2) * 2 - 1
        self.assertAlmostEqual(relative_l2_error(sol.eval(xy), xy, torch.ones(30,1)*0.1, sol).item(), 0.0, places=6)
        
    def test_relative_l2_bounded(self):
        sol = SineSolution(); xy = torch.rand(30, 2) * 2 - 1
        v = sol.eval(xy) + 0.01 * torch.randn(30, 1)
        rel = relative_l2_error(v, xy, torch.ones(30,1)*0.1, sol)
        self.assertGreater(rel.item(), 0.0); self.assertLess(rel.item(), 10.0)

if __name__ == "__main__":
    unittest.main()