import unittest
import torch
import torch.nn as nn
from networks.ntk_utils import (
    compute_jacobian, compute_ntk_from_jacobian,
    compute_empirical_ntk, ntk_predict, ntk_train_dynamics,
    ntk_spectrum_analysis, ntk_preconditioned_step,
    extract_learned_frequencies,
)

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))
    def forward(self, x):
        return self.net(x)

class TestJacobian(unittest.TestCase):
    def test_shape(self):
        model = SimpleNet()
        J = compute_jacobian(model, torch.randn(5, 2))
        P = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertEqual(J.shape, (5, P))

    def test_finite(self):
        J = compute_jacobian(SimpleNet(), torch.randn(3, 2))
        self.assertTrue(torch.all(torch.isfinite(J)))

    def test_works_inside_no_grad(self):
        model = SimpleNet()
        with torch.no_grad():
            J = compute_jacobian(model, torch.randn(3, 2))
        self.assertTrue(torch.all(torch.isfinite(J)))
        self.assertGreater(J.abs().sum().item(), 0)

class TestNTK(unittest.TestCase):
    def test_symmetric(self):
        K = compute_empirical_ntk(SimpleNet(), torch.randn(5, 2))
        torch.testing.assert_close(K, K.T, atol=1e-5, rtol=1e-5)

    def test_positive_semidefinite(self):
        K = compute_empirical_ntk(SimpleNet(), torch.randn(5, 2))
        eigenvalues = torch.linalg.eigvalsh(K)
        self.assertTrue(torch.all(eigenvalues >= -1e-6))

    def test_from_jacobian_matches(self):
        model = SimpleNet()
        X = torch.randn(4, 2)
        J = compute_jacobian(model, X)
        K1 = compute_ntk_from_jacobian(J)
        K2 = compute_empirical_ntk(model, X)
        torch.testing.assert_close(K1, K2, atol=1e-4, rtol=1e-4)

class TestNTKPredict(unittest.TestCase):
    def test_predict_shape(self):
        y = ntk_predict(SimpleNet(), torch.randn(8, 2), torch.randn(8, 1),
                        torch.randn(3, 2), lr=0.01, epoch=100)
        self.assertEqual(y.shape, (3, 1))

    def test_train_dynamics_shape(self):
        y = ntk_train_dynamics(SimpleNet(), torch.randn(5, 2),
                               torch.randn(5, 1), lr=0.01, epoch=100)
        self.assertEqual(y.shape, (5, 1))

    def test_convergence(self):
        model = SimpleNet()
        X, y = torch.randn(5, 2), torch.randn(5, 1)
        y1 = ntk_train_dynamics(model, X, y, lr=0.1, epoch=10)
        y2 = ntk_train_dynamics(model, X, y, lr=0.1, epoch=10000)
        self.assertLess((y2 - y).norm(), (y1 - y).norm() + 1e-6)

class TestNTKSpectrum(unittest.TestCase):
    def test_keys(self):
        result = ntk_spectrum_analysis(SimpleNet(), torch.randn(10, 2))
        for key in ["eigenvalues", "condition_number", "effective_rank", "trace"]:
            self.assertIn(key, result)

    def test_effective_rank_positive(self):
        result = ntk_spectrum_analysis(SimpleNet(), torch.randn(10, 2))
        self.assertGreater(result["effective_rank"], 0)

class TestNTKPreconditioner(unittest.TestCase):
    def test_step_runs(self):
        model = SimpleNet()
        X = torch.randn(10, 2)
        r = torch.randn(10, 1)
        norm = ntk_preconditioned_step(model, X, r, lr=0.01, reg=1e-3)
        self.assertGreater(norm, 0)

    def test_step_changes_params(self):
        model = SimpleNet()
        params_before = [p.clone() for p in model.parameters()]
        X = torch.randn(10, 2)
        r = torch.randn(10, 1) * 10
        ntk_preconditioned_step(model, X, r, lr=0.01, reg=1e-3)
        changed = False
        for p_before, p_after in zip(params_before, model.parameters()):
            if not torch.allclose(p_before, p_after.data):
                changed = True
                break
        self.assertTrue(changed)

class TestExtractFrequencies(unittest.TestCase):
    def test_simple_model_empty(self):
        freqs = extract_learned_frequencies(SimpleNet())
        self.assertEqual(len(freqs), 0)

    def test_pinn_has_frequencies(self):
        from networks.pinn import PINN
        model = PINN()
        freqs = extract_learned_frequencies(model)
        self.assertGreater(len(freqs), 0)

if __name__ == "__main__":
    unittest.main()
