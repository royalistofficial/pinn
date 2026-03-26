import unittest
import math
import torch
import numpy as np
from geometry.domains import make_domain
from networks.activations import CnActivation, make_activation, SoftplusSmooth
from networks.blocks import NTKLinear, ResBlock, FourierNet
from networks.pinn import PINN
from networks.corners import extract_corners, build_corner_enrichment, CornerEnrichment

class TestActivations(unittest.TestCase):
    def test_cn_output_shape(self):
        self.assertEqual(CnActivation(n=0, alpha=0.01)(torch.randn(10)).shape, (10,))

    def test_cn_positive_monotone(self):
        act = CnActivation(n=2, alpha=0.01)
        x = torch.linspace(0, 2, 20)
        y = act(x)
        self.assertTrue(torch.all(y >= 0))
        self.assertTrue(torch.all(y[1:] - y[:-1] >= 0))

    def test_make_activation_gelu(self):
        act = make_activation("gelu")
        out = act(torch.randn(10))
        self.assertEqual(out.shape, (10,))

    def test_make_activation_silu(self):
        act = make_activation("silu")
        out = act(torch.randn(10))
        self.assertEqual(out.shape, (10,))

    def test_make_activation_softplus(self):
        act = make_activation("softplus")
        self.assertIsInstance(act, SoftplusSmooth)

    def test_make_activation_cn(self):
        act = make_activation("cn", n=1, alpha=0.01)
        self.assertIsInstance(act, CnActivation)

    def test_make_activation_invalid(self):
        with self.assertRaises(ValueError):
            make_activation("nonexistent")

class TestNTKLinear(unittest.TestCase):
    def test_output_shape(self):
        layer = NTKLinear(10, 5)
        out = layer(torch.randn(3, 10))
        self.assertEqual(out.shape, (3, 5))

    def test_scale_is_correct(self):
        layer = NTKLinear(16, 8)
        self.assertAlmostEqual(layer.scale, 1.0 / math.sqrt(16), places=6)

    def test_gradient_flows(self):
        layer = NTKLinear(4, 2)
        x = torch.randn(5, 4, requires_grad=True)
        out = layer(x).sum()
        out.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(layer.weight.grad)

    def test_ntk_scale_invariance(self):
        for width in [4, 16, 64, 256]:
            layer = NTKLinear(2, width)
            x = torch.randn(100, 2)
            out = layer(x)

            std = out.std().item()
            self.assertLess(std, 5.0, f"std={std} для width={width}")
            self.assertGreater(std, 0.01, f"std={std} для width={width}")

class TestResBlock(unittest.TestCase):
    def test_output_shape(self):
        block = ResBlock(16, activation="gelu", use_ntk_param=True)
        out = block(torch.randn(5, 16))
        self.assertEqual(out.shape, (5, 16))

    def test_residual_connection(self):
        block = ResBlock(8, activation="gelu")
        x = torch.randn(3, 8)
        out = block(x)

        diff = (out - x).norm() / x.norm()
        self.assertGreater(diff.item(), 0)  

class TestPINN(unittest.TestCase):
    def test_output_shape(self):
        self.assertEqual(PINN()(torch.randn(16, 2)).shape, (16, 1))

    def test_gradient_exists(self):
        xy = torch.randn(8, 2, requires_grad=True)
        PINN()(xy).sum().backward()
        self.assertIsNotNone(xy.grad)

    def test_deterministic(self):
        m = PINN(); m.eval(); xy = torch.randn(4, 2)
        torch.testing.assert_close(m(xy), m(xy))

    def test_with_init_freqs(self):
        from config import PINN_ARCH
        n_fourier = PINN_ARCH["n_fourier"]
        init_freqs = torch.linspace(-1, 1, n_fourier)
        model = PINN(init_freqs=init_freqs)
        out = model(torch.randn(5, 2))
        self.assertEqual(out.shape, (5, 1))

class TestCornerEnrichment(unittest.TestCase):
    def test_l_shape_has_reentrant(self):
        corners, angles = extract_corners(make_domain("l_shape"))
        self.assertGreater(len(corners), 0)
        self.assertTrue(any(a * 180 / np.pi > 180 for a in angles.numpy()))

    def test_square_corners(self):
        self.assertEqual(len(extract_corners(make_domain("square"))[0]), 4)

    def test_circle_no_corners(self):
        self.assertEqual(len(extract_corners(make_domain("circle"))[0]), 0)

    def test_enrichment_output_shape(self):
        ce = CornerEnrichment(
            torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
            torch.tensor([np.pi / 2, np.pi / 2]),
            n_harmonics=2,
        )
        self.assertEqual(ce(torch.randn(10, 2)).shape, (10, 2 * (1 + 2 * 2)))

    def test_pinn_with_enrichment(self):
        ce = build_corner_enrichment(make_domain("l_shape"), torch.device("cpu"))
        self.assertEqual(PINN(corner_enrichment=ce)(torch.randn(8, 2)).shape, (8, 1))

if __name__ == "__main__":
    unittest.main()
