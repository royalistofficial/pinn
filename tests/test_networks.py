import unittest, torch
import numpy as np
from geometry.domains import make_domain
from geometry.mesher import Mesher
from geometry.quadrature import QuadratureBuilder
from networks.activations import CnActivation
from networks.pinn import PINN
from networks.corners import extract_corners, build_corner_enrichment, CornerEnrichment

class TestCnActivation(unittest.TestCase):
    def test_c0_output_shape(self):
        self.assertEqual(CnActivation(n=0, alpha=0.01)(torch.randn(10)).shape, (10,))
    def test_c2_positive_monotone(self):
        act = CnActivation(n=2, alpha=0.01); x = torch.linspace(0,2,20); y = act(x)
        self.assertTrue(torch.all(y >= 0)); self.assertTrue(torch.all(y[1:]-y[:-1] >= 0))

class TestPINN(unittest.TestCase):
    def test_output_shape(self): self.assertEqual(PINN()(torch.randn(16,2)).shape, (16,1))
    def test_gradient_exists(self):
        xy = torch.randn(8,2, requires_grad=True); PINN()(xy).sum().backward()
        self.assertIsNotNone(xy.grad)
    def test_deterministic(self):
        m = PINN(); m.eval(); xy = torch.randn(4,2)
        torch.testing.assert_close(m(xy), m(xy))

class TestCornerEnrichment(unittest.TestCase):
    def test_l_shape_has_reentrant(self):
        corners, angles = extract_corners(make_domain("l_shape"))
        self.assertGreater(len(corners), 0)
        self.assertTrue(any(a*180/np.pi > 180 for a in angles.numpy()))
    def test_square_corners(self): self.assertEqual(len(extract_corners(make_domain("square"))[0]), 4)
    def test_circle_no_corners(self): self.assertEqual(len(extract_corners(make_domain("circle"))[0]), 0)
    def test_enrichment_output_shape(self):
        ce = CornerEnrichment(torch.tensor([[0.0,0.0],[1.0,0.0]]), torch.tensor([np.pi/2, np.pi/2]), n_harmonics=2)
        self.assertEqual(ce(torch.randn(10,2)).shape, (10, 2*(1+2*2)))
    def test_build_returns_none_for_circle(self):
        self.assertIsNone(build_corner_enrichment(make_domain("circle"), torch.device("cpu")))
    def test_build_returns_module_for_l_shape(self):
        ce = build_corner_enrichment(make_domain("l_shape"), torch.device("cpu"))
        self.assertIsNotNone(ce); self.assertIsInstance(ce, CornerEnrichment)
    def test_pinn_with_enrichment(self):
        ce = build_corner_enrichment(make_domain("l_shape"), torch.device("cpu"))
        self.assertEqual(PINN(corner_enrichment=ce)(torch.randn(8,2)).shape, (8,1))

if __name__ == "__main__": unittest.main()