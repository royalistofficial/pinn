import unittest
import torch
import numpy as np
from geometry.domains import make_domain
from geometry.mesher import Mesher
from geometry.quadrature import QuadratureBuilder, ref_triangle_gauss

class TestRefTriangleGauss(unittest.TestCase):
    def test_weights_sum_to_half(self):
        for order in [1, 2, 3, 4, 5]:
            _, wts = ref_triangle_gauss(order)
            self.assertAlmostEqual(wts.sum(), 0.5, places=12)

    def test_points_inside_reference(self):
        for order in [1, 2, 5]:
            pts, _ = ref_triangle_gauss(order)
            self.assertTrue(np.all(pts >= -1e-12))
            self.assertTrue(np.all(pts.sum(axis=1) <= 1.0 + 1e-12))

class TestQuadratureBuilder(unittest.TestCase):
    def setUp(self):
        self.domain = make_domain("square")
        self.mesher = Mesher(max_area=0.1, lloyd_iters=1, boundary_density=40)
        self.mesh = self.mesher.build(self.domain)
        self.builder = QuadratureBuilder(tri_order=5, line_order=7, device=torch.device("cpu"))

    def test_quadrature_shapes(self):
        q = self.builder.build(self.mesh, self.domain)
        self.assertGreater(len(q.xy_in), 0)
        self.assertGreater(len(q.xy_bd), 0)

    def test_interior_weights_positive(self):
        q = self.builder.build(self.mesh, self.domain)
        self.assertTrue(torch.all(q.vol_w > 0))

    def test_integrate_constant_one(self):
        q = self.builder.build(self.mesh, self.domain)
        self.assertAlmostEqual(q.vol_w.sum().item(), 4.0, places=1)

    def test_boundary_normals_unit(self):
        q = self.builder.build(self.mesh, self.domain)
        norms = torch.norm(q.normals, dim=1)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-6, rtol=0)

if __name__ == "__main__":
    unittest.main()
