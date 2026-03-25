import unittest, math
import numpy as np, torch
from geometry.domains import make_domain
from geometry.mesher import Mesher
from geometry.quadrature import QuadratureBuilder, ref_triangle_gauss

class TestRefTriangleGauss(unittest.TestCase):
    def test_weights_sum_to_half(self):
        for order in [1,2,3,4,5]:
            _, wts = ref_triangle_gauss(order)
            self.assertAlmostEqual(wts.sum(), 0.5, places=12)
    def test_points_inside_reference(self):
        for order in [1,2,5]:
            pts, _ = ref_triangle_gauss(order)
            self.assertTrue(np.all(pts >= -1e-12))
            self.assertTrue(np.all(pts.sum(axis=1) <= 1.0+1e-12))

class TestQuadratureBuilder(unittest.TestCase):
    def setUp(self):
        self.domain = make_domain("square")
        self.mesher = Mesher(max_area=0.1, lloyd_iters=1, boundary_density=40)
        self.mesh = self.mesher.build(self.domain)
        self.builder = QuadratureBuilder(tri_order=5, line_order=7, device=torch.device("cpu"))
    def test_quadrature_shapes(self):
        q = self.builder.build(self.mesh, self.domain)
        N_in, N_bd = len(q.xy_in), len(q.xy_bd)
        self.assertGreater(N_in, 0); self.assertGreater(N_bd, 0)
        self.assertEqual(q.vol_w.shape, (N_in, 1)); self.assertEqual(q.surf_w.shape, (N_bd, 1))
    def test_tri_indices_valid_range(self):
        q = self.builder.build(self.mesh, self.domain)
        self.assertTrue(torch.all(q.tri_indices >= 0))
        self.assertTrue(torch.all(q.tri_indices < len(self.mesh["triangles"])))
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
    def test_dirichlet_mask_square(self):
        q = self.builder.build(self.mesh, self.domain)
        self.assertTrue(torch.all(q.bc_mask == 1.0))
    def test_mixed_mask_has_both(self):
        domain = make_domain("square_mixed")
        q = self.builder.build(self.mesher.build(domain), domain)
        self.assertTrue(torch.any(q.bc_mask == 1.0)); self.assertTrue(torch.any(q.bc_mask == 0.0))
    def test_tri_indices_bd_on_different_domains(self):
        for name in ["square", "l_shape", "circle", "hollow_square"]:
            with self.subTest(domain=name):
                domain = make_domain(name)
                mesh = Mesher(max_area=0.15, lloyd_iters=1, boundary_density=30).build(domain)
                q = self.builder.build(mesh, domain)
                self.assertEqual(q.tri_indices_bd.shape[0], q.xy_bd.shape[0])
                self.assertTrue(torch.all(q.tri_indices_bd >= 0))

if __name__ == "__main__": unittest.main()
