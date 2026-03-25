import unittest
import numpy as np
from geometry.domains import (make_domain, BaseDomain, DOMAIN_REGISTRY,
    SquareDomain, LShapeDomain, HollowSquareDomain, CircleDomain)

class TestDomainRegistry(unittest.TestCase):
    def test_all_registered(self): self.assertEqual(len(DOMAIN_REGISTRY), 10)
    def test_make_domain_valid(self):
        for name in DOMAIN_REGISTRY:
            self.assertIsInstance(make_domain(name), BaseDomain)
    def test_make_domain_invalid(self):
        with self.assertRaises(ValueError): make_domain("nonexistent_domain")

class TestDomainGeometry(unittest.TestCase):
    def _check_domain(self, domain):
        bv = domain.boundary_vertices(); bs = domain.boundary_segments(); hl = domain.holes()
        self.assertEqual(bv.ndim, 2); self.assertEqual(bv.shape[1], 2)
        self.assertEqual(bs.ndim, 2); self.assertEqual(bs.shape[1], 2)
        self.assertTrue(np.all(bs >= 0)); self.assertTrue(np.all(bs < len(bv)))
        for i in range(len(bs)): self.assertIn(domain.bc_type(i), [0.0, 1.0])
    def test_all_domains_valid(self):
        for name in DOMAIN_REGISTRY:
            with self.subTest(domain=name): self._check_domain(make_domain(name))

class TestSpecificDomains(unittest.TestCase):
    def test_square_vertices(self):
        bv = SquareDomain().boundary_vertices(); self.assertEqual(len(bv), 4)
    def test_square_all_dirichlet(self):
        d = SquareDomain(); bs = d.boundary_segments()
        for i in range(len(bs)): self.assertEqual(d.bc_type(i), 1.0)
    def test_square_mixed_has_neumann(self): self.assertTrue(make_domain("square_mixed").has_neumann)
    def test_square_pure_no_neumann(self): self.assertFalse(make_domain("square").has_neumann)
    def test_l_shape_vertices(self): self.assertEqual(len(LShapeDomain().boundary_vertices()), 6)
    def test_hollow_square_has_hole(self):
        hl = HollowSquareDomain().holes(); self.assertEqual(len(hl), 1)
        np.testing.assert_array_almost_equal(hl[0], [0.0, 0.0])
    def test_circle_n_vertices(self):
        bv = CircleDomain(n_bd=32).boundary_vertices(); self.assertEqual(len(bv), 32)
        np.testing.assert_allclose(np.sqrt(bv[:,0]**2 + bv[:,1]**2), 1.0, atol=1e-12)

if __name__ == "__main__":
    unittest.main()
