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
        with self.assertRaises(ValueError): make_domain("nonexistent")

class TestDomainGeometry(unittest.TestCase):
    def _check_domain(self, domain):
        bv = domain.boundary_vertices(); bs = domain.boundary_segments()
        self.assertEqual(bv.ndim, 2); self.assertEqual(bv.shape[1], 2)
        self.assertEqual(bs.ndim, 2); self.assertEqual(bs.shape[1], 2)
        self.assertTrue(np.all(bs >= 0)); self.assertTrue(np.all(bs < len(bv)))
        for i in range(len(bs)): self.assertIn(domain.bc_type(i), [0.0, 1.0])

    def test_all_domains_valid(self):
        for name in DOMAIN_REGISTRY:
            with self.subTest(domain=name):
                self._check_domain(make_domain(name))

class TestSpecificDomains(unittest.TestCase):
    def test_square_vertices(self):
        self.assertEqual(len(SquareDomain().boundary_vertices()), 4)
    def test_l_shape_vertices(self):
        self.assertEqual(len(LShapeDomain().boundary_vertices()), 6)
    def test_hollow_square_has_hole(self):
        hl = HollowSquareDomain().holes()
        self.assertEqual(len(hl), 1)
    def test_circle_n_vertices(self):
        bv = CircleDomain(n_bd=32).boundary_vertices()
        self.assertEqual(len(bv), 32)

if __name__ == "__main__":
    unittest.main()
