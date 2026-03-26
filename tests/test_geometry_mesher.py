import unittest
import numpy as np
from geometry.domains import make_domain
from geometry.mesher import Mesher, get_inside_mask

class TestGetInsideMask(unittest.TestCase):
    def test_square(self):
        bv = np.array([[-1,-1],[1,-1],[1,1],[-1,1]], dtype=np.float64)
        bs = np.array([[0,1],[1,2],[2,3],[3,0]], dtype=np.int32)
        mask = get_inside_mask(np.array([[0,0],[0.5,0.5],[2,2],[-2,0]]), bv, bs)
        self.assertTrue(mask[0]); self.assertTrue(mask[1])
        self.assertFalse(mask[2]); self.assertFalse(mask[3])

    def test_hollow_square(self):
        d = make_domain("hollow_square")
        bv = d.boundary_vertices(); bs = d.boundary_segments()
        mask = get_inside_mask(np.array([[0.0,0.0],[0.8,0.8]]), bv, bs)
        self.assertFalse(mask[0]); self.assertTrue(mask[1])

class TestMesher(unittest.TestCase):
    def setUp(self):
        self.mesher = Mesher(max_area=0.1, lloyd_iters=1, boundary_density=40)

    def _validate_mesh(self, mesh):
        pts, tris = mesh["points"], mesh["triangles"]
        self.assertEqual(pts.shape[1], 2); self.assertEqual(tris.shape[1], 3)
        self.assertTrue(np.all(tris >= 0)); self.assertTrue(np.all(tris < len(pts)))

    def test_square_mesh(self):
        m = self.mesher.build(make_domain("square"))
        self._validate_mesh(m)
        self.assertGreater(len(m["triangles"]), 10)

    def test_l_shape_mesh(self):
        self._validate_mesh(self.mesher.build(make_domain("l_shape")))

    def test_circle_mesh(self):
        self._validate_mesh(self.mesher.build(make_domain("circle")))

    def test_finer_mesh_more_triangles(self):
        d = make_domain("square")
        coarse = Mesher(max_area=0.2, lloyd_iters=1, boundary_density=20).build(d)
        fine = Mesher(max_area=0.02, lloyd_iters=1, boundary_density=60).build(d)
        self.assertGreater(len(fine["triangles"]), len(coarse["triangles"]))

if __name__ == "__main__":
    unittest.main()
