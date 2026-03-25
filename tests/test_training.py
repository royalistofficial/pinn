import os
import tempfile
import unittest
import torch

from geometry.domains import make_domain
from geometry.mesher import Mesher
from geometry.quadrature import QuadratureBuilder
from problems.solutions import SineSolution
from training.data_module import DataModule, prepare_sample
from training.trainer import Trainer
from file_io.logger import FileLogger

class TestDataModule(unittest.TestCase):

    def setUp(self):
        self.domain = make_domain("square")
        mesher = Mesher(max_area=0.1, lloyd_iters=1, boundary_density=30)
        mesh = mesher.build(self.domain)
        builder = QuadratureBuilder(tri_order=5, line_order=7,
                                    device=torch.device("cpu"))
        quad = builder.build(mesh, self.domain)
        solution = SineSolution()
        sample = prepare_sample(quad, solution)
        self.dm = DataModule(sample, batch_size=256)

    def test_in_loader_yields_batches(self):
        batch = next(iter(self.dm.in_loader))
        self.assertEqual(len(batch), 5)  
        xy_in, f_in, vol_w, tri_idx, idx_in = batch
        self.assertEqual(xy_in.shape[1], 2)
        self.assertEqual(f_in.shape[1], 1)

    def test_boundary_iter_infinite(self):
        bd_iter = iter(self.dm.boundary_iter())
        for _ in range(10):
            batch = next(bd_iter)
            self.assertEqual(len(batch), 8)  

    def test_scale_weights(self):
        batch_in = next(iter(self.dm.in_loader))
        batch_bd = next(iter(self.dm.boundary_iter()))
        vol_w_sc, surf_w_sc = self.dm.scale_weights(
            batch_in[0], batch_bd[0], batch_in[2], batch_bd[5])
        self.assertTrue(torch.all(vol_w_sc > 0))
        self.assertTrue(torch.all(surf_w_sc > 0))

    def test_sample_has_correct_fields(self):
        s = self.dm.sample
        self.assertIsNotNone(s.quad)
        self.assertIsNotNone(s.f_in)
        self.assertIsNotNone(s.g_D)
        self.assertIsNotNone(s.g_N)
        self.assertEqual(s.f_in.shape[1], 1)
        self.assertEqual(s.g_D.shape[1], 1)


class TestTrainerInit(unittest.TestCase):
    def test_trainer_initialization(self):
        domain = make_domain("square")
        mesher = Mesher(max_area=0.2, lloyd_iters=1, boundary_density=20)
        mesh = mesher.build(domain)
        builder = QuadratureBuilder(tri_order=3, line_order=5,
                                    device=torch.device("cpu"))
        quad = builder.build(mesh, domain)
        solution = SineSolution()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            log_path = f.name
            
        try:
            with FileLogger(log_path, also_print=False) as logger:
                trainer = Trainer(
                    domain=domain, 
                    quad=quad, 
                    solution=solution,
                    logger=logger
                )
                
                # Проверяем наличие нужных атрибутов для базового PINN
                self.assertIsNotNone(trainer.pinn)
                self.assertIsNotNone(trainer.opt_pinn)
                self.assertIsNotNone(trainer.evaluator)
                self.assertIsNotNone(trainer.data)
                
        finally:
            os.unlink(log_path)

if __name__ == "__main__":
    unittest.main()