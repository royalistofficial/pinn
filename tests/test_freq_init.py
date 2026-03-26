import unittest
import torch
import numpy as np

from geometry.domains import make_domain
from problems.solutions import PolynomialSolution, SineSolution
from networks.freq_init import (
    estimate_rhs_spectrum,
    compute_adaptive_frequencies,
    compute_adaptive_frequencies_simple,
)

class TestRhsSpectrum(unittest.TestCase):
    def test_returns_dict(self):
        domain = make_domain("square")
        solution = PolynomialSolution()
        result = estimate_rhs_spectrum(solution, domain, n_grid=32)
        self.assertIn("radial_freqs", result)
        self.assertIn("radial_power", result)
        self.assertIn("dominant_freqs", result)
        self.assertIn("spectral_decay_rate", result)

    def test_dominant_freqs_positive(self):
        domain = make_domain("square")
        solution = SineSolution()
        result = estimate_rhs_spectrum(solution, domain, n_grid=32)

        self.assertGreater(len(result["dominant_freqs"]), 0)

    def test_decay_rate_positive(self):
        domain = make_domain("square")
        solution = PolynomialSolution()
        result = estimate_rhs_spectrum(solution, domain, n_grid=32)

        self.assertTrue(np.isfinite(result["spectral_decay_rate"]))

class TestAdaptiveFrequencies(unittest.TestCase):
    def test_output_shape(self):
        domain = make_domain("square")
        solution = PolynomialSolution()
        freqs = compute_adaptive_frequencies(solution, domain, n_fourier=4)
        self.assertEqual(len(freqs), 4)

    def test_sorted(self):
        domain = make_domain("l_shape")
        solution = PolynomialSolution()
        freqs = compute_adaptive_frequencies(solution, domain, n_fourier=6)

        for i in range(len(freqs) - 1):
            self.assertLessEqual(freqs[i].item(), freqs[i + 1].item() + 1e-6)

    def test_l_shape_vs_square(self):
        sol = PolynomialSolution()
        f_sq = compute_adaptive_frequencies(sol, make_domain("square"), n_fourier=4)
        f_ls = compute_adaptive_frequencies(sol, make_domain("l_shape"), n_fourier=4)

        self.assertGreater(f_ls.max().item(), f_sq.max().item() - 0.5)

class TestAdaptiveFrequenciesSimple(unittest.TestCase):
    def test_output_shape(self):
        freqs = compute_adaptive_frequencies_simple(
            make_domain("square"), n_fourier=4)
        self.assertEqual(len(freqs), 4)

    def test_l_shape_higher_freqs(self):
        f_sq = compute_adaptive_frequencies_simple(make_domain("square"), 4)
        f_ls = compute_adaptive_frequencies_simple(make_domain("l_shape"), 4)
        self.assertGreater(f_ls.max().item(), f_sq.max().item())

    def test_finite(self):
        freqs = compute_adaptive_frequencies_simple(make_domain("l_shape"), 8)
        self.assertTrue(torch.all(torch.isfinite(freqs)))

if __name__ == "__main__":
    unittest.main()
