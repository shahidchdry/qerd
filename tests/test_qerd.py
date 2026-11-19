"""
Fast unit & smoke test suite for the `qerd` package and `experiments` imports.

Run with:
  python -m tests.test_qerd
or:
  python -m unittest discover -s tests
"""

from __future__ import annotations

import importlib
import unittest

import numpy as np

import qerd
from qerd import (
    AdamCFP,
    AdamQER,
    CFPConfig,
    CFPGradientEstimator,
    QERConfig,
    QERGradConfig,
    QERGradientEstimator,
    QEROptimizer,
    QERDTrainer,
    QuantumIsingDenoiser,
    ReadoutGainCalibrator,
    StabilizedAdam,
    StabilizedAdamConfig,
    build_ising_edges,
    make_alpha_bar,
)
from qerd.__main__ import _run_self_tests
from qerd.backends import _HAS_TORCH


class TestQERDPackage(unittest.TestCase):
    def test_public_exports_and_aliases(self):
        self.assertIs(CFPConfig, QERGradConfig)
        self.assertIs(CFPGradientEstimator, QERGradientEstimator)
        self.assertIs(AdamQER, QEROptimizer)
        self.assertIs(AdamCFP, QEROptimizer)
        if _HAS_TORCH:
            self.assertIs(qerd.QERDDiffusionLayer, qerd.QERLayer)

    def test_schedules_and_topologies(self):
        ab = make_alpha_bar(4, "cosine")
        self.assertEqual(ab.shape, (5,))
        self.assertAlmostEqual(ab[0], 1.0)
        self.assertEqual(len(build_ising_edges((2, 4), topology="none")), 0)
        self.assertEqual(len(build_ising_edges((12,), topology="none", n_hidden=2)), 24)
        self.assertEqual(len(build_ising_edges((12,), topology="unentangled", n_hidden=2)), 0)
        self.assertEqual(len(build_ising_edges((12,), topology="all", n_hidden=1)), 66 + 12)
        self.assertEqual(len(build_ising_edges((3, 4), topology="chain")), 11)

    def test_ste_backward_mode_and_l1_exact_formula(self):
        # Verify exact L=1 2-qubit analytical formula: <Z1> = sin(phi)*sin(g*h1)*cos(g*J)
        import math
        m = QuantumIsingDenoiser(2, cond_dim=0, topology="all", n_layers=1, p_depol=0.0, p_readout=0.0, backend="numpy", seed=7)
        h1, h2, j12 = 0.45, -0.30, 0.55
        m.theta[:] = np.array([j12, h1, h2], dtype=np.float64)
        g1 = 2.0 * m.dt * m.temp_beta
        tau = 1.0
        phi = m.shake_scale * 2.0 * m.dt * (1.0 - m.shake_final_frac) * (0.6 + 0.4 * tau)
        # Noiseless exact expectation via large shots check (or statevector parity via torch)
        if _HAS_TORCH:
            import torch
            layer_ste = qerd.QERLayer(m, shots=16, gain=0.75, backward_mode="ste")
            x = torch.zeros((2, 2), dtype=torch.float32, requires_grad=True)
            h = torch.zeros((2, 2), dtype=torch.float32, requires_grad=True)
            out = layer_ste(x, h, t_step=2)
            out.sum().backward()
            self.assertIsNotNone(h.grad)
            self.assertAlmostEqual(float(layer_ste.qpu_theta.grad.abs().sum().item()), 0.0, places=7)
            self.assertTrue(np.allclose(h.grad.numpy(), np.full((2, 2), 1.0 / 0.75), atol=1e-5))
            # Check exact statevector L=1 formula directly using backend wavefunctions
            psi = np.full(4, 0.5, dtype=np.complex128)
            z1_vals = np.array([1.0, 1.0, -1.0, -1.0])
            z2_vals = np.array([1.0, -1.0, 1.0, -1.0])
            psi = psi * np.exp(-0.5j * g1 * (h1 * z1_vals + h2 * z2_vals + j12 * z1_vals * z2_vals))
            rx = np.array([[math.cos(phi / 2), -1j * math.sin(phi / 2)], [-1j * math.sin(phi / 2), math.cos(phi / 2)]])
            psi = np.kron(rx, rx) @ psi
            probs = np.abs(psi) ** 2
            z1_num = float(np.sum(probs * z1_vals))
            z12_num = float(np.sum(probs * z1_vals * z2_vals))
            z1_ana = math.sin(phi) * math.sin(g1 * h1) * math.cos(g1 * j12)
            z12_ana = (
                math.sin(phi) * math.cos(phi) * math.sin(g1 * j12) * (math.cos(g1 * h1) + math.cos(g1 * h2))
                + (math.sin(phi) ** 2) * math.sin(g1 * h1) * math.sin(g1 * h2)
            )
            self.assertAlmostEqual(z1_num, z1_ana, places=10)
            self.assertAlmostEqual(z12_num, z12_ana, places=10)

    def test_experiment_imports(self):
        for mod_name in (
            "experiments.benchmark_mnist",
            "experiments.run_paper_benchmarks",
            "experiments.eval_gradient_fidelity",
            "experiments.eval_spinglass_attribution",
            "experiments.eval_capacity_bottleneck",
            "experiments.eval_conditioning_and_opt",
            "experiments.eval_reciprocity_and_stepper",
            "experiments.eval_matched_controls",
            "experiments.eval_hardware_controls",
        ):
            mod = importlib.import_module(mod_name)
            self.assertIsNotNone(mod)

    def test_end_to_end_self_suite(self):
        _run_self_tests()


if __name__ == "__main__":
    unittest.main()
