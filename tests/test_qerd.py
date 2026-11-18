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
        self.assertEqual(len(build_ising_edges((12,), topology="all", n_hidden=1)), 66 + 12)
        self.assertEqual(len(build_ising_edges((3, 4), topology="chain")), 11)

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
        ):
            mod = importlib.import_module(mod_name)
            self.assertIsNotNone(mod)

    def test_end_to_end_self_suite(self):
        _run_self_tests()


if __name__ == "__main__":
    unittest.main()
