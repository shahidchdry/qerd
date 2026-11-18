"""
Unit tests for Qiskit IBM Runtime hardware batching backend, hardware subgraph selection,
edge coloring, and vectorized gradient estimation.

Run with:
  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from qerd import (
    QERGradConfig,
    QERGradientEstimator,
    QERDTrainer,
    QuantumIsingDenoiser,
    build_ising_edges,
    color_ising_edges,
    select_hardware_subgraph,
)

try:
    from qiskit.primitives import StatevectorSampler as _StatevectorSampler
    from qiskit.providers.fake_provider import GenericBackendV2 as _GenericBackendV2
    from qiskit_ibm_runtime import SamplerV2 as _RuntimeSamplerV2

    _HAS_LIVE_QISKIT = True
except Exception:
    _HAS_LIVE_QISKIT = False


class _FakeParam:
    def __init__(self, name: str):
        self.name = name

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _FakeParam) and self.name == other.name


class _FakeBitArray:
    """Minimal Qiskit SamplerV2 BitArray lookalike implementing `.to_bool_array(order="little")`."""

    def __init__(self, bools_little: np.ndarray):
        self._bools_little = np.asarray(bools_little, dtype=bool)
        self.shape = self._bools_little.shape

    def to_bool_array(self, order: str = "big") -> np.ndarray:
        if order == "little":
            return self._bools_little.copy()
        if order == "big":
            return self._bools_little[..., ::-1].copy()
        raise ValueError(f"Unknown order: {order}")


class _FakeSamplerV2:
    """
    Fake Qiskit V2 Sampler verifying exact shuffled parameter-column permutation,
    multi-PUB submission counts, and little-endian bitstring measurement layout.
    """

    def __init__(self, expected_param_names: list[str], n_qubits: int, seed: int = 123):
        self.expected_param_names = expected_param_names
        self.n_qubits = n_qubits
        self.rng = np.random.default_rng(seed)
        self.pub_batches: list[list[tuple]] = []

    def run(self, pubs):
        self.pub_batches.append(list(pubs))
        results = []
        for isa_qc, param_mat, shots in pubs:
            assert list(p.name for p in isa_qc.parameters) == self.expected_param_names
            m = param_mat.shape[0]
            # Encode sign of layer-0 RZ parameters (named "rz_00_<q>") into output spin bias
            probs_one = np.full((m, self.n_qubits), 0.5, dtype=np.float64)
            name_to_col = {p.name: c for c, p in enumerate(isa_qc.parameters)}
            for q in range(self.n_qubits):
                col = name_to_col[f"rz_00_{q:03d}"]
                # Bit 1 corresponds to spin -1 (<Z> < 0); positive rz tilts toward +1 (bit 0)
                probs_one[:, q] = 1.0 / (1.0 + np.exp(np.clip(2.0 * param_mat[:, col], -5.0, 5.0)))
            rand_u = self.rng.random((m, int(shots), self.n_qubits))
            bits = rand_u < probs_one[:, None, :]
            pub_res = SimpleNamespace(data=SimpleNamespace(meas=_FakeBitArray(bits)))
            results.append(pub_res)

        return SimpleNamespace(result=lambda: results)


class TestHardwareQiskitBackend(unittest.TestCase):
    def test_hardware_subgraph_and_edge_coloring(self):
        # Construct a 156-qubit synthetic 12x13 grid coupling graph
        rows, cols = 12, 13
        raw_edges = []
        for r in range(rows):
            for c in range(cols):
                u = r * cols + c
                if c + 1 < cols:
                    raw_edges.append((u, u + 1))
                if r + 1 < rows:
                    raw_edges.append((u, u + cols))

        logical_edges, phys_qubits = select_hardware_subgraph(raw_edges, n_qubits=24, anchor=45)
        self.assertEqual(len(phys_qubits), 24)
        self.assertEqual(len(set(phys_qubits)), 24)
        self.assertTrue(all(0 <= u < v < 24 for u, v in logical_edges))

        colored = color_ising_edges(logical_edges)
        seen_indices = []
        for layer in colored:
            used_nodes = set()
            for idx, u, v in layer:
                self.assertNotIn(u, used_nodes)
                self.assertNotIn(v, used_nodes)
                used_nodes.update((u, v))
                seen_indices.append(idx)
        self.assertEqual(sorted(seen_indices), list(range(len(logical_edges))))
        self.assertLessEqual(len(colored), 4)

    def test_vectorized_gradient_estimator_parity(self):
        rng = np.random.default_rng(7)
        n, p_dim, n_vis, c_dim = 16, 22, 8, 4
        params = {"theta": np.zeros(p_dim), "cond_proj": np.zeros((n_vis, c_dim))}
        obs_p = rng.normal(size=(n, p_dim))
        obs_m = rng.normal(size=(n, p_dim))
        z_p = rng.normal(size=(n, n_vis))
        z_m = rng.normal(size=(n, n_vis))
        conds = rng.normal(size=(n, c_dim))
        nudges = rng.normal(size=(n, n_vis))

        est = QERGradientEstimator(QERGradConfig(beta=0.45, centered=True, auto_beta_guard=True))
        grads, align = est.estimate(params, obs_p, obs_m, z_p, z_m, conds=conds, nudge_dirs=nudges)

        ref_theta = np.mean((obs_p - obs_m) / 0.90, axis=0)
        ref_cond = sum(np.outer((z_p[i] - z_m[i]) / 0.90, conds[i]) for i in range(n)) / n
        ref_align = float(np.mean((z_p - z_m) * nudges))
        np.testing.assert_allclose(grads["theta"], ref_theta, atol=1e-12)
        np.testing.assert_allclose(grads["cond_proj"], ref_cond, atol=1e-12)
        self.assertAlmostEqual(align, ref_align, places=12)

        # Verify fallback path with mixed None entries in conds
        mixed_conds = [conds[i] if i % 2 == 0 else None for i in range(n)]
        grads_mix, _ = est.estimate(params, obs_p, obs_m, z_p, z_m, conds=mixed_conds)
        ref_mix = sum(np.outer((z_p[i] - z_m[i]) / 0.90, mixed_conds[i]) for i in range(0, n, 2)) / n
        np.testing.assert_allclose(grads_mix["cond_proj"], ref_mix, atol=1e-12)

    def test_qiskit_backend_transpile_once_permutation_and_training(self):
        n_qubits = 6
        transpile_counter = {"count": 0}
        shuffled_names_ref: list[str] = []

        def fake_circuit_builder(nq, edges, colored_layers, n_layers, total_params):
            transpile_counter["count"] += 1
            block = nq + len(edges) + 1
            params = []
            for layer in range(n_layers):
                for q in range(nq):
                    params.append(_FakeParam(f"rz_{layer:02d}_{q:03d}"))
                for e_idx in range(len(edges)):
                    params.append(_FakeParam(f"rzz_{layer:02d}_{e_idx:03d}"))
                params.append(_FakeParam(f"rx_{layer:02d}"))
            assert len(params) == total_params
            # Simulate Qiskit alphabetical parameter reordering after transpilation
            sorted_params = sorted(params, key=lambda pr: pr.name)
            shuffled_names_ref.clear()
            shuffled_names_ref.extend(pr.name for pr in sorted_params)
            isa_qc = SimpleNamespace(parameters=sorted_params)
            return isa_qc, params

        sampler = _FakeSamplerV2(shuffled_names_ref, n_qubits=n_qubits)
        # Note max_circuits_per_pub=6 forces batch of 2*8=16 circuits into 3 PUBs in 1 job!
        model = QuantumIsingDenoiser(
            data_shape=n_qubits,
            cond_dim=4,
            topology="chain",
            n_layers=2,
            backend="qiskit",
            real_qpu=True,
            device_name="ibm_fez",
            device_kwargs={
                "sampler": sampler,
                "circuit_builder": fake_circuit_builder,
                "skip_transpile": True,
                "max_circuits_per_pub": 6,
                "max_pubs_per_job": 4,
            },
            seed=9,
        )
        self.assertEqual(model.backend_name, "qiskit")

        data = np.array([
            [+1, +1, +1, -1, -1, -1],
            [-1, -1, -1, +1, +1, +1],
            [+1, -1, +1, -1, +1, -1],
            [-1, +1, -1, +1, -1, +1],
        ] * 4, dtype=np.float64)
        conds = np.tile(np.eye(4, dtype=np.float64), (4, 1))

        trainer = QERDTrainer(model)
        hist = trainer.train(
            data, conds, max_shots=1_280, shots_per_run=20, batch_size=8, eval_every_batches=2, eval_shots=40, verbose=True
        )
        self.assertGreaterEqual(len(hist), 1)
        # Template circuit built/transpiled ONCE despite multiple training + eval steps
        self.assertEqual(transpile_counter["count"], 1)
        # First training batch had 2*8=16 circuits chunked into 3 PUBs ([6, 6, 4]) in 1 single job call
        first_batch_pubs = sampler.pub_batches[0]
        self.assertEqual([pub[1].shape[0] for pub in first_batch_pubs], [6, 6, 4])

        # Verify context manager close behavior
        closed_flag = {"closed": False}
        model.backend._owns_mode = True
        model.backend._mode_ctx = SimpleNamespace(__exit__=lambda *a: closed_flag.__setitem__("closed", True))
        with model:
            pass
        self.assertTrue(closed_flag["closed"])

    @unittest.skipUnless(_HAS_LIVE_QISKIT, "qiskit / qiskit-ibm-runtime not installed")
    def test_live_qiskit_transpiled_parity_and_training(self):
        data_shape = (2, 3)
        n_steps = 3
        fake_hw = _GenericBackendV2(
            num_qubits=12,
            coupling_map=build_ising_edges((3, 4), topology="grid4"),
            seed=7,
        )
        hw_edges, phys_qubits = select_hardware_subgraph(fake_hw, n_qubits=6)

        model_np = QuantumIsingDenoiser(
            data_shape=data_shape, diffusion_steps=n_steps, edges=hw_edges,
            shots_per_traj=100_000, backend="numpy", p_depol=0.0, p_readout=0.0, seed=123,
        )
        model_qk = QuantumIsingDenoiser(
            data_shape=data_shape, diffusion_steps=n_steps, edges=hw_edges,
            shots_per_traj=100_000, backend="auto", real_qpu=True, seed=123,
            device_kwargs={
                "sampler": _StatevectorSampler(seed=123),
                "qiskit_backend": fake_hw,
                "initial_layout": phys_qubits,
                "optimization_level": 1,
            },
        )
        self.assertEqual(model_qk.backend_name, "qiskit")
        model_qk.load_state_dict(model_np.state_dict())

        rng = np.random.default_rng(42)
        x_batch = rng.uniform(-1.0, 1.0, size=(3, 6))
        t_batch = np.array([1, 2, 3], dtype=np.int64)
        obs_np, z_np = model_np.measure_observables_batch(x_batch, t_batch, shots=200_000)
        obs_qk, z_qk = model_qk.measure_observables_batch(x_batch, t_batch, shots=100_000)

        isa_qc, _ = model_qk.backend._get_isa_template(model_qk.n_layers)
        self.assertEqual(isa_qc.num_qubits, 12)
        self.assertEqual(isa_qc.count_ops().get("swap", 0), 0)
        self.assertLess(float(np.max(np.abs(obs_qk - obs_np))), 0.03)
        self.assertLess(float(np.max(np.abs(z_qk - z_np))), 0.03)

        # Verify SamplerV2(mode=GenericBackendV2) auto-configures runtime error mitigation without IBMService
        model_rt_cfg = QuantumIsingDenoiser(
            data_shape=data_shape, diffusion_steps=n_steps, edges=hw_edges,
            shots_per_traj=512, backend="auto", real_qpu=True, seed=123,
            device_kwargs={
                "qiskit_backend": fake_hw,
                "initial_layout": phys_qubits,
                "dynamical_decoupling": True,
                "twirling": True,
            },
        )
        self.assertIsInstance(model_rt_cfg.backend._sampler, _RuntimeSamplerV2)
        self.assertTrue(model_rt_cfg.backend._sampler.options.dynamical_decoupling.enable)
        self.assertTrue(model_rt_cfg.backend._sampler.options.twirling.enable_gates)

        # End-to-end 20-batch training accuracy parity
        patterns = np.array([
            [1, -1, 1, -1, 1, -1],
            [-1, 1, -1, 1, -1, 1],
            [1, 1, -1, -1, 1, 1],
            [-1, -1, 1, 1, -1, -1],
        ], dtype=np.float64)
        conds = np.eye(4, dtype=np.float64)

        m_np_tr = QuantumIsingDenoiser(
            data_shape=data_shape, cond_dim=4, diffusion_steps=n_steps, edges=hw_edges,
            shots_per_traj=1024, backend="numpy", p_depol=0.0, p_readout=0.0, seed=77,
        )
        m_qk_tr = QuantumIsingDenoiser(
            data_shape=data_shape, cond_dim=4, diffusion_steps=n_steps, edges=hw_edges,
            shots_per_traj=1024, backend="qiskit", real_qpu=True, seed=77,
            device_kwargs={
                "sampler": _StatevectorSampler(seed=77),
                "qiskit_backend": fake_hw,
                "initial_layout": phys_qubits,
            },
        )
        m_qk_tr.load_state_dict(m_np_tr.state_dict())

        QERDTrainer(m_np_tr).train(
            patterns, conds=conds, max_shots=163_840, shots_per_run=1024, batch_size=4, eval_every_batches=20, eval_shots=1024, verbose=False
        )
        QERDTrainer(m_qk_tr).train(
            patterns, conds=conds, max_shots=163_840, shots_per_run=1024, batch_size=4, eval_every_batches=20, eval_shots=1024, verbose=False
        )
        ev_np = m_np_tr.evaluate(patterns, conds=conds, shots=1024, n_trials=4)
        ev_qk = m_qk_tr.evaluate(patterns, conds=conds, shots=1024, n_trials=4)
        self.assertGreaterEqual(ev_np["sign_acc"], 0.99)
        self.assertGreaterEqual(ev_qk["sign_acc"], 0.99)
        self.assertLess(abs(ev_np["sign_acc"] - ev_qk["sign_acc"]), 0.02)
        self.assertLess(ev_qk["mse"], 0.05)


if __name__ == "__main__":
    unittest.main()
