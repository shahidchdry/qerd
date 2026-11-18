"""Classical ground-truth Simulated Annealing, greedy min-var screening, and 2^N GPU wall profiler."""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, Iterable, List

import numpy as np
import torch

from qerd import QuantumIsingDenoiser, build_ising_edges


def solve_simulated_annealing(
    n_qubits: int,
    cost_fn: Callable[[np.ndarray], np.ndarray],
    n_targets: int = 64,
    sweeps: int = 80,
    seed: int = 42,
) -> np.ndarray:
    """Vectorized batch Simulated Annealing optimality ceiling for any binary Ising objective."""
    rng = np.random.default_rng(seed)
    s = rng.choice([-1.0, 1.0], size=(n_targets, n_qubits)).astype(np.float64)
    cur_c = np.asarray(cost_fn(s), dtype=np.float64)
    temps = np.linspace(max(0.25, 0.25 * n_qubits / 12.0), 0.008, sweeps)

    for temp in temps:
        inv_t = 1.0 / max(temp, 1e-6)
        for q in rng.permutation(n_qubits):
            cand = s.copy()
            cand[:, q] = -cand[:, q]
            cand_c = np.asarray(cost_fn(cand), dtype=np.float64)
            dE = cand_c - cur_c
            rand_u = rng.random(n_targets)
            accept = (dE <= 0.0) | (rand_u < np.exp(-np.clip(dE * inv_t, -50.0, 50.0)))
            if np.any(accept):
                s[accept] = cand[accept]
                cur_c[accept] = cand_c[accept]
    return s


def greedy_minvar_portfolio(cov_train_ann: np.ndarray, target_k: int) -> np.ndarray:
    """Deterministic greedy volatility-screening baseline picking lowest pairwise variance stocks."""
    n = cov_train_ann.shape[0]
    chosen: List[int] = [int(np.argmin(np.diag(cov_train_ann)))]
    remaining = set(range(n)) - set(chosen)
    while len(chosen) < target_k and remaining:
        best_i = min(
            remaining,
            key=lambda i: float(np.mean(cov_train_ann[np.ix_(chosen + [i], chosen + [i])])),
        )
        chosen.append(best_i)
        remaining.remove(best_i)
    spins = np.full((1, n), -1.0, dtype=np.float64)
    spins[0, chosen] = 1.0
    return spins


def _grid_shape(n_qubits: int) -> tuple[int, int]:
    r = int(math.floor(math.sqrt(n_qubits)))
    while r > 1 and n_qubits % r != 0:
        r -= 1
    return r, n_qubits // r


def benchmark_t4_simulation_wall(
    qubit_counts: Iterable[int],
    n_layers: int = 2,
    batch_size: int = 10,
    shots_per_run: int = 24,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Empirically time PyTorch CUDA/CPU exact 2^N noisy trajectory simulation per qubit width."""
    use_cuda = torch.cuda.is_available()
    rows: List[Dict[str, Any]] = []
    for nq in qubit_counts:
        edges = build_ising_edges(_grid_shape(nq), topology="grid4")
        with QuantumIsingDenoiser(
            data_shape=nq,
            cond_dim=0,
            edges=edges,
            n_layers=n_layers,
            backend="torch" if use_cuda or nq >= 14 else "numpy",
            real_qpu=False,
            seed=seed,
        ) as m:
            rng = np.random.default_rng(seed)
            xt = rng.uniform(-0.8, 0.8, size=(2 * batch_size, nq))
            tc = rng.integers(1, 5, size=2 * batch_size)
            nt = rng.choice([-1.0, 1.0], size=(2 * batch_size, nq))
            nb = np.concatenate([np.full(batch_size, 0.45), np.full(batch_size, -0.45)])
            m.measure_observables_batch(xt, tc, shots=shots_per_run, nudge_target_batch=nt, nudge_beta_batch=nb)
            if use_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            m.measure_observables_batch(xt, tc, shots=shots_per_run, nudge_target_batch=nt, nudge_beta_batch=nb)
            if use_cuda:
                torch.cuda.synchronize()
            dt_sec = time.perf_counter() - t0

        sv_mb_single = (8.0 * (2 ** nq)) / (1024.0 ** 2)
        sv_mb_batch = sv_mb_single * (2 * batch_size * 4)
        n_params = len(edges) + nq
        rows.append({
            "n_qubits": int(nq),
            "hilbert_dim": int(2 ** nq),
            "n_params": int(n_params),
            "batch_ms": float(dt_sec * 1000.0),
            "sv_mb_single": float(sv_mb_single),
            "sv_mb_noisy_batch": float(sv_mb_batch),
            "qerd_circs_per_sample": 2,
            "psr_circs_per_sample": int(2 * n_params),
        })
    return rows
