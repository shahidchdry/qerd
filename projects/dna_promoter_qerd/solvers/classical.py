"""Classical DNA baselines (`BigramHMM`, `RealGenomeCeiling`) and exact 2^N scaling profiler."""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch

from qerd import QuantumIsingDenoiser, build_ising_edges


def benchmark_2n_simulation_wall(
    qubit_counts: Iterable[int],
    n_layers: int = 2,
    batch_size: int = 16,
    shots_per_run: int = 48,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Measure PyTorch/NumPy exact 2^N noisy trajectory simulation RAM and latency."""
    use_cuda = torch.cuda.is_available()
    rows: List[Dict[str, Any]] = []
    for nq in qubit_counts:
        r = max(2, int(math.floor(math.sqrt(nq))))
        while r > 1 and nq % r != 0:
            r -= 1
        edges = build_ising_edges((r, nq // r), topology="grid4")
        with QuantumIsingDenoiser(
            data_shape=nq, cond_dim=0, edges=edges, n_layers=n_layers,
            backend="torch" if use_cuda or nq >= 14 else "numpy", real_qpu=False, seed=seed,
        ) as m:
            rng = np.random.default_rng(seed)
            xt = rng.uniform(-0.8, 0.8, size=(2 * batch_size, nq))
            tc = rng.integers(1, 4, size=2 * batch_size)
            nt = rng.choice([-1.0, 1.0], size=(2 * batch_size, nq))
            nb = np.concatenate([np.full(batch_size, 0.45), np.full(batch_size, -0.45)])
            m.measure_observables_batch(xt, tc, shots=shots_per_run, nudge_target_batch=nt, nudge_beta_batch=nb)
            if use_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            m.measure_observables_batch(xt, tc, shots=shots_per_run, nudge_target_batch=nt, nudge_beta_batch=nb)
            if use_cuda:
                torch.cuda.synchronize()
            dt_ms = 1000.0 * (time.perf_counter() - t0)
        rows.append({
            "n_qubits": int(nq),
            "hilbert_dim": int(1 << nq),
            "batch_ms": float(dt_ms),
            "sv_mb_noisy_batch": float(((8.0 * (1 << nq)) / (1024.0 ** 2)) * (2 * batch_size * 4)),
        })
    return rows


def sample_classical_dinuc_baseline(
    train_seqs: Sequence[str],
    n_samples: int,
    seed: int = 42,
) -> List[str]:
    """
    Fit an inhomogeneous 1st-order Markov (position-specific dinucleotide) generative model
    on the real training promoters and sample novel sequences from it as a classical baseline.
    """
    if not train_seqs:
        return []
    L = len(train_seqs[0])
    bmap = {"A": 0, "C": 1, "G": 2, "T": 3}
    alphabet = ["A", "C", "G", "T"]

    p0 = np.full(4, 1.0, dtype=np.float64)
    trans = np.full((L - 1, 4, 4), 0.25, dtype=np.float64)
    for s in train_seqs:
        p0[bmap[s[0]]] += 1.0
        for i in range(L - 1):
            trans[i, bmap[s[i]], bmap[s[i + 1]]] += 1.0
    p0 /= p0.sum()
    trans /= trans.sum(axis=2, keepdims=True)

    rng = np.random.default_rng(seed)
    out: List[str] = []
    for _ in range(n_samples):
        cur = int(rng.choice(4, p=p0))
        chars = [alphabet[cur]]
        for i in range(L - 1):
            cur = int(rng.choice(4, p=trans[i, cur]))
            chars.append(alphabet[cur])
        out.append("".join(chars))
    return out


def fit_core8_pssm(pos_seqs: Sequence[str], neg_seqs: Sequence[str]) -> np.ndarray:
    """Fit empirical 8-bp core motif log-odds matrix `(8, 4)` directly from real human promoters."""
    bmap = {"A": 0, "C": 1, "G": 2, "T": 3}
    pos_c = np.ones((8, 4), dtype=np.float64)
    neg_c = np.ones((8, 4), dtype=np.float64)
    for s in pos_seqs:
        sub = s[18:22] + s[47:51]
        for j, ch in enumerate(sub):
            pos_c[j, bmap[ch]] += 1.0
    for s in neg_seqs:
        sub = s[18:22] + s[47:51]
        for j, ch in enumerate(sub):
            neg_c[j, bmap[ch]] += 1.0
    pos_p = pos_c / pos_c.sum(axis=1, keepdims=True)
    neg_p = neg_c / neg_c.sum(axis=1, keepdims=True)
    return np.log(pos_p / neg_p)
