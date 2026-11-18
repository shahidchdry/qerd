"""Native Heavy-Hex frustrated Ising spin-glass problem (zero SWAP overhead on IBM Heron R2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class HeavyHexSpinGlassProblem:
    n_qubits: int
    edges: List[Tuple[int, int]]
    bonds: np.ndarray
    cost_fn: Callable[[np.ndarray], np.ndarray]

    def evaluate(
        self,
        samples: np.ndarray,
        baseline_untrained_cost: float,
        ceiling_sa_cost: float,
    ) -> Dict[str, float]:
        spins = np.where(np.asarray(samples, dtype=np.float64) >= 0.0, 1.0, -1.0)
        eu = np.array([u for u, _ in self.edges], dtype=np.int64)
        ev = np.array([v for _, v in self.edges], dtype=np.int64)
        prod = spins[:, eu] * spins[:, ev]
        energies = np.sum(self.bonds[None, :] * prod, axis=1)
        satisfied_frac = float(np.mean((self.bonds[None, :] * prod) < 0.0))

        mean_e = float(np.mean(energies))
        denom = max(1e-8, baseline_untrained_cost - ceiling_sa_cost)
        gap_closed_pct = 100.0 * (baseline_untrained_cost - mean_e) / denom
        uniq = {tuple(r.astype(int).tolist()) for r in spins}

        return {
            "mean_energy": mean_e,
            "best_energy": float(np.min(energies)),
            "satisfied_bonds_pct": 100.0 * satisfied_frac,
            "gap_closed_pct": gap_closed_pct,
            "unique_frac": float(len(uniq) / max(1, len(spins))),
        }


def build_spinglass_problem(
    n_qubits: int,
    edges: List[Tuple[int, int]],
    seed: int = 42,
) -> HeavyHexSpinGlassProblem:
    rng = np.random.default_rng(seed)
    bonds = rng.choice([-1.0, 1.0], size=len(edges)).astype(np.float64)
    eu = np.array([u for u, _ in edges], dtype=np.int64)
    ev = np.array([v for _, v in edges], dtype=np.int64)

    def cost_fn(s_mat: np.ndarray) -> np.ndarray:
        spins = np.where(np.asarray(s_mat, dtype=np.float64) >= 0.0, 1.0, -1.0)
        return np.sum(bonds[None, :] * (spins[:, eu] * spins[:, ev]), axis=1)

    return HeavyHexSpinGlassProblem(n_qubits=n_qubits, edges=edges, bonds=bonds, cost_fn=cost_fn)
