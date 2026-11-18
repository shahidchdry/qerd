"""
Decoupled O(1)-circuit Contrastive Field Probing (QER-Grad / CFP) gradient estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class QERGradConfig:
    """Hyperparameters strictly for O(1) Contrastive Field Probing (QER-Grad / CFP)."""

    beta: float = 0.45
    beta_min: float = 0.15
    centered: bool = True         # (+beta, -beta) cancels quadratic RZ bias
    auto_beta_guard: bool = True  # Shrink beta if wraparound sign flips occur


CFPConfig = QERGradConfig


class QERGradientEstimator:
    """
    Standalone O(1)-circuit Contrastive Field Probing (QER-Grad / CFP) estimator.

    Designed for Hamiltonians whose trainable parameters multiply commuting diagonal
    Pauli generators (Z_i, Z_u Z_v) interleaved with fixed transverse mixers.
    Does not own Adam momentum buffers or step learning rates; can be paired with
    `StabilizedAdam`, standard SGD, Momentum, or PyTorch optimizers.
    """

    def __init__(self, config: Optional[QERGradConfig] = None):
        self.cfg = config or QERGradConfig()
        self.beta = float(self.cfg.beta)
        self._neg_align_streak = 0

    def estimate(
        self,
        params: Dict[str, np.ndarray],
        obs_plus_list: Sequence[np.ndarray],
        obs_minus_list: Sequence[np.ndarray],
        z_plus_list: Optional[Sequence[np.ndarray]] = None,
        z_minus_list: Optional[Sequence[np.ndarray]] = None,
        conds: Optional[Sequence[Optional[np.ndarray]]] = None,
        nudge_dirs: Optional[Sequence[np.ndarray]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Optional[float]]:
        """Compute O(1) contrastive ascent directions and update auto_beta_guard."""
        n = len(obs_plus_list)
        if n == 0:
            raise ValueError("Empty batch")
        denom = (2.0 * self.beta) if self.cfg.centered else self.beta

        has_theta = "theta" in params
        has_cond = "cond_proj" in params and conds is not None and z_plus_list is not None and z_minus_list is not None
        grads: Dict[str, np.ndarray] = {}

        if has_theta:
            obs_diff = np.asarray(obs_plus_list, dtype=np.float64) - np.asarray(obs_minus_list, dtype=np.float64)
            grads["theta"] = np.mean(obs_diff, axis=0) / denom

        mean_align: Optional[float] = None
        if z_plus_list is not None and z_minus_list is not None:
            z_diff = np.asarray(z_plus_list, dtype=np.float64) - np.asarray(z_minus_list, dtype=np.float64)
            if has_cond and conds is not None:
                if any(c is None for c in conds):
                    g_cond = np.zeros_like(params["cond_proj"], dtype=np.float64)
                    for i in range(n):
                        if conds[i] is not None:
                            g_cond += np.outer(z_diff[i] / denom, conds[i])
                    grads["cond_proj"] = g_cond / n
                else:
                    cond_arr = np.asarray(conds, dtype=np.float64)
                    grads["cond_proj"] = (z_diff.T @ cond_arr) / (n * denom)
            if nudge_dirs is not None:
                nudge_arr = np.asarray(nudge_dirs, dtype=np.float64)
                mean_align = float(np.mean(z_diff * nudge_arr))

        if self.cfg.auto_beta_guard and mean_align is not None:
            if mean_align < 0.0:
                self._neg_align_streak += 1
                if self._neg_align_streak >= 2 and self.beta > self.cfg.beta_min:
                    self.beta = max(self.cfg.beta_min, round(0.75 * self.beta, 4))
                    self._neg_align_streak = 0
            else:
                self._neg_align_streak = 0

        return grads, mean_align

    def state_dict(self) -> dict:
        return {"beta": self.beta, "neg_align_streak": self._neg_align_streak}

    def load_state_dict(self, state: dict):
        self.beta = float(state["beta"])
        self._neg_align_streak = int(state["neg_align_streak"])


CFPGradientEstimator = QERGradientEstimator
