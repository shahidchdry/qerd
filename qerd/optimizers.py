"""
Physics-stabilized parameter updaters (`StabilizedAdam`) and composite `QEROptimizer` (`Adam-QER`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from .gradients import QERGradConfig, QERGradientEstimator


@dataclass
class StabilizedAdamConfig:
    """Hyperparameters strictly for the parameter update rule (Adam / SGD + stabilizers)."""

    lr_max: float = 0.12
    lr_min: float = 0.015
    beta1: float = 0.75
    beta2: float = 0.98
    eps: float = 1e-6
    weight_decay: float = 0.01    # LR-coupled decay strength
    clip_bound: float = 0.90      # Box bound avoiding 2*pi/pi phase wraparound
    use_adam: bool = True         # Disable for fixed-LR SGD ablation


class StabilizedAdam:
    """
    Standalone stabilized Adam / SGD parameter updater with cosine LR annealing,
    LR-coupled weight decay, and periodic phase box clipping.
    """

    def __init__(
        self,
        params: Optional[Dict[str, np.ndarray]] = None,
        config: Optional[StabilizedAdamConfig] = None,
    ):
        self.cfg = config or StabilizedAdamConfig()
        self._m: Dict[str, np.ndarray] = {}
        self._v: Dict[str, np.ndarray] = {}
        if params is not None:
            for k, arr in params.items():
                self._m[k] = np.zeros_like(arr, dtype=np.float64)
                self._v[k] = np.zeros_like(arr, dtype=np.float64)
        self.step_count = 0
        self.shots_used = 0

    def current_lr(self, progress: float) -> float:
        p = float(np.clip(progress, 0.0, 1.0))
        return self.cfg.lr_min + 0.5 * (self.cfg.lr_max - self.cfg.lr_min) * (1.0 + math.cos(math.pi * p))

    def step(
        self,
        params: Dict[str, np.ndarray],
        ascent_grads: Dict[str, np.ndarray],
        progress: float = 0.0,
        shots_in_batch: int = 0,
    ) -> float:
        """Apply one ascent step in-place to each named array in `params` and return lr."""
        self.step_count += 1
        self.shots_used += int(shots_in_batch)
        lr = self.current_lr(progress)
        eff_decay = 1.0 - self.cfg.weight_decay * (lr / max(self.cfg.lr_max, 1e-8))
        b1, b2 = self.cfg.beta1, self.cfg.beta2

        for name, grad in ascent_grads.items():
            if name not in params:
                continue
            p = params[name]
            if name not in self._m:
                self._m[name] = np.zeros_like(p, dtype=np.float64)
                self._v[name] = np.zeros_like(p, dtype=np.float64)
            m = self._m[name]
            v = self._v[name]

            if self.cfg.use_adam:
                m[:] = b1 * m + (1.0 - b1) * grad
                v[:] = b2 * v + (1.0 - b2) * (grad ** 2)
                m_hat = m / (1.0 - b1 ** self.step_count)
                v_hat = v / (1.0 - b2 ** self.step_count)
                upd = lr * m_hat / (np.sqrt(v_hat) + self.cfg.eps)
            else:
                upd = lr * grad
            p[:] = np.clip(eff_decay * p + upd, -self.cfg.clip_bound, self.cfg.clip_bound)

        return lr

    def state_dict(self) -> dict:
        return {
            "m": {k: v.copy() for k, v in self._m.items()},
            "v": {k: v.copy() for k, v in self._v.items()},
            "step_count": self.step_count,
            "shots_used": self.shots_used,
        }

    def load_state_dict(self, state: dict):
        self._m = {k: v.copy() for k, v in state["m"].items()}
        self._v = {k: v.copy() for k, v in state["v"].items()}
        self.step_count = int(state["step_count"])
        self.shots_used = int(state["shots_used"])


@dataclass
class QERConfig:
    """Combined configuration for `QEROptimizer` (`Adam-QER` / `Adam-CFP`)."""

    lr_max: float = 0.12
    lr_min: float = 0.015
    beta: float = 0.45
    beta_min: float = 0.15
    centered: bool = True         # (+beta, -beta) cancels quadratic RZ bias
    beta1: float = 0.75
    beta2: float = 0.98
    eps: float = 1e-6
    weight_decay: float = 0.01    # LR-coupled decay strength
    clip_bound: float = 0.90      # Box bound avoiding phase wraparound
    auto_beta_guard: bool = True  # Shrink beta if wraparound sign flips occur
    use_adam: bool = True         # Disable for fixed-LR SGD ablation


class QEROptimizer:
    """
    Composite `Adam-QER` (`Adam-CFP`) wrapper pairing:
      1) `self.grad_estimator`: `QERGradientEstimator` (O(1) 2-run contrast gradients + `auto_beta_guard`)
      2) `self.stepper`:        `StabilizedAdam` (Adam + cosine LR + LR-coupled decay + box clip)

    Preserves the full `step_contrast` / `apply_gradients` API for backward compatibility.
    """

    def __init__(
        self,
        params: Optional[Dict[str, np.ndarray]] = None,
        config: Optional[QERConfig] = None,
    ):
        self.cfg = config or QERConfig()
        self.grad_estimator = QERGradientEstimator(
            QERGradConfig(
                beta=self.cfg.beta,
                beta_min=self.cfg.beta_min,
                centered=self.cfg.centered,
                auto_beta_guard=self.cfg.auto_beta_guard,
            )
        )
        self.stepper = StabilizedAdam(
            params=params,
            config=StabilizedAdamConfig(
                lr_max=self.cfg.lr_max,
                lr_min=self.cfg.lr_min,
                beta1=self.cfg.beta1,
                beta2=self.cfg.beta2,
                eps=self.cfg.eps,
                weight_decay=self.cfg.weight_decay,
                clip_bound=self.cfg.clip_bound,
                use_adam=self.cfg.use_adam,
            ),
        )

    @property
    def beta(self) -> float:
        return self.grad_estimator.beta

    @beta.setter
    def beta(self, val: float):
        self.grad_estimator.beta = float(val)

    @property
    def step_count(self) -> int:
        return self.stepper.step_count

    @step_count.setter
    def step_count(self, val: int):
        self.stepper.step_count = int(val)

    @property
    def shots_used(self) -> int:
        return self.stepper.shots_used

    @shots_used.setter
    def shots_used(self, val: int):
        self.stepper.shots_used = int(val)

    def current_lr(self, progress: float) -> float:
        return self.stepper.current_lr(progress)

    def apply_gradients(
        self,
        params: Dict[str, np.ndarray],
        ascent_grads: Dict[str, np.ndarray],
        progress: float = 0.0,
        nudge_align: Optional[float] = None,
        shots_in_batch: int = 0,
    ) -> Dict[str, float]:
        """Apply `StabilizedAdam` step in-place (plus optional `auto_beta_guard` fallback)."""
        lr = self.stepper.step(params, ascent_grads, progress=progress, shots_in_batch=shots_in_batch)
        if self.cfg.auto_beta_guard and nudge_align is not None:
            if nudge_align < 0.0:
                self.grad_estimator._neg_align_streak += 1
                if self.grad_estimator._neg_align_streak >= 2 and self.grad_estimator.beta > self.cfg.beta_min:
                    self.grad_estimator.beta = max(self.cfg.beta_min, round(0.75 * self.grad_estimator.beta, 4))
                    self.grad_estimator._neg_align_streak = 0
            else:
                self.grad_estimator._neg_align_streak = 0
        return {
            "lr": lr,
            "beta": self.grad_estimator.beta,
            "nudge_align": 0.0 if nudge_align is None else float(nudge_align),
        }

    def step_contrast(
        self,
        params: Dict[str, np.ndarray],
        obs_plus_list: Sequence[np.ndarray],
        obs_minus_list: Sequence[np.ndarray],
        z_plus_list: Optional[Sequence[np.ndarray]] = None,
        z_minus_list: Optional[Sequence[np.ndarray]] = None,
        conds: Optional[Sequence[Optional[np.ndarray]]] = None,
        nudge_dirs: Optional[Sequence[np.ndarray]] = None,
        progress: float = 0.0,
        shots_in_batch: int = 0,
    ) -> Dict[str, float]:
        """Estimate O(1) contrastive gradients via `QERGradientEstimator` and step via `StabilizedAdam`."""
        grads, mean_align = self.grad_estimator.estimate(
            params,
            obs_plus_list,
            obs_minus_list,
            z_plus_list=z_plus_list,
            z_minus_list=z_minus_list,
            conds=conds,
            nudge_dirs=nudge_dirs,
        )
        lr = self.stepper.step(params, grads, progress=progress, shots_in_batch=shots_in_batch)
        return {
            "lr": lr,
            "beta": self.grad_estimator.beta,
            "nudge_align": 0.0 if mean_align is None else float(mean_align),
        }

    def state_dict(self) -> dict:
        s = self.stepper.state_dict()
        s.update(self.grad_estimator.state_dict())
        return s

    def load_state_dict(self, state: dict):
        self.stepper.load_state_dict(state)
        self.grad_estimator.load_state_dict(state)


AdamQER = QEROptimizer
AdamCFP = QEROptimizer
