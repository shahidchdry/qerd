"""
Shot-free online NISQ readout gain mitigation via forward or inverse OLS EMA tracking.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class ReadoutGainCalibrator:
    """Online least-squares EMA calibrator compensating for per-qubit NISQ spin contraction per timestep."""

    def __init__(
        self,
        n_steps: int,
        n_visible: Optional[int] = None,
        init_gain: float = 0.75,
        ema: float = 0.10,
        bounds: Tuple[float, float] = (0.20, 1.00),
        prior_weight: float = 0.50,
        mode: str = "forward",
    ):
        if mode not in ("forward", "inverse"):
            raise ValueError(f"Unknown ReadoutGainCalibrator mode: {mode!r}")
        self.ema = float(ema)
        self.bounds = (float(bounds[0]), float(bounds[1]))
        self.mode = str(mode)
        shape = (int(n_steps) + 1,) if n_visible is None else (int(n_steps) + 1, int(n_visible))
        self.gain = np.full(shape, float(init_gain), dtype=np.float64)
        self._clip_hits = 0
        self._clip_total = 0
        self._upper_clip_hits = 0
        if self.mode == "forward":
            self._den = np.full(shape, float(prior_weight), dtype=np.float64)
            self._num = self.gain * self._den
        else:
            # Predictive inverse regression x_0 ~ a * z_free => a* = E[x_0 z_free] / E[z_free^2], gain = 1/a*
            self._den = np.full(shape, float(prior_weight) * (float(init_gain) ** 2), dtype=np.float64)
            self._num = np.full(shape, float(prior_weight) * float(init_gain), dtype=np.float64)

    @property
    def mean_gain(self) -> float:
        return float(np.mean(self.gain[1:]))

    @property
    def clip_fraction(self) -> float:
        return float(self._clip_hits / max(1, self._clip_total))

    @property
    def upper_clip_fraction(self) -> float:
        """Fraction of updates hitting the upper gain bound (inv_lo in inverse mode)."""
        return float(self._upper_clip_hits / max(1, self._clip_total))

    def update(self, t: int, z_free: np.ndarray, target: np.ndarray):
        """Fit optimal least-squares gain (forward: z~g*x0; inverse: x0~z/g) at zero extra QPU shots."""
        if self.ema <= 0.0:
            return
        zf = np.asarray(z_free, dtype=np.float64).reshape(-1)
        tg = np.asarray(target, dtype=np.float64).reshape(-1)
        if self.mode == "inverse":
            inv_lo = 1.0 / max(self.bounds[1], 1e-6)
            inv_hi = 1.0 / max(self.bounds[0], 1e-6)
            if self.gain.ndim == 1:
                num_val = float(np.dot(zf, tg))
                den_val = float(np.dot(zf, zf))
                self._num[t] = (1.0 - self.ema) * self._num[t] + self.ema * num_val
                self._den[t] = (1.0 - self.ema) * self._den[t] + self.ema * den_val
                raw_s = self._num[t] / max(self._den[t], 1e-6)
                scale = float(np.clip(raw_s, inv_lo, inv_hi))
                self._total_inc(raw_s <= inv_lo, raw_s < inv_lo or raw_s > inv_hi)
                self.gain[t] = 1.0 / scale
            else:
                self._num[t] = (1.0 - self.ema) * self._num[t] + self.ema * (zf * tg)
                self._den[t] = (1.0 - self.ema) * self._den[t] + self.ema * (zf * zf)
                raw_s = self._num[t] / np.maximum(self._den[t], 1e-6)
                scale = np.clip(raw_s, inv_lo, inv_hi)
                self._clip_hits += int(np.sum((raw_s <= inv_lo) | (raw_s >= inv_hi)))
                self._upper_clip_hits += int(np.sum(raw_s <= inv_lo))
                self._clip_total += int(raw_s.size)
                self.gain[t] = 1.0 / scale
            return

        if self.gain.ndim == 1:
            denom = float(np.dot(tg, tg))
            if denom < 1e-6:
                return
            raw_g = np.dot(zf, tg) / denom
            opt_g = float(np.clip(raw_g, self.bounds[0], self.bounds[1]))
            self._total_inc(raw_g >= self.bounds[1], raw_g < self.bounds[0] or raw_g > self.bounds[1])
            self.gain[t] = (1.0 - self.ema) * self.gain[t] + self.ema * opt_g
        else:
            self._num[t] = (1.0 - self.ema) * self._num[t] + self.ema * (zf * tg)
            self._den[t] = (1.0 - self.ema) * self._den[t] + self.ema * (tg * tg)
            ratio = self._num[t] / np.maximum(self._den[t], 1e-6)
            self._clip_hits += int(np.sum((ratio <= self.bounds[0]) | (ratio >= self.bounds[1])))
            self._upper_clip_hits += int(np.sum(ratio >= self.bounds[1]))
            self._clip_total += int(ratio.size)
            self.gain[t] = np.clip(ratio, self.bounds[0], self.bounds[1])

    def _total_inc(self, is_upper: bool, is_any: bool):
        self._clip_total += 1
        if is_any:
            self._clip_hits += 1
        if is_upper:
            self._upper_clip_hits += 1

    def calibrate(self, t: int, z_raw: np.ndarray) -> np.ndarray:
        return np.clip(z_raw / self.gain[t], -1.0, 1.0)

    def state_dict(self) -> dict:
        return {
            "gain": self.gain.copy(),
            "num": self._num.copy(),
            "den": self._den.copy(),
            "mode": self.mode,
            "clip_hits": self._clip_hits,
            "upper_clip_hits": self._upper_clip_hits,
            "clip_total": self._clip_total,
        }

    def load_state_dict(self, state: dict):
        self.gain[:] = state["gain"]
        if "mode" in state:
            self.mode = str(state["mode"])
        if "num" in state and "den" in state:
            self._num[:] = state["num"]
            self._den[:] = state["den"]
        self._clip_hits = int(state.get("clip_hits", 0))
        self._upper_clip_hits = int(state.get("upper_clip_hits", 0))
        self._clip_total = int(state.get("clip_total", 0))
