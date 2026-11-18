"""
Project 1: QPU Hardware Advantage & 156-Qubit Heron R2 Scaling Suite.

Compares QERD (`QER-Grad` + `QuantumIsingDenoiser`) against exact 2^N GPU statevector
limits and valid equal-shot optimization baselines (`SA`, `Untrained`, `PSR`, `SPSA`).
"""

from .configs import PRESETS, RunConfig, build_run_config

__all__ = ["PRESETS", "RunConfig", "build_run_config"]
