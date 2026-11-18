"""
Quantum Equilibrium-Response Diffusion (QERD) & Contrastive Field Probing (CFP) Stack
for Commuting-Diagonal Ising Circuits (NumPy / GPU-PyTorch / PennyLane).

Clean modular package layout:
  * `qerd.schedules`   : `make_alpha_bar`, `build_ising_edges`
  * `qerd.backends`    : `_NumPyTrajectoryBackend`, `_TorchTrajectoryBackend`, `_PennyLaneTrajectoryBackend`
  * `qerd.calibration` : `ReadoutGainCalibrator` (shot-free online OLS readout scaling)
  * `qerd.gradients`   : `QERGradConfig` (`CFPConfig`), `QERGradientEstimator` (`CFPGradientEstimator`)
  * `qerd.optimizers`  : `StabilizedAdamConfig`, `StabilizedAdam`, `QERConfig`, `QEROptimizer` (`AdamQER`)
  * `qerd.denoiser`    : `QuantumIsingDenoiser`
  * `qerd.trainer`     : `QERDTrainer`
  * `qerd.torch_layer` : `QERLayer` (`QERDDiffusionLayer`, when PyTorch is installed)
"""

from __future__ import annotations

from .backends import (
    _HAS_PENNYLANE,
    _HAS_QISKIT,
    _HAS_QISKIT_RUNTIME,
    _HAS_TORCH,
    QiskitRuntimeConfig,
    _BaseTrajectoryBackend,
    _NumPyTrajectoryBackend,
    _PennyLaneTrajectoryBackend,
    _QiskitRuntimeBackend,
    _TorchTrajectoryBackend,
)
from .calibration import ReadoutGainCalibrator
from .denoiser import QuantumIsingDenoiser
from .gradients import CFPConfig, CFPGradientEstimator, QERGradConfig, QERGradientEstimator
from .optimizers import (
    AdamCFP,
    AdamQER,
    QERConfig,
    QEROptimizer,
    StabilizedAdam,
    StabilizedAdamConfig,
)
from .schedules import build_ising_edges, color_ising_edges, make_alpha_bar, select_hardware_subgraph
from .trainer import QERDTrainer

__all__ = [
    "make_alpha_bar",
    "build_ising_edges",
    "select_hardware_subgraph",
    "color_ising_edges",
    "QiskitRuntimeConfig",
    "QERGradConfig",
    "CFPConfig",
    "QERGradientEstimator",
    "CFPGradientEstimator",
    "StabilizedAdamConfig",
    "StabilizedAdam",
    "QERConfig",
    "QEROptimizer",
    "AdamQER",
    "AdamCFP",
    "ReadoutGainCalibrator",
    "QuantumIsingDenoiser",
    "QERDTrainer",
]

if _HAS_TORCH:
    from .torch_layer import (
        QERDDiffusionLayer,
        QERLayer,
        _QERDFunction,
        _QERSurrogateFunction,
    )

    __all__.extend(["QERLayer", "QERDDiffusionLayer"])
