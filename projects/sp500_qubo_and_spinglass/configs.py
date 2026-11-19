"""Typed experiment and hardware configuration presets for Project 1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from typing import Dict, Tuple


PKG_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(PKG_DIR))


@dataclass(frozen=True)
class RunConfig:
    mode: str = "local_sim"               # local_sim | qiskit_sim | ibm_qpu
    device: str = "ibm_kingston"          # ibm_kingston | ibm_fez | ibm_marrakesh
    instance: str = ""
    api_key_json: str = ""
    execution_mode: str = "job"           # job (Open Plan compatible) | batch | session
    n_qubits: int = 12
    n_layers: int = 4
    diffusion_steps: int = 4
    batches: int = 25
    batch_size: int = 16
    shots_per_run: int = 48               # 25 * 2 * 16 * 48 = 38,400 training shots per problem
    eval_samples: int = 64
    eval_shots: int = 96
    cal_mode: str = "inverse"             # inverse MMSE avoids 0.35 floor shot-noise blowup
    nudge_mode: str = "residual"          # residual (3-ckt QER-Res MSE) | target (2-ckt QER-Grad)
    sample_temp: float = 1.15             # Gentle logit sharpening for stochastic MinVar sampling
    readout_mitigation: bool = True       # 1-job all-0/all-1 live marginal readout calibration
    run_live_spsa: bool = False           # Optional equal-shot live SPSA baseline on ibm_qpu
    run_live_nozz: bool = False           # Optional zero-ZZ ablation live on ibm_qpu
    skip_untrained: bool = False          # Skip QPU sampling at th=0 (use random baseline for gap_closed)
    skip_minvar: bool = False             # Run only Heavy-Hex Spin Glass (concentrates QPU quota)
    max_sim_qubits: int = 14
    skip_scaling: bool = False
    risk_aversion: float = 2.5
    card_penalty: float = 0.35
    qpu_shots_per_sec: float = 4000.0     # IBM Heron R2 empirical shot rate
    csv_path: str = os.path.join(REPO_ROOT, "data", "sp500_real", "daily_returns.csv")
    split_date: str = "2022-01-01"
    ckpt_dir: str = os.path.join(PKG_DIR, "checkpoints")
    artifact_dir: str = os.path.join(PKG_DIR, "artifacts")
    seed: int = 42

    @property
    def max_shots(self) -> int:
        return self.batches * 2 * self.shots_per_run * self.batch_size

    @property
    def scaling_qubits(self) -> Tuple[int, ...]:
        return tuple(range(8, min(18, self.max_sim_qubits) + 1, 2))

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS: Dict[str, Dict[str, object]] = {
    "quick": {
        "n_qubits": 8,
        "n_layers": 3,
        "diffusion_steps": 3,
        "batches": 20,
        "batch_size": 8,
        "shots_per_run": 16,
        "eval_samples": 32,
        "eval_shots": 64,
        "readout_mitigation": False,
        "max_sim_qubits": 12,
    },
    "standard": {
        "n_qubits": 12,
        "n_layers": 4,
        "diffusion_steps": 4,
        "batches": 25,
        "batch_size": 16,
        "shots_per_run": 48,              # 38,400 training shots for both problems
        "eval_samples": 64,
        "eval_shots": 96,
        "readout_mitigation": False,
        "max_sim_qubits": 14,
    },
    "t4_full": {
        "n_qubits": 16,
        "n_layers": 4,
        "diffusion_steps": 4,
        "batches": 40,
        "batch_size": 16,
        "shots_per_run": 48,
        "eval_samples": 64,
        "eval_shots": 96,
        "readout_mitigation": False,
        "max_sim_qubits": 16,
    },
    "heron_128q": {
        "mode": "ibm_qpu",
        "device": "ibm_kingston",
        "execution_mode": "job",
        "n_qubits": 128,                  # Clean noise-pruned Heron R2 sweet spot
        "n_layers": 4,
        "diffusion_steps": 4,
        "batches": 25,
        "batch_size": 16,
        "shots_per_run": 48,              # 25 * 2 * 16 * 48 = 38,400 training shots per problem
        "eval_samples": 32,
        "eval_shots": 96,
        "readout_mitigation": True,
        "run_live_spsa": False,
        "max_sim_qubits": 14,
        "skip_scaling": False,
    },
    "spinglass_only_128q": {
        "mode": "ibm_qpu",
        "device": "ibm_kingston",
        "execution_mode": "job",
        "n_qubits": 128,                  # Complete live 128q Spin Glass + SPSA + no-zz + untrained QPU
        "n_layers": 4,
        "diffusion_steps": 4,
        "batches": 17,
        "batch_size": 24,
        "shots_per_run": 48,              # 39,168 budget (11 QER-Res steps = 38,016 shots)
        "eval_samples": 64,
        "eval_shots": 96,
        "readout_mitigation": True,
        "run_live_spsa": True,
        "run_live_nozz": True,
        "skip_untrained": False,
        "skip_minvar": True,
        "max_sim_qubits": 14,
        "skip_scaling": True,
    },
    "heron_156q": {
        "mode": "ibm_qpu",
        "device": "ibm_kingston",
        "execution_mode": "job",
        "n_qubits": 156,
        "n_layers": 4,
        "diffusion_steps": 4,
        "batches": 25,
        "batch_size": 16,
        "shots_per_run": 48,              # 38,400 training shots per problem
        "eval_samples": 32,
        "eval_shots": 96,
        "readout_mitigation": True,
        "run_live_spsa": False,
        "max_sim_qubits": 14,
        "skip_scaling": False,
    },
}


def build_run_config(preset: str = "standard", **overrides) -> RunConfig:
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Choose from {list(PRESETS)}.")
    merged = dict(PRESETS[preset])
    for k, v in overrides.items():
        if v is not None:
            merged[k] = v
    return RunConfig(**merged)
