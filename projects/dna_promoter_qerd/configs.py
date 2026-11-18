"""Typed configurations and presets for Project 2 (`dna_promoter_qerd`)."""

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
    execution_mode: str = "job"           # job | batch | session
    n_qubits: int = 16                    # 16q (8bp core motif) local | 128q (64bp full promoter) on QPU
    n_layers: int = 3
    diffusion_steps: int = 3
    batches: int = 25
    batch_size: int = 16
    shots_per_run: int = 36
    eval_samples: int = 64
    eval_shots: int = 96
    cal_mode: str = "inverse"             # Wiener MMSE shrinkage avoids 1/0.35 floor noise blowup
    nudge_mode: str = "residual"          # residual (3-ckt QER-Res MSE) | target (2-ckt regularized QER-Grad)
    sample_temp: float = 1.15             # Gentle logit sharpening for stochastic binary DNA decoding
    stochastic_decode: bool = True        # Sample discrete spins stochastically instead of mean-sign thresholding
    readout_mitigation: bool = True       # 1-job all-0/all-1 hardware confusion correction
    run_live_spsa: bool = True            # Also run equal-shot SPSA live on QPU when mode=ibm_qpu
    oracle_epochs: int = 4
    max_train_targets: int = 600          # High-confidence biophysically valid promoter targets
    max_sim_qubits: int = 14
    skip_scaling: bool = False
    qpu_shots_per_sec: float = 4000.0
    data_dir: str = os.path.join(REPO_ROOT, "data", "dna_real")
    ckpt_dir: str = os.path.join(PKG_DIR, "checkpoints")
    artifact_dir: str = os.path.join(PKG_DIR, "artifacts")
    seed: int = 42

    @property
    def max_shots(self) -> int:
        return self.batches * 2 * self.shots_per_run * self.batch_size

    @property
    def seq_len(self) -> int:
        return self.n_qubits // 2

    @property
    def scaling_qubits(self) -> Tuple[int, ...]:
        return tuple(range(8, min(16, self.max_sim_qubits) + 1, 2))

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS: Dict[str, Dict[str, object]] = {
    "quick": {
        "n_qubits": 16,
        "n_layers": 3,
        "diffusion_steps": 3,
        "batches": 20,
        "batch_size": 12,
        "shots_per_run": 24,
        "eval_samples": 32,
        "eval_shots": 64,
        "oracle_epochs": 2,
        "max_sim_qubits": 12,
    },
    "local_cpu": {
        "n_qubits": 16,
        "n_layers": 3,
        "diffusion_steps": 3,
        "batches": 25,
        "batch_size": 16,
        "shots_per_run": 36,
        "eval_samples": 64,
        "eval_shots": 96,
        "oracle_epochs": 4,
        "max_sim_qubits": 14,
    },
    "heron_128q": {
        "mode": "ibm_qpu",
        "device": "ibm_kingston",
        "execution_mode": "job",
        "n_qubits": 128,
        "n_layers": 2,
        "diffusion_steps": 3,
        "batches": 25,
        "batch_size": 16,
        "shots_per_run": 48,      # Heavily packed: 25 * 2 * 16 * 48 = 38,400 training shots
        "eval_samples": 32,
        "eval_shots": 96,
        "oracle_epochs": 4,
        "readout_mitigation": True,
        "run_live_spsa": True,
        "max_sim_qubits": 14,
        "skip_scaling": False,
    },
    "heron_156q": {
        "mode": "ibm_qpu",
        "device": "ibm_kingston",
        "execution_mode": "job",
        "n_qubits": 156,
        "n_layers": 2,
        "diffusion_steps": 3,
        "batches": 25,
        "batch_size": 16,
        "shots_per_run": 48,
        "eval_samples": 32,
        "eval_shots": 96,
        "oracle_epochs": 4,
        "readout_mitigation": True,
        "run_live_spsa": False,
        "max_sim_qubits": 14,
        "skip_scaling": False,
    },
}


def build_run_config(preset: str = "local_cpu", **overrides) -> RunConfig:
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Choose from {list(PRESETS)}.")
    merged = dict(PRESETS[preset])
    for k, v in overrides.items():
        if v is not None:
            merged[k] = v
    return RunConfig(**merged)
