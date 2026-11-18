"""
Stateful non-adiabatic Quantum Ising diffusion model & reverse-diffusion sampler.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .backends import (
    _HAS_PENNYLANE,
    _HAS_QISKIT,
    _HAS_QISKIT_RUNTIME,
    _HAS_TORCH,
    _NumPyTrajectoryBackend,
    _PennyLaneTrajectoryBackend,
    _QiskitRuntimeBackend,
    _TorchTrajectoryBackend,
    torch,
)
from .calibration import ReadoutGainCalibrator
from .schedules import build_ising_edges, make_alpha_bar


class QuantumIsingDenoiser:
    """
    Stateful non-adiabatic QPU Ising denoiser & diffusion sampler for arbitrary shapes.

    Parameter partition is explicit:
      * `theta`     : Pure QPU weights multiplying commuting diagonal generators (Z_i, Z_u Z_v).
      * `cond_proj` : Optional classical linear conditioning matrix W_cond (when `cond_dim > 0`;
                      set `cond_dim=0` for strictly Pure-QPU operation with zero classical params).
      * `calibrator`: Optional shot-free OLS readout scaler (`init_gain=1.0, ema=0.0` disables it).
    """

    def __init__(
        self,
        data_shape: Union[int, Tuple[int, ...]],
        cond_dim: int = 0,
        n_hidden: int = 0,
        topology: str = "grid8",
        edges: Optional[Sequence[Tuple[int, int]]] = None,
        diffusion_steps: int = 4,
        noise_schedule: str = "cosine",
        p_depol: float = 0.035,
        p_readout: float = 0.04,
        p_depol_2q: float = 0.0,
        shots_per_traj: Optional[int] = None,
        n_layers: int = 2,
        dt: float = 0.42,
        temp_beta: float = 1.25,
        shake_scale: float = 2.2,
        shake_final_frac: float = 0.70,
        in_pinning_base: float = 0.28,
        in_pinning_slope: float = 0.52,
        init_gain: float = 0.75,
        gain_ema: float = 0.10,
        gain_bounds: Tuple[float, float] = (0.20, 1.00),
        cal_mode: str = "forward",
        backend: str = "auto",
        torch_device: str = "auto",
        device_name: str = "lightning.qubit",
        device_kwargs: Optional[dict] = None,
        real_qpu: bool = False,
        seed: int = 42,
    ):
        self.data_shape = (
            (int(data_shape),) if isinstance(data_shape, (int, np.integer)) else tuple(int(d) for d in data_shape)
        )
        self.n_visible = int(np.prod(self.data_shape))
        self.cond_dim = int(cond_dim)
        self.n_hidden = int(n_hidden)
        self.n_qubits = self.n_visible + self.n_hidden

        self.diffusion_steps = int(diffusion_steps)
        self.alpha_bar = make_alpha_bar(self.diffusion_steps, schedule=noise_schedule)
        self.n_layers = int(n_layers)
        self.dt = float(dt)
        self.temp_beta = float(temp_beta)
        self.shake_scale = float(shake_scale)
        self.shake_final_frac = float(shake_final_frac)
        self.in_pinning_base = float(in_pinning_base)
        self.in_pinning_slope = float(in_pinning_slope)
        self.rng = np.random.default_rng(seed)

        if edges is not None:
            self.edges = sorted({(min(int(u), int(v)), max(int(u), int(v))) for u, v in edges if u != v})
        else:
            self.edges = build_ising_edges(self.data_shape, topology=topology, n_hidden=self.n_hidden)

        self.n_edges = len(self.edges)
        self.n_biases = self.n_qubits
        self.n_qpu_params = self.n_edges + self.n_biases
        self._edge_u = np.array([u for u, _ in self.edges], dtype=np.int64)
        self._edge_v = np.array([v for _, v in self.edges], dtype=np.int64)

        # Stateful model weights & readout calibration
        self.theta = np.zeros(self.n_qpu_params, dtype=np.float64)
        if self.n_hidden > 0:
            # Break mirror symmetry gently on hidden qubits so hidden-visible ZZ responses are active from step 1
            for e_idx, (u, v) in enumerate(self.edges):
                if u >= self.n_visible or v >= self.n_visible:
                    self.theta[e_idx] = self.rng.normal(0.0, 0.08)
            self.theta[self.n_edges + self.n_visible :] = self.rng.normal(0.0, 0.04, size=self.n_hidden)
        self.cond_proj = (
            np.zeros((self.n_visible, self.cond_dim), dtype=np.float64) if self.cond_dim > 0 else None
        )
        self.calibrator = ReadoutGainCalibrator(
            self.diffusion_steps, n_visible=self.n_visible, init_gain=init_gain,
            ema=gain_ema, bounds=gain_bounds, mode=cal_mode,
        )

        if real_qpu:
            if backend != "auto":
                chosen = backend
            elif (_HAS_QISKIT or _HAS_QISKIT_RUNTIME) and (
                str(device_name).startswith("ibm_")
                or any(k in (device_kwargs or {}) for k in ("backend", "qiskit_backend", "sampler", "service", "execution_mode"))
            ):
                chosen = "qiskit"
            else:
                chosen = "pennylane"
            p_depol = 0.0
            p_readout = 0.0
            p_depol_2q = 0.0
            eff_shots_per_traj = 44 if shots_per_traj is None else int(shots_per_traj)
        else:
            if backend == "auto":
                if _HAS_TORCH and (torch.cuda.is_available() or self.n_qubits >= 14):
                    chosen = "torch"
                elif _HAS_PENNYLANE and self.n_qubits > 14:
                    chosen = "pennylane"
                else:
                    chosen = "numpy"
            else:
                chosen = backend
            eff_shots_per_traj = 4 if shots_per_traj is None else int(shots_per_traj)

        self.p_depol = float(p_depol)
        self.p_readout = float(p_readout)
        self.p_depol_2q = float(p_depol_2q)
        self.freeze_theta = False
        self.backend_name = chosen
        if chosen == "numpy":
            self.backend = _NumPyTrajectoryBackend(
                self.n_qubits, self.edges, p_depol=self.p_depol, p_readout=self.p_readout,
                p_depol_2q=self.p_depol_2q, shots_per_traj=eff_shots_per_traj, seed=seed,
            )
        elif chosen == "torch":
            self.backend = _TorchTrajectoryBackend(
                self.n_qubits, self.edges, p_depol=self.p_depol, p_readout=self.p_readout,
                p_depol_2q=self.p_depol_2q, shots_per_traj=eff_shots_per_traj, torch_device=torch_device, seed=seed,
            )
        elif chosen == "pennylane":
            self.backend = _PennyLaneTrajectoryBackend(
                self.n_qubits, self.edges, p_depol=self.p_depol, p_readout=self.p_readout,
                p_depol_2q=self.p_depol_2q, shots_per_traj=eff_shots_per_traj,
                device_name=device_name, device_kwargs=device_kwargs, seed=seed,
            )
        elif chosen == "qiskit":
            self.backend = _QiskitRuntimeBackend(
                self.n_qubits, self.edges, p_depol=self.p_depol, p_readout=self.p_readout,
                p_depol_2q=self.p_depol_2q, shots_per_traj=eff_shots_per_traj,
                device_name=device_name, device_kwargs=device_kwargs, seed=seed,
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def close(self) -> None:
        """Release any open hardware session/batch resources held by the backend."""
        if hasattr(self.backend, "close"):
            self.backend.close()

    def __enter__(self) -> "QuantumIsingDenoiser":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def parameters_dict(self) -> Dict[str, np.ndarray]:
        """Return mutable references to trainable model parameters for QEROptimizer."""
        d: Dict[str, np.ndarray] = {}
        if not self.freeze_theta:
            d["theta"] = self.theta
        if self.cond_proj is not None:
            d["cond_proj"] = self.cond_proj
        return d

    def state_dict(self) -> dict:
        return {
            "theta": self.theta.copy(),
            "cond_proj": None if self.cond_proj is None else self.cond_proj.copy(),
            "calibrator": self.calibrator.state_dict(),
        }

    def load_state_dict(self, state: dict):
        self.theta[:] = state["theta"]
        if self.cond_proj is not None and state["cond_proj"] is not None:
            self.cond_proj[:] = state["cond_proj"]
        self.calibrator.load_state_dict(state["calibrator"])

    def _build_fields(
        self,
        x_t: np.ndarray,
        t_norm: float,
        cond: Optional[np.ndarray],
        nudge_target: Optional[np.ndarray],
        nudge_beta: float,
        theta: Optional[np.ndarray] = None,
        cond_proj: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        th = self.theta if theta is None else theta
        cp = self.cond_proj if cond_proj is None else cond_proj
        j_couplings, biases = th[: self.n_edges], th[self.n_edges :]

        h_field = biases.copy()
        in_pinning = self.in_pinning_base + self.in_pinning_slope * (1.0 - t_norm)
        h_field[: self.n_visible] += in_pinning * x_t.reshape(-1)
        if cond is not None and cp is not None:
            h_field[: self.n_visible] += cp @ cond.reshape(-1)
        if nudge_target is not None and nudge_beta != 0.0:
            h_field[: self.n_visible] += nudge_beta * nudge_target.reshape(-1)
        return h_field, j_couplings

    def measure_observables(
        self,
        x_t: np.ndarray,
        t: int,
        shots: int,
        cond: Optional[np.ndarray] = None,
        nudge_target: Optional[np.ndarray] = None,
        nudge_beta: float = 0.0,
        theta: Optional[np.ndarray] = None,
        cond_proj: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run QPU circuit and return (all_param_obs, visible_z_exp)."""
        t_norm = float(t) / self.diffusion_steps
        h_field, j_couplings = self._build_fields(
            x_t, t_norm, cond, nudge_target, nudge_beta, theta=theta, cond_proj=cond_proj
        )
        raw_bits = self.backend.sample(
            h_field, j_couplings, t_norm, total_shots=shots,
            n_layers=self.n_layers, dt=self.dt, temp_beta=self.temp_beta,
            shake_scale=self.shake_scale, shake_final_frac=self.shake_final_frac,
        )
        spins = 1.0 - 2.0 * raw_bits.astype(np.float64)
        z_all = np.mean(spins, axis=0)
        if self.n_edges > 0:
            zz = np.mean(spins[:, self._edge_u] * spins[:, self._edge_v], axis=0)
            obs = np.concatenate([zz, z_all])
        else:
            obs = z_all
        return obs, z_all[: self.n_visible]

    def measure_observables_batch(
        self,
        x_t_batch: np.ndarray,
        t_batch: np.ndarray,
        shots: int,
        cond_batch: Optional[np.ndarray] = None,
        nudge_target_batch: Optional[np.ndarray] = None,
        nudge_beta_batch: Optional[np.ndarray] = None,
        theta_batch: Optional[np.ndarray] = None,
        cond_proj_batch: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Batched QPU observable evaluation returning (obs_batch, visible_z_batch)."""
        m = x_t_batch.shape[0]
        t_norms = np.asarray(t_batch, dtype=np.float64) / self.diffusion_steps
        x_flat = np.asarray(x_t_batch, dtype=np.float64).reshape(m, self.n_visible)

        if theta_batch is None and cond_proj_batch is None:
            j_couplings = np.broadcast_to(self.theta[: self.n_edges], (m, self.n_edges)).copy()
            h_fields = np.broadcast_to(self.theta[self.n_edges :], (m, self.n_qubits)).copy()
            in_pin = (self.in_pinning_base + self.in_pinning_slope * (1.0 - t_norms))[:, None]
            h_fields[:, : self.n_visible] += in_pin * x_flat
            if cond_batch is not None and self.cond_proj is not None:
                h_fields[:, : self.n_visible] += cond_batch.reshape(m, self.cond_dim) @ self.cond_proj.T
            if nudge_target_batch is not None and nudge_beta_batch is not None:
                h_fields[:, : self.n_visible] += (
                    np.asarray(nudge_beta_batch, dtype=np.float64)[:, None]
                    * np.asarray(nudge_target_batch, dtype=np.float64).reshape(m, self.n_visible)
                )
        else:
            h_fields = np.empty((m, self.n_qubits), dtype=np.float64)
            j_couplings = np.empty((m, self.n_edges), dtype=np.float64)
            for i in range(m):
                th_i = self.theta if theta_batch is None else theta_batch[i]
                cp_i = self.cond_proj if cond_proj_batch is None else cond_proj_batch[i]
                c_i = None if cond_batch is None else cond_batch[i]
                nt_i = None if nudge_target_batch is None else nudge_target_batch[i]
                nb_i = 0.0 if nudge_beta_batch is None else float(nudge_beta_batch[i])
                h_fields[i], j_couplings[i] = self._build_fields(
                    x_flat[i], float(t_norms[i]), c_i, nt_i, nb_i, theta=th_i, cond_proj=cp_i
                )

        return self.backend.sample_obs_batch(
            h_fields, j_couplings, t_norms, total_shots=shots, n_visible=self.n_visible,
            n_layers=self.n_layers, dt=self.dt, temp_beta=self.temp_beta,
            shake_scale=self.shake_scale, shake_final_frac=self.shake_final_frac,
        )

    def sample_batch(
        self,
        conds: Optional[np.ndarray] = None,
        n_samples: Optional[int] = None,
        shots_per_step: int = 160,
        step_noise_scale: float = 0.35,
        init_std: float = 0.90,
    ) -> np.ndarray:
        """Generate a batch of samples in parallel via reverse diffusion (t=T -> 0)."""
        n = int(conds.shape[0]) if conds is not None else int(n_samples or 1)
        x_t = np.clip(self.rng.normal(0.0, init_std, size=(n, self.n_visible)), -1.0, 1.0)
        c_arr = None if conds is None else np.asarray(conds, dtype=np.float64).reshape(n, self.cond_dim)

        for t in range(self.diffusion_steps, 0, -1):
            t_arr = np.full(n, t, dtype=np.int64)
            _, z_raw = self.measure_observables_batch(x_t, t_arr, shots=shots_per_step, cond_batch=c_arr)
            x0_hat = self.calibrator.calibrate(t, z_raw)
            if t > 1:
                a_next = self.alpha_bar[t - 1]
                eps = self.rng.normal(0.0, step_noise_scale, size=(n, self.n_visible))
                x_t = np.clip(math.sqrt(a_next) * x0_hat + math.sqrt(1.0 - a_next) * eps, -1.0, 1.0)
            else:
                x_t = x0_hat

        return x_t.reshape((n,) + self.data_shape)

    def sample(
        self,
        cond: Optional[np.ndarray] = None,
        shots_per_step: int = 160,
        step_noise_scale: float = 0.35,
        init_std: float = 0.90,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Generate a sample from pure Gaussian static via reverse diffusion (t=T -> 0)."""
        x_t = np.clip(self.rng.normal(0.0, init_std, size=self.n_visible), -1.0, 1.0)
        traj = [x_t.reshape(self.data_shape).copy()]

        for t in range(self.diffusion_steps, 0, -1):
            _, z_raw = self.measure_observables(x_t, t, shots=shots_per_step, cond=cond)
            x0_hat = self.calibrator.calibrate(t, z_raw)
            if t > 1:
                a_next = self.alpha_bar[t - 1]
                eps = self.rng.normal(0.0, step_noise_scale, size=self.n_visible)
                x_t = np.clip(math.sqrt(a_next) * x0_hat + math.sqrt(1.0 - a_next) * eps, -1.0, 1.0)
            else:
                x_t = x0_hat
            traj.append(x_t.reshape(self.data_shape).copy())

        return x_t.reshape(self.data_shape), traj

    def evaluate(
        self,
        data: np.ndarray,
        conds: Optional[np.ndarray] = None,
        shots: int = 160,
        n_trials: int = 3,
    ) -> Dict[str, float]:
        targets = np.asarray(data, dtype=np.float64).reshape(-1, self.n_visible)
        n_items = targets.shape[0]
        rep_targets = np.tile(targets, (n_trials, 1))
        if conds is not None:
            c_flat = np.asarray(conds, dtype=np.float64).reshape(n_items, self.cond_dim)
            rep_conds = np.tile(c_flat, (n_trials, 1))
        else:
            rep_conds = None
        gens = self.sample_batch(conds=rep_conds, n_samples=n_items * n_trials, shots_per_step=shots)
        gens_flat = gens.reshape(-1, self.n_visible)
        mse = float(np.mean((gens_flat - rep_targets) ** 2))
        s_gen = np.where(gens_flat >= 0, 1.0, -1.0)
        s_tgt = np.where(rep_targets >= 0, 1.0, -1.0)
        sign_acc = float(np.mean(s_gen == s_tgt))
        return {"mse": mse, "sign_acc": sign_acc}
