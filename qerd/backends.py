"""
Vectorized quantum circuit trajectory backends (NumPy, PyTorch CPU/CUDA, PennyLane, and IBM Qiskit Runtime).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .schedules import color_ising_edges

try:
    import pennylane as qml  # type: ignore

    _HAS_PENNYLANE = True
except ImportError:
    qml = None
    _HAS_PENNYLANE = False

try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore

    _HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    _HAS_TORCH = False

try:
    from qiskit import QuantumCircuit  # type: ignore
    from qiskit.circuit import ParameterVector  # type: ignore
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager  # type: ignore

    _HAS_QISKIT = True
except ImportError:
    QuantumCircuit = None
    ParameterVector = None
    generate_preset_pass_manager = None
    _HAS_QISKIT = False

try:
    from qiskit_ibm_runtime import (  # type: ignore
        Batch as _IBMBatch,
        QiskitRuntimeService as _IBMService,
        SamplerV2 as _IBMSamplerV2,
        Session as _IBMSession,
    )

    _HAS_QISKIT_RUNTIME = True
except ImportError:
    _IBMBatch = None
    _IBMService = None
    _IBMSamplerV2 = None
    _IBMSession = None
    _HAS_QISKIT_RUNTIME = False


class _BaseTrajectoryBackend:
    """Shared NISQ 1Q/2Q depolarizing + readout noise helpers for trajectory backends."""

    _PAULI_TO_XZ = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.int64)
    _XZ_TO_PAULI = np.array([[0, 3], [1, 2]], dtype=np.int64)

    def __init__(
        self,
        n_qubits: int,
        edges: List[Tuple[int, int]],
        p_depol: float = 0.035,
        p_readout: float = 0.04,
        p_depol_2q: float = 0.0,
        shots_per_traj: int = 4,
        seed: int = 42,
    ):
        self.n_qubits = int(n_qubits)
        self.edges = edges
        self.edge_u = np.array([u for u, _ in edges], dtype=np.int64)
        self.edge_v = np.array([v for _, v in edges], dtype=np.int64)
        self.p_depol = float(p_depol)
        self.p_readout = float(p_readout)
        self.p_depol_2q = float(p_depol_2q)
        self.shots_per_traj = int(shots_per_traj)
        self.rng = np.random.default_rng(seed)

    def _depol_probs(self) -> List[float]:
        return [
            1.0 - 0.75 * self.p_depol,
            0.25 * self.p_depol,
            0.25 * self.p_depol,
            0.25 * self.p_depol,
        ]

    def _compose_2q_faults_np(self, base_faults: Optional[np.ndarray], n_traj: int) -> np.ndarray:
        """Merge optional 1Q layer faults with correlated 2Q Pauli faults along active ZZ edges."""
        if base_faults is None:
            x_acc = np.zeros((n_traj, self.n_qubits), dtype=np.int64)
            z_acc = np.zeros((n_traj, self.n_qubits), dtype=np.int64)
        else:
            xz = self._PAULI_TO_XZ[base_faults]
            x_acc = xz[..., 0].copy()
            z_acc = xz[..., 1].copy()

        if self.p_depol_2q > 0.0 and len(self.edges) > 0:
            trigger = self.rng.random((n_traj, len(self.edges))) < self.p_depol_2q
            if np.any(trigger):
                codes = self.rng.integers(1, 16, size=(n_traj, len(self.edges))) * trigger
                pu = self._PAULI_TO_XZ[codes >> 2]
                pv = self._PAULI_TO_XZ[codes & 3]
                for e_idx, (u, v) in enumerate(self.edges):
                    mask = trigger[:, e_idx]
                    if np.any(mask):
                        x_acc[mask, u] ^= pu[mask, e_idx, 0]
                        z_acc[mask, u] ^= pu[mask, e_idx, 1]
                        x_acc[mask, v] ^= pv[mask, e_idx, 0]
                        z_acc[mask, v] ^= pv[mask, e_idx, 1]

        return self._XZ_TO_PAULI[x_acc, z_acc]

    def _inject_readout_noise(self, raw_bits: np.ndarray) -> np.ndarray:
        if self.p_readout > 0.0:
            flips = (self.rng.random(raw_bits.shape) < self.p_readout).astype(np.int64)
            raw_bits = raw_bits ^ flips
        return raw_bits

    def _bits_to_obs_batch(
        self,
        bits_3d: np.ndarray,
        n_visible: int,
        max_decode_elements: int = 16_000_000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert packed/unpacked bit array `(m, S, >=n_qubits)` to `(obs_batch, z_visible_batch)` without loops over shots."""
        m, s_cnt, _ = bits_3d.shape
        n_edges = len(self.edges)
        obs_out = np.empty((m, n_edges + self.n_qubits), dtype=np.float64)
        row_step = max(1, max_decode_elements // max(1, s_cnt * self.n_qubits))
        for st in range(0, m, row_step):
            en = min(m, st + row_step)
            sub_bits = bits_3d[st:en, :, : self.n_qubits]
            spins = 1.0 - 2.0 * sub_bits.astype(np.float32)
            z_all = spins.mean(axis=1, dtype=np.float64)
            if n_edges > 0:
                zz = (spins[:, :, self.edge_u] * spins[:, :, self.edge_v]).mean(axis=1, dtype=np.float64)
                obs_out[st:en, :n_edges] = zz
                obs_out[st:en, n_edges:] = z_all
            else:
                obs_out[st:en] = z_all
        return obs_out, obs_out[:, n_edges : n_edges + n_visible].copy()

    def sample_obs_batch(
        self,
        h_fields: np.ndarray,
        j_couplings: np.ndarray,
        t_norms: np.ndarray,
        total_shots: int,
        n_visible: int,
        n_layers: int = 2,
        dt: float = 0.42,
        temp_beta: float = 1.25,
        shake_scale: float = 2.2,
        shake_final_frac: float = 0.70,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run a batch of circuits and return (obs_batch, z_visible_batch)."""
        m = h_fields.shape[0]
        obs_out = np.empty((m, len(self.edges) + self.n_qubits), dtype=np.float64)
        for i in range(m):
            raw_bits = self.sample(
                h_fields[i], j_couplings[i], float(t_norms[i]), total_shots=total_shots,
                n_layers=n_layers, dt=dt, temp_beta=temp_beta,
                shake_scale=shake_scale, shake_final_frac=shake_final_frac,
            )
            spins = 1.0 - 2.0 * raw_bits.astype(np.float64)
            z_all = np.mean(spins, axis=0)
            if len(self.edges) > 0:
                zz = np.mean(spins[:, self.edge_u] * spins[:, self.edge_v], axis=0)
                obs_out[i] = np.concatenate([zz, z_all])
            else:
                obs_out[i] = z_all
        return obs_out, obs_out[:, len(self.edges) : len(self.edges) + n_visible].copy()


class _NumPyTrajectoryBackend(_BaseTrajectoryBackend):
    """Fast vectorized statevector simulator with NISQ depolarizing + readout noise."""

    def __init__(
        self,
        n_qubits: int,
        edges: List[Tuple[int, int]],
        p_depol: float = 0.035,
        p_readout: float = 0.04,
        p_depol_2q: float = 0.0,
        shots_per_traj: int = 4,
        seed: int = 42,
    ):
        if n_qubits > 14:
            raise ValueError(f"NumPy statevector backend supports <=14 qubits (got {n_qubits}); use backend='torch' or 'pennylane'.")
        super().__init__(
            n_qubits, edges, p_depol=p_depol, p_readout=p_readout,
            p_depol_2q=p_depol_2q, shots_per_traj=shots_per_traj, seed=seed,
        )

        n_states = 1 << n_qubits
        basis = np.arange(n_states, dtype=np.int64)
        self.z_basis = np.array([1.0 - 2.0 * ((basis >> (n_qubits - 1 - q)) & 1) for q in range(n_qubits)], dtype=np.float32)
        if edges:
            self.zz_basis = self.z_basis[self.edge_u] * self.z_basis[self.edge_v]
        else:
            self.zz_basis = np.zeros((0, n_states), dtype=np.float32)
        self.bit_table = ((basis[:, None] >> np.arange(n_qubits - 1, -1, -1)) & 1).astype(np.int64)
        self.idx0 = [np.flatnonzero((basis & (1 << (n_qubits - 1 - q))) == 0) for q in range(n_qubits)]
        self.idx1 = [self.idx0[q] | (1 << (n_qubits - 1 - q)) for q in range(n_qubits)]

    def _apply_rx(self, psi: np.ndarray, angle: float):
        ca = np.complex64(math.cos(0.5 * angle))
        sa = np.complex64(-1j * math.sin(0.5 * angle))
        b_size = psi.shape[0]
        for q in range(self.n_qubits):
            pre = 1 << q
            post = 1 << (self.n_qubits - 1 - q)
            v = psi.reshape(b_size * pre, 2, post)
            a0 = v[:, 0, :]
            a1 = v[:, 1, :]
            tmp = ca * a0 + sa * a1
            v[:, 1, :] = sa * a0 + ca * a1
            v[:, 0, :] = tmp

    def _apply_pauli(self, psi: np.ndarray, faults: np.ndarray):
        for q in range(self.n_qubits):
            f = faults[:, q]
            if not np.any(f):
                continue
            i0, i1 = self.idx0[q], self.idx1[q]
            r1 = np.flatnonzero(f == 1)
            if r1.size:
                tmp = psi[r1][:, i0]
                psi[r1[:, None], i0] = psi[r1[:, None], i1]
                psi[r1[:, None], i1] = tmp
            r2 = np.flatnonzero(f == 2)
            if r2.size:
                tmp = psi[r2][:, i0]
                psi[r2[:, None], i0] = np.complex64(-1j) * psi[r2[:, None], i1]
                psi[r2[:, None], i1] = np.complex64(1j) * tmp
            r3 = np.flatnonzero(f == 3)
            if r3.size:
                psi[r3[:, None], i1] *= -1.0

    def sample(
        self,
        h_field: np.ndarray,
        j_couplings: np.ndarray,
        t_norm: float,
        total_shots: int,
        n_layers: int = 2,
        dt: float = 0.42,
        temp_beta: float = 1.25,
        shake_scale: float = 2.2,
        shake_final_frac: float = 0.70,
    ) -> np.ndarray:
        n_traj = max(1, total_shots // self.shots_per_traj)
        n_states = 1 << self.n_qubits
        psi = np.full((n_traj, n_states), 1.0 / math.sqrt(n_states), dtype=np.complex64)

        diag_base = h_field.astype(np.float32) @ self.z_basis
        if len(self.edges) > 0:
            diag_base = diag_base + j_couplings.astype(np.float32) @ self.zz_basis

        probs_err = self._depol_probs()
        for layer in range(n_layers):
            s = (layer + 1) / n_layers
            mold_gain = np.float32(2.0 * dt * temp_beta * s)
            psi *= np.exp(np.complex64(-0.5j) * mold_gain * diag_base)[None, :]

            shake_angle = shake_scale * 2.0 * dt * (1.0 - shake_final_frac * s) * (0.6 + 0.4 * t_norm)
            self._apply_rx(psi, shake_angle)

            if self.p_depol > 0.0 or (self.p_depol_2q > 0.0 and len(self.edges) > 0):
                faults = self.rng.choice(4, size=(n_traj, self.n_qubits), p=probs_err) if self.p_depol > 0.0 else None
                if self.p_depol_2q > 0.0 and len(self.edges) > 0:
                    faults = self._compose_2q_faults_np(faults, n_traj)
                self._apply_pauli(psi, faults)

        probs = (psi.real * psi.real + psi.imag * psi.imag).astype(np.float64)
        probs /= probs.sum(axis=1, keepdims=True)
        cdfs = np.cumsum(probs, axis=1)
        cdfs[:, -1] = 1.0
        u_rand = self.rng.random((n_traj, self.shots_per_traj))
        idx_samples = np.empty((n_traj, self.shots_per_traj), dtype=np.int64)
        for tr in range(n_traj):
            idx_samples[tr] = np.searchsorted(cdfs[tr], u_rand[tr], side="right")
        np.minimum(idx_samples, n_states - 1, out=idx_samples)
        raw_bits = self.bit_table[idx_samples.reshape(-1)].copy()
        return self._inject_readout_noise(raw_bits)


class _TorchTrajectoryBackend(_BaseTrajectoryBackend):
    """GPU/CPU PyTorch batched statevector simulator with fused RX+Pauli NISQ channels."""

    def __init__(
        self,
        n_qubits: int,
        edges: List[Tuple[int, int]],
        p_depol: float = 0.035,
        p_readout: float = 0.04,
        p_depol_2q: float = 0.0,
        shots_per_traj: int = 4,
        torch_device: str = "auto",
        seed: int = 42,
    ):
        if not _HAS_TORCH:
            raise ImportError("PyTorch is not installed; use backend='numpy' or 'pennylane'.")
        if n_qubits > 18:
            raise ValueError(f"Torch statevector backend supports <=18 qubits (got {n_qubits}); use backend='pennylane'.")
        super().__init__(
            n_qubits, edges, p_depol=p_depol, p_readout=p_readout,
            p_depol_2q=p_depol_2q, shots_per_traj=shots_per_traj, seed=seed,
        )

        if torch_device == "auto":
            dev_name = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            dev_name = torch_device
        self.device = torch.device(dev_name)
        self._gen = torch.Generator(device=self.device)
        self._gen.manual_seed(int(seed))

        n_states = 1 << n_qubits
        basis = torch.arange(n_states, dtype=torch.int64, device=self.device)
        shifts = torch.arange(n_qubits - 1, -1, -1, dtype=torch.int64, device=self.device)
        self.bit_table = (basis[:, None] >> shifts[None, :]) & 1
        self.z_basis = (1.0 - 2.0 * self.bit_table.to(torch.float32)).t().contiguous()
        if edges:
            self.edge_u_t = torch.tensor(self.edge_u, dtype=torch.int64, device=self.device)
            self.edge_v_t = torch.tensor(self.edge_v, dtype=torch.int64, device=self.device)
            self.zz_basis = (self.z_basis[self.edge_u_t] * self.z_basis[self.edge_v_t]).contiguous()
        else:
            self.edge_u_t = torch.empty(0, dtype=torch.int64, device=self.device)
            self.edge_v_t = torch.empty(0, dtype=torch.int64, device=self.device)
            self.zz_basis = torch.zeros((0, n_states), dtype=torch.float32, device=self.device)

        self._pauli_mats = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.0, 1.0], [1.0, 0.0]],
                [[0.0, -1.0j], [1.0j, 0.0]],
                [[1.0, 0.0], [0.0, -1.0]],
            ],
            dtype=torch.complex64,
            device=self.device,
        )
        self._pauli_to_xz = torch.tensor(self._PAULI_TO_XZ, dtype=torch.int64, device=self.device)
        self._xz_to_pauli = torch.tensor(self._XZ_TO_PAULI, dtype=torch.int64, device=self.device)
        p0 = 1.0 - 0.75 * self.p_depol
        p1 = 0.25 * self.p_depol
        self._depol_cdf = torch.tensor([p0, p0 + p1, p0 + 2.0 * p1], dtype=torch.float32, device=self.device)

    def _sample_faults_torch(self, m: int, n_traj: int) -> "torch.Tensor":
        """Sample single-qubit depolarizing faults and compose edgewise 2Q faults mod 2."""
        if self.p_depol > 0.0:
            u_rand = torch.rand((m, n_traj, self.n_qubits), device=self.device, generator=self._gen)
            faults = (u_rand.unsqueeze(-1) >= self._depol_cdf).sum(dim=-1)
        else:
            faults = None

        if self.p_depol_2q > 0.0 and len(self.edges) > 0:
            n_e = len(self.edges)
            if faults is None:
                x_acc = torch.zeros((m, n_traj, self.n_qubits), dtype=torch.int64, device=self.device)
                z_acc = torch.zeros((m, n_traj, self.n_qubits), dtype=torch.int64, device=self.device)
            else:
                xz = self._pauli_to_xz[faults]
                x_acc = xz[..., 0].clone()
                z_acc = xz[..., 1].clone()

            trig = (torch.rand((m, n_traj, n_e), device=self.device, generator=self._gen) < self.p_depol_2q).to(torch.int64)
            codes = torch.randint(1, 16, (m, n_traj, n_e), dtype=torch.int64, device=self.device, generator=self._gen) * trig
            pu = self._pauli_to_xz[codes >> 2]
            pv = self._pauli_to_xz[codes & 3]
            idx_u = self.edge_u_t.view(1, 1, n_e).expand(m, n_traj, n_e)
            idx_v = self.edge_v_t.view(1, 1, n_e).expand(m, n_traj, n_e)
            # scatter_add + (& 1) avoids index-overwrite races when multiple edges share a qubit.
            x_acc = (x_acc.scatter_add(2, idx_u, pu[..., 0]).scatter_add(2, idx_v, pv[..., 0])) & 1
            z_acc = (z_acc.scatter_add(2, idx_u, pu[..., 1]).scatter_add(2, idx_v, pv[..., 1])) & 1
            faults = self._xz_to_pauli[x_acc, z_acc]
        return faults

    def _run_bits_chunk(
        self,
        h_fields: np.ndarray,
        j_couplings: np.ndarray,
        t_norms: np.ndarray,
        n_traj: int,
        n_layers: int,
        dt: float,
        temp_beta: float,
        shake_scale: float,
        shake_final_frac: float,
    ) -> "torch.Tensor":
        m = h_fields.shape[0]
        k_tot = m * n_traj
        n_states = 1 << self.n_qubits

        h_t = torch.as_tensor(h_fields, dtype=torch.float32, device=self.device)
        diag_base = h_t @ self.z_basis
        if len(self.edges) > 0:
            j_t = torch.as_tensor(j_couplings, dtype=torch.float32, device=self.device)
            diag_base = diag_base + j_t @ self.zz_basis
        t_norm_t = torch.as_tensor(t_norms, dtype=torch.float32, device=self.device)

        psi = torch.full((m, n_traj, n_states), 1.0 / math.sqrt(n_states), dtype=torch.complex64, device=self.device)
        has_pauli = self.p_depol > 0.0 or (self.p_depol_2q > 0.0 and len(self.edges) > 0)
        if has_pauli:
            m_grid = torch.arange(m, device=self.device)[:, None].expand(m, n_traj)

        for layer in range(n_layers):
            s = (layer + 1) / n_layers
            mold_gain = 2.0 * dt * temp_beta * s
            phase = torch.exp((-0.5j * mold_gain) * diag_base.to(torch.complex64)).unsqueeze(1)
            psi.mul_(phase)

            shake_angle = shake_scale * 2.0 * dt * (1.0 - shake_final_frac * s) * (0.6 + 0.4 * t_norm_t)
            ca = torch.cos(0.5 * shake_angle).to(torch.complex64)
            sa = (-1j * torch.sin(0.5 * shake_angle)).to(torch.complex64)

            if has_pauli:
                rx_mat = torch.stack(
                    [torch.stack([ca, sa], dim=-1), torch.stack([sa, ca], dim=-1)],
                    dim=-2,
                )  # (m, 2, 2)
                u_all = torch.einsum("fij,mjk->mfik", self._pauli_mats, rx_mat)  # (m, 4, 2, 2)
                faults = self._sample_faults_torch(m, n_traj)
                for q in range(self.n_qubits):
                    u_q = u_all[m_grid, faults[:, :, q]].reshape(k_tot, 2, 2)
                    u00 = u_q[:, 0, 0].view(k_tot, 1, 1)
                    u01 = u_q[:, 0, 1].view(k_tot, 1, 1)
                    u10 = u_q[:, 1, 0].view(k_tot, 1, 1)
                    u11 = u_q[:, 1, 1].view(k_tot, 1, 1)
                    v = psi.view(k_tot, 1 << q, 2, 1 << (self.n_qubits - 1 - q))
                    a0 = v[:, :, 0, :]
                    a1 = v[:, :, 1, :]
                    w0 = u00 * a0 + u01 * a1
                    w1 = u10 * a0 + u11 * a1
                    v[:, :, 0, :] = w0
                    v[:, :, 1, :] = w1
            else:
                ca_v = ca.view(m, 1, 1)
                sa_v = sa.view(m, 1, 1)
                for q in range(self.n_qubits):
                    v = psi.view(m, n_traj * (1 << q), 2, 1 << (self.n_qubits - 1 - q))
                    a0 = v[:, :, 0, :]
                    a1 = v[:, :, 1, :]
                    w0 = ca_v * a0 + sa_v * a1
                    w1 = sa_v * a0 + ca_v * a1
                    v[:, :, 0, :] = w0
                    v[:, :, 1, :] = w1

        probs = (psi.real * psi.real + psi.imag * psi.imag).view(k_tot, n_states)
        probs = probs / torch.clamp(probs.sum(dim=1, keepdim=True), min=1e-12)
        idx = torch.multinomial(probs, num_samples=self.shots_per_traj, replacement=True, generator=self._gen)
        bits = self.bit_table[idx.reshape(-1)].view(m, n_traj * self.shots_per_traj, self.n_qubits)
        if self.p_readout > 0.0:
            flips = (torch.rand(bits.shape, device=self.device, generator=self._gen) < self.p_readout).to(bits.dtype)
            bits = bits ^ flips
        return bits

    def sample(
        self,
        h_field: np.ndarray,
        j_couplings: np.ndarray,
        t_norm: float,
        total_shots: int,
        n_layers: int = 2,
        dt: float = 0.42,
        temp_beta: float = 1.25,
        shake_scale: float = 2.2,
        shake_final_frac: float = 0.70,
    ) -> np.ndarray:
        n_traj = max(1, total_shots // self.shots_per_traj)
        with torch.no_grad():
            bits = self._run_bits_chunk(
                h_field[None, :], j_couplings[None, :], np.array([t_norm], dtype=np.float32),
                n_traj, n_layers, dt, temp_beta, shake_scale, shake_final_frac,
            )
            return bits[0].cpu().numpy().astype(np.int64)

    def sample_obs_batch(
        self,
        h_fields: np.ndarray,
        j_couplings: np.ndarray,
        t_norms: np.ndarray,
        total_shots: int,
        n_visible: int,
        n_layers: int = 2,
        dt: float = 0.42,
        temp_beta: float = 1.25,
        shake_scale: float = 2.2,
        shake_final_frac: float = 0.70,
    ) -> Tuple[np.ndarray, np.ndarray]:
        m = h_fields.shape[0]
        n_traj = max(1, total_shots // self.shots_per_traj)
        n_states = 1 << self.n_qubits
        max_amp = 4_000_000
        chunk = max(1, max_amp // max(1, n_traj * n_states))

        obs_list = []
        with torch.no_grad():
            for st in range(0, m, chunk):
                en = min(m, st + chunk)
                bits = self._run_bits_chunk(
                    h_fields[st:en], j_couplings[st:en], t_norms[st:en],
                    n_traj, n_layers, dt, temp_beta, shake_scale, shake_final_frac,
                )
                spins = 1.0 - 2.0 * bits.to(torch.float32)  # (mc, S, n_qubits)
                z_all = spins.mean(dim=1)
                if len(self.edges) > 0:
                    zz = (spins[:, :, self.edge_u_t] * spins[:, :, self.edge_v_t]).mean(dim=1)
                    obs_t = torch.cat([zz, z_all], dim=1)
                else:
                    obs_t = z_all
                obs_list.append(obs_t.cpu().numpy().astype(np.float64))

        obs_out = np.concatenate(obs_list, axis=0) if len(obs_list) > 1 else obs_list[0]
        return obs_out, obs_out[:, len(self.edges) : len(self.edges) + n_visible].copy()


class _PennyLaneTrajectoryBackend(_BaseTrajectoryBackend):
    """PennyLane QNode backend compatible with lightning.qubit and real QPU hardware."""

    def __init__(
        self,
        n_qubits: int,
        edges: List[Tuple[int, int]],
        p_depol: float = 0.035,
        p_readout: float = 0.04,
        p_depol_2q: float = 0.0,
        shots_per_traj: int = 4,
        device_name: str = "lightning.qubit",
        device_kwargs: Optional[dict] = None,
        seed: int = 42,
    ):
        if not _HAS_PENNYLANE:
            raise ImportError("PennyLane is not installed; use backend='numpy' or install pennylane.")
        super().__init__(
            n_qubits, edges, p_depol=p_depol, p_readout=p_readout,
            p_depol_2q=p_depol_2q, shots_per_traj=shots_per_traj, seed=seed,
        )
        self.dev = qml.device(device_name, wires=n_qubits, shots=self.shots_per_traj, **(device_kwargs or {}))

        @qml.qnode(self.dev)
        def _circuit(
            h_field: np.ndarray,
            j_couplings: np.ndarray,
            t_norm: float,
            pauli_errors: Optional[np.ndarray],
            n_layers: int,
            dt: float,
            temp_beta: float,
            shake_scale: float,
            shake_final_frac: float,
        ):
            for q in range(n_qubits):
                qml.Hadamard(wires=q)

            for layer in range(n_layers):
                s = (layer + 1) / n_layers
                mold_gain = 2.0 * dt * temp_beta * s

                for q in range(n_qubits):
                    qml.RZ(mold_gain * h_field[q], wires=q)
                for idx, (u, v) in enumerate(edges):
                    qml.IsingZZ(mold_gain * j_couplings[idx], wires=[u, v])

                shake_angle = shake_scale * 2.0 * dt * (1.0 - shake_final_frac * s) * (0.6 + 0.4 * t_norm)
                for q in range(n_qubits):
                    qml.RX(shake_angle, wires=q)
                    if pauli_errors is not None:
                        err = pauli_errors[layer, q]
                        if err == 1:
                            qml.PauliX(wires=q)
                        elif err == 2:
                            qml.PauliY(wires=q)
                        elif err == 3:
                            qml.PauliZ(wires=q)

            return qml.sample(wires=range(n_qubits))

        self._circuit = _circuit

    def sample(
        self,
        h_field: np.ndarray,
        j_couplings: np.ndarray,
        t_norm: float,
        total_shots: int,
        n_layers: int = 2,
        dt: float = 0.42,
        temp_beta: float = 1.25,
        shake_scale: float = 2.2,
        shake_final_frac: float = 0.70,
    ) -> np.ndarray:
        n_traj = max(1, total_shots // self.shots_per_traj)
        probs_err = self._depol_probs()
        has_pauli = self.p_depol > 0.0 or (self.p_depol_2q > 0.0 and len(self.edges) > 0)
        all_bits = []
        for _ in range(n_traj):
            if has_pauli:
                faults = self.rng.choice(4, size=(n_layers, self.n_qubits), p=probs_err) if self.p_depol > 0.0 else None
                if self.p_depol_2q > 0.0 and len(self.edges) > 0:
                    faults = self._compose_2q_faults_np(faults, n_layers)
            else:
                faults = None
            bits = self._circuit(
                h_field, j_couplings, t_norm, faults,
                n_layers, dt, temp_beta, shake_scale, shake_final_frac,
            )
            all_bits.append(np.asarray(bits, dtype=np.int64))
        raw_bits = np.vstack(all_bits)
        return self._inject_readout_noise(raw_bits)


@dataclass
class QiskitRuntimeConfig:
    """Configuration for batched parameterized execution via `qiskit-ibm-runtime` `SamplerV2`."""

    execution_mode: str = "job"       # "job" (Open Plan compatible), "batch", or "session"
    optimization_level: int = 1
    initial_layout: Optional[Sequence[int]] = None
    max_circuits_per_pub: int = 500
    max_pubs_per_job: int = 4
    color_edges: bool = True
    dynamical_decoupling: bool = False
    twirling: bool = False
    skip_transpile: bool = False
    max_decode_elements: int = 16_000_000


class _QiskitRuntimeBackend(_BaseTrajectoryBackend):
    """
    Zero-recompilation, batched Qiskit Runtime (`SamplerV2`) hardware backend.

    Optimizations for large datasets and 128--156 qubit hardware:
      1) Transpiles ONE parametric `ParameterVector` ISA circuit per depth `(n_layers, ...)`
         and precomputes the exact column permutation matching `list(isa_qc.parameters)`.
      2) Greedy edge coloring compresses commuting `RZZ` gates into parallel 2Q layers.
      3) Vectorized NumPy broadcasting builds `(M, n_isa_params)` matrices with zero sample loops.
      4) Packs all `2M` contrastive circuits into multi-PUB `SamplerV2.run()` submissions inside
         a persistent `Session` / `Batch` / direct backend mode.
      5) Unpacks `BitArray.to_bool_array(order="little")` directly in C for vectorized `<Z>` / `<ZZ>`.
    """

    def __init__(
        self,
        n_qubits: int,
        edges: List[Tuple[int, int]],
        p_depol: float = 0.0,
        p_readout: float = 0.0,
        p_depol_2q: float = 0.0,
        shots_per_traj: int = 44,
        device_name: str = "ibm_fez",
        device_kwargs: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ):
        super().__init__(
            n_qubits, edges, p_depol=p_depol, p_readout=p_readout,
            p_depol_2q=p_depol_2q, shots_per_traj=shots_per_traj, seed=seed,
        )
        kw = dict(device_kwargs or {})
        self.cfg = QiskitRuntimeConfig(
            execution_mode=str(kw.get("execution_mode", "job")),
            optimization_level=int(kw.get("optimization_level", 1)),
            initial_layout=kw.get("initial_layout"),
            max_circuits_per_pub=max(1, int(kw.get("max_circuits_per_pub", 500))),
            max_pubs_per_job=max(1, int(kw.get("max_pubs_per_job", 4))),
            color_edges=bool(kw.get("color_edges", True)),
            dynamical_decoupling=bool(kw.get("dynamical_decoupling", False)),
            twirling=bool(kw.get("twirling", False)),
            skip_transpile=bool(kw.get("skip_transpile", False)),
            max_decode_elements=max(65_536, int(kw.get("max_decode_elements", 16_000_000))),
        )
        if self.cfg.execution_mode not in ("job", "batch", "session"):
            raise ValueError(f"execution_mode must be 'job', 'batch', or 'session' (got {self.cfg.execution_mode!r})")

        raw_backend = kw.get("backend") if kw.get("backend") is not None else kw.get("qiskit_backend")
        self._qiskit_backend = raw_backend
        self._pass_manager = kw.get("pass_manager")
        self._sampler = kw.get("sampler")
        self._meas_reg_name = str(kw.get("meas_reg_name", "meas"))
        self._mode_ctx = None
        self._owns_mode = False

        if isinstance(self._qiskit_backend, str) and _HAS_QISKIT_RUNTIME and _IBMService is not None:
            service = kw.get("service") or _IBMService(**(kw.get("service_kwargs") or {}))
            self._qiskit_backend = service.backend(self._qiskit_backend)

        if self._sampler is None:
            if not _HAS_QISKIT_RUNTIME:
                raise ImportError(
                    "qiskit-ibm-runtime is not installed; install `qiskit>=1.2 qiskit-ibm-runtime>=0.30` "
                    "or pass a local SamplerV2 instance via `device_kwargs={'sampler': ...}`."
                )
            if self._qiskit_backend is None:
                service = kw.get("service") or _IBMService(**(kw.get("service_kwargs") or {}))
                self._qiskit_backend = service.backend(device_name)

            if self.cfg.execution_mode == "job":
                mode_obj = self._qiskit_backend
            elif self.cfg.execution_mode == "batch":
                self._mode_ctx = _IBMBatch(backend=self._qiskit_backend)
                mode_obj = self._mode_ctx.__enter__()
                self._owns_mode = True
            else:
                self._mode_ctx = _IBMSession(backend=self._qiskit_backend)
                mode_obj = self._mode_ctx.__enter__()
                self._owns_mode = True

            self._sampler = _IBMSamplerV2(mode=mode_obj)

        opts = getattr(self._sampler, "options", None)
        if opts is not None:
            if self.cfg.dynamical_decoupling and hasattr(opts, "dynamical_decoupling"):
                opts.dynamical_decoupling.enable = True
            if self.cfg.twirling and hasattr(opts, "twirling"):
                opts.twirling.enable_gates = True
                opts.twirling.enable_measure = True
            for opt_k, opt_v in (kw.get("sampler_options") or {}).items():
                setattr(opts, opt_k, opt_v)

        if not _HAS_QISKIT and kw.get("circuit_builder") is None:
            raise ImportError("qiskit is not installed; install `qiskit>=1.2 qiskit-ibm-runtime>=0.30`.")

        self._circuit_builder = kw.get("circuit_builder")
        self._colored_layers = color_ising_edges(self.edges) if self.cfg.color_edges else []
        self._template_cache: Dict[int, Tuple[Any, np.ndarray]] = {}

    def close(self) -> None:
        """Close any opened Qiskit Runtime `Session` or `Batch` context cleanly."""
        if self._owns_mode and self._mode_ctx is not None:
            ctx, self._mode_ctx = self._mode_ctx, None
            self._owns_mode = False
            ctx.__exit__(None, None, None)

    def __enter__(self) -> "_QiskitRuntimeBackend":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _get_isa_template(self, n_layers: int) -> Tuple[Any, np.ndarray]:
        """Build and transpile one parametric template circuit per depth `n_layers` (cached)."""
        if n_layers in self._template_cache:
            return self._template_cache[n_layers]

        n_edges = len(self.edges)
        block = self.n_qubits + n_edges + 1
        total_params = n_layers * block

        if self._circuit_builder is not None:
            qc, param_objs = self._circuit_builder(
                self.n_qubits, self.edges, self._colored_layers, n_layers, total_params
            )
        else:
            p_vec = ParameterVector("p", total_params)
            qc = QuantumCircuit(self.n_qubits)
            for q in range(self.n_qubits):
                qc.h(q)
            for layer in range(n_layers):
                off = layer * block
                for q in range(self.n_qubits):
                    qc.rz(p_vec[off + q], q)
                if self.cfg.color_edges and self._colored_layers:
                    for edge_group in self._colored_layers:
                        for edge_idx, u, v in edge_group:
                            qc.rzz(p_vec[off + self.n_qubits + edge_idx], u, v)
                else:
                    for edge_idx, (u, v) in enumerate(self.edges):
                        qc.rzz(p_vec[off + self.n_qubits + edge_idx], u, v)
                rx_p = p_vec[off + self.n_qubits + n_edges]
                for q in range(self.n_qubits):
                    qc.rx(rx_p, q)
            qc.measure_all()
            param_objs = list(p_vec)

        if self.cfg.skip_transpile or (self._qiskit_backend is None and self._pass_manager is None):
            isa_qc = qc
        else:
            pm = self._pass_manager or generate_preset_pass_manager(
                backend=self._qiskit_backend,
                optimization_level=self.cfg.optimization_level,
                initial_layout=list(self.cfg.initial_layout) if self.cfg.initial_layout is not None else None,
            )
            isa_qc = pm.run(qc)

        orig_idx = {par: i for i, par in enumerate(param_objs)}
        isa_params = list(isa_qc.parameters)
        col_perm = np.array([orig_idx[par] for par in isa_params], dtype=np.int64)
        self._template_cache[n_layers] = (isa_qc, col_perm)
        return isa_qc, col_perm

    def _build_param_matrix(
        self,
        h_fields: np.ndarray,
        j_couplings: np.ndarray,
        t_norms: np.ndarray,
        col_perm: np.ndarray,
        n_layers: int,
        dt: float,
        temp_beta: float,
        shake_scale: float,
        shake_final_frac: float,
    ) -> np.ndarray:
        """Vectorized `(m, n_isa_params)` angle matrix construction with zero Python sample loops."""
        m = h_fields.shape[0]
        s_vec = np.arange(1, n_layers + 1, dtype=np.float64) / float(n_layers)
        mold_gains = (2.0 * dt * temp_beta * s_vec)[None, :, None]
        rz_blk = h_fields[:, None, :] * mold_gains  # (m, L, n_qubits)

        shake_s = (shake_scale * 2.0 * dt * (1.0 - shake_final_frac * s_vec))[None, :]
        rx_blk = (shake_s * (0.6 + 0.4 * t_norms[:, None]))[:, :, None]  # (m, L, 1)

        if len(self.edges) > 0:
            rzz_blk = j_couplings[:, None, :] * mold_gains  # (m, L, n_edges)
            raw = np.concatenate([rz_blk, rzz_blk, rx_blk], axis=2).reshape(m, -1)
        else:
            raw = np.concatenate([rz_blk, rx_blk], axis=2).reshape(m, -1)

        return np.ascontiguousarray(raw[:, col_perm], dtype=np.float64)

    def _extract_bits_from_pub_result(self, pub_result: Any) -> np.ndarray:
        """Decode Qiskit V2 `BitArray` into `(m_chunk, shots, n_qubits)` uint8/bool array without Python loops."""
        data_bin = pub_result.data
        bit_arr = getattr(data_bin, self._meas_reg_name, None)
        if bit_arr is None:
            for v in vars(data_bin).values():
                if hasattr(v, "to_bool_array") or hasattr(v, "array"):
                    bit_arr = v
                    break
        if bit_arr is None:
            raise KeyError(f"Could not find measurement BitArray register '{self._meas_reg_name}' in PUB result.")

        if hasattr(bit_arr, "to_bool_array"):
            bits = bit_arr.to_bool_array(order="little")
        else:
            raw_u8 = np.asarray(bit_arr.array, dtype=np.uint8)
            bits = np.unpackbits(raw_u8, axis=-1, bitorder="little")[..., : self.n_qubits].astype(bool)

        if bits.ndim == 2:
            bits = bits[None, :, :]
        return bits

    def _execute_param_matrix(
        self,
        isa_qc: Any,
        param_matrix: np.ndarray,
        total_shots: int,
    ) -> List[np.ndarray]:
        """Submit multi-PUB batched job(s) to Qiskit SamplerV2 and decode bit arrays."""
        m = param_matrix.shape[0]
        shots = max(1, int(total_shots))
        step = self.cfg.max_circuits_per_pub
        pubs = [(isa_qc, param_matrix[st : min(m, st + step)], shots) for st in range(0, m, step)]

        bit_chunks: List[np.ndarray] = []
        p_step = self.cfg.max_pubs_per_job
        for pst in range(0, len(pubs), p_step):
            job = self._sampler.run(pubs[pst : min(len(pubs), pst + p_step)])
            result = job.result()
            for pub_res in result:
                bit_chunks.append(self._extract_bits_from_pub_result(pub_res))
        return bit_chunks

    def sample(
        self,
        h_field: np.ndarray,
        j_couplings: np.ndarray,
        t_norm: float,
        total_shots: int,
        n_layers: int = 2,
        dt: float = 0.42,
        temp_beta: float = 1.25,
        shake_scale: float = 2.2,
        shake_final_frac: float = 0.70,
    ) -> np.ndarray:
        isa_qc, col_perm = self._get_isa_template(n_layers)
        param_mat = self._build_param_matrix(
            h_field[None, :],
            j_couplings[None, :],
            np.array([t_norm], dtype=np.float64),
            col_perm,
            n_layers=n_layers,
            dt=dt,
            temp_beta=temp_beta,
            shake_scale=shake_scale,
            shake_final_frac=shake_final_frac,
        )
        bits = self._execute_param_matrix(isa_qc, param_mat, total_shots)[0]
        return bits[0, :, : self.n_qubits].astype(np.int64)

    def sample_obs_batch(
        self,
        h_fields: np.ndarray,
        j_couplings: np.ndarray,
        t_norms: np.ndarray,
        total_shots: int,
        n_visible: int,
        n_layers: int = 2,
        dt: float = 0.42,
        temp_beta: float = 1.25,
        shake_scale: float = 2.2,
        shake_final_frac: float = 0.70,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run an entire contrastive/inference batch as packed SamplerV2 PUBs without per-sample CPU loops."""
        isa_qc, col_perm = self._get_isa_template(n_layers)
        param_mat = self._build_param_matrix(
            np.asarray(h_fields, dtype=np.float64),
            np.asarray(j_couplings, dtype=np.float64),
            np.asarray(t_norms, dtype=np.float64),
            col_perm,
            n_layers=n_layers,
            dt=dt,
            temp_beta=temp_beta,
            shake_scale=shake_scale,
            shake_final_frac=shake_final_frac,
        )
        bit_chunks = self._execute_param_matrix(isa_qc, param_mat, total_shots)

        if len(bit_chunks) == 1:
            return self._bits_to_obs_batch(bit_chunks[0], n_visible, self.cfg.max_decode_elements)

        obs_parts = []
        vis_parts = []
        for chunk_bits in bit_chunks:
            o_part, z_part = self._bits_to_obs_batch(chunk_bits, n_visible, self.cfg.max_decode_elements)
            obs_parts.append(o_part)
            vis_parts.append(z_part)
        return np.vstack(obs_parts), np.vstack(vis_parts)
