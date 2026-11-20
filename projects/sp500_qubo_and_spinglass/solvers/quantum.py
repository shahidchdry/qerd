"""Quantum solvers (`QER-Res` / `QER-Grad`, calibrated `Adam-SPSA`, `Exact-PSR`), sampler, and checkpoints."""

from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from qerd import (
    QERGradConfig,
    QERGradientEstimator,
    QuantumIsingDenoiser,
    StabilizedAdam,
    StabilizedAdamConfig,
)
from ..hardware.heron_r2 import mitigate_expectations


def _sample_noisy_batch(
    model: QuantumIsingDenoiser,
    x_tr: np.ndarray,
    eff_bs: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample balanced diffusion timesteps and noisy forward inputs `(tg, tc, xt)`."""
    ids = model.rng.choice(len(x_tr), size=eff_bs, replace=len(x_tr) < eff_bs)
    tg = x_tr[ids]
    tc = (np.arange(eff_bs) % model.diffusion_steps) + 1
    model.rng.shuffle(tc)
    ab = model.alpha_bar[tc][:, None]
    xt = np.clip(
        np.sqrt(ab) * tg + np.sqrt(1.0 - ab) * model.rng.normal(0.0, 0.70, size=tg.shape),
        -1.0,
        1.0,
    )
    return tg, tc, xt


def sample_qerd_batch(
    model: QuantumIsingDenoiser,
    n_samples: int,
    shots_per_step: int,
    step_noise_scale: float = 0.30,
    init_std: float = 0.90,
    sample_temp: float = 1.15,
    stochastic_decode: bool = False,
    ro_cal: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> np.ndarray:
    """Reverse diffusion sampler with optional 1-job live readout mitigation and stochastic decoding."""
    x_t = np.clip(model.rng.normal(0.0, init_std, size=(n_samples, model.n_visible)), -1.0, 1.0)
    for t in range(model.diffusion_steps, 0, -1):
        t_arr = np.full(n_samples, t, dtype=np.int64)
        _, z_raw = model.measure_observables_batch(x_t, t_arr, shots=shots_per_step)
        if ro_cal is not None:
            z_raw = mitigate_expectations(z_raw, ro_cal[0], ro_cal[1])
        x0_hat = model.calibrator.calibrate(t, z_raw)
        if stochastic_decode:
            p_pos = 0.5 * (1.0 + np.clip(sample_temp * x0_hat, -0.98, 0.98))
            x0_draw = np.where(model.rng.random(x0_hat.shape) < p_pos, 1.0, -1.0)
        else:
            x0_draw = x0_hat
        if t > 1:
            a_next = model.alpha_bar[t - 1]
            eps = model.rng.normal(0.0, step_noise_scale, size=(n_samples, model.n_visible))
            x_t = np.clip(math.sqrt(a_next) * x0_draw + math.sqrt(1.0 - a_next) * eps, -1.0, 1.0)
        else:
            x_t = x0_draw
    return np.where(x_t >= 0.0, 1.0, -1.0)


def train_qerd_explicit(
    model: QuantumIsingDenoiser,
    x_tr: np.ndarray,
    max_shots: int,
    shots_per_run: int,
    batch_size: int,
    lr_max: float = 0.13,
    lr_min: float = 0.015,
    clip_bound: float = 1.30,
    weight_decay: float = 0.015,
    nudge_mode: str = "residual",
    ro_cal: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Explicit `QERGradientEstimator` + `StabilizedAdam` loop supporting:
      * `"residual"` (`3` circuits/sample `QER-Res` MSE contrastive nudging, `2` packed QPU jobs/batch)
      * `"target"`   (`2` circuits/sample regularized `QER-Grad`, `1` packed QPU job/batch)
    """
    if nudge_mode not in ("residual", "target"):
        raise ValueError(f"Unknown nudge_mode={nudge_mode!r}; expected 'residual' or 'target'.")
    t_start = time.perf_counter()
    grad_est = QERGradientEstimator(
        QERGradConfig(beta=0.45, beta_min=0.15, centered=True, auto_beta_guard=True)
    )
    use_residual = nudge_mode == "residual"
    circuits_per_sample = 3 if use_residual else 2
    eff_bs = min(batch_size, len(x_tr))
    eff_lr = min(0.18, lr_max * (eff_bs / 16.0) ** 0.7) if (use_residual and eff_bs > 16) else lr_max
    if use_residual:
        adam = StabilizedAdam(
            model.parameters_dict(),
            StabilizedAdamConfig(lr_max=eff_lr, lr_min=lr_min, weight_decay=weight_decay, clip_bound=clip_bound),
        )
    else:
        adam = StabilizedAdam(
            model.parameters_dict(),
            StabilizedAdamConfig(
                lr_max=min(lr_max, 0.07), lr_min=lr_min, weight_decay=max(weight_decay, 0.03),
                clip_bound=clip_bound, use_adam=False,
            ),
        )
    n_batches = max(1, max_shots // (circuits_per_sample * eff_bs * shots_per_run))
    aligns: List[float] = []

    for b_idx in range(1, n_batches + 1):
        progress = (b_idx - 1) / max(1, n_batches - 1)
        tg, tc, xt = _sample_noisy_batch(model, x_tr, eff_bs)

        if use_residual:
            _, z_free = model.measure_observables_batch(xt, tc, shots=shots_per_run)
            if ro_cal is not None:
                z_free = mitigate_expectations(z_free, ro_cal[0], ro_cal[1])
            nudge_tg = np.empty_like(tg)
            for k in range(eff_bs):
                tv = int(tc[k])
                model.calibrator.update(tv, z_free[k], tg[k])
                x0_hat = model.calibrator.calibrate(tv, z_free[k])
                g_t = np.maximum(model.calibrator.gain[tv], 0.20)
                nudge_tg[k] = np.clip((tg[k] - x0_hat) / g_t, -2.0, 2.0)
        else:
            nudge_tg = tg

        beta_v = grad_est.beta
        obs_both, z_both = model.measure_observables_batch(
            np.concatenate([xt, xt], axis=0),
            np.concatenate([tc, tc], axis=0),
            shots=shots_per_run,
            nudge_target_batch=np.concatenate([nudge_tg, nudge_tg], axis=0),
            nudge_beta_batch=np.concatenate([np.full(eff_bs, beta_v), np.full(eff_bs, -beta_v)], axis=0),
        )
        zp, zm = z_both[:eff_bs], z_both[eff_bs:]
        if ro_cal is not None:
            zp = mitigate_expectations(zp, ro_cal[0], ro_cal[1])
            zm = mitigate_expectations(zm, ro_cal[0], ro_cal[1])

        grads, mean_align = grad_est.estimate(
            model.parameters_dict(),
            obs_plus_list=obs_both[:eff_bs],
            obs_minus_list=obs_both[eff_bs:],
            z_plus_list=zp,
            z_minus_list=zm,
            nudge_dirs=nudge_tg,
        )
        if not use_residual:
            z_mid = 0.5 * (zp + zm)
            for tv, z_v, tgv in zip(tc, z_mid, tg):
                model.calibrator.update(int(tv), z_v, tgv)

        adam.step(
            model.parameters_dict(),
            grads,
            progress=progress,
            shots_in_batch=circuits_per_sample * eff_bs * shots_per_run,
        )
        if mean_align is not None:
            aligns.append(float(mean_align))
        if verbose and (b_idx == 1 or b_idx % 5 == 0 or b_idx == n_batches):
            al_str = f"{mean_align:+.3f}" if mean_align is not None else "n/a"
            print(
                f"  [QPU Train] Batch {b_idx:02d}/{n_batches:02d} | Shots={adam.shots_used:5d} | "
                f"Beta={grad_est.beta:.2f} | Align={al_str} | Wall={time.perf_counter()-t_start:.1f}s",
                flush=True,
            )

    return {
        "steps": n_batches,
        "batch_size": eff_bs,
        "circuits_per_sample": circuits_per_sample,
        "shots_used": adam.shots_used,
        "final_beta": grad_est.beta,
        "mean_nudge_align": float(np.mean(aligns)) if aligns else 0.0,
        "wall_sec": time.perf_counter() - t_start,
    }


def train_spsa_equal_shot(
    model: QuantumIsingDenoiser,
    x_tr: np.ndarray,
    max_shots: int,
    shots_per_run: int,
    batch_size: int,
    lr_max: float = 0.13,
    lr_min: float = 0.015,
    c0: float = 0.22,
    clip_bound: float = 1.30,
    weight_decay: float = 0.015,
    spsa_shots_per_run: Optional[int] = None,
    ro_cal: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Strong equal-shot per-sample multi-direction SPSA + StabilizedAdam + online calibrator.

    Trades redundant shots-per-circuit for additional parallel Rademacher directions inside the
    exact same total shot budget and QPU job count (reducing O(P) mask-crosstalk variance while
    keeping 1.5x more steps than 3-circuit QER-Res).
    """
    t_start = time.perf_counter()
    n_p = model.n_qpu_params
    eff_bs = min(batch_size, len(x_tr))
    if spsa_shots_per_run is None:
        target_sh = min(shots_per_run, 12) if shots_per_run >= 24 else shots_per_run
    else:
        target_sh = max(4, min(int(spsa_shots_per_run), shots_per_run))
    dir_mult = max(1, shots_per_run // max(1, target_sh))
    eff_sh = max(4, shots_per_run // dir_mult)
    n_dirs = eff_bs * dir_mult
    step_shots = 2 * n_dirs * eff_sh
    n_steps = max(1, max_shots // max(1, step_shots))
    eff_lr = min(0.18, lr_max * (eff_bs / 16.0) ** 0.7) if eff_bs > 16 else lr_max
    adam = StabilizedAdam(
        model.parameters_dict(),
        StabilizedAdamConfig(lr_max=eff_lr, lr_min=lr_min, weight_decay=weight_decay, clip_bound=clip_bound),
    )

    for st in range(1, n_steps + 1):
        progress = (st - 1) / max(1, n_steps - 1)
        tg, tc, xt = _sample_noisy_batch(model, x_tr, n_dirs)
        c_k = c0 / (st ** 0.101)
        d_th = model.rng.choice([-1.0, 1.0], size=(n_dirs, n_p))
        th_p = np.clip(model.theta[None, :] + c_k * d_th, -clip_bound, clip_bound)
        th_m = np.clip(model.theta[None, :] - c_k * d_th, -clip_bound, clip_bound)

        _, z_2b = model.measure_observables_batch(
            np.concatenate([xt, xt], axis=0),
            np.concatenate([tc, tc], axis=0),
            shots=eff_sh,
            theta_batch=np.concatenate([th_p, th_m], axis=0),
        )
        zp, zm = z_2b[:n_dirs], z_2b[n_dirs:]
        if ro_cal is not None:
            zp = mitigate_expectations(zp, ro_cal[0], ro_cal[1])
            zm = mitigate_expectations(zm, ro_cal[0], ro_cal[1])

        z_mid = 0.5 * (zp + zm)
        for k in range(n_dirs):
            model.calibrator.update(int(tc[k]), z_mid[k], tg[k])
        gain_b = np.stack([model.calibrator.gain[int(t)] for t in tc], axis=0)
        if gain_b.ndim == 1:
            gain_b = gain_b[:, None]

        lp = np.mean((np.clip(zp / gain_b, -1.0, 1.0) - tg) ** 2, axis=1)
        lm = np.mean((np.clip(zm / gain_b, -1.0, 1.0) - tg) ** 2, axis=1)
        diff = (lp - lm) / (2.0 * c_k)
        grad_th = -np.mean(diff[:, None] * d_th, axis=0)

        adam.step(model.parameters_dict(), {"theta": grad_th}, progress=progress, shots_in_batch=step_shots)
        if verbose and (st == 1 or st % 5 == 0 or st == n_steps):
            print(
                f"  [SPSA Train] Batch {st:02d}/{n_steps:02d} ({n_dirs} dirs x {eff_sh} sh) | "
                f"Shots={adam.shots_used:5d} | Wall={time.perf_counter()-t_start:.1f}s",
                flush=True,
            )

    return {
        "steps": n_steps,
        "n_dirs": n_dirs,
        "shots_per_run_spsa": eff_sh,
        "shots_used": adam.shots_used,
        "wall_sec": time.perf_counter() - t_start,
    }


def train_psr_equal_shot(
    model: QuantumIsingDenoiser,
    x_tr: np.ndarray,
    max_shots: int,
    shots_per_run: int,
    lr_max: float = 0.10,
    lr_min: float = 0.015,
    clip_bound: float = 1.30,
    weight_decay: float = 0.015,
    ro_cal: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Dict[str, int]:
    """Equal-shot finite-difference Parameter-Shift + StabilizedAdam + online calibrator (`2P` circuits/step)."""
    n_p = model.n_qpu_params
    step_shots = 2 * n_p * shots_per_run
    n_steps = max(0, max_shots // max(1, step_shots))
    adam = StabilizedAdam(
        model.parameters_dict(),
        StabilizedAdamConfig(lr_max=lr_max, lr_min=lr_min, weight_decay=weight_decay, clip_bound=clip_bound),
    )
    eye = np.arange(n_p)

    for st in range(1, n_steps + 1):
        progress = (st - 1) / max(1, n_steps - 1)
        tg = x_tr[int(model.rng.integers(0, len(x_tr)))]
        tv = ((st - 1) % model.diffusion_steps) + 1
        ab = model.alpha_bar[tv]
        xt = np.clip(math.sqrt(ab) * tg + math.sqrt(1.0 - ab) * model.rng.normal(0.0, 0.70, size=tg.shape), -1.0, 1.0)
        th_p = np.broadcast_to(model.theta, (n_p, n_p)).copy()
        th_m = th_p.copy()
        th_p[eye, eye] += 0.25
        th_m[eye, eye] -= 0.25
        _, z_all = model.measure_observables_batch(
            np.broadcast_to(xt, (2 * n_p, model.n_visible)),
            np.full(2 * n_p, tv, dtype=np.int64),
            shots=shots_per_run,
            theta_batch=np.concatenate([th_p, th_m], axis=0),
        )
        if ro_cal is not None:
            z_all = mitigate_expectations(z_all, ro_cal[0], ro_cal[1])
        model.calibrator.update(tv, np.mean(z_all, axis=0), tg)
        g_t = model.calibrator.gain[tv]
        zp = np.clip(z_all[:n_p] / g_t, -1.0, 1.0)
        zm = np.clip(z_all[n_p:] / g_t, -1.0, 1.0)
        grad_th = -(np.mean((zp - tg[None, :]) ** 2, axis=1) - np.mean((zm - tg[None, :]) ** 2, axis=1)) / 0.50
        adam.step(model.parameters_dict(), {"theta": grad_th}, progress=progress, shots_in_batch=step_shots)

    return {"steps": n_steps, "shots_used": adam.shots_used}


def train_coord_psr_equal_shot(
    model: QuantumIsingDenoiser,
    x_tr: np.ndarray,
    max_shots: int,
    shots_per_shift: int = 4,
    coords_per_step: int = 8,
    lr_max: float = 0.10,
    lr_min: float = 0.015,
    clip_bound: float = 1.30,
    weight_decay: float = 0.015,
    ro_cal: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    verbose: bool = False,
) -> Dict[str, int]:
    """
    Doubly stochastic randomized-coordinate PSR (Sweke et al., 2020) under equal shot budget.

    Samples `k = min(coords_per_step, P)` random parameter coordinates per step and evaluates
    symmetric shifts with `shots_per_shift` shots per query, giving `O(max_shots / (2*k*shots_per_shift))`
    updates instead of full-sweep PSR's step starvation.
    """
    t_start = time.perf_counter()
    n_p = model.n_qpu_params
    k = max(1, min(int(coords_per_step), n_p))
    sh = max(1, int(shots_per_shift))
    step_shots = 2 * k * sh
    n_steps = max(0, max_shots // max(1, step_shots))
    m_buf = np.zeros(n_p, dtype=np.float64)
    v_buf = np.zeros(n_p, dtype=np.float64)
    coord_steps = np.zeros(n_p, dtype=np.int64)
    shots_used = 0
    b1, b2, eps = 0.75, 0.98, 1e-6

    for st in range(1, n_steps + 1):
        progress = (st - 1) / max(1, n_steps - 1)
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))
        eff_decay = 1.0 - weight_decay * (lr / max(lr_max, 1e-8))
        tg = x_tr[int(model.rng.integers(0, len(x_tr)))]
        tv = ((st - 1) % model.diffusion_steps) + 1
        ab = model.alpha_bar[tv]
        xt = np.clip(math.sqrt(ab) * tg + math.sqrt(1.0 - ab) * model.rng.normal(0.0, 0.70, size=tg.shape), -1.0, 1.0)
        sel = model.rng.choice(n_p, size=k, replace=False)
        th_p = np.broadcast_to(model.theta, (k, n_p)).copy()
        th_m = th_p.copy()
        rows = np.arange(k)
        th_p[rows, sel] += 0.30
        th_m[rows, sel] -= 0.30
        _, z_all = model.measure_observables_batch(
            np.broadcast_to(xt, (2 * k, model.n_visible)),
            np.full(2 * k, tv, dtype=np.int64),
            shots=sh,
            theta_batch=np.concatenate([th_p, th_m], axis=0),
        )
        shots_used += step_shots
        if ro_cal is not None:
            z_all = mitigate_expectations(z_all, ro_cal[0], ro_cal[1])
        model.calibrator.update(tv, np.mean(z_all, axis=0), tg)
        g_t = model.calibrator.gain[tv]
        zp = np.clip(z_all[:k] / g_t, -1.0, 1.0)
        zm = np.clip(z_all[k:] / g_t, -1.0, 1.0)
        g_sel = -(np.mean((zp - tg[None, :]) ** 2, axis=1) - np.mean((zm - tg[None, :]) ** 2, axis=1)) / 0.60
        # Sparse per-coordinate Adam: only update moments and weights for sampled coordinates
        coord_steps[sel] += 1
        m_buf[sel] = b1 * m_buf[sel] + (1.0 - b1) * g_sel
        v_buf[sel] = b2 * v_buf[sel] + (1.0 - b2) * (g_sel ** 2)
        mh = m_buf[sel] / (1.0 - b1 ** coord_steps[sel])
        vh = v_buf[sel] / (1.0 - b2 ** coord_steps[sel])
        model.theta[sel] = np.clip(eff_decay * model.theta[sel] + lr * mh / (np.sqrt(vh) + eps), -clip_bound, clip_bound)
        if verbose and (st == 1 or st % 5 == 0 or st == n_steps):
            print(
                f"  [Coord-PSR] Batch {st:02d}/{n_steps:02d} ({k} coords x {sh} sh) | "
                f"Shots={shots_used:5d} | Wall={time.perf_counter()-t_start:.1f}s",
                flush=True,
            )

    return {"steps": n_steps, "shots_used": shots_used, "coords_per_step": k, "shots_per_shift": sh}


def save_qerd_checkpoint(
    path: str,
    model: QuantumIsingDenoiser,
    edges: List[Tuple[int, int]],
    problem_name: str,
    tickers: List[str] | None = None,
    ro_cal: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ro_off = ro_cal[0] if ro_cal is not None else np.zeros(model.n_qubits, dtype=np.float64)
    ro_sc = ro_cal[1] if ro_cal is not None else np.ones(model.n_qubits, dtype=np.float64)
    np.savez(
        path,
        theta=model.theta,
        calibrator_gain=model.calibrator.gain,
        cal_mode=np.array([model.calibrator.mode]),
        edges=np.array(edges, dtype=np.int64),
        n_qubits=np.array([model.n_visible], dtype=np.int64),
        n_layers=np.array([model.n_layers], dtype=np.int64),
        diffusion_steps=np.array([model.diffusion_steps], dtype=np.int64),
        problem_name=np.array([problem_name]),
        tickers=np.array(tickers or []),
        ro_offset=ro_off,
        ro_scale=ro_sc,
    )


def load_qerd_checkpoint(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at {path}")
    ck = np.load(path, allow_pickle=False)
    n_q = int(ck["n_qubits"][0])
    return {
        "theta": ck["theta"],
        "calibrator_gain": ck["calibrator_gain"],
        "cal_mode": str(ck["cal_mode"][0]) if "cal_mode" in ck else "forward",
        "edges": [tuple(e) for e in ck["edges"].tolist()],
        "n_qubits": n_q,
        "n_layers": int(ck["n_layers"][0]),
        "diffusion_steps": int(ck["diffusion_steps"][0]) if "diffusion_steps" in ck else 4,
        "problem_name": str(ck["problem_name"][0]),
        "tickers": ck["tickers"].tolist(),
        "ro_offset": ck["ro_offset"] if "ro_offset" in ck else np.zeros(n_q, dtype=np.float64),
        "ro_scale": ck["ro_scale"] if "ro_scale" in ck else np.ones(n_q, dtype=np.float64),
    }
