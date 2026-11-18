"""
Heavily packed conditional Quantum solvers for Project 2 (`QERD + cal_mode='inverse'`, `SPSA`, `PSR`),
plus optional 1-job marginal hardware readout error mitigation and checkpoint I/O.
"""

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
from ..hardware.kingston_clean import mitigate_expectations


def sample_dna_batch(
    model: QuantumIsingDenoiser,
    conds: Optional[np.ndarray],
    n_samples: int,
    shots_per_step: int,
    step_noise_scale: float = 0.70,
    init_std: float = 0.90,
    sample_temp: float = 1.15,
    stochastic_decode: bool = True,
    ro_cal: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> np.ndarray:
    """Conditional reverse diffusion sampler with optional 1-job live marginal readout mitigation."""
    n = int(conds.shape[0]) if conds is not None else int(n_samples)
    x_t = np.clip(model.rng.normal(0.0, init_std, size=(n, model.n_visible)), -1.0, 1.0)
    c_arr = None if conds is None else np.asarray(conds, dtype=np.float64).reshape(n, model.cond_dim)

    for t in range(model.diffusion_steps, 0, -1):
        t_arr = np.full(n, t, dtype=np.int64)
        _, z_raw = model.measure_observables_batch(x_t, t_arr, shots=shots_per_step, cond_batch=c_arr)
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
            eps = model.rng.normal(0.0, step_noise_scale, size=(n, model.n_visible))
            x_t = np.clip(math.sqrt(a_next) * x0_draw + math.sqrt(1.0 - a_next) * eps, -1.0, 1.0)
        else:
            x_t = x0_draw
    return np.where(x_t >= 0.0, 1.0, -1.0)


def train_qerd_dna(
    model: QuantumIsingDenoiser,
    x_tr: np.ndarray,
    cond_tr: np.ndarray,
    max_shots: int,
    shots_per_run: int,
    batch_size: int,
    lr_max: float = 0.12,
    lr_min: float = 0.012,
    clip_bound: float = 1.05,
    nudge_mode: str = "residual",
    ro_cal: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Conditional QERD training using either:
      * `"residual"` (`3` circuits/sample `QER-Res` optimizing true MSE `y = (x_0 - x0_hat)/gain`)
      * `"target"`   (`2` circuits/sample `QER-Grad` with regularized updates preventing corner saturation)
    """
    if nudge_mode not in ("residual", "target"):
        raise ValueError(f"Unknown nudge_mode={nudge_mode!r}; expected 'residual' or 'target'.")
    t0 = time.perf_counter()
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
            StabilizedAdamConfig(lr_max=eff_lr, lr_min=lr_min, weight_decay=0.02, clip_bound=clip_bound),
        )
    else:
        # Avoid Adam second-moment sign runaway on linear target-nudge gradients
        adam = StabilizedAdam(
            model.parameters_dict(),
            StabilizedAdamConfig(lr_max=0.06, lr_min=0.01, weight_decay=0.04, clip_bound=clip_bound, use_adam=False),
        )
    n_batches = max(1, max_shots // (circuits_per_sample * eff_bs * shots_per_run))
    aligns: List[float] = []

    for b_idx in range(1, n_batches + 1):
        progress = (b_idx - 1) / max(1, n_batches - 1)
        ids = model.rng.choice(len(x_tr), size=eff_bs, replace=False)
        tg = x_tr[ids]
        cb = cond_tr[ids]
        tc = (np.arange(eff_bs) % model.diffusion_steps) + 1
        model.rng.shuffle(tc)
        ab = model.alpha_bar[tc][:, None]
        xt = np.clip(
            np.sqrt(ab) * tg + np.sqrt(1.0 - ab) * model.rng.normal(0.0, 0.70, size=tg.shape),
            -1.0,
            1.0,
        )
        if use_residual:
            _, z_free = model.measure_observables_batch(xt, tc, shots=shots_per_run, cond_batch=cb)
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
            cond_batch=np.concatenate([cb, cb], axis=0),
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
            conds=cb,
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
            al_s = f"{mean_align:+.3f}" if mean_align is not None else "n/a"
            print(
                f"  [QERD QPU] Batch {b_idx:02d}/{n_batches:02d} | Shots={adam.shots_used:6d} | "
                f"Beta={grad_est.beta:.2f} | Align={al_s} | Wall={time.perf_counter()-t0:.1f}s",
                flush=True,
            )

    return {
        "steps": n_batches,
        "batch_size": eff_bs,
        "circuits_per_sample": circuits_per_sample,
        "shots_used": adam.shots_used,
        "mean_nudge_align": float(np.mean(aligns)) if aligns else 0.0,
        "wall_sec": time.perf_counter() - t0,
    }


def train_spsa_dna(
    model: QuantumIsingDenoiser,
    x_tr: np.ndarray,
    cond_tr: np.ndarray,
    max_shots: int,
    shots_per_run: int,
    batch_size: int,
    lr_max: float = 0.12,
    lr_min: float = 0.012,
    c0: float = 0.22,
    clip_bound: float = 1.05,
    ro_cal: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Fair equal-shot per-sample Rademacher 2-run SPSA + StabilizedAdam + online calibrator."""
    t0 = time.perf_counter()
    n_th = model.n_qpu_params
    has_cp = model.cond_proj is not None
    cp_shape = model.cond_proj.shape if has_cp else (0, 0)

    eff_bs = min(batch_size, len(x_tr))
    n_steps = max(1, max_shots // (2 * eff_bs * shots_per_run))
    adam = StabilizedAdam(
        model.parameters_dict(),
        StabilizedAdamConfig(lr_max=lr_max, lr_min=lr_min, weight_decay=0.02, clip_bound=clip_bound),
    )

    for st in range(1, n_steps + 1):
        progress = (st - 1) / max(1, n_steps - 1)
        ids = model.rng.choice(len(x_tr), size=eff_bs, replace=False)
        tg = x_tr[ids]
        cb = cond_tr[ids]
        tc = (np.arange(eff_bs) % model.diffusion_steps) + 1
        model.rng.shuffle(tc)
        ab = model.alpha_bar[tc][:, None]
        xt = np.clip(np.sqrt(ab) * tg + np.sqrt(1.0 - ab) * model.rng.normal(0.0, 0.70, size=tg.shape), -1.0, 1.0)

        c_k = c0 / (st ** 0.101)
        d_th = model.rng.choice([-1.0, 1.0], size=(eff_bs, n_th))
        th_b = np.concatenate([
            np.clip(model.theta[None, :] + c_k * d_th, -clip_bound, clip_bound),
            np.clip(model.theta[None, :] - c_k * d_th, -clip_bound, clip_bound),
        ], axis=0)
        d_cp = None
        cp_b = None
        if has_cp:
            d_cp = model.rng.choice([-1.0, 1.0], size=(eff_bs,) + cp_shape)
            cp_b = np.concatenate([
                np.clip(model.cond_proj[None, :, :] + c_k * d_cp, -clip_bound, clip_bound),
                np.clip(model.cond_proj[None, :, :] - c_k * d_cp, -clip_bound, clip_bound),
            ], axis=0)

        _, z_2b = model.measure_observables_batch(
            np.concatenate([xt, xt], axis=0),
            np.concatenate([tc, tc], axis=0),
            shots=shots_per_run,
            cond_batch=np.concatenate([cb, cb], axis=0),
            theta_batch=th_b,
            cond_proj_batch=cp_b,
        )
        zp, zm = z_2b[:eff_bs], z_2b[eff_bs:]
        if ro_cal is not None:
            zp = mitigate_expectations(zp, ro_cal[0], ro_cal[1])
            zm = mitigate_expectations(zm, ro_cal[0], ro_cal[1])

        z_mid = 0.5 * (zp + zm)
        for k in range(eff_bs):
            model.calibrator.update(int(tc[k]), z_mid[k], tg[k])
        gain_b = np.stack([model.calibrator.gain[int(t)] for t in tc], axis=0)
        if gain_b.ndim == 1:
            gain_b = gain_b[:, None]

        lp = np.mean((np.clip(zp / gain_b, -1.0, 1.0) - tg) ** 2, axis=1)
        lm = np.mean((np.clip(zm / gain_b, -1.0, 1.0) - tg) ** 2, axis=1)
        diff = (lp - lm) / (2.0 * c_k)

        grads: Dict[str, np.ndarray] = {"theta": -np.mean(diff[:, None] * d_th, axis=0)}
        if has_cp and d_cp is not None:
            grads["cond_proj"] = -np.mean(diff[:, None, None] * d_cp, axis=0)
        adam.step(
            model.parameters_dict(),
            grads,
            progress=progress,
            shots_in_batch=2 * eff_bs * shots_per_run,
        )
        if verbose and (st == 1 or st % 5 == 0 or st == n_steps):
            print(f"  [SPSA QPU] Batch {st:02d}/{n_steps:02d} | Shots={adam.shots_used:6d} | Wall={time.perf_counter()-t0:.1f}s", flush=True)

    return {"steps": n_steps, "shots_used": adam.shots_used, "wall_sec": time.perf_counter() - t0}


def train_psr_dna(
    model: QuantumIsingDenoiser,
    x_tr: np.ndarray,
    cond_tr: np.ndarray,
    max_shots: int,
    shots_per_run: int,
    lr_max: float = 0.08,
    lr_min: float = 0.012,
    clip_bound: float = 1.05,
) -> Dict[str, int]:
    """Equal-shot finite-difference Parameter-Shift baseline (`2P` circuits/step) + calibrator + chain-rule `cond_proj`."""
    n_p = model.n_qpu_params
    step_shots = 2 * n_p * shots_per_run
    n_steps = max(0, max_shots // max(1, step_shots))
    adam = StabilizedAdam(
        model.parameters_dict(),
        StabilizedAdamConfig(lr_max=lr_max, lr_min=lr_min, weight_decay=0.02, clip_bound=clip_bound),
    )
    eye = np.arange(n_p)

    for st in range(1, n_steps + 1):
        progress = (st - 1) / max(1, n_steps - 1)
        idx = int(model.rng.integers(0, len(x_tr)))
        tg = x_tr[idx]
        cb = cond_tr[idx]
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
            cond_batch=np.broadcast_to(cb, (2 * n_p, model.cond_dim)),
            theta_batch=np.concatenate([th_p, th_m], axis=0),
        )
        model.calibrator.update(tv, np.mean(z_all, axis=0), tg)
        g_t = model.calibrator.gain[tv]
        zp = np.clip(z_all[:n_p] / g_t, -1.0, 1.0)
        zm = np.clip(z_all[n_p:] / g_t, -1.0, 1.0)
        grad_th = -(np.mean((zp - tg[None, :]) ** 2, axis=1) - np.mean((zm - tg[None, :]) ** 2, axis=1)) / 0.50
        grads: Dict[str, np.ndarray] = {"theta": grad_th}
        if model.cond_proj is not None:
            grads["cond_proj"] = np.outer(grad_th[model.n_edges : model.n_edges + model.n_visible], cb)
        adam.step(model.parameters_dict(), grads, progress=progress, shots_in_batch=step_shots)

    return {"steps": n_steps, "shots_used": adam.shots_used}


def save_dna_checkpoint(
    path: str,
    model: QuantumIsingDenoiser,
    edges: List[Tuple[int, int]],
    ro_cal: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ro_off = ro_cal[0] if ro_cal is not None else np.zeros(model.n_qubits, dtype=np.float64)
    ro_sc = ro_cal[1] if ro_cal is not None else np.ones(model.n_qubits, dtype=np.float64)
    np.savez(
        path,
        theta=model.theta,
        cond_proj=model.cond_proj if model.cond_proj is not None else np.zeros((model.n_visible, 0)),
        calibrator_gain=model.calibrator.gain,
        cal_mode=np.array([model.calibrator.mode]),
        edges=np.array(edges, dtype=np.int64),
        n_qubits=np.array([model.n_visible], dtype=np.int64),
        n_layers=np.array([model.n_layers], dtype=np.int64),
        diffusion_steps=np.array([model.diffusion_steps], dtype=np.int64),
        ro_offset=ro_off,
        ro_scale=ro_sc,
    )


def load_dna_checkpoint(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at {path}")
    ck = np.load(path, allow_pickle=False)
    return {
        "theta": ck["theta"],
        "cond_proj": ck["cond_proj"],
        "calibrator_gain": ck["calibrator_gain"],
        "cal_mode": str(ck["cal_mode"][0]),
        "edges": [tuple(e) for e in ck["edges"].tolist()],
        "n_qubits": int(ck["n_qubits"][0]),
        "n_layers": int(ck["n_layers"][0]),
        "diffusion_steps": int(ck["diffusion_steps"][0]),
        "ro_offset": ck["ro_offset"],
        "ro_scale": ck["ro_scale"],
    }
