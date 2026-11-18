#!/usr/bin/env python3
"""
Checkpoint Reciprocity, Stepper Factor Isolation, and Schedule Control Suite:
  [Check 1] Trained L=2 checkpoint reciprocity (clean & NISQ) + trained-checkpoint 4k-shot SNR.
  [Check 2] StabilizedAdam factor isolation (Momentum-only LRs, RMSprop-only, full Adam)
            + active auto_beta_guard wraparound rescue (beta0=11.0).
  [Check 3] W_cond-only Exact-PSR, Hybrid noisy-eval & noise-dose Pareto, and L=4 Spin-Glass Exact-AD.

Outputs:
  ./analysis/reciprocity_stepper_results.json
  ./analysis/fig_reciprocity_stepper_checks.pdf / .png

Usage:
  python -u experiments/eval_reciprocity_and_stepper.py --quick
  python -u experiments/eval_reciprocity_and_stepper.py
  python -u experiments/eval_reciprocity_and_stepper.py --replot
"""

from __future__ import annotations

try:
    import _paths  # noqa: F401
except ImportError:
    from . import _paths  # noqa: F401

import argparse
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import benchmark_mnist as bm
import eval_capacity_bottleneck as pcd
import eval_spinglass_attribution as psp
import eval_conditioning_and_opt as prf
from qerd import (
    QERConfig,
    QERGradConfig,
    QERGradientEstimator,
    QEROptimizer,
    QERDTrainer,
    QuantumIsingDenoiser,
    StabilizedAdam,
    StabilizedAdamConfig,
)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _eval_reciprocity_at_weights(
    circ: bm._DiffIsingCircuit,
    th_np: np.ndarray,
    cp_np: Optional[np.ndarray],
    xt_t: torch.Tensor,
    t_t: torch.Tensor,
    cond_t: Optional[torch.Tensor],
    y_t: torch.Tensor,
    dev: torch.device,
    fault_masks: Optional[torch.Tensor] = None,
    p_readout: float = 0.0,
) -> Dict[str, Any]:
    """Compute Jacobian asymmetry Ratio and JVP/VJP/QER directional cosines."""
    nq = circ.n_qubits
    ne = len(circ.edges)
    bs = xt_t.shape[0]
    th_t = torch.tensor(th_np, dtype=torch.float32, device=dev, requires_grad=True)
    cp_t = None if cp_np is None else torch.tensor(cp_np, dtype=torch.float32, device=dev)

    dh = torch.zeros((bs, nq), dtype=torch.float32, device=dev, requires_grad=True)
    _, z_dh = circ.expectation(
        xt_t, t_t, cond_t, th_t, cp_t,
        nudge_target=dh, nudge_beta=1.0,
        fault_masks=fault_masks, p_readout=p_readout,
    )
    jac_rows = []
    for qi in range(nq):
        (g_qi,) = torch.autograd.grad(z_dh[:, qi].sum(), [dh], retain_graph=True)
        jac_rows.append(g_qi)
    jac = torch.stack(jac_rows, dim=1).detach()

    diff_f = torch.linalg.norm(jac - jac.transpose(1, 2), dim=(1, 2))
    norm_f = torch.linalg.norm(jac, dim=(1, 2)).clamp(min=1e-12)
    asym_ratio = float((diff_f / norm_f).mean().item())

    _, z0 = circ.expectation(
        xt_t, t_t, cond_t, th_t, cp_t,
        fault_masks=fault_masks, p_readout=p_readout,
    )
    (g_vjp_th,) = torch.autograd.grad(torch.sum(z0 * y_t, dim=1).mean(), [th_t])
    g_vjp_np = g_vjp_th.detach().cpu().numpy()

    with torch.no_grad():
        vjp_h = torch.bmm(jac.transpose(1, 2), y_t.unsqueeze(-1)).squeeze(-1).mean(dim=0).cpu().numpy()
        jvp_h = torch.bmm(jac, y_t.unsqueeze(-1)).squeeze(-1).mean(dim=0).cpu().numpy()

        out_beta = {}
        for beta in (0.45, 0.10):
            op, zp = circ.expectation(
                xt_t, t_t, cond_t, th_t, cp_t,
                nudge_target=y_t, nudge_beta=beta,
                fault_masks=fault_masks, p_readout=p_readout,
            )
            om, zm = circ.expectation(
                xt_t, t_t, cond_t, th_t, cp_t,
                nudge_target=y_t, nudge_beta=-beta,
                fault_masks=fault_masks, p_readout=p_readout,
            )
            gq_th = ((op - om) / (2.0 * beta)).mean(dim=0).cpu().numpy()
            gq_h = ((zp - zm) / (2.0 * beta)).mean(dim=0).cpu().numpy()
            out_beta[str(beta)] = {
                "cos_qer_vjp_h": _cos(gq_h, vjp_h),
                "cos_qer_jvp_h": _cos(gq_h, jvp_h),
                "cos_qer_vjp_zz": _cos(gq_th[:ne], g_vjp_np[:ne]),
                "cos_qer_vjp_z": _cos(gq_th[ne:], g_vjp_np[ne:]),
            }

    return {
        "asym_ratio": asym_ratio,
        "cos_jvp_vjp_h": _cos(jvp_h, vjp_h),
        "beta_045": out_beta["0.45"],
        "beta_010": out_beta["0.1"],
    }


def _sample_depol_masks(
    k_traj: int, n_layers: int, batch_size: int, n_qubits: int, p_depol: float, dev: torch.device, rng: np.random.Generator
) -> torch.Tensor:
    probs = np.array([1.0 - 0.75 * p_depol, 0.25 * p_depol, 0.25 * p_depol, 0.25 * p_depol], dtype=np.float64)
    masks_np = rng.choice(4, size=(k_traj, n_layers, batch_size, n_qubits), p=probs).astype(np.int64)
    return torch.tensor(masks_np, dtype=torch.long, device=dev)


def _measure_4k_snr_at_theta(
    m_den: QuantumIsingDenoiser,
    circ: bm._DiffIsingCircuit,
    th_ref: np.ndarray,
    xt: np.ndarray,
    x0: np.ndarray,
    t_arr: np.ndarray,
    dev: torch.device,
    n_mc: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Fixed 4,000-shot/step NISQ cosine SNR evaluated at an arbitrary checkpoint theta."""
    nq, nl, ne = m_den.n_qubits, m_den.n_layers, m_den.n_edges
    n_p = th_ref.size
    m_den.theta[:] = th_ref
    bs = xt.shape[0]
    xt_t = torch.tensor(xt, dtype=torch.float32, device=dev)
    x0_t = torch.tensor(x0, dtype=torch.float32, device=dev)
    t_t = torch.tensor(t_arr, dtype=torch.long, device=dev)
    th_t = torch.tensor(th_ref, dtype=torch.float32, device=dev, requires_grad=True)

    _, zf = circ.expectation(xt_t, t_t, None, th_t, None)
    (g_ec_t,) = torch.autograd.grad(torch.sum(zf * x0_t, dim=1).mean(), [th_t], retain_graph=True)
    (g_em_t,) = torch.autograd.grad(-0.5 * torch.sum((zf - x0_t) ** 2, dim=1).mean(), [th_t])
    g_ec = g_ec_t.detach().cpu().numpy()
    g_em = g_em_t.detach().cpu().numpy()

    sh_2run = max(1, 4000 // (2 * bs))
    sh_3run = max(1, 4000 // (3 * bs))
    sh_exact_psr = max(1, 4000 // (2 * nl * n_p))
    mold_gains = [2.0 * m_den.dt * m_den.temp_beta * ((l + 1) / nl) for l in range(nl)]

    c_qer_s, c_qer_r, c_spsa_s, c_spsa_m, c_epsr_s, c_epsr_m = [], [], [], [], [], []
    for mc in range(n_mc):
        ob2, _ = m_den.measure_observables_batch(
            np.concatenate([xt, xt]), np.concatenate([t_arr, t_arr]), shots=sh_2run,
            nudge_target_batch=np.concatenate([x0, x0]),
            nudge_beta_batch=np.concatenate([np.full(bs, 0.45), np.full(bs, -0.45)]),
        )
        c_qer_s.append(_cos(np.mean((ob2[:bs] - ob2[bs:]) / 0.90, axis=0), g_ec))

        _, zf_res = m_den.measure_observables_batch(xt, t_arr, shots=sh_3run)
        y_res = x0 - zf_res
        ob_r, _ = m_den.measure_observables_batch(
            np.concatenate([xt, xt]), np.concatenate([t_arr, t_arr]), shots=sh_3run,
            nudge_target_batch=np.concatenate([y_res, y_res]),
            nudge_beta_batch=np.concatenate([np.full(bs, 0.45), np.full(bs, -0.45)]),
        )
        c_qer_r.append(_cos(np.mean((ob_r[:bs] - ob_r[bs:]) / 0.90, axis=0), g_em))

        d_th = rng.choice([-1.0, 1.0], size=(bs, n_p))
        _, zs = m_den.measure_observables_batch(
            np.concatenate([xt, xt]), np.concatenate([t_arr, t_arr]), shots=sh_2run,
            theta_batch=np.concatenate([th_ref[None] + 0.18 * d_th, th_ref[None] - 0.18 * d_th]),
        )
        df_s = (-np.sum(zs[:bs] * x0, axis=1) - (-np.sum(zs[bs:] * x0, axis=1))) / 0.36
        df_m = (0.5 * np.sum((zs[:bs] - x0) ** 2, axis=1) - 0.5 * np.sum((zs[bs:] - x0) ** 2, axis=1)) / 0.36
        c_spsa_s.append(_cos(-np.mean(df_s[:, None] * d_th, axis=0), g_ec))
        c_spsa_m.append(_cos(-np.mean(df_m[:, None] * d_th, axis=0), g_em))

        sid = mc % bs
        in_pin = m_den.in_pinning_base + m_den.in_pinning_slope * (1.0 - float(t_arr[sid]) / m_den.diffusion_steps)
        h_b = th_ref[ne:] + in_pin * xt[sid]
        j_b = th_ref[:ne]
        h_stk = np.broadcast_to(h_b[None, None, :], (1 + 2 * nl * n_p, nl, nq)).copy()
        j_stk = np.broadcast_to(j_b[None, None, :], (1 + 2 * nl * n_p, nl, ne)).copy()
        ptr, meta_e = 1, []
        for l_i, mg in enumerate(mold_gains):
            ang = (0.5 * math.pi) / mg
            for ei in range(ne):
                j_stk[ptr, l_i, ei] += ang; j_stk[ptr + 1, l_i, ei] -= ang
                meta_e.append((ei, mg)); ptr += 2
            for qi in range(nq):
                h_stk[ptr, l_i, qi] += ang; h_stk[ptr + 1, l_i, qi] -= ang
                meta_e.append((ne + qi, mg)); ptr += 2
        z_ep = bm._sample_layer_perturbed_z(m_den, h_stk, j_stk, float(t_arr[sid]) / m_den.diffusion_steps, sh_exact_psr, dev)
        ge_s, ge_m = np.zeros(n_p), np.zeros(n_p)
        err_m = z_ep[0] - x0[sid]
        for idx_p, (th_i, mg) in enumerate(meta_e):
            dz = 0.5 * mg * (z_ep[1 + 2 * idx_p] - z_ep[2 + 2 * idx_p])
            ge_s[th_i] += float(np.dot(x0[sid], dz))
            ge_m[th_i] += -float(np.dot(err_m, dz))
        c_epsr_s.append(_cos(ge_s, g_ec))
        c_epsr_m.append(_cos(ge_m, g_em))

    return {
        "qer_surr_snr": float(np.mean(c_qer_s)),
        "qer_res_mse_snr": float(np.mean(c_qer_r)),
        "epsr_surr_snr": float(np.mean(c_epsr_s)),
        "epsr_mse_snr": float(np.mean(c_epsr_m)),
        "spsa_surr_snr": float(np.mean(c_spsa_s)),
        "spsa_mse_snr": float(np.mean(c_spsa_m)),
    }


def run_check1_real_checkpoints(
    z_tr: np.ndarray, y_tr: np.ndarray, dev: torch.device, tdev_str: str, quick: bool
) -> Dict[str, Any]:
    """Check 1: Reciprocity (clean + noisy depol/readout) & 4k-shot SNR on real trained checkpoints."""
    print("\n=== [Check 1] Real Trained-Checkpoint Reciprocity (Clean vs NISQ) & 4k-Shot SNR ===", flush=True)
    nq = 12
    edges = bm.build_ising_edges(nq, "all")
    c_tr = np.eye(10, dtype=np.float64)[y_tr]
    pools = [np.flatnonzero(y_tr == c) for c in range(10)]
    bs = 12 if quick else 16
    k_traj = 12 if quick else 32
    n_mc = 4 if quick else 12
    n_steps = 40 if quick else 200
    ckpt_steps = [0, 20, 40] if quick else [0, 50, 100, 200]

    # Train one real MNIST QuantumIsingDenoiser with QEROptimizer and record checkpoints
    m = QuantumIsingDenoiser(nq, cond_dim=10, n_hidden=0, topology="all", n_layers=2, p_depol=0.035, p_readout=0.04, init_gain=1.0, gain_ema=0.0, backend="torch", torch_device=tdev_str, seed=0)
    opt = QEROptimizer(m.parameters_dict(), QERConfig(clip_bound=0.90))
    ckpts: Dict[int, Dict[str, np.ndarray]] = {0: {"theta": m.theta.copy(), "cond_proj": m.cond_proj.copy()}}
    for st in range(1, n_steps + 1):
        prog = (st - 1) / max(1, n_steps - 1)
        ids = bm._sample_stratified_ids(m.rng, pools, 10)
        tb = (np.arange(10) % 4 + 1).astype(np.int64); m.rng.shuffle(tb)
        tg, cb = z_tr[ids], c_tr[ids]
        xt = np.clip(np.sqrt(m.alpha_bar[tb][:, None]) * tg + np.sqrt(1.0 - m.alpha_bar[tb][:, None]) * m.rng.normal(0.0, 0.70, size=tg.shape), -1.0, 1.0)
        ob2, zb2 = m.measure_observables_batch(
            np.concatenate([xt, xt]), np.concatenate([tb, tb]), shots=20,
            cond_batch=np.concatenate([cb, cb]), nudge_target_batch=np.concatenate([tg, tg]),
            nudge_beta_batch=np.concatenate([np.full(10, opt.beta), np.full(10, -opt.beta)]),
        )
        opt.step_contrast(m.parameters_dict(), ob2[:10], ob2[10:], zb2[:10], zb2[10:], cb, tg, progress=prog, shots_in_batch=400)
        if st in ckpt_steps:
            ckpts[st] = {"theta": m.theta.copy(), "cond_proj": m.cond_proj.copy()}

    # Fixed evaluation probe batch (stratified targets & noisy inputs)
    rng_p = np.random.default_rng(77)
    p_ids = bm._sample_stratified_ids(rng_p, pools, bs)
    p_x0 = z_tr[p_ids].astype(np.float32)
    p_cb = c_tr[p_ids].astype(np.float32)
    p_t = ((np.arange(bs) % 4) + 1).astype(np.int64)
    p_ab = m.alpha_bar[p_t][:, None]
    p_xt = np.clip(np.sqrt(p_ab) * p_x0 + np.sqrt(1.0 - p_ab) * rng_p.normal(0.0, 0.70, size=p_x0.shape), -1.0, 1.0).astype(np.float32)

    xt_t = torch.tensor(p_xt, dtype=torch.float32, device=dev)
    x0_t = torch.tensor(p_x0, dtype=torch.float32, device=dev)
    cb_t = torch.tensor(p_cb, dtype=torch.float32, device=dev)
    t_t = torch.tensor(p_t, dtype=torch.long, device=dev)
    circ_cond = bm._DiffIsingCircuit(nq, cond_dim=10, edges=edges, n_layers=2, device=dev)
    fm = _sample_depol_masks(k_traj, 2, bs, nq, 0.035, dev, rng_p)

    mnist_rows = []
    for st in ckpt_steps:
        th_c, cp_c = ckpts[st]["theta"], ckpts[st]["cond_proj"]
        clean_m = _eval_reciprocity_at_weights(circ_cond, th_c, cp_c, xt_t, t_t, cb_t, x0_t, dev, fault_masks=None, p_readout=0.0)
        noisy_m = _eval_reciprocity_at_weights(circ_cond, th_c, cp_c, xt_t, t_t, cb_t, x0_t, dev, fault_masks=fm, p_readout=0.04)
        mnist_rows.append({"step": st, "max_abs_th": float(np.max(np.abs(th_c))), "clean": clean_m, "nisq": noisy_m})
        print(
            f"  [MNIST L=2 ckpt step={st:3d}] Clean Asym={clean_m['asym_ratio']:.3f} ZZ45={clean_m['beta_045']['cos_qer_vjp_zz']:+.3f} | "
            f"NISQ Asym={noisy_m['asym_ratio']:.3f} JVP/VJP={noisy_m['cos_jvp_vjp_h']:+.3f} ZZ45={noisy_m['beta_045']['cos_qer_vjp_zz']:+.3f}",
            flush=True,
        )

    # Also evaluate 4k-shot SNR at real trained spin-glass L=2 checkpoint vs init
    modes, _, _, _ = psp._build_spinglass_problem()
    sg_l2 = QuantumIsingDenoiser(nq, cond_dim=0, n_hidden=0, topology="all", n_layers=2, p_depol=0.035, p_readout=0.04, init_gain=0.75, gain_ema=0.10, backend="torch", torch_device=tdev_str, seed=0)
    th_sg_init = sg_l2.theta.copy()
    QERDTrainer(sg_l2, QEROptimizer(sg_l2.parameters_dict(), QERConfig(clip_bound=0.90))).train(modes, None, max_shots=(12_000 if quick else 40_000), shots_per_run=20, batch_size=8, eval_every_batches=999999, verbose=False)
    th_sg_trained = sg_l2.theta.copy()

    # Use residual MSE QER checkpoint from Study 4 setup at L=2 on trained weights
    m_snr = QuantumIsingDenoiser(nq, cond_dim=0, n_hidden=0, topology="all", n_layers=2, p_depol=0.035, p_readout=0.04, init_gain=1.0, gain_ema=0.0, backend="torch", torch_device=tdev_str, seed=0)
    circ_snr = bm._DiffIsingCircuit(nq, cond_dim=0, edges=edges, n_layers=2, device=dev)
    sg_bs = 10
    sg_ids = rng_p.integers(0, len(modes), size=sg_bs)
    sg_x0 = modes[sg_ids].astype(np.float32)
    sg_t = ((np.arange(sg_bs) % 4) + 1).astype(np.int64)
    sg_ab = m_snr.alpha_bar[sg_t][:, None]
    sg_xt = np.clip(np.sqrt(sg_ab) * sg_x0 + np.sqrt(1.0 - sg_ab) * rng_p.normal(0.0, 0.70, size=sg_x0.shape), -1.0, 1.0).astype(np.float32)

    snr_init = _measure_4k_snr_at_theta(m_snr, circ_snr, th_sg_init, sg_xt, sg_x0, sg_t, dev, n_mc, rng_p)
    snr_trained = _measure_4k_snr_at_theta(m_snr, circ_snr, th_sg_trained, sg_xt, sg_x0, sg_t, dev, n_mc, rng_p)
    # Also evaluate unconditional trained MNIST QPU theta in 4k-shot SNR
    snr_mnist_ckpt = _measure_4k_snr_at_theta(m_snr, circ_snr, ckpts[ckpt_steps[-1]]["theta"], p_xt[:10], p_x0[:10], p_t[:10], dev, n_mc, rng_p)
    print(f"  [4k-shot SNR] SpinGlass Trained: QER_Surr={snr_trained['qer_surr_snr']:+.3f} QER_Res={snr_trained['qer_res_mse_snr']:+.3f} ePSR={snr_trained['epsr_surr_snr']:+.3f} SPSA={snr_trained['spsa_surr_snr']:+.3f}", flush=True)

    return {
        "mnist_reciprocity_ckpts": mnist_rows,
        "snr_4k": {"sg_init": snr_init, "sg_trained": snr_trained, "mnist_ckpt": snr_mnist_ckpt},
    }


def _train_custom_stepper(
    z_tr: np.ndarray, y_tr: np.ndarray, z_te: np.ndarray, x_te: np.ndarray, y_te: np.ndarray,
    ae: nn.Module, clf: bm.EvalCNN, dev: torch.device, tdev_str: str,
    seeds: List[int], max_shots: int, beta1: float, beta2: float, lr_max: float,
    beta0: float = 0.45, auto_beta_guard: bool = True,
) -> Dict[str, Any]:
    """Run decoupled QERGradientEstimator + parameterized (beta1, beta2) cosine/decay/clip stepper."""
    eye_c = np.eye(10, dtype=np.float64)
    c_tr = eye_c[y_tr]
    ev_lbl = np.repeat(np.arange(10, dtype=np.int64), 40 if max_shots < 50_000 else 100)
    ev_c = eye_c[ev_lbl]
    pools = [np.flatnonzero(y_tr == c) for c in range(10)]

    seed_mets: List[Dict[str, float]] = []
    beta_traces: List[List[float]] = []
    for sd in seeds:
        m = QuantumIsingDenoiser(12, cond_dim=10, n_hidden=0, topology="all", n_layers=2, p_depol=0.035, p_readout=0.04, init_gain=1.0, gain_ema=0.0, backend="torch", torch_device=tdev_str, seed=sd)
        est = QERGradientEstimator(QERGradConfig(beta=beta0, beta_min=0.15, auto_beta_guard=auto_beta_guard))
        params = m.parameters_dict()
        m_buf = {k: np.zeros_like(v, dtype=np.float64) for k, v in params.items()}
        v_buf = {k: np.zeros_like(v, dtype=np.float64) for k, v in params.items()}
        tot_b = max(1, max_shots // 400)
        b_hist = [float(est.beta)]
        shrinks = 0
        for st in range(1, tot_b + 1):
            prog = (st - 1) / max(1, tot_b - 1)
            lr_k = 0.015 + 0.5 * max(0.0, lr_max - 0.015) * (1.0 + math.cos(math.pi * prog))
            eff_decay = 1.0 - 0.01 * (lr_k / max(lr_max, 1e-8))
            ids = bm._sample_stratified_ids(m.rng, pools, 10)
            tb = (np.arange(10) % 4 + 1).astype(np.int64); m.rng.shuffle(tb)
            tg, cb = z_tr[ids], c_tr[ids]
            ab_s = m.alpha_bar[tb][:, None]
            xt = np.clip(np.sqrt(ab_s) * tg + np.sqrt(1.0 - ab_s) * m.rng.normal(0.0, 0.70, size=tg.shape), -1.0, 1.0)
            prev_b = est.beta
            ob2, zb2 = m.measure_observables_batch(
                np.concatenate([xt, xt]), np.concatenate([tb, tb]), shots=20,
                cond_batch=np.concatenate([cb, cb]), nudge_target_batch=np.concatenate([tg, tg]),
                nudge_beta_batch=np.concatenate([np.full(10, est.beta), np.full(10, -est.beta)]),
            )
            grads, _ = est.estimate(params, ob2[:10], ob2[10:], z_plus_list=zb2[:10], z_minus_list=zb2[10:], conds=cb, nudge_dirs=tg)
            if est.beta < prev_b:
                shrinks += 1
            b_hist.append(float(est.beta))
            for name, g in grads.items():
                if beta1 > 0.0 and beta2 > 0.0:
                    m_buf[name][:] = beta1 * m_buf[name] + (1.0 - beta1) * g
                    v_buf[name][:] = beta2 * v_buf[name] + (1.0 - beta2) * (g ** 2)
                    m_hat = m_buf[name] / (1.0 - beta1 ** st)
                    v_hat = v_buf[name] / (1.0 - beta2 ** st)
                    upd = lr_k * m_hat / (np.sqrt(v_hat) + 1e-6)
                elif beta1 > 0.0 and beta2 == 0.0:
                    m_buf[name][:] = beta1 * m_buf[name] + (1.0 - beta1) * g
                    upd = lr_k * (m_buf[name] / (1.0 - beta1 ** st))
                elif beta1 == 0.0 and beta2 > 0.0:
                    v_buf[name][:] = beta2 * v_buf[name] + (1.0 - beta2) * (g ** 2)
                    v_hat = v_buf[name] / (1.0 - beta2 ** st)
                    upd = lr_k * g / (np.sqrt(v_hat) + 1e-6)
                else:
                    upd = lr_k * g
                params[name][:] = np.clip(eff_decay * params[name] + upd, -0.90, 0.90)

        met, _ = bm.evaluate_generated(f"b1_{beta1}_b2_{beta2}_lr_{lr_max}", m.sample_batch(conds=ev_c, shots_per_step=160, step_noise_scale=0.35), ev_lbl, ae, clf, z_te, x_te, y_te, dev)
        met["final_beta"] = float(est.beta)
        met["guard_shrinks"] = float(shrinks)
        seed_mets.append(met)
        beta_traces.append(b_hist)
    return {"aggregate": prf._agg_dicts(seed_mets), "beta_trace_seed0": beta_traces[0]}


def run_check2_adam_and_guard(
    z_tr: np.ndarray, y_tr: np.ndarray, z_te: np.ndarray, x_te: np.ndarray, y_te: np.ndarray,
    ae: nn.Module, clf: bm.EvalCNN, dev: torch.device, tdev_str: str,
    seeds: List[int], max_shots: int, quick: bool,
) -> Dict[str, Any]:
    """Check 2: Decouple beta1 vs beta2 & test active auto_beta_guard rescue at wraparound beta0."""
    print("\n=== [Check 2] StabilizedAdam Factor Isolation (beta1 vs beta2) & auto_beta_guard Rescue ===", flush=True)
    mom_lrs = [0.08, 0.25] if quick else [0.04, 0.08, 0.12, 0.25, 0.50]
    rms_lrs = [0.12] if quick else [0.06, 0.12]

    configs: List[Tuple[str, Dict[str, Any]]] = [("full_adam_0.12", dict(b1=0.75, b2=0.98, lr=0.12, b0=0.45, guard=True))]
    for lr in mom_lrs:
        configs.append((f"mom_only_lr{lr}", dict(b1=0.75, b2=0.0, lr=lr, b0=0.45, guard=True)))
    for lr in rms_lrs:
        configs.append((f"rmsprop_only_lr{lr}", dict(b1=0.0, b2=0.98, lr=lr, b0=0.45, guard=True)))
    configs.append(("guard_on_b110", dict(b1=0.75, b2=0.98, lr=0.12, b0=11.0, guard=True)))
    configs.append(("guard_off_b110", dict(b1=0.75, b2=0.98, lr=0.12, b0=11.0, guard=False)))

    results: Dict[str, Any] = {}
    for name, kw in configs:
        r = _train_custom_stepper(
            z_tr, y_tr, z_te, x_te, y_te, ae, clf, dev, tdev_str,
            seeds=seeds, max_shots=max_shots, beta1=kw["b1"], beta2=kw["b2"], lr_max=kw["lr"],
            beta0=kw["b0"], auto_beta_guard=kw["guard"],
        )
        results[name] = r
        ag = r["aggregate"]
        print(
            f"  [{name:18s}] Acc={ag['clf_acc']['mean']*100:5.1f}% | FeatFID={ag['feature_fid']['mean']:6.1f} | "
            f"FinalBeta={ag['final_beta']['mean']:.2f} | Shrinks={ag['guard_shrinks']['mean']:.1f}",
            flush=True,
        )
    return results


def _train_wcond_only_exact_psr(
    z_tr: np.ndarray, y_tr: np.ndarray, z_te: np.ndarray, x_te: np.ndarray, y_te: np.ndarray,
    ae: nn.Module, clf: bm.EvalCNN, dev: torch.device, tdev_str: str,
    seeds: List[int], max_shots: int,
) -> Dict[str, Any]:
    """Fair active W_cond-only Exact-PSR baseline (theta=0, 12 active cols -> 49 layered queries/sample)."""
    eye_c = np.eye(10, dtype=np.float64)
    c_tr = eye_c[y_tr]
    ev_lbl = np.repeat(np.arange(10, dtype=np.int64), 40 if max_shots < 50_000 else 100)
    ev_c = eye_c[ev_lbl]
    pools = [np.flatnonzero(y_tr == c) for c in range(10)]
    rows = []
    n_act_q = 12
    for sd in seeds:
        m = QuantumIsingDenoiser(12, cond_dim=10, n_hidden=0, topology="none", n_layers=2, p_depol=0.035, p_readout=0.04, init_gain=1.0, gain_ema=0.0, backend="torch", torch_device=tdev_str, seed=sd)
        m_cp = np.zeros_like(m.cond_proj)
        v_cp = np.zeros_like(m.cond_proj)
        nl = m.n_layers
        mold_gains = [2.0 * m.dt * m.temp_beta * ((l + 1) / nl) for l in range(nl)]
        n_queries = 1 + 2 * nl * n_act_q  # 49 queries/sample
        sh_q = 8
        step_sh = n_queries * sh_q  # 392 shots per single-sample Exact-PSR step (~204 steps in 80k)
        tot_steps = max(1, max_shots // step_sh)
        j_stk = np.zeros((n_queries, nl, 0), dtype=np.float64)
        for st in range(1, tot_steps + 1):
            cls_i = st % 10
            ts = (st % 4) + 1
            tg = z_tr[int(m.rng.choice(pools[cls_i]))]
            xt = np.clip(math.sqrt(m.alpha_bar[ts]) * tg + math.sqrt(1.0 - m.alpha_bar[ts]) * m.rng.normal(0.0, 0.70, size=12), -1.0, 1.0)
            in_pin = m.in_pinning_base + m.in_pinning_slope * (1.0 - float(ts) / m.diffusion_steps)
            h_base = in_pin * xt + m.cond_proj[:, cls_i]
            h_stk = np.broadcast_to(h_base[None, None, :], (n_queries, nl, 12)).copy()
            ptr, meta = 1, []
            for l_i, mg in enumerate(mold_gains):
                ang = (0.5 * math.pi) / mg
                for qi in range(n_act_q):
                    h_stk[ptr, l_i, qi] += ang; h_stk[ptr + 1, l_i, qi] -= ang
                    meta.append((qi, mg)); ptr += 2
            z_all = bm._sample_layer_perturbed_z(m, h_stk, j_stk, float(ts) / m.diffusion_steps, sh_q, dev)
            err = (2.0 / 12.0) * (z_all[0] - tg)
            g_col = np.zeros(12, dtype=np.float64)
            for idx_p, (qi, mg) in enumerate(meta):
                dz = 0.5 * mg * (z_all[1 + 2 * idx_p] - z_all[2 + 2 * idx_p])
                g_col[qi] += -float(np.dot(err, dz))
            g_cp = np.zeros_like(m.cond_proj)
            g_cp[:, cls_i] = g_col
            lrk = 0.015 + 0.5 * (0.12 - 0.015) * (1.0 + math.cos(math.pi * (st - 1) / max(1, tot_steps - 1)))
            m_cp[:] = 0.75 * m_cp + 0.25 * g_cp
            v_cp[:] = 0.98 * v_cp + 0.02 * (g_cp ** 2)
            upd = lrk * (m_cp / (1.0 - 0.75 ** st)) / (np.sqrt(v_cp / (1.0 - 0.98 ** st)) + 1e-6)
            m.cond_proj[:] = np.clip(m.cond_proj + upd, -0.90, 0.90)
        met, _ = bm.evaluate_generated("wcond_exact_psr", m.sample_batch(conds=ev_c, shots_per_step=160, step_noise_scale=0.35), ev_lbl, ae, clf, z_te, x_te, y_te, dev)
        met["steps_executed"] = float(tot_steps)
        rows.append(met)
    return prf._agg_dicts(rows)


def _sample_exact_ad_through_nisq(
    m_ad: prf.HybridResMLPExactAD,
    cond_np: np.ndarray,
    dev: torch.device,
    tdev_str: str,
    shots: int,
    step_noise_scale: float,
    seed: int,
) -> np.ndarray:
    """Evaluate trained HybridResMLPExactAD weights through noisy 24-shot NISQ QPU at sampling time."""
    qpu_eval = QuantumIsingDenoiser(
        12, cond_dim=0, n_hidden=0, topology="all", n_layers=2,
        p_depol=0.035, p_readout=0.04, init_gain=m_ad.gain, gain_ema=0.0,
        backend="torch", torch_device=tdev_str, seed=seed,
    )
    qpu_eval.theta[:] = m_ad.qpu_theta.detach().cpu().numpy().astype(np.float64)
    rng = np.random.default_rng(seed + 100)
    n = cond_np.shape[0]
    x_t = np.clip(rng.normal(0.0, 0.90, size=(n, 12)), -1.0, 1.0).astype(np.float32)
    c_t = torch.from_numpy(cond_np.astype(np.float32)).to(dev)
    with torch.no_grad():
        for t in range(4, 0, -1):
            h_cond = m_ad._trunk(torch.from_numpy(x_t).to(dev), t, c_t).cpu().numpy().astype(np.float64)
            _, z_raw = qpu_eval.measure_observables_batch(
                x_t.astype(np.float64), np.full(n, t, dtype=np.int64), shots=shots,
                nudge_target_batch=h_cond, nudge_beta_batch=np.ones(n, dtype=np.float64),
            )
            x0_hat = np.clip(z_raw / m_ad.gain, -1.0, 1.0).astype(np.float32)
            if t > 1:
                a_next = m_ad.alpha_bar[t - 1]
                eps = rng.normal(0.0, step_noise_scale, size=(n, 12)).astype(np.float32)
                x_t = np.clip(math.sqrt(a_next) * x0_hat + math.sqrt(1.0 - a_next) * eps, -1.0, 1.0)
            else:
                x_t = x0_hat
    return x_t.astype(np.float64)


def run_check3_missing_controls(
    z_tr: np.ndarray, y_tr: np.ndarray, z_te: np.ndarray, x_te: np.ndarray, y_te: np.ndarray,
    ae: nn.Module, clf: bm.EvalCNN, dev: torch.device, tdev_str: str,
    seeds: List[int], max_shots: int, quick: bool,
) -> Dict[str, Any]:
    """Check 3: W_cond-only Exact-PSR, Hybrid noisy-eval & noise-dose Pareto, and Spin-Glass Exact-AD."""
    print("\n=== [Check 3] Missing Controls: W_cond Exact-PSR, Hybrid Noise Pareto, Spin-Glass Exact-AD ===", flush=True)

    # 3a) W_cond-only Exact-PSR baseline (49 queries/sample, 0 QPU weights)
    wcond_epsr = _train_wcond_only_exact_psr(z_tr, y_tr, z_te, x_te, y_te, ae, clf, dev, tdev_str, seeds, max_shots)
    print(f"  [3a W_cond Exact-PSR] Acc={wcond_epsr['clf_acc']['mean']*100:.1f}% | FeatFID={wcond_epsr['feature_fid']['mean']:.1f} | Steps={wcond_epsr['steps_executed']['mean']:.0f}", flush=True)

    # 3b) Hybrid noisy-eval Exact-AD & Classical noise-dose Pareto sweep (h=48; h in [48, 128] full)
    eye_c = np.eye(10, dtype=np.float64)
    c_tr = eye_c[y_tr]
    ev_lbl = np.repeat(np.arange(10, dtype=np.int64), 40 if quick else 200)
    ev_c = eye_c[ev_lbl]
    pools = [np.flatnonzero(y_tr == c) for c in range(10)]
    cached_real = {sd: pcd.precompute_real_knn_ref(clf, x_te, dev, seed=sd) for sd in seeds}
    edges_all = bm.build_ising_edges(12, "all")
    n_steps_hyb = 60 if quick else 300
    bs_hyb = 16 if quick else 32
    widths = [48] if quick else [48, 128]
    noise_doses = [24, 6] if quick else [24, 12, 6, 4]

    def _fit_hybrid(model: prf._HybridResBase, rng_s: np.random.Generator) -> None:
        opt = torch.optim.Adam(model.parameters(), lr=3.5e-3)
        ab = model.alpha_bar
        model.train()
        for st in range(1, n_steps_hyb + 1):
            lr_k = 0.08 * 3.5e-3 + 0.92 * 3.5e-3 * 0.5 * (1.0 + math.cos(math.pi * (st - 1) / max(1, n_steps_hyb - 1)))
            for pg in opt.param_groups:
                pg["lr"] = lr_k
            opt.zero_grad()
            for ts in ((1, 3) if (st % 2 == 1) else (2, 4)):
                ids = bm._sample_stratified_ids(rng_s, pools, bs_hyb)
                tg = torch.from_numpy(z_tr[ids].astype(np.float32)).to(dev)
                cb = torch.from_numpy(c_tr[ids].astype(np.float32)).to(dev)
                xt = torch.clamp(math.sqrt(ab[ts]) * tg + math.sqrt(1.0 - ab[ts]) * torch.from_numpy(rng_s.normal(0.0, 0.70, size=tg.shape).astype(np.float32)).to(dev), -1.0, 1.0)
                loss = 0.5 * F.mse_loss(model.forward_t(xt, ts, cb), tg)
                loss.backward()
            opt.step()
        model.eval()

    hyb_controls: Dict[str, Any] = {}
    for hid in widths:
        exact_clean_eval: Dict[float, List[Dict[str, float]]] = {0.35: [], 0.85: []}
        exact_nisq_eval: Dict[float, List[Dict[str, float]]] = {0.35: [], 0.85: []}
        cl_dose: Dict[Tuple[int, float], List[Dict[str, float]]] = {(sh, sn): [] for sh in noise_doses for sn in (0.35, 0.85)}

        for sd in seeds:
            torch.manual_seed(sd)
            m_ad = prf.HybridResMLPExactAD(edges_all, hidden=hid, gain=0.70, seed=sd).to(dev)
            _fit_hybrid(m_ad, np.random.default_rng(sd))
            for sn in (0.35, 0.85):
                zg_cl = m_ad.sample_batch(ev_c, dev, seed=100 + sd, step_noise_scale=sn)
                mc, _ = bm.evaluate_generated(f"ExactAD-Clean-h{hid}-sn{int(sn*100)}", zg_cl, ev_lbl, ae, clf, z_te, x_te, y_te, dev)
                mc.update(pcd.measure_knn_radius_geometry(ae, clf, zg_cl, x_te, dev, seed=sd, cached_real=cached_real[sd]))
                exact_clean_eval[sn].append(mc)

                zg_nq = _sample_exact_ad_through_nisq(m_ad, ev_c, dev, tdev_str, shots=24, step_noise_scale=sn, seed=100 + sd)
                mn, _ = bm.evaluate_generated(f"ExactAD-NISQEval-h{hid}-sn{int(sn*100)}", zg_nq, ev_lbl, ae, clf, z_te, x_te, y_te, dev)
                mn.update(pcd.measure_knn_radius_geometry(ae, clf, zg_nq, x_te, dev, seed=sd, cached_real=cached_real[sd]))
                exact_nisq_eval[sn].append(mn)

            for sh_d in noise_doses:
                torch.manual_seed(sd)
                m_cl = prf.ClassicalResMLPNoiseMatched(hidden=hid, shots=sh_d, gain=0.70).to(dev)
                _fit_hybrid(m_cl, np.random.default_rng(sd))
                for sn in (0.35, 0.85):
                    zg_c = m_cl.sample_batch(ev_c, dev, seed=100 + sd, step_noise_scale=sn)
                    md, _ = bm.evaluate_generated(f"ClNoise{sh_d}-h{hid}-sn{int(sn*100)}", zg_c, ev_lbl, ae, clf, z_te, x_te, y_te, dev)
                    md.update(pcd.measure_knn_radius_geometry(ae, clf, zg_c, x_te, dev, seed=sd, cached_real=cached_real[sd]))
                    cl_dose[(sh_d, sn)].append(md)

        hyb_controls[f"h{hid}"] = {
            "exact_ad_clean_eval": {str(sn): prf._agg_dicts(exact_clean_eval[sn]) for sn in (0.35, 0.85)},
            "exact_ad_nisq24_eval": {str(sn): prf._agg_dicts(exact_nisq_eval[sn]) for sn in (0.35, 0.85)},
            "cl_noise_dose": {f"sh{sh}_sn{int(sn*100)}": prf._agg_dicts(cl_dose[(sh, sn)]) for sh in noise_doses for sn in (0.35, 0.85)},
        }
        e_nq85 = hyb_controls[f"h{hid}"]["exact_ad_nisq24_eval"]["0.85"]
        e_cl85 = hyb_controls[f"h{hid}"]["exact_ad_clean_eval"]["0.85"]
        c_hi85 = hyb_controls[f"h{hid}"]["cl_noise_dose"][f"sh{noise_doses[-1]}_sn85"]
        print(
            f"  [3b Hybrid h={hid} sn=0.85] ExactAD Clean Rec={e_cl85['recall']['mean']:.3f} -> NISQ24 Eval Rec={e_nq85['recall']['mean']:.3f} Acc={e_nq85['clf_acc']['mean']*100:.1f}% | "
            f"Cl-{noise_doses[-1]}sh Rec={c_hi85['recall']['mean']:.3f} Acc={c_hi85['clf_acc']['mean']*100:.1f}%",
            flush=True,
        )

    # 3c) Spin-Glass L=4 Exact-AD initialization & LR convergence diagnosis
    modes, valid_set, intra_pairs, c_ref = psp._build_spinglass_problem()
    edges_sg = bm.build_ising_edges((3, 4), topology="all")
    sg_diag: Dict[str, Any] = {}
    sg_steps_base = 50 if quick else 200
    sg_steps_long = 80 if quick else 400
    for tag, init_std, use_cos, n_st in [
        ("exact_ad_zero_const", 0.0, False, sg_steps_base),
        ("exact_ad_rand08_cos200", 0.08, True, sg_steps_base),
        ("exact_ad_rand08_cos400", 0.08, True, sg_steps_long),
    ]:
        tag_rows = []
        for sd in seeds:
            torch.manual_seed(sd)
            rng_sg = np.random.default_rng(sd + 11)
            circ_sg = bm._DiffIsingCircuit(12, cond_dim=0, edges=edges_sg, n_hidden=0, n_layers=4, device=dev)
            if init_std > 0.0:
                th = (init_std * torch.randn(len(edges_sg) + 12, dtype=torch.float32, device=dev)).detach().requires_grad_(True)
            else:
                th = torch.zeros(len(edges_sg) + 12, dtype=torch.float32, device=dev, requires_grad=True)
            opt = torch.optim.Adam([th], lr=(0.12 if use_cos else 0.08))
            ab = bm.make_alpha_bar(4, "cosine")
            x_all = torch.tensor(modes, dtype=torch.float32, device=dev)
            losses = []
            for st_i in range(1, n_st + 1):
                if use_cos:
                    lr_c = 0.015 + 0.5 * (0.12 - 0.015) * (1.0 + math.cos(math.pi * (st_i - 1) / max(1, n_st - 1)))
                    for pg in opt.param_groups:
                        pg["lr"] = lr_c
                ids = rng_sg.integers(0, len(modes), size=32)
                tc = rng_sg.integers(1, 5, size=32)
                tg = x_all[ids]
                a = torch.tensor(ab[tc], dtype=torch.float32, device=dev)[:, None]
                xt = torch.clamp(torch.sqrt(a) * tg + torch.sqrt(1.0 - a) * torch.tensor(rng_sg.normal(0.0, 0.70, size=(32, 12)), dtype=torch.float32, device=dev), -1.0, 1.0)
                _, z = circ_sg.expectation(xt, torch.tensor(tc, dtype=torch.long, device=dev), None, th, None)
                loss = F.mse_loss(z, tg)
                opt.zero_grad(); loss.backward(); opt.step()
                with torch.no_grad():
                    th.clamp_(-2.0, 2.0)
                losses.append(float(loss.item()))

            with torch.no_grad():
                xt = np.clip(rng_sg.normal(0.0, 0.90, size=(320, 12)), -1.0, 1.0)
                for ts in range(4, 0, -1):
                    _, z = circ_sg.expectation(torch.tensor(xt, dtype=torch.float32, device=dev), torch.full((320,), ts, dtype=torch.long, device=dev), None, th, None)
                    x0_h = z.cpu().numpy()
                    if ts > 1:
                        xt = np.clip(math.sqrt(ab[ts - 1]) * x0_h + math.sqrt(1.0 - ab[ts - 1]) * rng_sg.normal(0.0, 0.30, x0_h.shape), -1.0, 1.0)
                    else:
                        xt = x0_h
            met = psp._eval_spin_samples(xt, valid_set, intra_pairs, c_ref, len(modes))
            met["final_loss"] = float(np.mean(losses[-10:]))
            tag_rows.append(met)
        ag = prf._agg_dicts(tag_rows)
        sg_diag[tag] = ag
        print(
            f"  [3c SpinGlass {tag:22s}] Loss={ag['final_loss']['mean']:.4f} | Corr={ag['intra_corr']['mean']:+.3f} | "
            f"Acc={ag['pattern_acc']['mean']*100:.1f}% | ModeCov={ag['mode_cov']['mean']*100:.1f}%",
            flush=True,
        )

    return {"wcond_exact_psr": wcond_epsr, "hybrid_controls": hyb_controls, "spinglass_exact_ad": sg_diag}


def make_figure(res: Dict[str, Any], out_dir: str) -> None:
    """Plot summary 3-panel figure covering trained reciprocity, stepper isolation, and spin-glass schedules."""
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8.6, "axes.titlesize": 9.6,
        "axes.titleweight": "bold", "axes.labelsize": 8.6, "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.6, "legend.fontsize": 7.3,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.3))

    # Panel (a): Trained L=2 Checkpoint Clean vs NISQ Reciprocity
    ax = axes[0]
    ck = res["check1"]["mnist_reciprocity_ckpts"]
    xs = [r["step"] for r in ck]
    ax.plot(xs, [r["clean"]["beta_045"]["cos_qer_vjp_zz"] for r in ck], "o-", color="#0072B2", lw=2.0, label="Clean $\\cos(\\mathrm{QER}, \\mathrm{VJP})_{ZZ}$")
    ax.plot(xs, [r["nisq"]["beta_045"]["cos_qer_vjp_zz"] for r in ck], "s--", color="#D55E00", lw=2.0, label="NISQ (3.5%/4%) $ZZ$")
    ax.plot(xs, [r["nisq"]["asym_ratio"] for r in ck], "^:", color="#009E73", lw=1.8, label="NISQ Asym. Ratio")
    ax.set_xlabel("Training Step $t$")
    ax.set_ylabel("Cosine / Asymmetry Ratio")
    ax.set_title("(a) Trained $L=2$ Checkpoints: Clean vs. NISQ")
    ax.set_ylim(-0.05, 1.08)
    ax.grid(True, ls="--", alpha=0.25)
    ax.legend(frameon=False, loc="best")

    # Panel (b): StabilizedAdam Factor Isolation + auto_beta_guard Rescue
    ax = axes[1]
    c2 = res["check2"]
    order = [k for k in c2.keys()]
    pretty_labels = {
        "full_adam_0.12": "Adam (0.12)",
        "mom_only_lr0.04": "Mom (0.04)",
        "mom_only_lr0.08": "Mom (0.08)",
        "mom_only_lr0.12": "Mom (0.12)",
        "mom_only_lr0.25": "Mom (0.25)",
        "mom_only_lr0.5": "Mom (0.50)",
        "rmsprop_only_lr0.06": "RMSprop (0.06)",
        "rmsprop_only_lr0.12": "RMSprop (0.12)",
        "guard_on_b110": "Guard On ($\\beta_0{=}11$)",
        "guard_off_b110": "Guard Off ($\\beta_0{=}11$)",
    }
    vals = [c2[k]["aggregate"]["clf_acc"]["mean"] * 100.0 for k in order]
    stds = [c2[k]["aggregate"]["clf_acc"]["std"] * 100.0 for k in order]
    cols = ["#0072B2" if "full" in k or "guard_on" in k else ("#CC79A7" if "guard_off" in k else "#E69F00") for k in order]
    xp = np.arange(len(order))
    ax.bar(xp, vals, yerr=stds, capsize=3.0, color=cols, edgecolor="#222222", lw=0.7)
    ax.set_xticks(xp)
    ax.set_xticklabels([pretty_labels.get(k, k) for k in order], rotation=32, ha="right", fontsize=7.2)
    ax.set_ylabel("ClfAcc (%) $\\uparrow$")
    ax.set_title("(b) Stepper Factor Isolation & $\\beta$-Guard")
    ax.set_ylim(0, 115)
    ax.grid(axis="y", ls="--", alpha=0.25)

    # Panel (c): L=4 Spin-Glass Exact-AD Initialization / Schedule Check
    ax = axes[2]
    sg = res["check3"]["spinglass_exact_ad"]
    skeys = list(sg.keys())
    xp2 = np.arange(len(skeys))
    w = 0.36
    corrs = [sg[k]["intra_corr"]["mean"] for k in skeys]
    covs = [sg[k]["mode_cov"]["mean"] for k in skeys]
    ax.bar(xp2 - w / 2, corrs, width=w, color="#0072B2", edgecolor="#222222", lw=0.7, label="IntraCorr")
    ax.bar(xp2 + w / 2, covs, width=w, color="#009E73", edgecolor="#222222", lw=0.7, label="ModeCov")
    ax.set_xticks(xp2)
    ax.set_xticklabels(["Zero / Const", "Rand08 / Cos (200)", "Rand08 / Cos (400)"], rotation=15, ha="right", fontsize=7.6)
    ax.set_ylabel("Score [0, 1]")
    ax.set_title("(c) $L=4$ Spin-Glass Exact-AD Init/Schedule")
    ax.set_ylim(0, 1.12)
    ax.grid(axis="y", ls="--", alpha=0.25)
    ax.legend(frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_reciprocity_stepper_checks.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig_reciprocity_stepper_checks.png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default="./analysis")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--replot", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    json_p = os.path.join(args.out_dir, "reciprocity_stepper_results.json")
    if args.replot:
        with open(json_p, "r", encoding="utf-8") as f:
            res = json.load(f)
        make_figure(res, args.out_dir)
        print(f"[Replotted] {json_p} -> fig_reciprocity_stepper_checks.pdf", flush=True)
        return

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tdev_str = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [0] if args.quick else [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    max_shots = 16_000 if args.quick else 80_000
    print(f"[Setup] Reciprocity & Stepper Controls | Device={dev} | Seeds={seeds} | quick={args.quick}", flush=True)

    x_tr, y_tr, x_te, y_te, _ = bm.load_dataset("mnist", data_dir="./data")
    cache_p = os.path.join(args.out_dir, "mnist_d12_ae8_clf5_helpers.pt")
    ae, clf, z_tr, z_te = bm.train_classical_helpers(
        x_tr, y_tr, x_te, y_te, latent_dim=12, ae_epochs=8, clf_epochs=5, device=dev, cache_path=cache_p, use_bn=True,
    )

    t0 = time.time()
    c1 = run_check1_real_checkpoints(z_tr, y_tr, dev, tdev_str, args.quick)
    c2 = run_check2_adam_and_guard(z_tr, y_tr, z_te, x_te, y_te, ae, clf, dev, tdev_str, seeds, max_shots, args.quick)
    c3 = run_check3_missing_controls(z_tr, y_tr, z_te, x_te, y_te, ae, clf, dev, tdev_str, seeds, max_shots, args.quick)

    res = {
        "device": str(dev),
        "seeds": seeds,
        "quick": args.quick,
        "wall_sec": round(time.time() - t0, 1),
        "check1": c1,
        "check2": c2,
        "check3": c3,
    }
    with open(json_p, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    make_figure(res, args.out_dir)
    print(f"[Done] Saved {json_p} & fig_reciprocity_stepper_checks.pdf in {res['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
