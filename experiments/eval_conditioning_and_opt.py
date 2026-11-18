#!/usr/bin/env python3
"""
Conditioning Disentanglement, Optimizer Controls, Hybrid QERLayer Scaling, and Unit-Consistent SNR Suite.
  [Study 0] Deterministic Proposition 3 Jacobian Reciprocity Verification (JVP vs VJP, L in {2,4}).
  [Study 1] Zero-W_cond, Shared-W_cond, Chain-Rule W_cond, Prototype, & DDPM-LR Controls on MNIST (80k Shots).
  [Study 2] Hybrid ClassicalTrunk + QERLayer vs Exact-AD vs ClassicalResMLP+ShotNoise Diversity Scaling.
  [Study 3] Tuned SPSA Grid, Chain-Rule/4sh SPSA, QN-SPSA, Exact-PSR (+-pi/2), iCANS/Rosalin PSR,
            Proposition 2 FwdOLS vs InvOLS Calibrator Bounds, StabilizedAdam Ablations, & Standalone QERLayer.
  [Study 4] Unit-Consistent Sum-Normalized BP Variance & Equal-4k-Shot Cosine SNR (Surrogate & MSE).

Outputs:
  ./analysis/conditioning_opt_results.json
  ./analysis/fig_hybrid_qpu_diversity.pdf / .png
  ./analysis/fig_conditioning_opt_controls.pdf / .png

Usage:
  python -u experiments/eval_conditioning_and_opt.py
  python -u experiments/eval_conditioning_and_opt.py --reuse-spsa-grid
  python -u experiments/eval_conditioning_and_opt.py --quick
  python -u experiments/eval_conditioning_and_opt.py --replot
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
from qerd import (
    QERConfig,
    QERGradConfig,
    QERGradientEstimator,
    QERLayer,
    QEROptimizer,
    QERDTrainer,
    QuantumIsingDenoiser,
    StabilizedAdam,
    StabilizedAdamConfig,
)


def _agg_dicts(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for k in rows[0]:
        vals = [float(r[k]) for r in rows]
        out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return out


def run_study0_reciprocity(dev: torch.device) -> Dict[str, Any]:
    """Verify Proposition 3: exact Jacobian asymmetry ||J-J^T||_F/||J||_F and VJP vs JVP alignment."""
    print("\n=== [Study 0] Deterministic Proposition 3 Jacobian Reciprocity (VJP vs JVP) ===", flush=True)
    rng = np.random.default_rng(123)
    nq = 12
    edges = bm.build_ising_edges(nq, "all")
    ne = len(edges)
    bs = 16
    xt_np = np.clip(rng.normal(0.0, 0.70, size=(bs, nq)), -1.0, 1.0).astype(np.float32)
    x0_np = rng.choice([-1.0, 1.0], size=(bs, nq)).astype(np.float32)
    t_np = ((np.arange(bs) % 4) + 1).astype(np.int64)

    xt_t = torch.tensor(xt_np, dtype=torch.float32, device=dev)
    x0_t = torch.tensor(x0_np, dtype=torch.float32, device=dev)
    t_t = torch.tensor(t_np, dtype=torch.long, device=dev)

    rows = []
    for nl in (2, 4):
        circ = bm._DiffIsingCircuit(nq, cond_dim=0, edges=edges, n_layers=nl, device=dev)
        for scale_name, th_scale in [("init_0.2", 0.20), ("trained_0.6", 0.60)]:
            th_np = rng.uniform(-th_scale, th_scale, size=ne + nq).astype(np.float32)
            th_t = torch.tensor(th_np, dtype=torch.float32, device=dev, requires_grad=True)

            # Exact field Jacobian J_zh [B, nq, nq] via autograd on explicit bias perturbation
            dh = torch.zeros((bs, nq), dtype=torch.float32, device=dev, requires_grad=True)
            _, z_dh = circ.expectation(xt_t, t_t, None, th_t, None, nudge_target=dh, nudge_beta=1.0)
            jac_rows = []
            for qi in range(nq):
                (g_qi,) = torch.autograd.grad(z_dh[:, qi].sum(), [dh], retain_graph=True)
                jac_rows.append(g_qi)
            jac = torch.stack(jac_rows, dim=1).detach()  # [B, nq, nq], jac[b, i, j] = d<Z_i>/dh_j

            diff_f = torch.linalg.norm(jac - jac.transpose(1, 2), dim=(1, 2))
            norm_f = torch.linalg.norm(jac, dim=(1, 2)).clamp(min=1e-12)
            asym_ratio = float((diff_f / norm_f).mean().item())

            # Compare visible-field and parameter gradients for surrogate sum(z * x0)
            _, z0 = circ.expectation(xt_t, t_t, None, th_t, None)
            (g_vjp_th,) = torch.autograd.grad(torch.sum(z0 * x0_t, dim=1).mean(), [th_t])
            g_vjp_np = g_vjp_th.detach().cpu().numpy()

            with torch.no_grad():
                vjp_h = torch.bmm(jac.transpose(1, 2), x0_t.unsqueeze(-1)).squeeze(-1).mean(dim=0).cpu().numpy()
                jvp_h = torch.bmm(jac, x0_t.unsqueeze(-1)).squeeze(-1).mean(dim=0).cpu().numpy()

                cos_by_beta = {}
                for beta in (0.45, 0.10):
                    op, zp = circ.expectation(xt_t, t_t, None, th_t, None, nudge_target=x0_t, nudge_beta=beta)
                    om, zm = circ.expectation(xt_t, t_t, None, th_t, None, nudge_target=x0_t, nudge_beta=-beta)
                    gq_th = ((op - om) / (2.0 * beta)).mean(dim=0).cpu().numpy()
                    gq_h = ((zp - zm) / (2.0 * beta)).mean(dim=0).cpu().numpy()

                    def _cos(a: np.ndarray, b: np.ndarray) -> float:
                        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

                    cos_by_beta[str(beta)] = {
                        "cos_qer_vjp_h": _cos(gq_h, vjp_h),
                        "cos_qer_jvp_h": _cos(gq_h, jvp_h),
                        "cos_qer_vjp_zz": _cos(gq_th[:ne], g_vjp_np[:ne]),
                        "cos_qer_vjp_z": _cos(gq_th[ne:], g_vjp_np[ne:]),
                        "norm_ratio_zz": float(np.linalg.norm(gq_th[:ne]) / (np.linalg.norm(g_vjp_np[:ne]) + 1e-12)),
                        "norm_ratio_z": float(np.linalg.norm(gq_th[ne:]) / (np.linalg.norm(g_vjp_np[ne:]) + 1e-12)),
                    }

            row = {
                "n_layers": nl,
                "regime": scale_name,
                "asym_ratio": asym_ratio,
                "cos_jvp_vjp_h": float(np.dot(jvp_h, vjp_h) / (np.linalg.norm(jvp_h) * np.linalg.norm(vjp_h) + 1e-12)),
                "beta_045": cos_by_beta["0.45"],
                "beta_010": cos_by_beta["0.1"],
            }
            rows.append(row)
            b45 = row["beta_045"]
            print(
                f"  L={nl} ({scale_name:11s}) | Asym={asym_ratio:.3f} | cos(JVP,VJP)_h={row['cos_jvp_vjp_h']:+.3f} | "
                f"beta=0.45 cos(QER,VJP) ZZ={b45['cos_qer_vjp_zz']:+.3f} Z={b45['cos_qer_vjp_z']:+.3f} | "
                f"beta=0.10 cos(QER,JVP)_h={row['beta_010']['cos_qer_jvp_h']:+.3f}",
                flush=True,
            )
    return {"rows": rows}


def _eval_subset_logits(
    ae: nn.Module,
    clf: bm.EvalCNN,
    x_latents: np.ndarray,
    cls_sub: List[int],
    eval_orig: np.ndarray,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate 10-way and subset classifier accuracy of decoded visible latents."""
    with torch.no_grad():
        zt = torch.tensor(x_latents, dtype=torch.float32, device=device)
        imgs = ae.decode(zt)
        logits, _ = clf.forward_features(imgs)
        p10 = logits.argmax(dim=1).cpu().numpy()
        psub_idx = logits[:, cls_sub].argmax(dim=1).cpu().numpy()
        psub = np.array([cls_sub[int(i)] for i in psub_idx], dtype=np.int64)
    return {
        "sub_acc": float(np.mean(psub == eval_orig)),
        "ten_way_acc": float(np.mean(p10 == eval_orig)),
        "lat_std": float(np.mean([np.mean(np.std(x_latents[eval_orig == c], axis=0)) for c in cls_sub])),
    }


def _train_label_qubit_qer(
    z_sub: np.ndarray,
    y_idx: np.ndarray,
    cls_sub: List[int],
    codes: np.ndarray,
    topo: str,
    n_layers: int,
    max_shots: int,
    seed: int,
    tdev_str: str,
    dev: torch.device,
    ae: nn.Module,
    clf: bm.EvalCNN,
    n_eval_per_class: int = 120,
    lr_max: float = 0.16,
    clip_bound: float = 1.50,
    field_amp: float = 1.15,
) -> Dict[str, float]:
    n_lbl = codes.shape[1]
    n_q = 12 + n_lbl
    n_cls = len(cls_sub)
    rng = np.random.default_rng(seed)
    m = QuantumIsingDenoiser(
        n_q, cond_dim=0, n_hidden=0, topology=topo, n_layers=n_layers,
        p_depol=0.035, p_readout=0.04, init_gain=1.0, gain_ema=0.0,
        backend="torch", torch_device=tdev_str, seed=seed,
    )
    opt = QEROptimizer(
        m.parameters_dict(),
        QERConfig(lr_max=lr_max, lr_min=0.015, beta=0.45, beta_min=0.15, clip_bound=clip_bound),
    )
    pools = [np.flatnonzero(y_idx == i) for i in range(n_cls)]
    ab = m.alpha_bar
    shots_per_run = 20
    bs = 10
    n_batches = max(1, max_shots // (2 * shots_per_run * bs))
    for b in range(1, n_batches + 1):
        prog = (b - 1) / max(1, n_batches - 1)
        c_ord = rng.permutation(n_cls)
        ids = np.array([rng.choice(pools[int(c_ord[k % n_cls])]) for k in range(bs)], dtype=np.int64)
        tc = (np.arange(bs) % 4) + 1
        rng.shuffle(tc)
        zt = z_sub[ids]
        a = ab[tc][:, None]
        xv = np.clip(np.sqrt(a) * zt + np.sqrt(1.0 - a) * rng.normal(0.0, 0.70, size=zt.shape), -1.0, 1.0)
        in_pin = m.in_pinning_base + m.in_pinning_slope * (1.0 - tc / 4.0)
        lbl_x = (field_amp / in_pin)[:, None] * codes[y_idx[ids]]
        xf = np.concatenate([xv, lbl_x], axis=1)
        yn = np.concatenate([zt, np.zeros_like(lbl_x)], axis=1)
        obs_b, z_b = m.measure_observables_batch(
            np.concatenate([xf, xf]), np.concatenate([tc, tc]), shots=shots_per_run,
            nudge_target_batch=np.concatenate([yn, yn]),
            nudge_beta_batch=np.concatenate([np.full(bs, opt.beta), np.full(bs, -opt.beta)]),
        )
        opt.step_contrast(
            m.parameters_dict(), obs_b[:bs], obs_b[bs:],
            z_plus_list=z_b[:bs], z_minus_list=z_b[bs:],
            nudge_dirs=yn, progress=prog, shots_in_batch=2 * bs * shots_per_run,
        )

    ev_idx = np.repeat(np.arange(n_cls, dtype=np.int64), n_eval_per_class)
    ev_orig = np.array([cls_sub[int(i)] for i in ev_idx], dtype=np.int64)
    xt_v = np.clip(rng.normal(0.0, 0.90, size=(len(ev_idx), 12)), -1.0, 1.0)
    for ts in range(4, 0, -1):
        in_pin_t = m.in_pinning_base + m.in_pinning_slope * (1.0 - ts / 4.0)
        lbl_e = (field_amp / in_pin_t) * codes[ev_idx]
        _, zo = m.measure_observables_batch(
            np.concatenate([xt_v, lbl_e], axis=1), np.full(len(ev_idx), ts, dtype=np.int64), shots=160,
        )
        x0h = zo[:, :12]
        xt_v = (
            np.clip(math.sqrt(ab[ts - 1]) * x0h + math.sqrt(1.0 - ab[ts - 1]) * rng.normal(0.0, 0.35, size=x0h.shape), -1.0, 1.0)
            if ts > 1 else x0h
        )
    res = _eval_subset_logits(ae, clf, xt_v, cls_sub, ev_orig, dev)
    res["n_qpu_params"] = float(m.n_qpu_params)
    return res


def _train_label_qubit_spsa_or_icans(
    mode: str,
    z_sub: np.ndarray,
    y_idx: np.ndarray,
    cls_sub: List[int],
    codes: np.ndarray,
    n_layers: int,
    max_shots: int,
    seed: int,
    tdev_str: str,
    dev: torch.device,
    ae: nn.Module,
    clf: bm.EvalCNN,
    n_eval_per_class: int = 120,
    clip_bound: float = 1.50,
    field_amp: float = 1.15,
) -> Dict[str, float]:
    n_lbl = codes.shape[1]
    n_q = 12 + n_lbl
    n_cls = len(cls_sub)
    rng = np.random.default_rng(seed)
    m = QuantumIsingDenoiser(
        n_q, cond_dim=0, n_hidden=0, topology="all", n_layers=n_layers,
        p_depol=0.035, p_readout=0.04, init_gain=1.0, gain_ema=0.0,
        backend="torch", torch_device=tdev_str, seed=seed,
    )
    pools = [np.flatnonzero(y_idx == i) for i in range(n_cls)]
    ab = m.alpha_bar
    m_th = np.zeros_like(m.theta)
    v_th = np.zeros_like(m.theta)
    n_th = m.theta.size

    if mode == "spsa":
        bs, shots_per_run, lr, c0 = 10, 20, 0.25, 0.30
        n_batches = max(1, max_shots // (2 * shots_per_run * bs))
        for st in range(1, n_batches + 1):
            ck = c0 / (st ** 0.101)
            lrk = 0.015 + 0.5 * (lr - 0.015) * (1.0 + math.cos(math.pi * (st - 1) / max(1, n_batches - 1)))
            c_ord = rng.permutation(n_cls)
            ids = np.array([rng.choice(pools[int(c_ord[k % n_cls])]) for k in range(bs)], dtype=np.int64)
            tc = (np.arange(bs) % 4) + 1
            rng.shuffle(tc)
            zt = z_sub[ids]
            a = ab[tc][:, None]
            xv = np.clip(np.sqrt(a) * zt + np.sqrt(1.0 - a) * rng.normal(0.0, 0.70, size=zt.shape), -1.0, 1.0)
            in_pin = m.in_pinning_base + m.in_pinning_slope * (1.0 - tc / 4.0)
            lbl_x = (field_amp / in_pin)[:, None] * codes[y_idx[ids]]
            xf = np.concatenate([xv, lbl_x], axis=1)
            d_th = rng.choice([-1.0, 1.0], size=(bs, n_th))
            _, zb = m.measure_observables_batch(
                np.concatenate([xf, xf]), np.concatenate([tc, tc]), shots=shots_per_run,
                theta_batch=np.concatenate([m.theta[None] + ck * d_th, m.theta[None] - ck * d_th]),
            )
            lp = np.mean((zb[:bs, :12] - zt) ** 2, axis=1)
            lm = np.mean((zb[bs:, :12] - zt) ** 2, axis=1)
            g_th = -np.mean(((lp - lm) / (2.0 * ck))[:, None] * d_th, axis=0)
            m_th[:] = 0.75 * m_th + 0.25 * g_th
            v_th[:] = 0.98 * v_th + 0.02 * (g_th ** 2)
            m.theta[:] = np.clip(m.theta + lrk * (m_th / (1.0 - 0.75 ** st)) / (np.sqrt(v_th / (1.0 - 0.98 ** st)) + 1e-6), -clip_bound, clip_bound)
    else:  # iCANS stochastic PSR
        k_sub, sh_q = 15, 6
        ema_g = np.ones(n_th, dtype=np.float64) * 0.1
        shots_used, st = 0, 0
        while shots_used + 2 * k_sub * sh_q <= max_shots:
            st += 1
            cls_i = st % n_cls
            ts = (st % 4) + 1
            sid = int(rng.choice(pools[cls_i]))
            zt = z_sub[sid]
            in_pin = m.in_pinning_base + m.in_pinning_slope * (1.0 - ts / 4.0)
            xv = np.clip(math.sqrt(ab[ts]) * zt + math.sqrt(1.0 - ab[ts]) * rng.normal(0.0, 0.70, size=12), -1.0, 1.0)
            xf = np.concatenate([xv, (field_amp / in_pin) * codes[cls_i]])
            probs = ema_g + 0.05
            probs /= probs.sum()
            pick = rng.choice(n_th, size=k_sub, replace=False, p=probs)
            tp = np.broadcast_to(m.theta, (k_sub, n_th)).copy()
            tm = tp.copy()
            tp[np.arange(k_sub), pick] += 0.25
            tm[np.arange(k_sub), pick] -= 0.25
            _, z2 = m.measure_observables_batch(
                np.broadcast_to(xf, (2 * k_sub, n_q)), np.full(2 * k_sub, ts, dtype=np.int64),
                shots=sh_q, theta_batch=np.concatenate([tp, tm]),
            )
            shots_used += 2 * k_sub * sh_q
            lp = np.mean((z2[:k_sub, :12] - zt[None, :]) ** 2, axis=1)
            lm = np.mean((z2[k_sub:, :12] - zt[None, :]) ** 2, axis=1)
            g_sub = -(lp - lm) / 0.50
            ema_g[pick] = 0.7 * ema_g[pick] + 0.3 * np.abs(g_sub)
            g_th = np.zeros(n_th, dtype=np.float64)
            g_th[pick] = g_sub / np.maximum(k_sub * probs[pick], 1e-3)
            lrk = 0.015 + 0.5 * (0.08 - 0.015) * (1.0 + math.cos(math.pi * min(1.0, shots_used / max(1, max_shots))))
            m_th[:] = 0.75 * m_th + 0.25 * g_th
            v_th[:] = 0.98 * v_th + 0.02 * (g_th ** 2)
            m.theta[:] = np.clip(m.theta + lrk * (m_th / (1.0 - 0.75 ** st)) / (np.sqrt(v_th / (1.0 - 0.98 ** st)) + 1e-6), -clip_bound, clip_bound)

    ev_idx = np.repeat(np.arange(n_cls, dtype=np.int64), n_eval_per_class)
    ev_orig = np.array([cls_sub[int(i)] for i in ev_idx], dtype=np.int64)
    xt_v = np.clip(rng.normal(0.0, 0.90, size=(len(ev_idx), 12)), -1.0, 1.0)
    for ts in range(4, 0, -1):
        in_pin_t = m.in_pinning_base + m.in_pinning_slope * (1.0 - ts / 4.0)
        lbl_e = (field_amp / in_pin_t) * codes[ev_idx]
        _, zo = m.measure_observables_batch(
            np.concatenate([xt_v, lbl_e], axis=1), np.full(len(ev_idx), ts, dtype=np.int64), shots=160,
        )
        x0h = zo[:, :12]
        xt_v = (
            np.clip(math.sqrt(ab[ts - 1]) * x0h + math.sqrt(1.0 - ab[ts - 1]) * rng.normal(0.0, 0.35, size=x0h.shape), -1.0, 1.0)
            if ts > 1 else x0h
        )
    res = _eval_subset_logits(ae, clf, xt_v, cls_sub, ev_orig, dev)
    res["n_qpu_params"] = float(m.n_qpu_params)
    return res


def run_study1_wcond_controls(
    z_tr: np.ndarray, y_tr: np.ndarray, z_te: np.ndarray, x_te: np.ndarray, y_te: np.ndarray,
    ae: nn.Module, clf: bm.EvalCNN, dev: torch.device, tdev_str: str,
    seeds: List[int], max_shots: int, n_eval_per_cls: int,
) -> Dict[str, Any]:
    print("\n=== [Study 1] Zero-W_cond Label-Qubit Conditioning & 10-Class Frozen/Removed W_cond ===", flush=True)
    eye_c = np.eye(10, dtype=np.float64)
    c_tr = eye_c[y_tr]
    ev_lbl = np.repeat(np.arange(10, dtype=np.int64), 30 if n_eval_per_cls < 60 else 100)
    ev_c = eye_c[ev_lbl]
    rng_q = np.random.default_rng(777)
    q_orth, _ = np.linalg.qr(rng_q.normal(size=(12, 10)))
    w_frozen = q_orth * 0.45

    # Part 1A: Zero-classical-weight 13q {0, 1} (and 14q {0, 1, 4})
    cls2 = [0, 1]
    codes2 = np.array([[-1.0], [+1.0]], dtype=np.float64)
    mask2 = np.isin(y_tr, cls2)
    z2 = z_tr[mask2]
    y2 = np.array([0 if v == 0 else 1 for v in y_tr[mask2]], dtype=np.int64)

    cls3 = [0, 1, 4]
    codes3 = np.array([[-1.0, -1.0], [-1.0, +1.0], [+1.0, -1.0]], dtype=np.float64)
    mask3 = np.isin(y_tr, cls3)
    z3 = z_tr[mask3]
    cmap3 = {0: 0, 1: 1, 4: 2}
    y3 = np.array([cmap3[int(v)] for v in y_tr[mask3]], dtype=np.int64)

    lq_keys = [
        ("lbl13_nozz", "13q {0,1} QER no-zz (0 cls, L=3)"),
        ("lbl13_all_l2", "13q {0,1} QER all (0 cls, L=2)"),
        ("lbl13_all_l3", "13q {0,1} QER all (0 cls, L=3)"),
        ("lbl13_spsa_l3", "13q {0,1} Tuned SPSA all (L=3)"),
        ("lbl13_icans_l3", "13q {0,1} iCANS-PSR all (L=3)"),
        ("lbl14_nozz", "14q {0,1,4} QER no-zz (0 cls, L=3)"),
        ("lbl14_all_l3", "14q {0,1,4} QER all (0 cls, L=3)"),
    ]
    lq_rows: Dict[str, List[Dict[str, float]]] = {k: [] for k, _ in lq_keys}

    # Part 1B: 10-class MNIST Frozen-Ortho, Shared-W_cond, Chain-Rule W_cond, Prototype, & DDPM-LR Controls
    full_keys = [
        ("std_qer_198p", "MNIST 10-Cls Standard QER (198p)"),
        ("frozen_qer_all", "MNIST 10-Cls Frozen-Ortho W_cond (QER, 78p)"),
        ("frozen_qer_nozz", "MNIST 10-Cls Frozen-Ortho W_cond (no-zz, 12p)"),
        ("frozen_spsa_all", "MNIST 10-Cls Frozen-Ortho W_cond (SPSA, 78p)"),
        ("shared_wcond_qer", "MNIST 10-Cls Shared Learned W_cond (QER, 78p)"),
        ("shared_wcond_spsa", "MNIST 10-Cls Shared Learned W_cond (SPSA, 78p)"),
        ("shared_wcond_exact_psr", "MNIST 10-Cls Shared Learned W_cond (Exact-PSR, 78p)"),
        ("chainrule_spsa", "MNIST 10-Cls Chain-Rule W_cond (SPSA, 78p QPU)"),
        ("chainrule_psr_fd", "MNIST 10-Cls Chain-Rule W_cond (PSR-FD, 78p QPU)"),
        ("chainrule_exact_psr", "MNIST 10-Cls Chain-Rule W_cond (Exact-PSR +-pi/2)"),
        ("qnspsa_chainrule", "MNIST 10-Cls QN-SPSA + Chain-Rule W_cond (80k)"),
        ("class_mean_proto", "MNIST 10-Cls Class-Mean Prototype + sigma=0.35"),
        ("class_mean_det", "MNIST 10-Cls Class-Mean Prototype (sigma=0.0)"),
        ("ddpm_80k_cos", "Classical-DDPM (~200p, 80k-eq 200st Cosine LR)"),
        ("ddpm_160k_const", "Classical-DDPM (~200p, 160k-eq 400st Const LR)"),
        ("ddpm_160k_cos", "Classical-DDPM (~200p, 160k-eq 400st Cosine LR)"),
        ("uncond_qer_all", "MNIST Unconditional QER all (cond_dim=0, 78p)"),
        ("uncond_qer_nozz", "MNIST Unconditional QER no-zz (cond_dim=0, 12p)"),
    ]
    full_rows: Dict[str, List[Dict[str, float]]] = {k: [] for k, _ in full_keys}

    # Pre-train one independent shared learned W_cond from an auxiliary QER seed (seed=999)
    m_shared_ref = QuantumIsingDenoiser(12, cond_dim=10, n_hidden=0, topology="all", n_layers=2, p_depol=0.035, p_readout=0.04, init_gain=1.0, gain_ema=0.0, backend="torch", torch_device=tdev_str, seed=999)
    QERDTrainer(m_shared_ref, QEROptimizer(m_shared_ref.parameters_dict(), QERConfig(clip_bound=0.90))).train(
        z_tr, c_tr, stratify_labels=y_tr, max_shots=max_shots, shots_per_run=20, batch_size=10, eval_every_batches=999999, verbose=False
    )
    w_shared = m_shared_ref.cond_proj.copy()
    cls_means = np.stack([z_tr[y_tr == c].mean(axis=0) for c in range(10)], axis=0)

    def _eval_ddpm(dm: nn.Module, name: str, sd: int) -> Dict[str, float]:
        r_d = np.random.default_rng(sd + 500)
        xt_d = np.clip(r_d.normal(0.0, 0.90, size=(len(ev_lbl), 12)), -1.0, 1.0).astype(np.float32)
        c_d = torch.from_numpy(ev_c.astype(np.float32))
        ab_d = bm.make_alpha_bar(4, "cosine")
        dm.eval()
        with torch.no_grad():
            for ts in range(4, 0, -1):
                tn = torch.full((len(ev_lbl), 1), float(ts) / 4.0, dtype=torch.float32)
                x0h = dm(torch.from_numpy(xt_d), tn, c_d).numpy()
                if ts > 1:
                    xt_d = np.clip(math.sqrt(ab_d[ts - 1]) * x0h + math.sqrt(1.0 - ab_d[ts - 1]) * r_d.normal(0.0, 0.35, size=x0h.shape), -1.0, 1.0).astype(np.float32)
                else:
                    xt_d = x0h
        return bm.evaluate_generated(name, xt_d.astype(np.float64), ev_lbl, ae, clf, z_te, x_te, y_te, dev)[0]

    for sd in seeds:
        print(f"  [Study 1 Seed {sd}]", flush=True)
        lq_rows["lbl13_nozz"].append(_train_label_qubit_qer(z2, y2, cls2, codes2, "none", 3, max_shots, sd, tdev_str, dev, ae, clf, n_eval_per_cls))
        lq_rows["lbl13_all_l2"].append(_train_label_qubit_qer(z2, y2, cls2, codes2, "all", 2, max_shots, sd, tdev_str, dev, ae, clf, n_eval_per_cls))
        lq_rows["lbl13_all_l3"].append(_train_label_qubit_qer(z2, y2, cls2, codes2, "all", 3, max_shots, sd, tdev_str, dev, ae, clf, n_eval_per_cls))
        lq_rows["lbl13_spsa_l3"].append(_train_label_qubit_spsa_or_icans("spsa", z2, y2, cls2, codes2, 3, max_shots, sd, tdev_str, dev, ae, clf, n_eval_per_cls))
        lq_rows["lbl13_icans_l3"].append(_train_label_qubit_spsa_or_icans("icans", z2, y2, cls2, codes2, 3, max_shots, sd, tdev_str, dev, ae, clf, n_eval_per_cls))
        lq_rows["lbl14_nozz"].append(_train_label_qubit_qer(z3, y3, cls3, codes3, "none", 3, max_shots, sd, tdev_str, dev, ae, clf, n_eval_per_cls))
        lq_rows["lbl14_all_l3"].append(_train_label_qubit_qer(z3, y3, cls3, codes3, "all", 3, max_shots, sd, tdev_str, dev, ae, clf, n_eval_per_cls))
        for k, _ in lq_keys:
            r = lq_rows[k][-1]
            print(f"    [{k:15s}] SubAcc={r['sub_acc']*100:5.1f}% | 10-Way={r['ten_way_acc']*100:5.1f}%", flush=True)

        # Standard 198p QER
        m = QuantumIsingDenoiser(12, cond_dim=10, n_hidden=0, topology="all", n_layers=2, p_depol=0.035, p_readout=0.04, init_gain=1.0, gain_ema=0.0, backend="torch", torch_device=tdev_str, seed=sd)
        QERDTrainer(m, QEROptimizer(m.parameters_dict(), QERConfig(clip_bound=0.90))).train(z_tr, c_tr, stratify_labels=y_tr, max_shots=max_shots, shots_per_run=20, batch_size=10, eval_every_batches=999999, verbose=False)
        met, _ = bm.evaluate_generated("StdQER", m.sample_batch(conds=ev_c, shots_per_step=160, step_noise_scale=0.35), ev_lbl, ae, clf, z_te, x_te, y_te, dev)
        full_rows["std_qer_198p"].append(met)

        # Frozen-Ortho W_cond (QER all, QER no-zz, Tuned SPSA all)
        for fk, topo in [("frozen_qer_all", "all"), ("frozen_qer_nozz", "none")]:
            mf = QuantumIsingDenoiser(12, cond_dim=10, n_hidden=0, topology=topo, n_layers=2, p_depol=0.035, p_readout=0.04, init_gain=0.75, gain_ema=0.10, backend="torch", torch_device=tdev_str, seed=sd)
            mf.cond_proj[:] = w_frozen
            mf.parameters_dict = lambda _m=mf: {"theta": _m.theta}
            QERDTrainer(mf, QEROptimizer({"theta": mf.theta}, QERConfig(clip_bound=1.20))).train(z_tr, c_tr, stratify_labels=y_tr, max_shots=max_shots, shots_per_run=20, batch_size=10, eval_every_batches=999999, verbose=False)
            metf, _ = bm.evaluate_generated(fk, mf.sample_batch(conds=ev_c, shots_per_step=160, step_noise_scale=0.35), ev_lbl, ae, clf, z_te, x_te, y_te, dev)
            full_rows[fk].append(metf)

        # Frozen SPSA (only perturbing theta, W_cond frozen)
        ms = QuantumIsingDenoiser(12, cond_dim=10, n_hidden=0, topology="all", n_layers=2, p_depol=0.035, p_readout=0.04, init_gain=0.75, gain_ema=0.10, backend="torch", torch_device=tdev_str, seed=sd)
        ms.cond_proj[:] = w_frozen
        pools10 = [np.flatnonzero(y_tr == c) for c in range(10)]
        mth, vth = np.zeros_like(ms.theta), np.zeros_like(ms.theta)
        totb = max(1, max_shots // 400)
        for st in range(1, totb + 1):
            ck = 0.25 / (st ** 0.101)
            lrk = 0.015 + 0.5 * (0.20 - 0.015) * (1.0 + math.cos(math.pi * (st - 1) / max(1, totb - 1)))
            ids = bm._sample_stratified_ids(ms.rng, pools10, 10)
            tb = (np.arange(10) % 4 + 1).astype(np.int64); ms.rng.shuffle(tb)
            tg, cb = z_tr[ids], c_tr[ids]
            ab_s = ms.alpha_bar[tb][:, None]
            xt = np.clip(np.sqrt(ab_s) * tg + np.sqrt(1.0 - ab_s) * ms.rng.normal(0.0, 0.70, size=tg.shape), -1.0, 1.0)
            d_th = ms.rng.choice([-1.0, 1.0], size=(10,) + ms.theta.shape)
            _, zb = ms.measure_observables_batch(
                np.concatenate([xt, xt]), np.concatenate([tb, tb]), shots=20,
                cond_batch=np.concatenate([cb, cb]),
                theta_batch=np.concatenate([ms.theta[None] + ck * d_th, ms.theta[None] - ck * d_th]),
            )
            diff = (np.mean((zb[:10] - tg) ** 2, axis=1) - np.mean((zb[10:] - tg) ** 2, axis=1)) / (2.0 * ck)
            gth = -np.mean(diff[:, None] * d_th, axis=0)
            mth[:] = 0.75 * mth + 0.25 * gth; vth[:] = 0.98 * vth + 0.02 * (gth ** 2)
            ms.theta[:] = np.clip(ms.theta + lrk * (mth / (1.0 - 0.75 ** st)) / (np.sqrt(vth / (1.0 - 0.98 ** st)) + 1e-6), -1.20, 1.20)
        mets, _ = bm.evaluate_generated("frozen_spsa", ms.sample_batch(conds=ev_c, shots_per_step=160, step_noise_scale=0.35), ev_lbl, ae, clf, z_te, x_te, y_te, dev)
        full_rows["frozen_spsa_all"].append(mets)

        # Shared-W_cond QPU-Only Bake-off (exact same frozen learned w_shared across QER / SPSA / Exact-PSR)
        mq_s = QuantumIsingDenoiser(12, cond_dim=10, n_hidden=0, topology="all", n_layers=2, p_depol=0.035, p_readout=0.04, init_gain=1.0, gain_ema=0.0, backend="torch", torch_device=tdev_str, seed=sd)
        mq_s.cond_proj[:] = w_shared
        mq_s.parameters_dict = lambda _m=mq_s: {"theta": _m.theta}
        QERDTrainer(mq_s, QEROptimizer({"theta": mq_s.theta}, QERConfig(clip_bound=0.90))).train(z_tr, c_tr, stratify_labels=y_tr, max_shots=max_shots, shots_per_run=20, batch_size=10, eval_every_batches=999999, verbose=False)
        full_rows["shared_wcond_qer"].append(bm.evaluate_generated("shared_wcond_qer", mq_s.sample_batch(conds=ev_c, shots_per_step=160, step_noise_scale=0.35), ev_lbl, ae, clf, z_te, x_te, y_te, dev)[0])

        ms_s = QuantumIsingDenoiser(12, cond_dim=10, n_hidden=0, topology="all", n_layers=2, p_depol=0.035, p_readout=0.04, init_gain=1.0, gain_ema=0.0, backend="torch", torch_device=tdev_str, seed=sd)
        ms_s.cond_proj[:] = w_shared
        # Disable cond_proj training via dummy array or direct 0-update: run custom theta-only SPSA loop via mask (cond_dim>0 passed to circuit)
        mth_s, vth_s = np.zeros_like(ms_s.theta), np.zeros_like(ms_s.theta)
        for st in range(1, totb + 1):
            ck = 0.18 / (st ** 0.101)
            lrk = 0.015 + 0.5 * (0.08 - 0.015) * (1.0 + math.cos(math.pi * (st - 1) / max(1, totb - 1)))
            ids = bm._sample_stratified_ids(ms_s.rng, pools10, 10)
            tb = (np.arange(10) % 4 + 1).astype(np.int64); ms_s.rng.shuffle(tb)
            tg, cb = z_tr[ids], c_tr[ids]
            ab_s = ms_s.alpha_bar[tb][:, None]
            xt = np.clip(np.sqrt(ab_s) * tg + np.sqrt(1.0 - ab_s) * ms_s.rng.normal(0.0, 0.70, size=tg.shape), -1.0, 1.0)
            d_th = ms_s.rng.choice([-1.0, 1.0], size=(10,) + ms_s.theta.shape)
            _, zb = ms_s.measure_observables_batch(np.concatenate([xt, xt]), np.concatenate([tb, tb]), shots=20, cond_batch=np.concatenate([cb, cb]), theta_batch=np.concatenate([ms_s.theta[None] + ck * d_th, ms_s.theta[None] - ck * d_th]))
            diff = (np.mean((zb[:10] - tg) ** 2, axis=1) - np.mean((zb[10:] - tg) ** 2, axis=1)) / (2.0 * ck)
            gth = -np.mean(diff[:, None] * d_th, axis=0)
            mth_s[:] = 0.75 * mth_s + 0.25 * gth; vth_s[:] = 0.98 * vth_s + 0.02 * (gth ** 2)
            ms_s.theta[:] = np.clip(ms_s.theta + lrk * (mth_s / (1.0 - 0.75 ** st)) / (np.sqrt(vth_s / (1.0 - 0.98 ** st)) + 1e-6), -0.90, 0.90)
        full_rows["shared_wcond_spsa"].append(bm.evaluate_generated("shared_wcond_spsa", ms_s.sample_batch(conds=ev_c, shots_per_step=160, step_noise_scale=0.35), ev_lbl, ae, clf, z_te, x_te, y_te, dev)[0])

        me_s = QuantumIsingDenoiser(12, cond_dim=10, n_hidden=0, topology="all", n_layers=2, p_depol=0.035, p_readout=0.04, init_gain=1.0, gain_ema=0.0, backend="torch", torch_device=tdev_str, seed=sd)
        me_s.cond_proj[:] = w_shared
        # Freeze w_shared via post-step restore in train_exact_psr_baseline (lr_cp=0 equivalent)
        bm.train_exact_psr_baseline(me_s, z_tr, c_tr, y_tr, max_shots=max_shots, shots_per_run=max(1, max_shots // 3130), lr=0.08, clip_bound=0.90, device=tdev_str)
        me_s.cond_proj[:] = w_shared
        full_rows["shared_wcond_exact_psr"].append(bm.evaluate_generated("shared_wcond_e_psr", me_s.sample_batch(conds=ev_c, shots_per_step=160, step_noise_scale=0.35), ev_lbl, ae, clf, z_te, x_te, y_te, dev)[0])

        # Chain-Rule W_cond Full 198p Baselines (78 QPU params + exact outer-product W_cond grad)
        mcr_s = QuantumIsingDenoiser(12, cond_dim=10, n_hidden=0, topology="all", n_layers=2, p_depol=0.035, p_readout=0.04, init_gain=1.0, gain_ema=0.0, backend="torch", torch_device=tdev_str, seed=sd)
        bm.train_adam_spsa_baseline(mcr_s, z_tr, c_tr, y_tr, max_shots=max_shots, shots_per_run=20, lr=0.08, c0=0.18, clip_bound=0.90, chain_rule_wcond=True)
        full_rows["chainrule_spsa"].append(bm.evaluate_generated("chainrule_spsa", mcr_s.sample_batch(conds=ev_c, shots_per_step=160, step_noise_scale=0.35), ev_lbl, ae, clf, z_te, x_te, y_te, dev)[0])

        mcr_p = QuantumIsingDenoiser(12, cond_dim=10, n_hidden=0, topology="all", n_layers=2, p_depol=0.035, p_readout=0.04, init_gain=1.0, gain_ema=0.0, backend="torch", torch_device=tdev_str, seed=sd)
        bm.train_adam_psr_baseline(mcr_p, z_tr, c_tr, y_tr, max_shots=max_shots, shots_per_run=20, lr=0.08, delta=0.25, clip_bound=0.90, chain_rule_wcond=True)
        full_rows["chainrule_psr_fd"].append(bm.evaluate_generated("chainrule_psr_fd", mcr_p.sample_batch(conds=ev_c, shots_per_step=160, step_noise_scale=0.35), ev_lbl, ae, clf, z_te, x_te, y_te, dev)[0])

        mcr_ep = QuantumIsingDenoiser(12, cond_dim=10, n_hidden=0, topology="all", n_layers=2, p_depol=0.035, p_readout=0.04, init_gain=1.0, gain_ema=0.0, backend="torch", torch_device=tdev_str, seed=sd)
        bm.train_exact_psr_baseline(mcr_ep, z_tr, c_tr, y_tr, max_shots=max_shots, shots_per_run=max(4, max_shots // 3130), lr=0.08, clip_bound=0.90, device=tdev_str)
        full_rows["chainrule_exact_psr"].append(bm.evaluate_generated("chainrule_exact_psr", mcr_ep.sample_batch(conds=ev_c, shots_per_step=160, step_noise_scale=0.35), ev_lbl, ae, clf, z_te, x_te, y_te, dev)[0])

        mqn = QuantumIsingDenoiser(12, cond_dim=10, n_hidden=0, topology="all", n_layers=2, p_depol=0.035, p_readout=0.04, init_gain=1.0, gain_ema=0.0, backend="torch", torch_device=tdev_str, seed=sd)
        bm.train_qnspsa_baseline(mqn, z_tr, c_tr, y_tr, max_shots=max_shots, shots_per_run=20, lr=0.08, c0=0.18, clip_bound=0.90, chain_rule_wcond=True, device=dev)
        full_rows["qnspsa_chainrule"].append(bm.evaluate_generated("qnspsa_cr", mqn.sample_batch(conds=ev_c, shots_per_step=160, step_noise_scale=0.35), ev_lbl, ae, clf, z_te, x_te, y_te, dev)[0])

        # Class-Mean Prototype + Noise / Deterministic Controls
        rng_p = np.random.default_rng(sd + 101)
        z_proto_n = np.clip(cls_means[ev_lbl] + rng_p.normal(0.0, 0.35, size=(len(ev_lbl), 12)), -1.0, 1.0)
        z_proto_0 = np.clip(cls_means[ev_lbl], -1.0, 1.0)
        full_rows["class_mean_proto"].append(bm.evaluate_generated("class_mean_proto", z_proto_n, ev_lbl, ae, clf, z_te, x_te, y_te, dev)[0])
        full_rows["class_mean_det"].append(bm.evaluate_generated("class_mean_det", z_proto_0, ev_lbl, ae, clf, z_te, x_te, y_te, dev)[0])

        # Classical-DDPM 80k (200st) vs 160k (400st) Cosine LR vs Const LR Control
        n198 = m.n_qpu_params + m.cond_proj.size
        steps80 = max(1, max_shots // 400)
        full_rows["ddpm_80k_cos"].append(_eval_ddpm(bm.train_classical_baseline(z_tr, c_tr, y_tr, n198, steps80, seed=sd, use_cosine_lr=True), "ddpm_80k_cos", sd))
        full_rows["ddpm_160k_const"].append(_eval_ddpm(bm.train_classical_baseline(z_tr, c_tr, y_tr, n198, 2 * steps80, seed=sd, use_cosine_lr=False), "ddpm_160k_const", sd))
        full_rows["ddpm_160k_cos"].append(_eval_ddpm(bm.train_classical_baseline(z_tr, c_tr, y_tr, n198, 2 * steps80, seed=sd, use_cosine_lr=True), "ddpm_160k_cos", sd))

        # Unconditional (cond_dim=0, 0 classical weights)
        for uk, topo in [("uncond_qer_all", "all"), ("uncond_qer_nozz", "none")]:
            mu = QuantumIsingDenoiser(12, cond_dim=0, n_hidden=0, topology=topo, n_layers=2, p_depol=0.035, p_readout=0.04, init_gain=0.75, gain_ema=0.10, backend="torch", torch_device=tdev_str, seed=sd)
            QERDTrainer(mu, QEROptimizer(mu.parameters_dict(), QERConfig(clip_bound=0.90))).train(z_tr, None, stratify_labels=y_tr, max_shots=max_shots, shots_per_run=20, batch_size=10, eval_every_batches=999999, verbose=False)
            metu, _ = bm.evaluate_generated(uk, mu.sample_batch(n_samples=len(ev_lbl), shots_per_step=160, step_noise_scale=0.70), ev_lbl, ae, clf, z_te, x_te, y_te, dev)
            full_rows[uk].append(metu)

    return {
        "label_qubits": [{"key": k, "label": lbl, "aggregate": _agg_dicts(lq_rows[k])} for k, lbl in lq_keys],
        "full_mnist": [{"key": k, "label": lbl, "aggregate": _agg_dicts(full_rows[k])} for k, lbl in full_keys],
    }


class _HybridResBase(nn.Module):
    """Shared residual classical trunk producing bounded field conditioning h_cond [B, 12] (no skip bypass)."""

    def __init__(self, hidden: int = 48, diffusion_steps: int = 4, gain: float = 0.70):
        super().__init__()
        self.diffusion_steps = int(diffusion_steps)
        self.alpha_bar = bm.make_alpha_bar(self.diffusion_steps, schedule="cosine")
        self.gain = float(gain)
        self.in_proj = nn.Sequential(nn.Linear(23, hidden), nn.SiLU())
        self.res = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.head = nn.Linear(hidden, 12)

    def _trunk(self, x_t: torch.Tensor, t_step: int, cond: torch.Tensor) -> torch.Tensor:
        t_norm = torch.full((x_t.shape[0], 1), float(t_step) / self.diffusion_steps, dtype=x_t.dtype, device=x_t.device)
        h = self.in_proj(torch.cat([x_t, t_norm, cond], dim=1))
        h = h + self.res(h)
        return 2.5 * torch.tanh(self.head(h) * 0.5)

    @torch.no_grad()
    def sample_batch(self, cond_np: np.ndarray, device: torch.device, seed: int = 123, step_noise_scale: float = 0.85, init_std: float = 0.90) -> np.ndarray:
        rng = np.random.default_rng(seed)
        n = cond_np.shape[0]
        x_t = np.clip(rng.normal(0.0, init_std, size=(n, 12)), -1.0, 1.0).astype(np.float32)
        c_t = torch.from_numpy(cond_np.astype(np.float32)).to(device)
        for t in range(self.diffusion_steps, 0, -1):
            x0_hat = self.forward_t(torch.from_numpy(x_t).to(device), t, c_t).cpu().numpy()
            if t > 1:
                a_next = self.alpha_bar[t - 1]
                eps = rng.normal(0.0, step_noise_scale, size=(n, 12)).astype(np.float32)
                x_t = np.clip(math.sqrt(a_next) * x0_hat + math.sqrt(1.0 - a_next) * eps, -1.0, 1.0)
            else:
                x_t = x0_hat
        return x_t.astype(np.float64)


class HybridResMLPQER(_HybridResBase):
    """Residual classical trunk driving a noisy 12-qubit QERLayer bottleneck (no skip bypass)."""

    def __init__(self, denoiser: QuantumIsingDenoiser, hidden: int = 48, shots: int = 24, gain: float = 0.70):
        super().__init__(hidden=hidden, diffusion_steps=denoiser.diffusion_steps, gain=gain)
        self.denoiser = denoiser
        self.alpha_bar = denoiser.alpha_bar
        self.qpu = QERLayer(denoiser, shots=shots, beta=0.45, gain=gain)

    def forward_t(self, x_t: torch.Tensor, t_step: int, cond: torch.Tensor) -> torch.Tensor:
        return self.qpu(x_t, self._trunk(x_t, t_step, cond), t_step=t_step)


HybridResMLPQERD = HybridResMLPQER


class HybridResMLPExactAD(_HybridResBase):
    """Identical ResMLP + 12-qubit unitary Ising bottleneck trained with noiseless Exact-AD VJP."""

    def __init__(self, edges: List[Tuple[int, int]], hidden: int = 48, gain: float = 0.70, seed: int = 0):
        super().__init__(hidden=hidden, diffusion_steps=4, gain=gain)
        self.edges = list(edges)
        self.circ: Optional[bm._DiffIsingCircuit] = None
        gen = torch.Generator().manual_seed(seed)
        init_th = 0.1 * torch.randn(len(self.edges) + 12, generator=gen)
        self.qpu_theta = nn.Parameter(init_th)

    def forward_t(self, x_t: torch.Tensor, t_step: int, cond: torch.Tensor) -> torch.Tensor:
        if self.circ is None or self.circ.device != x_t.device:
            self.circ = bm._DiffIsingCircuit(12, cond_dim=0, edges=self.edges, n_hidden=0, n_layers=2, device=x_t.device)
        h_cond = self._trunk(x_t, t_step, cond)
        t_arr = torch.full((x_t.shape[0],), int(t_step), dtype=torch.long, device=x_t.device)
        _, z_vis = self.circ.expectation(
            x_t, t_arr, cond=None, theta=self.qpu_theta, cond_proj=None, nudge_target=h_cond, nudge_beta=1.0
        )
        return torch.clamp(z_vis / self.gain, -1.0, 1.0)


class ClassicalResMLPNoiseMatched(_HybridResBase):
    """Identical ResMLP trunk + parameter-matched classical readout + 24-shot Binomial readout noise."""

    def __init__(self, hidden: int = 48, shots: int = 24, gain: float = 0.70):
        super().__init__(hidden=hidden, diffusion_steps=4, gain=gain)
        self.shots = int(shots)
        self.readout = nn.Linear(24, 12)

    def forward_t(self, x_t: torch.Tensor, t_step: int, cond: torch.Tensor) -> torch.Tensor:
        h_cond = self._trunk(x_t, t_step, cond)
        z_clean = torch.tanh(self.readout(torch.cat([h_cond, x_t], dim=1)))
        with torch.no_grad():
            std = torch.sqrt(torch.clamp(1.0 - z_clean * z_clean, min=0.05) / float(self.shots))
            noise = torch.randn_like(z_clean) * std
        return torch.clamp(z_clean + noise, -1.0, 1.0)


def run_study2_hybrid_capacity(
    z_tr: np.ndarray, y_tr: np.ndarray, z_te: np.ndarray, x_te: np.ndarray, y_te: np.ndarray,
    ae: nn.Module, clf: bm.EvalCNN, dev: torch.device, tdev_str: str,
    seeds: List[int], n_steps: int, batch_size: int, shots: int, eval_per_cls: int,
) -> Dict[str, Any]:
    print("\n=== [Study 2] Hybrid ClassicalTrunk + QERLayer vs Exact-AD vs ClassicalNoiseMatched ===", flush=True)
    eye_c = np.eye(10, dtype=np.float64)
    c_tr = eye_c[y_tr]
    ev_lbl = np.repeat(np.arange(10, dtype=np.int64), eval_per_cls)
    ev_c = eye_c[ev_lbl]
    pools = [np.flatnonzero(y_tr == c) for c in range(10)]
    cached_real = {sd: pcd.precompute_real_knn_ref(clf, x_te, dev, seed=sd) for sd in seeds}

    widths = [48, 128]
    edges_all = bm.build_ising_edges(12, "all")
    out_rows = []
    controls: Dict[str, List[Dict[str, float]]] = {
        "exact_ad_h48": [], "exact_ad_h128": [],
        "cl_noise_h48": [], "cl_noise_h128": [],
    }

    def _train_hybrid(model: _HybridResBase, rng_s: np.random.Generator) -> None:
        lr = 3.5e-3
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        ab = model.alpha_bar
        model.train()
        for st in range(1, n_steps + 1):
            lr_k = 0.08 * lr + 0.92 * lr * 0.5 * (1.0 + math.cos(math.pi * (st - 1) / max(1, n_steps - 1)))
            for pg in opt.param_groups:
                pg["lr"] = lr_k
            opt.zero_grad()
            t_pair = (1, 3) if (st % 2 == 1) else (2, 4)
            for ts in t_pair:
                ids = bm._sample_stratified_ids(rng_s, pools, batch_size)
                tg = torch.from_numpy(z_tr[ids].astype(np.float32)).to(dev)
                cb = torch.from_numpy(c_tr[ids].astype(np.float32)).to(dev)
                a = ab[ts]
                eps = torch.from_numpy(rng_s.normal(0.0, 0.70, size=tg.shape).astype(np.float32)).to(dev)
                xt = torch.clamp(math.sqrt(a) * tg + math.sqrt(1.0 - a) * eps, -1.0, 1.0)
                loss = 0.5 * F.mse_loss(model.forward_t(xt, ts, cb), tg)
                loss.backward()
            opt.step()
        model.eval()

    for hid in widths:
        per_sn: Dict[float, List[Dict[str, float]]] = {0.35: [], 0.85: []}
        n_params = 0
        for sd in seeds:
            torch.manual_seed(sd)
            rng = np.random.default_rng(sd)
            qpu_den = QuantumIsingDenoiser(
                12, cond_dim=12, n_hidden=0, topology="all", n_layers=2,
                p_depol=0.035, p_readout=0.04, init_gain=0.70, gain_ema=0.0,
                backend="torch", torch_device=tdev_str, seed=sd,
            )
            model = HybridResMLPQER(qpu_den, hidden=hid, shots=shots, gain=0.70).to(dev)
            n_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
            t0 = time.time()
            _train_hybrid(model, rng)
            for sn in (0.35, 0.85):
                zg = model.sample_batch(ev_c, dev, seed=100 + sd, step_noise_scale=sn)
                met, _ = bm.evaluate_generated(f"HybridQER-h{hid}-sn{int(sn*100)}", zg, ev_lbl, ae, clf, z_te, x_te, y_te, dev)
                geo = pcd.measure_knn_radius_geometry(ae, clf, zg, x_te, dev, seed=sd, cached_real=cached_real[sd])
                met.update(geo)
                per_sn[sn].append(met)
            print(f"  [HybridQER h={hid} P={n_params} Seed {sd}] {time.time()-t0:.1f}s | sn0.85 Rec={per_sn[0.85][-1]['recall']:.3f} Acc={per_sn[0.85][-1]['clf_acc']*100:.1f}%", flush=True)

            # Control 1: Hybrid Exact-AD QPU bottleneck (noiseless, exact VJP) at sn=0.85
            torch.manual_seed(sd)
            m_ad = HybridResMLPExactAD(edges_all, hidden=hid, gain=0.70, seed=sd).to(dev)
            _train_hybrid(m_ad, np.random.default_rng(sd))
            zg_ad = m_ad.sample_batch(ev_c, dev, seed=100 + sd, step_noise_scale=0.85)
            met_ad, _ = bm.evaluate_generated(f"HybridExactAD-h{hid}", zg_ad, ev_lbl, ae, clf, z_te, x_te, y_te, dev)
            met_ad.update(pcd.measure_knn_radius_geometry(ae, clf, zg_ad, x_te, dev, seed=sd, cached_real=cached_real[sd]))
            met_ad["n_params"] = float(sum(p.numel() for p in m_ad.parameters() if p.requires_grad))
            controls[f"exact_ad_h{hid}"].append(met_ad)

            # Control 2: Classical ResMLP + 24-shot Matched Readout Noise (no QPU) at sn=0.85
            torch.manual_seed(sd)
            m_cn = ClassicalResMLPNoiseMatched(hidden=hid, shots=shots, gain=0.70).to(dev)
            _train_hybrid(m_cn, np.random.default_rng(sd))
            zg_cn = m_cn.sample_batch(ev_c, dev, seed=100 + sd, step_noise_scale=0.85)
            met_cn, _ = bm.evaluate_generated(f"ClassicalNoise-h{hid}", zg_cn, ev_lbl, ae, clf, z_te, x_te, y_te, dev)
            met_cn.update(pcd.measure_knn_radius_geometry(ae, clf, zg_cn, x_te, dev, seed=sd, cached_real=cached_real[sd]))
            met_cn["n_params"] = float(sum(p.numel() for p in m_cn.parameters() if p.requires_grad))
            controls[f"cl_noise_h{hid}"].append(met_cn)

        out_rows.append({
            "hidden": hid,
            "n_params": n_params,
            "sn035": _agg_dicts(per_sn[0.35]),
            "sn085": _agg_dicts(per_sn[0.85]),
        })
    return {"models": out_rows, "controls": {k: _agg_dicts(v) for k, v in controls.items()}}


class _InvOLSCalibrator:
    """Inverse OLS readout calibrator (x0_hat = a_inv * z) testing Theorem 2 gain clamping."""

    def __init__(self, diffusion_steps: int, n_visible: int, init_gain: float = 1.0, ema: float = 0.10, bounds: Tuple[float, float] = (0.20, 1.00)):
        self.bounds = (float(bounds[0]), float(bounds[1]))
        self.ema = float(ema)
        self.gain = np.full(diffusion_steps + 1, float(np.clip(init_gain, *self.bounds)), dtype=np.float64)
        self._clip_hits = 0
        self._upper_clip_hits = 0
        self._clip_total = 0

    @property
    def mean_gain(self) -> float:
        return float(np.mean(self.gain[1:]))

    @property
    def clip_fraction(self) -> float:
        return float(self._clip_hits / self._clip_total) if self._clip_total > 0 else 0.0

    @property
    def upper_clip_fraction(self) -> float:
        return float(self._upper_clip_hits / self._clip_total) if self._clip_total > 0 else 0.0

    def update(self, t: int, z_obs: np.ndarray, x0_target: np.ndarray) -> None:
        if self.ema <= 0.0:
            return
        denom = float(np.dot(z_obs, z_obs)) + 1e-6
        slope = float(np.dot(x0_target, z_obs)) / denom
        lo, hi = self.bounds
        cand = float((1.0 - self.ema) * self.gain[t] + self.ema * slope)
        self._clip_total += 1
        if cand <= lo or cand >= hi:
            self._clip_hits += 1
        if cand >= hi:
            self._upper_clip_hits += 1
        self.gain[t] = float(np.clip(cand, lo, hi))

    def calibrate(self, t: int, z_obs: np.ndarray) -> np.ndarray:
        return np.clip(self.gain[t] * np.asarray(z_obs, dtype=np.float64), -1.0, 1.0)


def _train_standalone_qer_layer(
    z_tr: np.ndarray, y_tr: np.ndarray, z_te: np.ndarray, y_te: np.ndarray,
    dev: torch.device, tdev_str: str, seed: int, n_steps: int, batch_size: int, shots: int,
) -> Dict[str, float]:
    """Standalone non-diffusion 10-way classification/denoising check using QERLayer(x, h) vs Exact-AD."""
    rng = np.random.default_rng(seed)
    n_tr, n_te = z_tr.shape[0], z_te.shape[0]
    x_tr_n = np.clip(0.75 * z_tr + 0.50 * rng.normal(0.0, 1.0, size=z_tr.shape), -1.0, 1.0).astype(np.float32)
    x_te_n = np.clip(0.75 * z_te + 0.50 * rng.normal(0.0, 1.0, size=z_te.shape), -1.0, 1.0).astype(np.float32)
    xt_tr_t = torch.from_numpy(x_tr_n).to(dev)
    y_tr_t = torch.from_numpy(y_tr.astype(np.int64)).to(dev)
    xt_te_t = torch.from_numpy(x_te_n).to(dev)
    edges_all = bm.build_ising_edges(12, "all")

    def _run_head(mode: str) -> Tuple[float, float]:
        torch.manual_seed(seed)
        r_loc = np.random.default_rng(seed + 19)
        enc = nn.Sequential(nn.Linear(12, 24), nn.SiLU(), nn.Linear(24, 12)).to(dev)
        cls_head = nn.Linear(12, 10).to(dev)
        if mode == "qer_layer":
            q_den = QuantumIsingDenoiser(12, cond_dim=0, n_hidden=0, topology="all", n_layers=2, p_depol=0.035, p_readout=0.04, init_gain=0.75, gain_ema=0.0, backend="torch", torch_device=tdev_str, seed=seed)
            bot: nn.Module = QERLayer(q_den, shots=shots, beta=0.45, gain=0.75).to(dev)
            params = list(enc.parameters()) + list(bot.parameters()) + list(cls_head.parameters())
        else:
            bot = nn.Identity()
            circ = bm._DiffIsingCircuit(12, cond_dim=0, edges=edges_all, n_layers=2, device=dev)
            th_ad = nn.Parameter(torch.zeros(len(edges_all) + 12, dtype=torch.float32, device=dev))
            params = list(enc.parameters()) + [th_ad] + list(cls_head.parameters())
        opt = torch.optim.Adam(params, lr=5e-3)
        t_eval_fixed = torch.full((n_te,), 2, dtype=torch.long, device=dev)
        for _ in range(n_steps):
            ids = r_loc.integers(0, n_tr, size=batch_size)
            xb = xt_tr_t[ids]; yb = y_tr_t[ids]
            hc = 2.0 * torch.tanh(0.5 * enc(xb))
            if mode == "qer_layer":
                zf = bot(xb, hc)  # t_step=None standalone call!
            else:
                tb = torch.full((batch_size,), 2, dtype=torch.long, device=dev)
                _, zv = circ.expectation(xb, tb, None, th_ad, None, nudge_target=hc, nudge_beta=1.0)
                zf = torch.clamp(zv / 0.75, -1.0, 1.0)
            loss = F.cross_entropy(cls_head(zf), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            hc_e = 2.0 * torch.tanh(0.5 * enc(xt_te_t))
            if mode == "qer_layer":
                zf_e = bot(xt_te_t, hc_e)
            else:
                _, zv_e = circ.expectation(xt_te_t, t_eval_fixed, None, th_ad, None, nudge_target=hc_e, nudge_beta=1.0)
                zf_e = torch.clamp(zv_e / 0.75, -1.0, 1.0)
            logits = cls_head(zf_e)
            acc = float((logits.argmax(dim=1).cpu().numpy() == y_te).mean())
            ce = float(F.cross_entropy(logits, torch.from_numpy(y_te.astype(np.int64)).to(dev)).item())
        return acc, ce

    acc_q, ce_q = _run_head("qer_layer")
    acc_a, ce_a = _run_head("exact_ad")
    return {"qer_layer_acc": acc_q, "qer_layer_ce": ce_q, "exact_ad_acc": acc_a, "exact_ad_ce": ce_a}


def run_study3_optimizer_tuning(
    z_tr: np.ndarray, y_tr: np.ndarray, z_te: np.ndarray, x_te: np.ndarray, y_te: np.ndarray,
    ae: nn.Module, clf: bm.EvalCNN, dev: torch.device, tdev_str: str,
    seeds: List[int], max_shots: int, quick: bool, reuse_spsa_grid_cache: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    print("\n=== [Study 3] SPSA Grid, Chain-Rule/4sh/QN-SPSA, Exact-PSR/iCANS, Prop. 2 Calibrator & StabilizedAdam ===", flush=True)
    eye_c = np.eye(10, dtype=np.float64)
    c_tr = eye_c[y_tr]
    ev_lbl = np.repeat(np.arange(10, dtype=np.int64), 30 if quick else 100)
    ev_c = eye_c[ev_lbl]
    pools = [np.flatnonzero(y_tr == c) for c in range(10)]

    if reuse_spsa_grid_cache:
        print("  [Study 3] Reusing cached 3x3 SPSA grid results (--reuse-spsa-grid).", flush=True)
        grid_results = reuse_spsa_grid_cache
    else:
        grid_cfgs = [(0.08, 0.18)] if quick else [
            (0.04, 0.10), (0.04, 0.22), (0.04, 0.40),
            (0.12, 0.10), (0.12, 0.22), (0.12, 0.40),
            (0.25, 0.10), (0.25, 0.22), (0.25, 0.40),
        ]
        grid_results = []
        for lr, c0 in grid_cfgs:
            per_s = []
            for sd in seeds:
                m = QuantumIsingDenoiser(12, cond_dim=10, n_hidden=0, topology="all", n_layers=2, p_depol=0.035, p_readout=0.04, init_gain=1.0, gain_ema=0.0, backend="torch", torch_device=tdev_str, seed=sd)
                bm.train_adam_spsa_baseline(m, z_tr, c_tr, y_tr, max_shots=max_shots, shots_per_run=20, batch_size=10, lr=lr, c0=c0)
                met, _ = bm.evaluate_generated(f"SPSA({lr},{c0})", m.sample_batch(conds=ev_c, shots_per_step=160, step_noise_scale=0.35), ev_lbl, ae, clf, z_te, x_te, y_te, dev)
                per_s.append(met)
            grid_results.append({"lr": lr, "c0": c0, "aggregate": _agg_dicts(per_s)})

    special_keys = [
        ("spsa_act_198p", "Active-Col SPSA (90 active/198p, lr=0.25, c0=0.30)"),
        ("spsa_cr_default", "Chain-Rule W_cond SPSA (78 QPU params, 20 sh/run)"),
        ("spsa_4sh_198p", "Chain-Rule W_cond SPSA (78 QPU params, 4 sh/run)"),
        ("qnspsa_cr", "Quantum Natural SPSA + Chain-Rule W_cond (20 sh/run)"),
        ("spsa_wcond_std", "W_cond-Only SPSA (theta=0, 120p, lr=0.08, c0=0.18)"),
        ("spsa_wcond_act", "W_cond-Only Active-Col SPSA (theta=0, 12 active, lr=0.25)"),
        ("icans_psr", "iCANS Stochastic FD (k=15, 6 sh/q)"),
        ("icans_exact_psr", "iCANS Exact-PSR (+-pi/2, k=15, 6 sh/q, Chain-Rule W_cond)"),
        ("exact_psr_198p", "Exact-PSR (+-pi/2, Fixed 313 q/step, Chain-Rule W_cond)"),
        ("rosalin_psr", "Rosalin Shot-Allocated PSR (all 90 active, ~4 sh/q)"),
    ]
    special_rows: Dict[str, List[Dict[str, float]]] = {k: [] for k, _ in special_keys}

    cal_keys = ["fwd_ols_c10", "inv_ols_c10", "inv_ols_c20"]
    cal_rows: Dict[str, List[Dict[str, float]]] = {k: [] for k in cal_keys}
    step_keys = ["adam_qer_full", "no_autobeta", "no_wd", "no_cos_lr", "momentum_only"]
    step_rows: Dict[str, List[Dict[str, float]]] = {k: [] for k in step_keys}
    standalone_rows: List[Dict[str, float]] = []

    for sd in seeds:
        for sk in ["spsa_act_198p", "spsa_wcond_std", "spsa_wcond_act"]:
            m = QuantumIsingDenoiser(12, cond_dim=10, n_hidden=0, topology=("none" if "wcond" in sk else "all"), n_layers=2, p_depol=0.035, p_readout=0.04, init_gain=1.0, gain_ema=0.0, backend="torch", torch_device=tdev_str, seed=sd)
            lr = 0.08 if sk == "spsa_wcond_std" else 0.25
            c0 = 0.18 if sk == "spsa_wcond_std" else 0.30
            m_th, v_th = np.zeros_like(m.theta), np.zeros_like(m.theta)
            m_cp, v_cp = np.zeros_like(m.cond_proj), np.zeros_like(m.cond_proj)
            totb = max(1, max_shots // 400)
            for st in range(1, totb + 1):
                ck = c0 / (st ** 0.101)
                lrk = 0.015 + 0.5 * (lr - 0.015) * (1.0 + math.cos(math.pi * (st - 1) / max(1, totb - 1)))
                ids = bm._sample_stratified_ids(m.rng, pools, 10)
                tb = (np.arange(10) % 4 + 1).astype(np.int64); m.rng.shuffle(tb)
                tg, cb = z_tr[ids], c_tr[ids]
                ab_s = m.alpha_bar[tb][:, None]
                xt = np.clip(np.sqrt(ab_s) * tg + np.sqrt(1.0 - ab_s) * m.rng.normal(0.0, 0.70, size=tg.shape), -1.0, 1.0)
                d_cp = m.rng.choice([-1.0, 1.0], size=(10,) + m.cond_proj.shape)
                if "act" in sk:
                    d_cp = d_cp * (np.abs(cb[:, None, :]) > 1e-8)
                d_th = np.zeros((10,) + m.theta.shape) if "wcond" in sk else m.rng.choice([-1.0, 1.0], size=(10,) + m.theta.shape)
                _, zb = m.measure_observables_batch(
                    np.concatenate([xt, xt]), np.concatenate([tb, tb]), shots=20,
                    cond_batch=np.concatenate([cb, cb]),
                    theta_batch=None if "wcond" in sk else np.concatenate([m.theta[None] + ck * d_th, m.theta[None] - ck * d_th]),
                    cond_proj_batch=np.concatenate([m.cond_proj[None] + ck * d_cp, m.cond_proj[None] - ck * d_cp]),
                )
                diff = (np.mean((zb[:10] - tg) ** 2, axis=1) - np.mean((zb[10:] - tg) ** 2, axis=1)) / (2.0 * ck)
                g_th = -np.mean(diff[:, None] * d_th, axis=0)
                g_cp = -np.mean(diff[:, None, None] * d_cp, axis=0)
                if "wcond" in sk:
                    m_cp[:] = 0.75 * m_cp + 0.25 * g_cp; v_cp[:] = 0.98 * v_cp + 0.02 * (g_cp ** 2)
                    m.cond_proj[:] = np.clip(m.cond_proj + lrk * (m_cp / (1.0 - 0.75 ** st)) / (np.sqrt(v_cp / (1.0 - 0.98 ** st)) + 1e-6), -0.90, 0.90)
                else:
                    bm._adam_ascent_update(m, g_th, g_cp, m_th, v_th, m_cp, v_cp, lrk, st, 0.90)
            mets, _ = bm.evaluate_generated(sk, m.sample_batch(conds=ev_c, shots_per_step=160, step_noise_scale=0.35), ev_lbl, ae, clf, z_te, x_te, y_te, dev)
            special_rows[sk].append(mets)

        # Chain-rule W_cond SPSA (20sh, 4sh) and QN-SPSA
        for sk_cr, sh_r in [("spsa_cr_default", 20), ("spsa_4sh_198p", 4)]:
            mcr = QuantumIsingDenoiser(12, cond_dim=10, n_hidden=0, topology="all", n_layers=2, p_depol=0.035, p_readout=0.04, init_gain=1.0, gain_ema=0.0, backend="torch", torch_device=tdev_str, seed=sd)
            bm.train_adam_spsa_baseline(mcr, z_tr, c_tr, y_tr, max_shots=max_shots, shots_per_run=sh_r, lr=0.08, c0=0.18, clip_bound=0.90, chain_rule_wcond=True)
            special_rows[sk_cr].append(bm.evaluate_generated(sk_cr, mcr.sample_batch(conds=ev_c, shots_per_step=160, step_noise_scale=0.35), ev_lbl, ae, clf, z_te, x_te, y_te, dev)[0])

        mqn = QuantumIsingDenoiser(12, cond_dim=10, n_hidden=0, topology="all", n_layers=2, p_depol=0.035, p_readout=0.04, init_gain=1.0, gain_ema=0.0, backend="torch", torch_device=tdev_str, seed=sd)
        bm.train_qnspsa_baseline(mqn, z_tr, c_tr, y_tr, max_shots=max_shots, shots_per_run=20, lr=0.08, c0=0.18, clip_bound=0.90, chain_rule_wcond=True, device=dev)
        special_rows["qnspsa_cr"].append(bm.evaluate_generated("QN-SPSA-CR", mqn.sample_batch(conds=ev_c, shots_per_step=160, step_noise_scale=0.35), ev_lbl, ae, clf, z_te, x_te, y_te, dev)[0])

        # iCANS Stochastic FD (k_sub=15, 6 shots/query)
        mi = QuantumIsingDenoiser(12, cond_dim=10, n_hidden=0, topology="all", n_layers=2, p_depol=0.035, p_readout=0.04, init_gain=1.0, gain_ema=0.0, backend="torch", torch_device=tdev_str, seed=sd)
        m_th, v_th, m_cp, v_cp = np.zeros_like(mi.theta), np.zeros_like(mi.theta), np.zeros_like(mi.cond_proj), np.zeros_like(mi.cond_proj)
        ema_g = np.ones(198, dtype=np.float64) * 0.1
        shots_used, st = 0, 0
        k_sub, sh_q = 15, 6
        while shots_used + 2 * k_sub * sh_q <= max_shots:
            st += 1
            cls_i = st % 10; ts = (st % 4) + 1
            tg = z_tr[int(mi.rng.choice(pools[cls_i]))]; cv = c_tr[int(np.flatnonzero(y_tr == cls_i)[0])]
            xt = np.clip(math.sqrt(mi.alpha_bar[ts]) * tg + math.sqrt(1.0 - mi.alpha_bar[ts]) * mi.rng.normal(0.0, 0.70, size=12), -1.0, 1.0)
            act = np.concatenate([np.arange(78), 78 + np.arange(12) * 10 + cls_i])
            probs = ema_g[act] + 0.05; probs /= probs.sum()
            pick_pos = mi.rng.choice(len(act), size=k_sub, replace=False, p=probs)
            pick = act[pick_pos]
            base_f = np.concatenate([mi.theta, mi.cond_proj.ravel()])
            fp = np.broadcast_to(base_f, (k_sub, 198)).copy(); fm = fp.copy()
            fp[np.arange(k_sub), pick] += 0.25; fm[np.arange(k_sub), pick] -= 0.25
            _, z2 = mi.measure_observables_batch(
                np.broadcast_to(xt, (2 * k_sub, 12)), np.full(2 * k_sub, ts, dtype=np.int64), shots=sh_q,
                cond_batch=np.broadcast_to(cv, (2 * k_sub, 10)),
                theta_batch=np.concatenate([fp[:, :78], fm[:, :78]]),
                cond_proj_batch=np.concatenate([fp[:, 78:].reshape(k_sub, 12, 10), fm[:, 78:].reshape(k_sub, 12, 10)]),
            )
            shots_used += 2 * k_sub * sh_q
            g_sub = -(np.mean((z2[:k_sub] - tg[None, :]) ** 2, axis=1) - np.mean((z2[k_sub:] - tg[None, :]) ** 2, axis=1)) / 0.50
            ema_g[pick] = 0.7 * ema_g[pick] + 0.3 * np.abs(g_sub)
            g_full = np.zeros(198, dtype=np.float64)
            g_full[pick] = g_sub / np.maximum(k_sub * probs[pick_pos], 1e-3)
            lrk = 0.015 + 0.5 * (0.08 - 0.015) * (1.0 + math.cos(math.pi * min(1.0, shots_used / max(1, max_shots))))
            bm._adam_ascent_update(mi, g_full[:78], g_full[78:].reshape(12, 10), m_th, v_th, m_cp, v_cp, lrk, st, 0.90)
        special_rows["icans_psr"].append(bm.evaluate_generated("iCANS-FD", mi.sample_batch(conds=ev_c, shots_per_step=160, step_noise_scale=0.35), ev_lbl, ae, clf, z_te, x_te, y_te, dev)[0])

        # iCANS Exact-PSR (+-pi/2 per layer, k_sub=15, 6 shots/query, + 6-shot base z_0, Chain-Rule W_cond)
        mie = QuantumIsingDenoiser(12, cond_dim=10, n_hidden=0, topology="all", n_layers=2, p_depol=0.035, p_readout=0.04, init_gain=1.0, gain_ema=0.0, backend="torch", torch_device=tdev_str, seed=sd)
        m_th, v_th, m_cp, v_cp = np.zeros_like(mie.theta), np.zeros_like(mie.theta), np.zeros_like(mie.cond_proj), np.zeros_like(mie.cond_proj)
        n_pairs = mie.theta.size * mie.n_layers
        ema_pair = np.ones(n_pairs, dtype=np.float64) * 0.1
        shots_used, st = 0, 0
        sh_q_e = 6
        mold_gains = [2.0 * mie.dt * mie.temp_beta * ((l + 1) / mie.n_layers) for l in range(mie.n_layers)]
        step_sh_e = (1 + 2 * k_sub) * sh_q_e
        while shots_used + step_sh_e <= max_shots:
            st += 1
            cls_i = st % 10; ts = (st % 4) + 1
            tg = z_tr[int(mie.rng.choice(pools[cls_i]))]; cv = c_tr[int(np.flatnonzero(y_tr == cls_i)[0])]
            xt = np.clip(math.sqrt(mie.alpha_bar[ts]) * tg + math.sqrt(1.0 - mie.alpha_bar[ts]) * mie.rng.normal(0.0, 0.70, size=12), -1.0, 1.0)
            in_pin = mie.in_pinning_base + mie.in_pinning_slope * (1.0 - float(ts) / mie.diffusion_steps)
            h_base = mie.theta[mie.n_edges:] + in_pin * xt + mie.cond_proj @ cv
            j_base = mie.theta[:mie.n_edges]
            probs_p = ema_pair + 0.05; probs_p /= probs_p.sum()
            pick_p = mie.rng.choice(n_pairs, size=k_sub, replace=False, p=probs_p)
            h_stk = np.broadcast_to(h_base[None, None, :], (1 + 2 * k_sub, mie.n_layers, 12)).copy()
            j_stk = np.broadcast_to(j_base[None, None, :], (1 + 2 * k_sub, mie.n_layers, mie.n_edges)).copy()
            meta = []
            for idx_p, pid in enumerate(pick_p):
                l_idx, th_idx = divmod(int(pid), mie.theta.size)
                mg = mold_gains[l_idx]
                ang = (0.5 * math.pi) / mg
                if th_idx < mie.n_edges:
                    j_stk[1 + 2 * idx_p, l_idx, th_idx] += ang
                    j_stk[2 + 2 * idx_p, l_idx, th_idx] -= ang
                else:
                    qi = th_idx - mie.n_edges
                    h_stk[1 + 2 * idx_p, l_idx, qi] += ang
                    h_stk[2 + 2 * idx_p, l_idx, qi] -= ang
                meta.append((idx_p, int(pid), th_idx, mg))
            z_all_e = bm._sample_layer_perturbed_z(mie, h_stk, j_stk, float(ts) / mie.diffusion_steps, sh_q_e, dev)
            shots_used += step_sh_e
            err_v = 2.0 / 12.0 * (z_all_e[0] - tg)
            g_th_e = np.zeros_like(mie.theta)
            for idx_p, pid, th_idx, mg in meta:
                dz = 0.5 * mg * (z_all_e[1 + 2 * idx_p] - z_all_e[2 + 2 * idx_p])
                dL_i = -float(np.dot(err_v, dz))
                ema_pair[pid] = 0.7 * ema_pair[pid] + 0.3 * abs(dL_i)
                g_th_e[th_idx] += dL_i / max(k_sub * probs_p[pid], 1e-3)
            g_cp_e = np.outer(g_th_e[mie.n_edges:], cv)
            lrk = 0.015 + 0.5 * (0.08 - 0.015) * (1.0 + math.cos(math.pi * min(1.0, shots_used / max(1, max_shots))))
            bm._adam_ascent_update(mie, g_th_e, g_cp_e, m_th, v_th, m_cp, v_cp, lrk, st, 0.90)
        special_rows["icans_exact_psr"].append(bm.evaluate_generated("iCANS-ExactPSR", mie.sample_batch(conds=ev_c, shots_per_step=160, step_noise_scale=0.35), ev_lbl, ae, clf, z_te, x_te, y_te, dev)[0])

        # Fixed Exact-PSR (313 queries/step on L=2, Chain-Rule W_cond)
        mep = QuantumIsingDenoiser(12, cond_dim=10, n_hidden=0, topology="all", n_layers=2, p_depol=0.035, p_readout=0.04, init_gain=1.0, gain_ema=0.0, backend="torch", torch_device=tdev_str, seed=sd)
        bm.train_exact_psr_baseline(mep, z_tr, c_tr, y_tr, max_shots=max_shots, shots_per_run=max(4, max_shots // 3130), lr=0.08, clip_bound=0.90, device=tdev_str)
        special_rows["exact_psr_198p"].append(bm.evaluate_generated("Exact-PSR", mep.sample_batch(conds=ev_c, shots_per_step=160, step_noise_scale=0.35), ev_lbl, ae, clf, z_te, x_te, y_te, dev)[0])

        # Rosalin-style low-shot all-active PSR (4 shots per query across all 90 active coords)
        mr = QuantumIsingDenoiser(12, cond_dim=10, n_hidden=0, topology="all", n_layers=2, p_depol=0.035, p_readout=0.04, init_gain=1.0, gain_ema=0.0, backend="torch", torch_device=tdev_str, seed=sd)
        bm.train_adam_psr_baseline(mr, z_tr, c_tr, y_tr, max_shots=max_shots, shots_per_run=4, lr=0.08, delta=0.25, clip_bound=0.90)
        special_rows["rosalin_psr"].append(bm.evaluate_generated("Rosalin-PSR-4sh", mr.sample_batch(conds=ev_c, shots_per_step=160, step_noise_scale=0.35), ev_lbl, ae, clf, z_te, x_te, y_te, dev)[0])

        # Theorem 2 Calibrator Study (FwdOLS [0.2,1.0] vs InvOLS [0.2,1.0] vs InvOLS [0.2,2.0])
        for ck in cal_keys:
            mc = QuantumIsingDenoiser(12, cond_dim=10, n_hidden=0, topology="all", n_layers=2, p_depol=0.035, p_readout=0.04, init_gain=0.75, gain_ema=0.10, backend="torch", torch_device=tdev_str, seed=sd)
            if ck == "inv_ols_c10":
                mc.calibrator = _InvOLSCalibrator(4, 12, init_gain=1.0, ema=0.10, bounds=(0.20, 1.00))
            elif ck == "inv_ols_c20":
                mc.calibrator = _InvOLSCalibrator(4, 12, init_gain=1.0, ema=0.10, bounds=(0.20, 2.00))
            QERDTrainer(mc, QEROptimizer(mc.parameters_dict(), QERConfig(clip_bound=0.90))).train(z_tr, c_tr, stratify_labels=y_tr, max_shots=max_shots, shots_per_run=20, batch_size=10, eval_every_batches=999999, verbose=False)
            mcal, _ = bm.evaluate_generated(ck, mc.sample_batch(conds=ev_c, shots_per_step=160, step_noise_scale=0.35), ev_lbl, ae, clf, z_te, x_te, y_te, dev)
            for ti in range(1, 5):
                mcal[f"gain_t{ti}"] = float(np.mean(mc.calibrator.gain[ti]))
            mcal["clip_fraction"] = float(mc.calibrator.clip_fraction)
            mcal["upper_clip_fraction"] = float(mc.calibrator.upper_clip_fraction)
            cal_rows[ck].append(mcal)

        # Standalone StabilizedAdam Leave-One-Out Stepper Ablation via QERGradientEstimator
        for st_k, ab_kw in [
            ("adam_qer_full", dict(auto_beta=True, wd=0.01, lr_min=0.015, use_adam=True)),
            ("no_autobeta", dict(auto_beta=False, wd=0.01, lr_min=0.015, use_adam=True)),
            ("no_wd", dict(auto_beta=True, wd=0.0, lr_min=0.015, use_adam=True)),
            ("no_cos_lr", dict(auto_beta=True, wd=0.01, lr_min=0.12, use_adam=True)),
            ("momentum_only", dict(auto_beta=True, wd=0.01, lr_min=0.015, use_adam=False)),
        ]:
            ms_a = QuantumIsingDenoiser(12, cond_dim=10, n_hidden=0, topology="all", n_layers=2, p_depol=0.035, p_readout=0.04, init_gain=1.0, gain_ema=0.0, backend="torch", torch_device=tdev_str, seed=sd)
            est_a = QERGradientEstimator(QERGradConfig(auto_beta_guard=ab_kw["auto_beta"]))
            stp_a = StabilizedAdam(ms_a.parameters_dict(), StabilizedAdamConfig(lr_max=0.12, lr_min=ab_kw["lr_min"], weight_decay=ab_kw["wd"], use_adam=ab_kw["use_adam"]))
            totb_a = max(1, max_shots // 400)
            for ba in range(1, totb_a + 1):
                prog = (ba - 1) / max(1, totb_a - 1)
                ids = bm._sample_stratified_ids(ms_a.rng, pools, 10)
                tc = (np.arange(10) % 4 + 1).astype(np.int64); ms_a.rng.shuffle(tc)
                tg_a, cb_a = z_tr[ids], c_tr[ids]
                xt_a = np.clip(np.sqrt(ms_a.alpha_bar[tc][:, None]) * tg_a + np.sqrt(1.0 - ms_a.alpha_bar[tc][:, None]) * ms_a.rng.normal(0.0, 0.70, size=tg_a.shape), -1.0, 1.0)
                ob2, zb2 = ms_a.measure_observables_batch(
                    np.concatenate([xt_a, xt_a]), np.concatenate([tc, tc]), shots=20,
                    cond_batch=np.concatenate([cb_a, cb_a]), nudge_target_batch=np.concatenate([tg_a, tg_a]),
                    nudge_beta_batch=np.concatenate([np.full(10, est_a.beta), np.full(10, -est_a.beta)]),
                )
                grads_a, _ = est_a.estimate(ms_a.parameters_dict(), ob2[:10], ob2[10:], z_plus_list=zb2[:10], z_minus_list=zb2[10:], conds=cb_a, nudge_dirs=tg_a)
                stp_a.step(ms_a.parameters_dict(), grads_a, progress=prog, shots_in_batch=400)
            step_rows[st_k].append(bm.evaluate_generated(st_k, ms_a.sample_batch(conds=ev_c, shots_per_step=160, step_noise_scale=0.35), ev_lbl, ae, clf, z_te, x_te, y_te, dev)[0])

        # Standalone Non-Diffusion QERLayer(x, h) Classification Check
        standalone_rows.append(
            _train_standalone_qer_layer(z_tr, y_tr, z_te, y_te, dev, tdev_str, sd, n_steps=(80 if quick else 160), batch_size=32, shots=40)
        )

    return {
        "spsa_grid": grid_results,
        "specials": [{"key": k, "label": lbl, "aggregate": _agg_dicts(special_rows[k])} for k, lbl in special_keys],
        "calibrators": {k: _agg_dicts(cal_rows[k]) for k in cal_keys},
        "stepper_ablation": {k: _agg_dicts(step_rows[k]) for k in step_keys},
        "standalone_layer": _agg_dicts(standalone_rows),
    }


def run_study4_bp_unit_snr(dev: torch.device, tdev_str: str, n_trials: int, n_mc_shot: int) -> Dict[str, Any]:
    print("\n=== [Study 4] Unit-Consistent Sum Variance & Equal-4k-Shot Cosine SNR (Surrogate & MSE) ===", flush=True)
    rng = np.random.default_rng(0)
    widths = [6, 8, 10, 12, 14]
    depths = [2, 6]
    rows = []

    for nl in depths:
        for nq in widths:
            edges = bm.build_ising_edges(nq, "all")
            ne = len(edges)
            n_p = ne + nq
            circ = bm._DiffIsingCircuit(nq, cond_dim=0, edges=edges, n_layers=nl, device=dev)
            x0 = rng.choice([-1.0, 1.0], size=(10, nq)).astype(np.float32)
            xt = np.clip(0.5 * x0 + 0.5 * rng.normal(0.0, 0.70, size=x0.shape), -1.0, 1.0).astype(np.float32)
            t_arr = (np.arange(10) % 4) + 1
            t_t = torch.tensor(t_arr, dtype=torch.int64, device=dev)
            xt_t = torch.tensor(xt, dtype=torch.float32, device=dev)
            x0_t = torch.tensor(x0, dtype=torch.float32, device=dev)

            gq_l, gc_l, gm_sum_l, gm_mean_l = [], [], [], []
            for _ in range(n_trials):
                th_np = rng.uniform(-0.5, 0.5, size=n_p).astype(np.float32)
                th_t = torch.tensor(th_np, dtype=torch.float32, device=dev, requires_grad=True)
                _, zf = circ.expectation(xt_t, t_t, None, th_t, None)
                l_sum_mse = 0.5 * torch.sum((zf - x0_t) ** 2, dim=1).mean()
                l_mean_mse = F.mse_loss(zf, x0_t)
                l_sum_corr = torch.sum(zf * x0_t, dim=1).mean()
                (gm_s,) = torch.autograd.grad(-l_sum_mse, [th_t], retain_graph=True)
                (gm_m,) = torch.autograd.grad(-l_mean_mse, [th_t], retain_graph=True)
                (gc_s,) = torch.autograd.grad(l_sum_corr, [th_t])
                with torch.no_grad():
                    op, _ = circ.expectation(xt_t, t_t, None, th_t, None, nudge_target=x0_t, nudge_beta=0.45)
                    om, _ = circ.expectation(xt_t, t_t, None, th_t, None, nudge_target=x0_t, nudge_beta=-0.45)
                    gq = ((op - om) / 0.90).mean(dim=0).cpu().numpy()
                gq_l.append(gq[:ne])
                gc_l.append(gc_s.detach().cpu().numpy()[:ne])
                gm_sum_l.append(gm_s.detach().cpu().numpy()[:ne])
                gm_mean_l.append(gm_m.detach().cpu().numpy()[:ne])

            # True 4,000-shot/step NISQ cosine SNR across BOTH equal targets (g_ec surrogate & g_em MSE)
            cos_qer_s, cos_spsa_s, cos_epsr_s = [], [], []
            cos_qer_res_m, cos_spsa_m, cos_psrfd_m, cos_epsr_m = [], [], [], []
            if nl == 2:
                m_den = QuantumIsingDenoiser(nq, cond_dim=0, n_hidden=0, topology="all", n_layers=nl, p_depol=0.035, p_readout=0.04, init_gain=1.0, gain_ema=0.0, backend="torch", torch_device=tdev_str, seed=0)
                th_ref = rng.uniform(-0.4, 0.4, size=n_p).astype(np.float64)
                m_den.theta[:] = th_ref
                th_t = torch.tensor(th_ref, dtype=torch.float32, device=dev, requires_grad=True)
                _, zf = circ.expectation(xt_t, t_t, None, th_t, None)
                (g_exact_corr_t,) = torch.autograd.grad(torch.sum(zf * x0_t, dim=1).mean(), [th_t], retain_graph=True)
                (g_exact_mse_t,) = torch.autograd.grad(-0.5 * torch.sum((zf - x0_t) ** 2, dim=1).mean(), [th_t])
                g_ec = g_exact_corr_t.detach().cpu().numpy()
                g_em = g_exact_mse_t.detach().cpu().numpy()

                sh_2run = max(1, 4000 // (2 * 10))      # 200 shots/circuit x 20 circuits = 4,000 shots
                sh_3run = max(1, 4000 // (3 * 10))      # 133 shots/circuit x 30 circuits = 3,990 shots
                sh_psr_fd = max(1, 4000 // (2 * n_p))
                sh_exact_psr = max(1, 4000 // (2 * nl * n_p))
                mold_gains = [2.0 * m_den.dt * m_den.temp_beta * ((l + 1) / nl) for l in range(nl)]

                def _cos_v(a: np.ndarray, b: np.ndarray) -> float:
                    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

                for mc in range(n_mc_shot):
                    # (1) Surrogate 4k-shot group scored vs g_ec: QER, SPSA-Surr, Exact-PSR-Surr (+-pi/2)
                    ob2, _ = m_den.measure_observables_batch(
                        np.concatenate([xt, xt]), np.concatenate([t_arr, t_arr]), shots=sh_2run,
                        nudge_target_batch=np.concatenate([x0, x0]),
                        nudge_beta_batch=np.concatenate([np.full(10, 0.45), np.full(10, -0.45)]),
                    )
                    gq_s = np.mean((ob2[:10] - ob2[10:]) / 0.90, axis=0)
                    cos_qer_s.append(_cos_v(gq_s, g_ec))

                    d_th = rng.choice([-1.0, 1.0], size=(10, n_p))
                    _, zs = m_den.measure_observables_batch(
                        np.concatenate([xt, xt]), np.concatenate([t_arr, t_arr]), shots=sh_2run,
                        theta_batch=np.concatenate([th_ref[None] + 0.18 * d_th, th_ref[None] - 0.18 * d_th]),
                    )
                    df_surr = (-np.sum(zs[:10] * x0, axis=1) - (-np.sum(zs[10:] * x0, axis=1))) / 0.36
                    cos_spsa_s.append(_cos_v(-np.mean(df_surr[:, None] * d_th, axis=0), g_ec))

                    # (2) MSE 4k-shot group scored vs g_em: QER-Residual (3-run), SPSA-MSE, PSR-FD, Exact-PSR (+-pi/2)
                    _, zf_res = m_den.measure_observables_batch(xt, t_arr, shots=sh_3run)
                    y_res = x0 - zf_res
                    ob_r, _ = m_den.measure_observables_batch(
                        np.concatenate([xt, xt]), np.concatenate([t_arr, t_arr]), shots=sh_3run,
                        nudge_target_batch=np.concatenate([y_res, y_res]),
                        nudge_beta_batch=np.concatenate([np.full(10, 0.45), np.full(10, -0.45)]),
                    )
                    cos_qer_res_m.append(_cos_v(np.mean((ob_r[:10] - ob_r[10:]) / 0.90, axis=0), g_em))

                    df_mse = (0.5 * np.sum((zs[:10] - x0) ** 2, axis=1) - 0.5 * np.sum((zs[10:] - x0) ** 2, axis=1)) / 0.36
                    cos_spsa_m.append(_cos_v(-np.mean(df_mse[:, None] * d_th, axis=0), g_em))

                    sid = mc % 10
                    tp = np.broadcast_to(th_ref, (n_p, n_p)).copy(); tm = tp.copy()
                    tp[np.arange(n_p), np.arange(n_p)] += 0.25; tm[np.arange(n_p), np.arange(n_p)] -= 0.25
                    _, zp2 = m_den.measure_observables_batch(
                        np.broadcast_to(xt[sid], (2 * n_p, nq)), np.full(2 * n_p, t_arr[sid], dtype=np.int64),
                        shots=sh_psr_fd, theta_batch=np.concatenate([tp, tm]),
                    )
                    gp_fd = -(0.5 * np.sum((zp2[:n_p] - x0[sid][None]) ** 2, axis=1) - 0.5 * np.sum((zp2[n_p:] - x0[sid][None]) ** 2, axis=1)) / 0.50
                    cos_psrfd_m.append(_cos_v(gp_fd, g_em))

                    # Exact PSR (+-pi/2 per layer) using shared _sample_layer_perturbed_z -> scored on BOTH g_ec and g_em!
                    in_pin = m_den.in_pinning_base + m_den.in_pinning_slope * (1.0 - float(t_arr[sid]) / m_den.diffusion_steps)
                    h_b = th_ref[ne:] + in_pin * xt[sid]
                    j_b = th_ref[:ne]
                    h_stk = np.broadcast_to(h_b[None, None, :], (1 + 2 * nl * n_p, nl, nq)).copy()
                    j_stk = np.broadcast_to(j_b[None, None, :], (1 + 2 * nl * n_p, nl, ne)).copy()
                    ptr = 1; meta_e = []
                    for l_i, mg in enumerate(mold_gains):
                        ang = (0.5 * math.pi) / mg
                        for ei in range(ne):
                            j_stk[ptr, l_i, ei] += ang; j_stk[ptr + 1, l_i, ei] -= ang
                            meta_e.append((ei, mg)); ptr += 2
                        for qi in range(nq):
                            h_stk[ptr, l_i, qi] += ang; h_stk[ptr + 1, l_i, qi] -= ang
                            meta_e.append((ne + qi, mg)); ptr += 2
                    ze_all = bm._sample_layer_perturbed_z(m_den, h_stk, j_stk, float(t_arr[sid]) / m_den.diffusion_steps, sh_exact_psr, dev)
                    err_m = ze_all[0] - x0[sid]
                    ge_surr = np.zeros(n_p, dtype=np.float64)
                    ge_mse = np.zeros(n_p, dtype=np.float64)
                    for pidx, (p_id, mg) in enumerate(meta_e):
                        dz = 0.5 * mg * (ze_all[1 + 2 * pidx] - ze_all[2 + 2 * pidx])
                        ge_surr[p_id] += float(np.dot(x0[sid], dz))
                        ge_mse[p_id] -= float(np.dot(err_m, dz))
                    cos_epsr_s.append(_cos_v(ge_surr, g_ec))
                    cos_epsr_m.append(_cos_v(ge_mse, g_em))

            vq = float(np.mean(np.var(gq_l, axis=0)))
            vc = float(np.mean(np.var(gc_l, axis=0)))
            vms = float(np.mean(np.var(gm_sum_l, axis=0)))
            vmm = float(np.mean(np.var(gm_mean_l, axis=0)))
            row = {
                "n_qubits": nq, "n_layers": nl, "n_edges": ne, "n_params": n_p,
                "var_qer_sum_zz": vq, "var_corr_sum_zz": vc, "var_mse_sum_zz": vms, "var_mse_mean_zz": vmm,
                "ratio_sum_qer_mse": vq / max(vms, 1e-12),
                "ratio_mean_artifact": vq / max(vmm, 1e-12),
                "cos4k_qer": float(np.mean(cos_qer_s)) if cos_qer_s else 0.0,
                "cos4k_qer_std": float(np.std(cos_qer_s)) if cos_qer_s else 0.0,
                "cos4k_spsa_surr": float(np.mean(cos_spsa_s)) if cos_spsa_s else 0.0,
                "cos4k_spsa_surr_std": float(np.std(cos_spsa_s)) if cos_spsa_s else 0.0,
                "cos4k_epsr_surr": float(np.mean(cos_epsr_s)) if cos_epsr_s else 0.0,
                "cos4k_epsr_surr_std": float(np.std(cos_epsr_s)) if cos_epsr_s else 0.0,
                "cos4k_qer_res_mse": float(np.mean(cos_qer_res_m)) if cos_qer_res_m else 0.0,
                "cos4k_qer_res_mse_std": float(np.std(cos_qer_res_m)) if cos_qer_res_m else 0.0,
                "cos4k_spsa": float(np.mean(cos_spsa_m)) if cos_spsa_m else 0.0,
                "cos4k_spsa_std": float(np.std(cos_spsa_m)) if cos_spsa_m else 0.0,
                "cos4k_psr": float(np.mean(cos_psrfd_m)) if cos_psrfd_m else 0.0,
                "cos4k_psr_std": float(np.std(cos_psrfd_m)) if cos_psrfd_m else 0.0,
                "cos4k_epsr_mse": float(np.mean(cos_epsr_m)) if cos_epsr_m else 0.0,
                "cos4k_epsr_mse_std": float(np.std(cos_epsr_m)) if cos_epsr_m else 0.0,
            }
            rows.append(row)
            extra = (
                f" | 4k Surr(g_ec): QER={row['cos4k_qer']:+.2f} ePSR={row['cos4k_epsr_surr']:+.2f} SPSA={row['cos4k_spsa_surr']:+.2f}"
                f" | MSE(g_em): QER-Res={row['cos4k_qer_res_mse']:+.2f} ePSR={row['cos4k_epsr_mse']:+.2f} SPSA={row['cos4k_spsa']:+.2f}"
                if nl == 2 else ""
            )
            print(f"  L={nl} n={nq:2d} (P={n_p:3d}) | Var(QER)={vq:.2e} Var(Corr)={vc:.2e} Var(MSE_sum)={vms:.2e} ratio={row['ratio_sum_qer_mse']:.2f}x{extra}", flush=True)

    return {"rows": rows}


def make_figures(results: Dict[str, Any], out_dir: str) -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8.6, "axes.titlesize": 9.5,
        "axes.titleweight": "bold", "axes.labelsize": 8.6, "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.6, "legend.fontsize": 7.0,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
    })

    # ---- Figure A: Hybrid ClassicalTrunk + QERLayer Diversity Unlock ----
    cap_path = os.path.join(out_dir, "capacity_diversity_results.json")
    if not os.path.exists(cap_path) and os.path.exists("./analysis/capacity_diversity_results.json"):
        cap_path = "./analysis/capacity_diversity_results.json"
    s2 = results["study2_hybrid"]["models"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.8))

    if os.path.exists(cap_path):
        with open(cap_path, "r", encoding="utf-8") as f:
            cap = json.load(f)
        sweep35 = cap.get("resmlp_sweep", [])
        sweep85 = cap.get("resmlp_sweep_sn85", sweep35)
        ps85 = [r["n_params"] for r in sweep85]
        ps35 = [r["n_params"] for r in sweep35]
        rec35 = [r["aggregate"]["recall"]["mean"] for r in sweep35]
        rec85 = [r["aggregate"]["recall"]["mean"] for r in sweep85]
        cov85 = [r["aggregate"]["coverage"]["mean"] for r in sweep85]
        fid85 = [r["aggregate"]["feature_fid"]["mean"] for r in sweep85]
        ax1.semilogx(ps85, rec85, "o-", color="#56B4E9", lw=1.9, ms=4.5, label="Classical ResMLP Recall ($\\sigma_{\\mathrm{step}}=0.85$)")
        ax1.semilogx(ps35, rec35, "s:", color="#56B4E9", lw=1.4, ms=3.8, alpha=0.75, label="Classical ResMLP Recall ($\\sigma_{\\mathrm{step}}=0.35$)")
        ax1.semilogx(ps85, cov85, "^--", color="#009E73", lw=1.5, ms=4.2, alpha=0.8, label="Classical ResMLP Coverage ($\\sigma_{\\mathrm{step}}=0.85$)")
        ax2.loglog(ps85, fid85, "o-", color="#56B4E9", lw=1.9, ms=4.5, label="Classical ResMLP FeatFID ($\\sigma_{\\mathrm{step}}=0.85$)")
        ax1.axhline(cap["real_baselines"]["real_ae_recon"]["recall"], color="#777777", ls="--", lw=1.1, label="Frozen 12-d AE Recall Ceiling")

    std_qer_s1 = {r["key"]: r["aggregate"] for r in results["study1_wcond"]["full_mnist"]}["std_qer_198p"]
    qer_acc_s1 = std_qer_s1["clf_acc"]["mean"] * 100.0
    qer_fid_s1 = std_qer_s1["feature_fid"]["mean"]

    hps = [r["n_params"] for r in s2]
    hrec35 = [r["sn035"]["recall"]["mean"] for r in s2]
    hrec85 = [r["sn085"]["recall"]["mean"] for r in s2]
    hcov85 = [r["sn085"]["coverage"]["mean"] for r in s2]
    hfid35 = [r["sn035"]["feature_fid"]["mean"] for r in s2]
    hfid85 = [r["sn085"]["feature_fid"]["mean"] for r in s2]
    hacc35 = [r["sn035"]["clf_acc"]["mean"] * 100.0 for r in s2]
    hacc85 = [r["sn085"]["clf_acc"]["mean"] * 100.0 for r in s2]

    ax1.scatter([198], [0.0], marker="*", s=180, color="#D55E00", edgecolor="black", zorder=7, label=f"Pure QPU QER ($198$p: Acc={qer_acc_s1:.0f}%, Rec=0.000)")
    ax1.scatter(hps, hrec85, marker="D", s=95, color="#D55E00", edgecolor="black", zorder=8, label="Hybrid QERLayer (No Skip, $\\sigma=0.85$) Rec")
    ax1.scatter(hps, hrec35, marker="P", s=95, color="#CC79A7", edgecolor="black", zorder=8, label="Hybrid QERLayer ($\\sigma=0.35$) Rec")
    ax1.scatter(hps, hcov85, marker="X", s=85, color="#E69F00", edgecolor="black", zorder=7, label="Hybrid QERLayer ($\\sigma=0.85$) Coverage")

    s2_ctrl = results["study2_hybrid"].get("controls", {})
    if s2_ctrl:
        ad_ps = [s2_ctrl[k]["n_params"]["mean"] for k in ("exact_ad_h48", "exact_ad_h128") if k in s2_ctrl]
        ad_rec = [s2_ctrl[k]["recall"]["mean"] for k in ("exact_ad_h48", "exact_ad_h128") if k in s2_ctrl]
        ad_fid = [s2_ctrl[k]["feature_fid"]["mean"] for k in ("exact_ad_h48", "exact_ad_h128") if k in s2_ctrl]
        cn_ps = [s2_ctrl[k]["n_params"]["mean"] for k in ("cl_noise_h48", "cl_noise_h128") if k in s2_ctrl]
        cn_rec = [s2_ctrl[k]["recall"]["mean"] for k in ("cl_noise_h48", "cl_noise_h128") if k in s2_ctrl]
        cn_fid = [s2_ctrl[k]["feature_fid"]["mean"] for k in ("cl_noise_h48", "cl_noise_h128") if k in s2_ctrl]
        if ad_ps:
            ax1.scatter(ad_ps, ad_rec, marker="^", s=85, color="#004D7A", edgecolor="black", zorder=9, label="Hybrid Exact-AD QPU ($\\sigma=0.85$) Rec")
            ax2.scatter(ad_ps, ad_fid, marker="^", s=85, color="#004D7A", edgecolor="black", zorder=9, label="Hybrid Exact-AD QPU ($\\sigma=0.85$) FeatFID")
        if cn_ps:
            ax1.scatter(cn_ps, cn_rec, marker="o", s=75, color="#999999", edgecolor="black", zorder=7, label="Classical ResMLP + 24sh Noise Rec")
            ax2.scatter(cn_ps, cn_fid, marker="o", s=75, color="#999999", edgecolor="black", zorder=7, label="Classical ResMLP + 24sh Noise FeatFID")

    ann1_offsets = [(0.24, 0.08), (1.75, 0.04)]
    for idx_a, (p_v, r_v, a_v) in enumerate(zip(hps, hrec85, hacc85)):
        xm, dy = ann1_offsets[idx_a % len(ann1_offsets)]
        ax1.annotate(f"Rec={r_v:.2f}\nAcc={a_v:.0f}%", xy=(p_v, r_v), xytext=(p_v * xm, min(0.92, max(0.05, r_v + dy))),
                     fontsize=6.8, fontweight="bold", arrowprops=dict(arrowstyle="->", lw=0.7))
    ax1.set_xscale("log")
    ax1.set_ylim(-0.04, 1.04)
    ax1.set_xlabel("Trainable Denoiser Parameters $P$ (log scale)")
    ax1.set_ylabel("5-NN Recall / Coverage [0, 1]")
    ax1.set_title("(a) Hybrid QERLayer vs Exact-AD & Classical Noise-Matched")
    ax1.grid(True, which="both", ls="--", alpha=0.25)
    ax1.legend(frameon=True, facecolor="white", framealpha=0.95, fontsize=6.4,
               loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)

    ax2.scatter([198], [qer_fid_s1], marker="*", s=180, color="#D55E00", edgecolor="black", zorder=7, label=f"Pure QPU QER ($198$p: FeatFID={qer_fid_s1:.1f})")
    ax2.scatter(hps, hfid35, marker="P", s=95, color="#CC79A7", edgecolor="black", zorder=8, label="Hybrid QERLayer ($\\sigma=0.35$) FeatFID")
    ax2.scatter(hps, hfid85, marker="D", s=95, color="#D55E00", edgecolor="black", zorder=8, label="Hybrid QERLayer ($\\sigma=0.85$) FeatFID")
    ann2_offsets = [(0.24, 0.92), (1.85, 1.15)]
    for idx_a, (p_v, f_v, a_v) in enumerate(zip(hps, hfid35, hacc35)):
        xm, ym = ann2_offsets[idx_a % len(ann2_offsets)]
        ax2.annotate(f"FID={f_v:.1f}\nAcc={a_v:.0f}%", xy=(p_v, f_v), xytext=(p_v * xm, f_v * ym),
                     fontsize=6.8, fontweight="bold", arrowprops=dict(arrowstyle="->", lw=0.7))
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_ylim(45, 1150)
    ax2.set_xlabel("Trainable Denoiser Parameters $P$ (log scale)")
    ax2.set_ylabel("Perceptual Feature FID $\\downarrow$")
    ax2.set_title("(b) End-to-End O(1) VJP Feature FID Across NISQ QPU")
    ax2.grid(True, which="both", ls="--", alpha=0.25)
    ax2.legend(frameon=True, facecolor="white", framealpha=0.95, fontsize=6.4,
               loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)

    fig.suptitle("Hybrid Classical-Quantum Scaling via QERLayer (No Skip Bypass, NISQ Noise, MNIST D=12)", fontsize=10.8, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(os.path.join(out_dir, "fig_hybrid_qpu_diversity.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig_hybrid_qpu_diversity.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure B: Four-Panel Controls (Label-Qubit, SPSA/iCANS, W_cond, Unit-Consistent BP SNR) ----
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.8))

    # Panel (a): Label-Qubit Zero-W_cond MNIST
    ax = axes[0, 0]
    lq = results["study1_wcond"]["label_qubits"]
    lbls_a = ["13q no-zz\n(0 cls)", "13q all\nL=2 (0 cls)", "13q all\nL=3 (0 cls)", "13q SPSA\nL=3", "13q iCANS\nL=3", "14q no-zz\n{0,1,4}", "14q all L=3\n{0,1,4}"]
    cols_a = ["#CC79A7", "#56B4E9", "#0072B2", "#D55E00", "#E69F00", "#CC79A7", "#009E73"]
    m_a = [r["aggregate"]["sub_acc"]["mean"] * 100.0 for r in lq]
    s_a = [r["aggregate"]["sub_acc"]["std"] * 100.0 for r in lq]
    xa = np.arange(len(lq))
    bars = ax.bar(xa, m_a, yerr=s_a, capsize=3.0, width=0.58, color=cols_a, edgecolor="#222222", lw=0.7)
    ax.axhline(50.0, color="#666666", ls=":", lw=1.0, label="2-Class Chance (50%)")
    ax.axhline(33.3, color="#999999", ls="--", lw=1.0, label="3-Class Chance (33.3%)")
    for b, mv, sv in zip(bars, m_a, s_a):
        ax.text(b.get_x() + b.get_width() / 2.0, mv + sv + 1.5, f"{mv:.1f}%", ha="center", va="bottom", fontsize=6.9, fontweight="bold")
    ax.set_xticks(xa); ax.set_xticklabels(lbls_a, fontsize=6.7)
    ax.set_ylim(0, 118); ax.set_ylabel("Generated Subset ClfAcc (%) $\\uparrow$")
    ax.set_title("(a) Zero-$W_{\\mathrm{cond}}$ Label-Qubit MNIST (Pure Quantum $ZZ$ Routing)")
    ax.grid(axis="y", ls="--", alpha=0.25); ax.legend(frameon=False, fontsize=6.8, loc="upper right")

    # Panel (b): Chain-Rule W_cond, 4-sh SPSA, QN-SPSA, Exact-PSR (+-pi/2), & iCANS/Rosalin PSR
    ax = axes[0, 1]
    sp = results["study3_opt"]["specials"]
    best_g = max(results["study3_opt"]["spsa_grid"], key=lambda r: r["aggregate"]["clf_acc"]["mean"])
    sp_map = {r["key"]: r["aggregate"] for r in sp}
    std_qer = {r["key"]: r["aggregate"] for r in results["study1_wcond"]["full_mnist"]}["std_qer_198p"]
    items_b = [
        ("W_cond\nOnly", sp_map["spsa_wcond_act"], "#56B4E9"),
        (f"Grid SPSA\n({best_g['lr']},{best_g['c0']})", best_g["aggregate"], "#D55E00"),
        ("Chain-Rule\nSPSA (20sh)", sp_map["spsa_cr_default"], "#CC6600"),
        ("Chain-Rule\nSPSA (4sh)", sp_map["spsa_4sh_198p"], "#AA4400"),
        ("QN-SPSA\n(CR, 80k)", sp_map["qnspsa_cr"], "#CC79A7"),
        ("iCANS\nExact-PSR", sp_map["icans_exact_psr"], "#E69F00"),
        ("Exact-PSR\n(+-pi/2)", sp_map["exact_psr_198p"], "#B07C00"),
        ("Rosalin\nPSR (4sh)", sp_map["rosalin_psr"], "#999999"),
        ("Adam-QER\n(All 198p)", std_qer, "#0072B2"),
    ]
    names_b = [it[0] for it in items_b]
    accs_b = [it[1]["clf_acc"]["mean"] * 100.0 for it in items_b]
    stds_b = [it[1]["clf_acc"]["std"] * 100.0 for it in items_b]
    cols_b = [it[2] for it in items_b]
    xb = np.arange(len(names_b))
    bars_b = ax.bar(xb, accs_b, yerr=stds_b, capsize=3.0, width=0.58, color=cols_b, edgecolor="#222222", lw=0.7)
    for b, mv, sv in zip(bars_b, accs_b, stds_b):
        ax.text(b.get_x() + b.get_width() / 2.0, mv + sv + 1.5, f"{mv:.1f}%", ha="center", va="bottom", fontsize=6.4, fontweight="bold")
    ax.set_xticks(xb); ax.set_xticklabels(names_b, fontsize=5.9)
    ax.set_ylim(0, 118); ax.set_ylabel("MNIST 10-Class ClfAcc (%) $\\uparrow$")
    ax.set_title("(b) Chain-Rule $W_{\\mathrm{cond}}$, QN-SPSA, & Exact-PSR Baselines")
    ax.grid(axis="y", ls="--", alpha=0.25)

    # Panel (c): Unit-Consistent Sum-Normalized Barren-Plateau Variance
    ax = axes[1, 0]
    bp_rows = results["study4_bp"]["rows"]
    for nl, col, mk in [(2, "#0072B2", "o-"), (6, "#D55E00", "s--")]:
        sub = [r for r in bp_rows if int(r["n_layers"]) == nl]
        xs = [r["n_qubits"] for r in sub]
        ax.semilogy(xs, [r["var_qer_sum_zz"] for r in sub], mk, color=col, lw=1.9, label=f"QER Surrogate ($L={nl}$, sum)")
        ax.semilogy(xs, [r["var_mse_sum_zz"] for r in sub], ":", marker="x", color=col, lw=1.7, label=f"Exact MSE ($L={nl}$, sum)")
    bp_vals = [r[k] for r in bp_rows for k in ("var_qer_sum_zz", "var_mse_sum_zz") if r.get(k, 0) > 0]
    ax.set_ylim(min(bp_vals) * 0.55 if bp_vals else 5e-4, max(bp_vals) * 4.5 if bp_vals else 4e-1)
    ax.set_xlabel("Circuit Qubits $n$"); ax.set_ylabel("Sum-Normalized $ZZ$ Grad Variance")
    ax.set_title("(c) Unit-Consistent Sum Variance Scaling ($n=6\\to 14$)")
    ax.grid(True, which="both", ls="--", alpha=0.25)
    ax.legend(frameon=True, facecolor="white", framealpha=0.90, fontsize=6.6, ncol=2, loc="upper right")

    # Panel (d): Fair Equal-Objective Fixed 4,000-Shot/Step NISQ Cosine SNR vs Width n
    ax = axes[1, 1]
    l2_rows = [r for r in bp_rows if int(r["n_layers"]) == 2]
    xs = [r["n_qubits"] for r in l2_rows]
    plot_d_lines = [
        ("cos4k_qer", "cos4k_qer_std", "QER (4k vs Surr $g_{ec}$)", "#0072B2", "o-"),
        ("cos4k_epsr_surr", "cos4k_epsr_surr_std", "Exact-PSR ($\\pm\\pi/2$, vs $g_{ec}$)", "#56B4E9", "^-."),
        ("cos4k_spsa_surr", "cos4k_spsa_surr_std", "SPSA-Surr (4k vs $g_{ec}$)", "#E69F00", "s:"),
        ("cos4k_qer_res_mse", "cos4k_qer_res_mse_std", "QER-Residual (3-run vs $g_{em}$)", "#009E73", "D-"),
        ("cos4k_epsr_mse", "cos4k_epsr_mse_std", "Exact-PSR ($\\pm\\pi/2$, vs $g_{em}$)", "#CC79A7", "v--"),
        ("cos4k_spsa", "cos4k_spsa_std", "SPSA-MSE (4k vs $g_{em}$)", "#D55E00", "x:"),
    ]
    for key_m, key_s, lbl, col, mk in plot_d_lines:
        mv = np.array([r[key_m] for r in l2_rows]); sv = np.array([r[key_s] for r in l2_rows])
        ax.plot(xs, mv, mk, color=col, lw=1.8, ms=4.5, label=lbl)
        ax.fill_between(xs, mv - sv, mv + sv, color=col, alpha=0.10)
    ax.axhline(0.0, color="#444444", lw=0.8)
    ax.set_ylim(-0.22, 1.32); ax.set_xlabel("Circuit Qubits $n$ ($P=n(n+1)/2$)")
    ax.set_ylabel("Equal-Target 4k-Shot Cosine SNR $\\uparrow$")
    ax.set_title("(d) Fixed 4k-Shot NISQ Cosine Alignment ($L=2$, Surrogate & MSE)")
    ax.grid(True, ls="--", alpha=0.25); ax.legend(frameon=True, facecolor="white", framealpha=0.92, fontsize=6.0, ncol=2, loc="upper center")

    fig.suptitle("Quantum-Conditioned Vision, Optimizer Sweeps, & Equal-Target Scaling Controls", fontsize=11.0, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    fig.savefig(os.path.join(out_dir, "fig_conditioning_opt_controls.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig_conditioning_opt_controls.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {os.path.join(out_dir, 'fig_hybrid_qpu_diversity.pdf')} & fig_conditioning_opt_controls.pdf", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default="./analysis")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--replot", action="store_true")
    ap.add_argument("--reuse-spsa-grid", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    json_p = os.path.join(args.out_dir, "conditioning_opt_results.json")
    if args.replot:
        with open(json_p, "r", encoding="utf-8") as f:
            res = json.load(f)
        make_figures(res, args.out_dir)
        return

    spsa_grid_cache = None
    if args.reuse_spsa_grid and not args.quick and os.path.exists(json_p):
        with open(json_p, "r", encoding="utf-8") as f:
            prev = json.load(f)
        if prev.get("study3_opt", {}).get("spsa_grid"):
            spsa_grid_cache = prev["study3_opt"]["spsa_grid"]

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tdev_str = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [0] if args.quick else [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    max_shots = 20_000 if args.quick else 80_000
    n_eval = 40 if args.quick else 120
    hyb_steps = 80 if args.quick else 300
    hyb_bs = 16 if args.quick else 32
    hyb_shots = 16 if args.quick else 24
    hyb_eval_cls = 40 if args.quick else 200
    bp_trials = 6 if args.quick else 24
    bp_mc = 4 if args.quick else 16

    print(f"[Setup] Device={dev} | Seeds={seeds} | quick={args.quick} | reuse_spsa_grid={args.reuse_spsa_grid}", flush=True)
    x_tr, y_tr, x_te, y_te, _ = bm.load_dataset("mnist", data_dir="./data")
    cache_p = os.path.join(args.out_dir, "mnist_d12_ae8_clf5_helpers.pt")
    ae, clf, z_tr, z_te = bm.train_classical_helpers(
        x_tr, y_tr, x_te, y_te, latent_dim=12, ae_epochs=8, clf_epochs=5, device=dev, cache_path=cache_p, use_bn=True,
    )

    t0 = time.time()
    s0 = run_study0_reciprocity(dev)
    s1 = run_study1_wcond_controls(z_tr, y_tr, z_te, x_te, y_te, ae, clf, dev, tdev_str, seeds, max_shots, n_eval)
    s2 = run_study2_hybrid_capacity(z_tr, y_tr, z_te, x_te, y_te, ae, clf, dev, tdev_str, seeds, hyb_steps, hyb_bs, hyb_shots, hyb_eval_cls)
    s3 = run_study3_optimizer_tuning(
        z_tr, y_tr, z_te, x_te, y_te, ae, clf, dev, tdev_str, seeds, max_shots, args.quick,
        reuse_spsa_grid_cache=spsa_grid_cache,
    )
    s4 = run_study4_bp_unit_snr(dev, tdev_str, bp_trials, bp_mc)

    res = {
        "seeds": seeds,
        "quick": args.quick,
        "wall_sec": round(time.time() - t0, 1),
        "study0_reciprocity": s0,
        "study1_wcond": s1,
        "study2_hybrid": s2,
        "study3_opt": s3,
        "study4_bp": s4,
    }
    with open(json_p, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"[Saved] {json_p} ({res['wall_sec']}s)", flush=True)
    make_figures(res, args.out_dir)


if __name__ == "__main__":
    main()
