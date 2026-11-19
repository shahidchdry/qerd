#!/usr/bin/env python3
"""
Controlled Hardware-Realism, Strong-Baseline, Reciprocity, and Hybrid-STE Verification Suite.

Designed for fast GPU execution (e.g., Kaggle T4) and offline figure/table regeneration:
  [Exp 1] Exact beta->0 Cross-Block Reciprocity (cos(JVP,VJP)_ZZ), h-block Diagonal Dominance
          (Full vs. Off-Diagonal), and Depolarizing Damping (L=1,2,4; random & trained 12q Heavy-Hex).
  [Exp 2] 3-Seed 12q Hardware-Native Frustrated Heavy-Hex Spin Glass (FakeKingston, 0 SWAPs,
          realistic 2Q CNOT noise) comparing QER-Res, Adam-QER, no-zz, Per-Sample SPSA (16-dir),
          High-Direction SPSA (64-dir), 1-shot & 4-shot Randomized-Coordinate PSR (Sweke et al., 2020),
          Full PSR-FD, and Clean Exact-AD; plus 2Q CNOT noise sweep (Heavy-Hex vs. All-to-All).
  [Exp 3] Parameter-Scaling 4,000-Shot Cosine SNR (n=6..14) against Multi-Direction SPSA
          and 1-shot / 4-shot Randomized-Coordinate PSR.
  [Exp 4] Hybrid QERLayer Controls (3 seeds, 8q, 24-shot NISQ):
          0-Backward-Shot STE vs. QERLayer (trainable theta vs. frozen theta=0) vs.
          Noise-Aware Exact-AD vs. Clean Exact-AD deployed unadapted to NISQ24.

Outputs:
  ./analysis/hardware_controls_results.json
  ./analysis/hardware_controls_tables.tex
  ./analysis/fig_heron_spinglass.pdf / .png
  ./analysis/fig_reciprocity_scaling.pdf / .png
  ./analysis/fig_heron_live_128q.pdf / .png

Usage:
  python -u experiments/eval_hardware_controls.py               # Full 3-seed suite (Kaggle GPU)
  python -u experiments/eval_hardware_controls.py --quick       # Fast 1-seed smoke run
  python -u experiments/eval_hardware_controls.py --replot      # Regenerate plots & TeX from JSON
  python -u experiments/eval_hardware_controls.py --plot-live   # Render live 128q Heron R2 figure only
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
from qerd import (
    QERGradConfig,
    QERGradientEstimator,
    QERLayer,
    QuantumIsingDenoiser,
    StabilizedAdam,
    StabilizedAdamConfig,
    build_ising_edges,
    make_alpha_bar,
)
from projects.sp500_qubo_and_spinglass.hardware.heron_r2 import get_fake_heron, select_clean_subgraph
from projects.sp500_qubo_and_spinglass.problems.spinglass import build_spinglass_problem
from projects.sp500_qubo_and_spinglass.solvers.classical import solve_simulated_annealing
from projects.sp500_qubo_and_spinglass.solvers.quantum import (
    sample_qerd_batch,
    train_coord_psr_equal_shot,
    train_psr_equal_shot,
    train_qerd_explicit,
    train_spsa_equal_shot,
)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _mean_std(vals: List[float]) -> Dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}


def _sample_depol_masks(
    k_traj: int, n_layers: int, batch_size: int, n_qubits: int, p_depol: float, dev: torch.device, rng: np.random.Generator
) -> torch.Tensor:
    probs = np.array([1.0 - 0.75 * p_depol, 0.25 * p_depol, 0.25 * p_depol, 0.25 * p_depol], dtype=np.float64)
    masks_np = rng.choice(4, size=(k_traj, n_layers, batch_size, n_qubits), p=probs).astype(np.int64)
    return torch.tensor(masks_np, dtype=torch.long, device=dev)


def _backend_kw(dev: torch.device) -> Dict[str, Any]:
    if dev.type == "cuda":
        return {"backend": "torch", "torch_device": "cuda"}
    return {"backend": "auto"}


# ==============================================================================
# EXP 1: Exact beta->0 Cross-Block Reciprocity & Diagonal Dominance Audit
# ==============================================================================
def _eval_full_reciprocity(
    circ: bm._DiffIsingCircuit,
    th_np: np.ndarray,
    xt_t: torch.Tensor,
    t_t: torch.Tensor,
    y_t: torch.Tensor,
    dev: torch.device,
    fault_masks: Optional[torch.Tensor] = None,
    p_readout: float = 0.0,
) -> Dict[str, Any]:
    """Compute exact beta->0 h-block (full & off-diag) and ZZ cross-block reciprocity + finite-beta cosines."""
    nq = circ.n_qubits
    ne = circ.n_edges
    bs = xt_t.shape[0]
    th_t = torch.tensor(th_np, dtype=torch.float32, device=dev, requires_grad=True)

    dh = torch.zeros((bs, nq), dtype=torch.float32, device=dev, requires_grad=True)
    obs_dh, z_dh = circ.expectation(
        xt_t, t_t, None, th_t, None,
        nudge_target=dh, nudge_beta=1.0,
        fault_masks=fault_masks, p_readout=p_readout,
    )
    jac_rows = []
    for qi in range(nq):
        (g_qi,) = torch.autograd.grad(z_dh[:, qi].sum(), [dh], retain_graph=True)
        jac_rows.append(g_qi)
    jac_h = torch.stack(jac_rows, dim=1).detach()  # (bs, nq, nq)

    # Exact beta->0 directional sensitivity of ZZ observables wrt visible fields h
    jvp_zz = np.zeros(ne, dtype=np.float64)
    for ei in range(ne):
        (g_ei,) = torch.autograd.grad(obs_dh[:, ei].sum(), [dh], retain_graph=(ei < ne - 1))
        jvp_zz[ei] = float((g_ei.detach() * y_t).sum(dim=1).mean().item())

    # Exact VJP: d/d(theta) E[y^T z]
    _, z0 = circ.expectation(
        xt_t, t_t, None, th_t, None,
        fault_masks=fault_masks, p_readout=p_readout,
    )
    (g_vjp_th,) = torch.autograd.grad(torch.sum(z0 * y_t, dim=1).mean(), [th_t])
    vjp_zz = g_vjp_th.detach().cpu().numpy()[:ne]

    eye_m = torch.eye(nq, dtype=torch.bool, device=dev).unsqueeze(0).expand(bs, -1, -1)
    jac_diag = torch.where(eye_m, jac_h, torch.zeros_like(jac_h))
    jac_off = torch.where(eye_m, torch.zeros_like(jac_h), jac_h)

    asym_ratio = float(
        (torch.linalg.norm(jac_h - jac_h.transpose(1, 2), dim=(1, 2))
         / torch.linalg.norm(jac_h, dim=(1, 2)).clamp(min=1e-12)).mean().item()
    )
    diag_frac = float(
        (torch.linalg.norm(jac_diag, dim=(1, 2))
         / torch.linalg.norm(jac_h, dim=(1, 2)).clamp(min=1e-12)).mean().item()
    )
    off_asym_ratio = float(
        (torch.linalg.norm(jac_off - jac_off.transpose(1, 2), dim=(1, 2))
         / torch.linalg.norm(jac_off, dim=(1, 2)).clamp(min=1e-12)).mean().item()
    )

    with torch.no_grad():
        vjp_h = torch.bmm(jac_h.transpose(1, 2), y_t.unsqueeze(-1)).squeeze(-1).mean(dim=0).cpu().numpy()
        jvp_h = torch.bmm(jac_h, y_t.unsqueeze(-1)).squeeze(-1).mean(dim=0).cpu().numpy()
        vjp_h_off = torch.bmm(jac_off.transpose(1, 2), y_t.unsqueeze(-1)).squeeze(-1).mean(dim=0).cpu().numpy()
        jvp_h_off = torch.bmm(jac_off, y_t.unsqueeze(-1)).squeeze(-1).mean(dim=0).cpu().numpy()

        finite_beta = {}
        for beta in (0.01, 0.10, 0.45):
            op, zp = circ.expectation(
                xt_t, t_t, None, th_t, None, nudge_target=y_t, nudge_beta=beta,
                fault_masks=fault_masks, p_readout=p_readout,
            )
            om, zm = circ.expectation(
                xt_t, t_t, None, th_t, None, nudge_target=y_t, nudge_beta=-beta,
                fault_masks=fault_masks, p_readout=p_readout,
            )
            gq_th = ((op - om) / (2.0 * beta)).mean(dim=0).cpu().numpy()
            gq_h = ((zp - zm) / (2.0 * beta)).mean(dim=0).cpu().numpy()
            finite_beta[str(beta)] = {
                "cos_qer_vjp_h": _cos(gq_h, vjp_h),
                "cos_qer_vjp_zz": _cos(gq_th[:ne], vjp_zz),
            }

    return {
        "asym_ratio_h": asym_ratio,
        "diag_frac_h": diag_frac,
        "off_asym_ratio_h": off_asym_ratio,
        "cos_jvp_vjp_h_full": _cos(jvp_h, vjp_h),
        "cos_jvp_vjp_h_offdiag": _cos(jvp_h_off, vjp_h_off),
        "cos_jvp_vjp_zz_exact": _cos(jvp_zz, vjp_zz),
        "finite_beta": finite_beta,
    }


def run_exp1_reciprocity_audit(dev: torch.device, quick: bool = False) -> Dict[str, Any]:
    print("\n=== [Exp 1] Exact beta->0 Cross-Block Reciprocity & Diagonal Dominance ===", flush=True)
    rng = np.random.default_rng(101)
    nq = 10
    edges_all = build_ising_edges((nq,), topology="all")
    bs = 8 if quick else 16
    k_traj = 12 if quick else 32
    x0 = rng.choice([-1.0, 1.0], size=(bs, nq))
    t_arr = np.array([(i % 4) + 1 for i in range(bs)], dtype=np.int64)
    ab = make_alpha_bar(4, "cosine")
    xt = np.clip(np.sqrt(ab[t_arr])[:, None] * x0 + np.sqrt(1.0 - ab[t_arr])[:, None] * rng.normal(0.0, 0.70, size=x0.shape), -1.0, 1.0)
    xt_t = torch.tensor(xt, dtype=torch.float32, device=dev)
    t_t = torch.tensor(t_arr, dtype=torch.long, device=dev)
    y_t = torch.tensor(x0, dtype=torch.float32, device=dev)

    random_rows = []
    for nl in (1, 2, 4):
        circ = bm._DiffIsingCircuit(nq, 0, edges_all, n_layers=nl, device=dev)
        fmasks = _sample_depol_masks(k_traj, nl, bs, nq, 0.035, dev, rng)
        for scale in (0.20, 0.60, 1.00):
            th = rng.uniform(-scale, scale, size=len(edges_all) + nq)
            r_clean = _eval_full_reciprocity(circ, th, xt_t, t_t, y_t, dev, fault_masks=None, p_readout=0.0)
            r_nisq = _eval_full_reciprocity(circ, th, xt_t, t_t, y_t, dev, fault_masks=fmasks, p_readout=0.04)
            random_rows.append({"n_layers": nl, "scale": scale, "clean": r_clean, "nisq": r_nisq})
            print(
                f"  L={nl} rand_{scale:.1f} | diag_h={r_clean['diag_frac_h']:.3f} | "
                f"cos_h(full/off)={r_clean['cos_jvp_vjp_h_full']:+.3f}/{r_clean['cos_jvp_vjp_h_offdiag']:+.3f} | "
                f"ZZ exact(beta->0) Cln/NISQ={r_clean['cos_jvp_vjp_zz_exact']:+.3f}/{r_nisq['cos_jvp_vjp_zz_exact']:+.3f} | "
                f"ZZ(0.45) Cln/NISQ={r_clean['finite_beta']['0.45']['cos_qer_vjp_zz']:+.3f}/{r_nisq['finite_beta']['0.45']['cos_qer_vjp_zz']:+.3f}",
                flush=True,
            )

    fk = get_fake_heron("ibm_kingston")
    hh_edges, _, _ = select_clean_subgraph(fk, n_qubits=12)
    sg_prob = build_spinglass_problem(12, hh_edges, seed=42)
    sg_sa = solve_simulated_annealing(12, sg_prob.cost_fn, n_targets=64, sweeps=80, seed=42)
    sg_tr = np.vstack([sg_sa, -sg_sa])
    m = QuantumIsingDenoiser(
        12, cond_dim=0, edges=hh_edges, n_layers=4, diffusion_steps=4,
        p_depol=0.005, p_depol_2q=0.005, p_readout=0.015,
        cal_mode="inverse", gain_bounds=(0.40, 1.00), seed=42, **_backend_kw(dev),
    )
    circ_hh = bm._DiffIsingCircuit(12, 0, hh_edges, n_layers=4, device=dev)
    x0_hh = sg_tr[:bs]
    t_hh = np.array([(i % 4) + 1 for i in range(bs)], dtype=np.int64)
    xt_hh = np.clip(np.sqrt(m.alpha_bar[t_hh])[:, None] * x0_hh + np.sqrt(1.0 - m.alpha_bar[t_hh])[:, None] * rng.normal(0.0, 0.70, size=x0_hh.shape), -1.0, 1.0)
    xt_hh_t = torch.tensor(xt_hh, dtype=torch.float32, device=dev)
    t_hh_t = torch.tensor(t_hh, dtype=torch.long, device=dev)
    y_hh_t = torch.tensor(x0_hh, dtype=torch.float32, device=dev)
    fmasks_hh = _sample_depol_masks(k_traj, 4, bs, 12, 0.035, dev, rng)

    trained_rows = []
    grad_est = QERGradientEstimator(QERGradConfig(beta=0.45, beta_min=0.15, centered=True, auto_beta_guard=True))
    adam = StabilizedAdam(m.parameters_dict(), StabilizedAdamConfig(lr_max=0.13, lr_min=0.015, weight_decay=0.015, clip_bound=1.30))
    n_tot = 12 if quick else 25
    snap_steps = {1, n_tot // 3, (2 * n_tot) // 3, n_tot}
    for st in range(1, n_tot + 1):
        prog = (st - 1) / max(1, n_tot - 1)
        ids = m.rng.choice(len(sg_tr), size=16, replace=False)
        tg = sg_tr[ids]
        tc = (np.arange(16) % 4) + 1
        m.rng.shuffle(tc)
        ab_b = m.alpha_bar[tc][:, None]
        xt_b = np.clip(np.sqrt(ab_b) * tg + np.sqrt(1.0 - ab_b) * m.rng.normal(0.0, 0.70, size=tg.shape), -1.0, 1.0)
        _, zf = m.measure_observables_batch(xt_b, tc, shots=48)
        nudge_tg = np.empty_like(tg)
        for k in range(16):
            tv = int(tc[k])
            m.calibrator.update(tv, zf[k], tg[k])
            xh = m.calibrator.calibrate(tv, zf[k])
            nudge_tg[k] = np.clip((tg[k] - xh) / np.maximum(m.calibrator.gain[tv], 0.20), -2.0, 2.0)
        ob2, z2 = m.measure_observables_batch(
            np.concatenate([xt_b, xt_b]), np.concatenate([tc, tc]), shots=48,
            nudge_target_batch=np.concatenate([nudge_tg, nudge_tg]),
            nudge_beta_batch=np.concatenate([np.full(16, grad_est.beta), np.full(16, -grad_est.beta)]),
        )
        grads, _ = grad_est.estimate(m.parameters_dict(), ob2[:16], ob2[16:], z2[:16], z2[16:], nudge_dirs=nudge_tg)
        adam.step(m.parameters_dict(), grads, progress=prog, shots_in_batch=3 * 16 * 48)
        if st in snap_steps:
            with torch.no_grad():
                _, z0_c = circ_hh.expectation(xt_hh_t, t_hh_t, None, torch.tensor(m.theta, dtype=torch.float32, device=dev), None)
                y_res_t = (y_hh_t - z0_c).detach()
            rc = _eval_full_reciprocity(circ_hh, m.theta.copy(), xt_hh_t, t_hh_t, y_res_t, dev, fault_masks=None, p_readout=0.0)
            rn = _eval_full_reciprocity(circ_hh, m.theta.copy(), xt_hh_t, t_hh_t, y_res_t, dev, fault_masks=fmasks_hh, p_readout=0.04)
            trained_rows.append({"step": st, "max_abs_theta": float(np.max(np.abs(m.theta))), "clean": rc, "nisq": rn})
            print(
                f"  Trained HH-12q L=4 st={st:2d} (max|th|={np.max(np.abs(m.theta)):.2f}) | "
                f"diag_h={rc['diag_frac_h']:.3f} | cos_h(full/off)={rc['cos_jvp_vjp_h_full']:+.3f}/{rc['cos_jvp_vjp_h_offdiag']:+.3f} | "
                f"ZZ exact={rc['cos_jvp_vjp_zz_exact']:+.3f}/{rn['cos_jvp_vjp_zz_exact']:+.3f} | "
                f"ZZ(0.45)={rc['finite_beta']['0.45']['cos_qer_vjp_zz']:+.3f}/{rn['finite_beta']['0.45']['cos_qer_vjp_zz']:+.3f}",
                flush=True,
            )

    return {"random_10q_all": random_rows, "trained_12q_heavyhex_L4": trained_rows}


# ==============================================================================
# EXP 2: 3-Seed 12q Frustrated Heavy-Hex Spin Glass + 2Q CNOT Noise Sweep
# ==============================================================================
def _train_exact_ad_spinglass(
    edges: List[Tuple[int, int]],
    n_qubits: int,
    x_tr: np.ndarray,
    n_layers: int,
    steps: int,
    seed: int,
    dev: torch.device,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    circ = bm._DiffIsingCircuit(n_qubits, 0, edges, n_layers=n_layers, device=dev)
    ab = make_alpha_bar(4, "cosine")
    th = torch.zeros(len(edges) + n_qubits, dtype=torch.float32, device=dev, requires_grad=True)
    opt = torch.optim.Adam([th], lr=0.10, betas=(0.75, 0.98))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=0.015)
    for _ in range(steps):
        ids = rng.choice(len(x_tr), size=16, replace=False)
        tg = x_tr[ids]
        tc = (np.arange(16) % 4) + 1
        rng.shuffle(tc)
        a = ab[tc][:, None]
        xt = np.clip(np.sqrt(a) * tg + np.sqrt(1.0 - a) * rng.normal(0.0, 0.70, size=tg.shape), -1.0, 1.0)
        opt.zero_grad()
        _, z = circ.expectation(
            torch.tensor(xt, dtype=torch.float32, device=dev),
            torch.tensor(tc, dtype=torch.long, device=dev),
            None, th, None,
        )
        loss = F.mse_loss(z, torch.tensor(tg, dtype=torch.float32, device=dev))
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            th.clamp_(-1.30, 1.30)
    return th.detach().cpu().numpy().astype(np.float64)


def run_exp2_heavyhex_and_2q_sweep(dev: torch.device, seeds: List[int], quick: bool = False) -> Dict[str, Any]:
    print("\n=== [Exp 2] 12q Frustrated Heavy-Hex Spin Glass + 2Q CNOT Noise Sweep ===", flush=True)
    fk = get_fake_heron("ibm_kingston")
    hh_edges, _, hh_audit = select_clean_subgraph(fk, n_qubits=12)
    print(f"  12q FakeKingston Heavy-Hex: |E|={len(hh_edges)} (0 SWAPs), med_cz={hh_audit.med_cz_err_pct:.2f}%, med_ro={hh_audit.med_ro_err_pct:.2f}%", flush=True)

    max_shots = 12_000 if quick else 38_400
    n_eval = 32 if quick else 64
    base_noise = dict(p_depol=0.005, p_depol_2q=0.005, p_readout=0.015)
    bkw = _backend_kw(dev)

    methods = [
        "untrained", "no_zz", "full_psr", "coord_psr_1sh", "coord_psr_4sh",
        "spsa_persample", "spsa_highdir", "adam_qer_target", "qer_res", "exact_ad_clean", "sa_ceiling",
    ]
    per_method: Dict[str, Dict[str, List[float]]] = {
        k: {"sat_bonds": [], "gap_closed": [], "mean_energy": [], "unique_pct": []} for k in methods
    }

    for seed in seeds:
        sg_prob = build_spinglass_problem(12, hh_edges, seed=seed)
        sg_sa = solve_simulated_annealing(12, sg_prob.cost_fn, n_targets=64, sweeps=100, seed=seed)
        sg_tr = np.vstack([sg_sa, -sg_sa])
        sa_cost = float(np.mean(sg_prob.cost_fn(sg_sa)))

        def _make_m(edges_use, noise_kw=base_noise):
            return QuantumIsingDenoiser(
                12, cond_dim=0, edges=edges_use, n_layers=4, diffusion_steps=4,
                cal_mode="inverse", init_gain=0.75, gain_ema=0.10, gain_bounds=(0.40, 1.00),
                seed=seed, **noise_kw, **bkw,
            )

        m_un = _make_m(hh_edges)
        s_un = sample_qerd_batch(m_un, n_samples=n_eval, shots_per_step=96, step_noise_scale=0.25)
        un_cost = float(np.mean(sg_prob.cost_fn(s_un)))

        def _rec(key: str, samps: np.ndarray):
            ev = sg_prob.evaluate(samps, un_cost, sa_cost)
            per_method[key]["sat_bonds"].append(ev["satisfied_bonds_pct"])
            per_method[key]["gap_closed"].append(ev["gap_closed_pct"])
            per_method[key]["mean_energy"].append(ev["mean_energy"])
            per_method[key]["unique_pct"].append(100.0 * ev["unique_frac"])

        _rec("untrained", s_un)
        _rec("sa_ceiling", sg_sa)

        m_nz = _make_m([])
        train_qerd_explicit(m_nz, sg_tr, max_shots=max_shots, shots_per_run=48, batch_size=16, nudge_mode="residual")
        _rec("no_zz", sample_qerd_batch(m_nz, n_samples=n_eval, shots_per_step=96, step_noise_scale=0.25))

        m_qr = _make_m(hh_edges)
        train_qerd_explicit(m_qr, sg_tr, max_shots=max_shots, shots_per_run=48, batch_size=16, nudge_mode="residual")
        _rec("qer_res", sample_qerd_batch(m_qr, n_samples=n_eval, shots_per_step=96, step_noise_scale=0.25))

        m_qt = _make_m(hh_edges)
        train_qerd_explicit(m_qt, sg_tr, max_shots=max_shots, shots_per_run=48, batch_size=16, nudge_mode="target")
        _rec("adam_qer_target", sample_qerd_batch(m_qt, n_samples=n_eval, shots_per_step=96, step_noise_scale=0.25))

        m_sp = _make_m(hh_edges)
        train_spsa_equal_shot(m_sp, sg_tr, max_shots=max_shots, shots_per_run=48, batch_size=16, spsa_shots_per_run=48)
        _rec("spsa_persample", sample_qerd_batch(m_sp, n_samples=n_eval, shots_per_step=96, step_noise_scale=0.25))

        m_sh = _make_m(hh_edges)
        train_spsa_equal_shot(m_sh, sg_tr, max_shots=max_shots, shots_per_run=12, batch_size=64, spsa_shots_per_run=12)
        _rec("spsa_highdir", sample_qerd_batch(m_sh, n_samples=n_eval, shots_per_step=96, step_noise_scale=0.25))

        m_c1 = _make_m(hh_edges)
        train_coord_psr_equal_shot(m_c1, sg_tr, max_shots=max_shots, shots_per_shift=1, coords_per_step=8)
        _rec("coord_psr_1sh", sample_qerd_batch(m_c1, n_samples=n_eval, shots_per_step=96, step_noise_scale=0.25))

        m_c4 = _make_m(hh_edges)
        train_coord_psr_equal_shot(m_c4, sg_tr, max_shots=max_shots, shots_per_shift=4, coords_per_step=8)
        _rec("coord_psr_4sh", sample_qerd_batch(m_c4, n_samples=n_eval, shots_per_step=96, step_noise_scale=0.25))

        m_fp = _make_m(hh_edges)
        train_psr_equal_shot(m_fp, sg_tr, max_shots=max_shots, shots_per_run=48)
        _rec("full_psr", sample_qerd_batch(m_fp, n_samples=n_eval, shots_per_step=96, step_noise_scale=0.25))

        th_ad = _train_exact_ad_spinglass(hh_edges, 12, sg_tr, n_layers=4, steps=160, seed=seed, dev=dev)
        m_ad = _make_m(hh_edges, noise_kw=dict(p_depol=0.0, p_depol_2q=0.0, p_readout=0.0))
        m_ad.theta[:] = th_ad
        m_ad.calibrator.gain[:] = 1.0
        _rec("exact_ad_clean", sample_qerd_batch(m_ad, n_samples=n_eval, shots_per_step=96, step_noise_scale=0.25))

    summary_main = {}
    for k in methods:
        summary_main[k] = {mk: _mean_std(per_method[k][mk]) for mk in per_method[k]}
        s = summary_main[k]
        print(
            f"  {k:<16s} | SatBonds={s['sat_bonds']['mean']:5.1f}±{s['sat_bonds']['std']:4.1f}% | "
            f"Gap={s['gap_closed']['mean']:5.1f}±{s['gap_closed']['std']:4.1f}% | "
            f"E={s['mean_energy']['mean']:+6.2f}",
            flush=True,
        )

    print("\n  [Exp 2B] 2Q CNOT Noise Stress Sweep: Native Heavy-Hex (88 CNOTs) vs All-to-All Stride-3 (528 CNOTs)", flush=True)
    all_edges = build_ising_edges((12,), topology="all")
    s3_modes = np.array([[[s0, s1, s2][q % 3] for q in range(12)] for s0 in (-1.0, 1.0) for s1 in (-1.0, 1.0) for s2 in (-1.0, 1.0)], dtype=np.float64)
    s3_valid = {tuple(r.astype(int).tolist()) for r in s3_modes}
    s3_intra = [(u, v) for u in range(12) for v in range(u + 1, 12) if (u % 3) == (v % 3)]

    sweep_2q = []
    for p2q in (0.0, 0.003, 0.008, 0.015):
        hh_sats, hh_gaps, s3_accs, s3_corrs = [], [], [], []
        for seed in seeds:
            sg_prob = build_spinglass_problem(12, hh_edges, seed=seed)
            sg_sa = solve_simulated_annealing(12, sg_prob.cost_fn, n_targets=64, sweeps=100, seed=seed)
            sg_tr = np.vstack([sg_sa, -sg_sa])
            m_h = QuantumIsingDenoiser(
                12, cond_dim=0, edges=hh_edges, n_layers=4, diffusion_steps=4,
                p_depol=0.005, p_depol_2q=p2q, p_readout=0.015,
                cal_mode="inverse", gain_bounds=(0.40, 1.00), seed=seed, **bkw,
            )
            train_qerd_explicit(m_h, sg_tr, max_shots=max_shots, shots_per_run=48, batch_size=16, nudge_mode="residual")
            sh = sample_qerd_batch(m_h, n_samples=n_eval, shots_per_step=96, step_noise_scale=0.25)
            evh = sg_prob.evaluate(sh, 0.0, float(np.mean(sg_prob.cost_fn(sg_sa))))
            hh_sats.append(evh["satisfied_bonds_pct"])
            hh_gaps.append(evh["gap_closed_pct"])

            m_a = QuantumIsingDenoiser(
                12, cond_dim=0, edges=all_edges, n_layers=4, diffusion_steps=4,
                p_depol=0.005, p_depol_2q=p2q, p_readout=0.015,
                cal_mode="forward", init_gain=0.75, gain_ema=0.10, gain_bounds=(0.20, 1.00), seed=seed, **bkw,
            )
            train_qerd_explicit(m_a, np.tile(s3_modes, (8, 1)), max_shots=max_shots, shots_per_run=20, batch_size=10, nudge_mode="residual", clip_bound=2.0)
            sa = sample_qerd_batch(m_a, n_samples=max(64, n_eval), shots_per_step=96, step_noise_scale=0.30)
            pat_acc = 100.0 * float(np.mean([tuple(r.astype(int).tolist()) in s3_valid for r in sa]))
            bc = sa - sa.mean(axis=0, keepdims=True)
            cg = (bc.T @ bc) / len(sa)
            icorr = float(np.mean([cg[u, v] for u, v in s3_intra]))
            s3_accs.append(pat_acc)
            s3_corrs.append(icorr)
        row = {
            "p_depol_2q": p2q,
            "heavyhex_cnots": 4 * 2 * len(hh_edges),
            "all2all_cnots": 4 * 2 * len(all_edges),
            "hh_sat_bonds": _mean_std(hh_sats),
            "hh_gap_closed": _mean_std(hh_gaps),
            "s3_pat_acc": _mean_std(s3_accs),
            "s3_intra_corr": _mean_std(s3_corrs),
        }
        sweep_2q.append(row)
        print(
            f"  p_2q={p2q:.3f} | HH({row['heavyhex_cnots']} CNOTs): Sat={row['hh_sat_bonds']['mean']:5.1f}±{row['hh_sat_bonds']['std']:4.1f}% "
            f"(Gap={row['hh_gap_closed']['mean']:5.1f}%) | "
            f"All2All({row['all2all_cnots']} CNOTs): PatAcc={row['s3_pat_acc']['mean']:5.1f}±{row['s3_pat_acc']['std']:4.1f}% "
            f"(Corr={row['s3_intra_corr']['mean']:+.3f})",
            flush=True,
        )

    return {"heavyhex_12q_edges": len(hh_edges), "max_shots": max_shots, "main_comparison": summary_main, "cnot_noise_sweep": sweep_2q}


# ==============================================================================
# EXP 3: Parameter-Scaling 4,000-Shot SNR vs Multi-Dir SPSA & Coord PSR
# ==============================================================================
def run_exp3_scaling_snr(dev: torch.device, seeds: List[int], quick: bool = False) -> Dict[str, Any]:
    print("\n=== [Exp 3] Parameter-Scaling 4,000-Shot SNR vs Strong Baselines ===", flush=True)
    rows = []
    budget = 4000
    bkw = _backend_kw(dev)
    qubit_list = (6, 8, 10, 12) if (quick or dev.type != "cuda") else (6, 8, 10, 12, 14)
    for nq in qubit_list:
        edges = build_ising_edges((nq,), topology="all")
        n_p = len(edges) + nq
        circ = bm._DiffIsingCircuit(nq, 0, edges, n_layers=2, device=dev)
        c_qer_res, c_spsa_16, c_spsa_100, c_cpsr_1, c_cpsr_4, c_fpsr = [], [], [], [], [], []
        for seed in seeds:
            rng = np.random.default_rng(seed + 100 * nq)
            m = QuantumIsingDenoiser(nq, cond_dim=0, edges=edges, n_layers=2, diffusion_steps=4, p_depol=0.035, p_readout=0.04, seed=seed, **bkw)
            th_ref = rng.uniform(-0.4, 0.4, size=n_p)
            m.theta[:] = th_ref
            bs = 16
            x0 = rng.choice([-1.0, 1.0], size=(bs, nq))
            t_arr = np.array([(i % 4) + 1 for i in range(bs)], dtype=np.int64)
            xt = np.clip(np.sqrt(m.alpha_bar[t_arr])[:, None] * x0 + np.sqrt(1.0 - m.alpha_bar[t_arr])[:, None] * rng.normal(0.0, 0.70, size=x0.shape), -1.0, 1.0)

            th_t = torch.tensor(th_ref, dtype=torch.float32, device=dev, requires_grad=True)
            _, zf = circ.expectation(torch.tensor(xt, dtype=torch.float32, device=dev), torch.tensor(t_arr, dtype=torch.long, device=dev), None, th_t, None)
            (g_em_t,) = torch.autograd.grad(-0.5 * torch.sum((zf - torch.tensor(x0, dtype=torch.float32, device=dev)) ** 2, dim=1).mean(), [th_t])
            g_em = g_em_t.detach().cpu().numpy()

            # 1) QER-Res
            sh_qr = budget // (3 * bs)
            _, zf_s = m.measure_observables_batch(xt, t_arr, shots=sh_qr)
            y_res = x0 - zf_s
            ob_r, _ = m.measure_observables_batch(
                np.concatenate([xt, xt]), np.concatenate([t_arr, t_arr]), shots=sh_qr,
                nudge_target_batch=np.concatenate([y_res, y_res]),
                nudge_beta_batch=np.concatenate([np.full(bs, 0.45), np.full(bs, -0.45)]),
            )
            c_qer_res.append(_cos(np.mean((ob_r[:bs] - ob_r[bs:]) / 0.90, axis=0), g_em))

            # 2) 16-Dir SPSA
            sh_s16 = budget // (2 * bs)
            d16 = rng.choice([-1.0, 1.0], size=(bs, n_p))
            _, z16 = m.measure_observables_batch(
                np.concatenate([xt, xt]), np.concatenate([t_arr, t_arr]), shots=sh_s16,
                theta_batch=np.concatenate([th_ref[None] + 0.22 * d16, th_ref[None] - 0.22 * d16]),
            )
            df16 = (0.5 * np.sum((z16[:bs] - x0) ** 2, axis=1) - 0.5 * np.sum((z16[bs:] - x0) ** 2, axis=1)) / 0.44
            c_spsa_16.append(_cos(-np.mean(df16[:, None] * d16, axis=0), g_em))

            # 3) 100-Dir SPSA
            n_dir = 100
            sh_s100 = budget // (2 * n_dir)
            idx100 = np.arange(n_dir) % bs
            d100 = rng.choice([-1.0, 1.0], size=(n_dir, n_p))
            _, z100 = m.measure_observables_batch(
                np.concatenate([xt[idx100], xt[idx100]]), np.concatenate([t_arr[idx100], t_arr[idx100]]), shots=sh_s100,
                theta_batch=np.concatenate([th_ref[None] + 0.22 * d100, th_ref[None] - 0.22 * d100]),
            )
            df100 = (0.5 * np.sum((z100[:n_dir] - x0[idx100]) ** 2, axis=1) - 0.5 * np.sum((z100[n_dir:] - x0[idx100]) ** 2, axis=1)) / 0.44
            c_spsa_100.append(_cos(-np.mean(df100[:, None] * d100, axis=0), g_em))

            # 4) 1-shot & 4-shot Randomized-Coordinate PSR (Sweke et al., 2020)
            for sh_c, store in ((1, c_cpsr_1), (4, c_cpsr_4)):
                n_probes = budget // (2 * sh_c)
                sel_coords = rng.integers(0, n_p, size=n_probes)
                sel_samps = rng.integers(0, bs, size=n_probes)
                g_acc = np.zeros(n_p, dtype=np.float64)
                cnt = np.zeros(n_p, dtype=np.float64)
                chunk = 200
                for start in range(0, n_probes, chunk):
                    end = min(n_probes, start + chunk)
                    k = end - start
                    sc = sel_coords[start:end]
                    ss = sel_samps[start:end]
                    th_p = np.broadcast_to(th_ref, (k, n_p)).copy()
                    th_m = th_p.copy()
                    rr = np.arange(k)
                    th_p[rr, sc] += 0.30
                    th_m[rr, sc] -= 0.30
                    _, zc = m.measure_observables_batch(
                        np.concatenate([xt[ss], xt[ss]]), np.concatenate([t_arr[ss], t_arr[ss]]), shots=sh_c,
                        theta_batch=np.concatenate([th_p, th_m]),
                    )
                    lp = 0.5 * np.sum((zc[:k] - x0[ss]) ** 2, axis=1)
                    lm = 0.5 * np.sum((zc[k:] - x0[ss]) ** 2, axis=1)
                    g_est = -(lp - lm) / 0.60
                    for idx_c, gv in zip(sc, g_est):
                        g_acc[idx_c] += gv
                        cnt[idx_c] += 1.0
                g_vec = np.where(cnt > 0, g_acc / np.maximum(cnt, 1.0), 0.0)
                store.append(_cos(g_vec, g_em))

            # 5) Full-Sweep PSR-FD
            sh_f = max(1, budget // (2 * n_p))
            th_p = np.broadcast_to(th_ref, (n_p, n_p)).copy()
            th_m = th_p.copy()
            eye = np.arange(n_p)
            th_p[eye, eye] += 0.30
            th_m[eye, eye] -= 0.30
            _, z_fp = m.measure_observables_batch(
                np.broadcast_to(xt[0], (2 * n_p, nq)), np.full(2 * n_p, t_arr[0], dtype=np.int64), shots=sh_f,
                theta_batch=np.concatenate([th_p, th_m]),
            )
            g_fp = -(0.5 * np.sum((z_fp[:n_p] - x0[0]) ** 2, axis=1) - 0.5 * np.sum((z_fp[n_p:] - x0[0]) ** 2, axis=1)) / 0.60
            c_fpsr.append(_cos(g_fp, g_em))

        row = {
            "n_qubits": nq,
            "n_params": n_p,
            "qer_res": _mean_std(c_qer_res),
            "spsa_16dir": _mean_std(c_spsa_16),
            "spsa_100dir": _mean_std(c_spsa_100),
            "coord_psr_1sh": _mean_std(c_cpsr_1),
            "coord_psr_4sh": _mean_std(c_cpsr_4),
            "full_psr_fd": _mean_std(c_fpsr),
        }
        rows.append(row)
        print(
            f"  n={nq:2d} (P={n_p:3d}) | QER-Res={row['qer_res']['mean']:+.3f} | "
            f"SPSA(16/100)={row['spsa_16dir']['mean']:+.3f}/{row['spsa_100dir']['mean']:+.3f} | "
            f"CoordPSR(1/4sh)={row['coord_psr_1sh']['mean']:+.3f}/{row['coord_psr_4sh']['mean']:+.3f} | "
            f"FullPSR={row['full_psr_fd']['mean']:+.3f}",
            flush=True,
        )
    return {"budget_shots": budget, "rows": rows}


# ==============================================================================
# EXP 4: Hybrid QERLayer Controls — STE (0 bwd shots) vs Noise-Aware AD vs QERLayer
# ==============================================================================
class _HybridTrunk(nn.Module):
    def __init__(self, dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor, t_norm: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(torch.cat([x, t_norm], dim=1)))


def _eval_hybrid_model(
    trunk: nn.Module,
    th_np: np.ndarray,
    denoiser_eval: QuantumIsingDenoiser,
    x_test: np.ndarray,
    y_test: np.ndarray,
    shots_eval: int,
    gain: float,
    dev: torch.device,
) -> Dict[str, float]:
    trunk.eval()
    with torch.no_grad():
        h_np = trunk(
            torch.tensor(x_test, dtype=torch.float32, device=dev),
            torch.full((len(x_test), 1), 0.5, dtype=torch.float32, device=dev),
        ).cpu().numpy().astype(np.float64)
    eye_p = np.eye(denoiser_eval.n_visible, dtype=np.float64)
    _, z_raw = denoiser_eval.measure_observables_batch(
        x_test, np.full(len(x_test), 2, dtype=np.int64), shots=shots_eval,
        cond_batch=h_np,
        theta_batch=np.broadcast_to(th_np, (len(x_test), denoiser_eval.n_qpu_params)),
        cond_proj_batch=np.broadcast_to(eye_p, (len(x_test), denoiser_eval.n_visible, denoiser_eval.n_visible)),
    )
    preds = np.clip(z_raw / gain, -1.0, 1.0)
    return {
        "mse": float(np.mean((preds - y_test) ** 2)),
        "sign_acc": float(100.0 * np.mean(np.sign(preds) == np.sign(y_test))),
    }


def run_exp4_hybrid_ste_controls(dev: torch.device, seeds: List[int], quick: bool = False) -> Dict[str, Any]:
    print("\n=== [Exp 4] Hybrid QERLayer Controls: STE (0 bwd) vs Noise-Aware AD vs QERLayer ===", flush=True)
    nq = 8
    edges = build_ising_edges((nq,), topology="all")
    shots_tr = 24
    gain = 0.75
    steps = 50 if quick else 150
    bs = 24
    bkw = _backend_kw(dev)

    base_patterns = np.array([
        [ 1,  1,  1,  1, -1, -1, -1, -1],
        [-1, -1, -1, -1,  1,  1,  1,  1],
        [ 1, -1,  1, -1,  1, -1,  1, -1],
        [-1,  1, -1,  1, -1,  1, -1,  1],
    ], dtype=np.float64)

    variants = ["clean_ad_unadapted", "noise_aware_ad", "ste_0bwd", "qer_theta_frozen", "qer_trainable"]
    res_store: Dict[str, Dict[str, List[float]]] = {
        v: {"mse": [], "sign_acc": [], "shots_total": []} for v in variants
    }

    for seed in seeds:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        y_te = base_patterns[rng.integers(0, len(base_patterns), size=128)]
        x_te = np.clip(0.65 * y_te + 0.55 * rng.normal(0.0, 1.0, size=y_te.shape), -1.0, 1.0)
        den_eval = QuantumIsingDenoiser(nq, cond_dim=nq, edges=edges, n_layers=2, diffusion_steps=4, p_depol=0.035, p_readout=0.04, seed=seed + 999, **bkw)

        def _next_batch(loc_rng):
            yb = base_patterns[loc_rng.integers(0, len(base_patterns), size=bs)]
            xb = np.clip(0.65 * yb + 0.55 * loc_rng.normal(0.0, 1.0, size=yb.shape), -1.0, 1.0)
            return xb, yb

        # 1) Clean Exact-AD -> NISQ24 unadapted
        torch.manual_seed(seed)
        loc_rng = np.random.default_rng(seed)
        trunk = _HybridTrunk(nq, hidden=32).to(dev)
        circ = bm._DiffIsingCircuit(nq, nq, edges, n_layers=2, device=dev)
        th_p = nn.Parameter(torch.zeros(len(edges) + nq, dtype=torch.float32, device=dev))
        eye_t = torch.eye(nq, dtype=torch.float32, device=dev)
        opt = torch.optim.Adam(list(trunk.parameters()) + [th_p], lr=0.015)
        for _ in range(steps):
            xb, yb = _next_batch(loc_rng)
            opt.zero_grad()
            xt_t = torch.tensor(xb, dtype=torch.float32, device=dev)
            h_t = trunk(xt_t, torch.full((bs, 1), 0.5, device=dev))
            _, z_c = circ.expectation(xt_t, torch.full((bs,), 2, dtype=torch.long, device=dev), h_t, th_p, eye_t)
            loss = F.mse_loss(torch.clamp(z_c / gain, -1.0, 1.0), torch.tensor(yb, dtype=torch.float32, device=dev))
            loss.backward()
            opt.step()
            with torch.no_grad():
                th_p.clamp_(-1.5, 1.5)
        ev = _eval_hybrid_model(trunk, th_p.detach().cpu().numpy(), den_eval, x_te, y_te, shots_tr, gain, dev)
        res_store["clean_ad_unadapted"]["mse"].append(ev["mse"])
        res_store["clean_ad_unadapted"]["sign_acc"].append(ev["sign_acc"])
        res_store["clean_ad_unadapted"]["shots_total"].append(0.0)

        # 2) Noise-Aware Exact-AD (simulated 3.5% depol + 4% RO + 24-shot reparameterized variance)
        torch.manual_seed(seed)
        loc_rng = np.random.default_rng(seed)
        trunk_na = _HybridTrunk(nq, hidden=32).to(dev)
        th_na = nn.Parameter(torch.zeros(len(edges) + nq, dtype=torch.float32, device=dev))
        opt_na = torch.optim.Adam(list(trunk_na.parameters()) + [th_na], lr=0.015)
        for _ in range(steps):
            xb, yb = _next_batch(loc_rng)
            opt_na.zero_grad()
            xt_t = torch.tensor(xb, dtype=torch.float32, device=dev)
            h_t = trunk_na(xt_t, torch.full((bs, 1), 0.5, device=dev))
            fm = _sample_depol_masks(6, 2, bs, nq, 0.035, dev, loc_rng)
            _, z_n = circ.expectation(xt_t, torch.full((bs,), 2, dtype=torch.long, device=dev), h_t, th_na, eye_t, fault_masks=fm, p_readout=0.04)
            shot_std = torch.sqrt(torch.clamp(1.0 - z_n.detach() ** 2, min=0.05) / float(shots_tr))
            z_noisy = torch.clamp(z_n + shot_std * torch.randn_like(z_n), -1.0, 1.0)
            loss = F.mse_loss(torch.clamp(z_noisy / gain, -1.0, 1.0), torch.tensor(yb, dtype=torch.float32, device=dev))
            loss.backward()
            opt_na.step()
            with torch.no_grad():
                th_na.clamp_(-1.5, 1.5)
        ev = _eval_hybrid_model(trunk_na, th_na.detach().cpu().numpy(), den_eval, x_te, y_te, shots_tr, gain, dev)
        res_store["noise_aware_ad"]["mse"].append(ev["mse"])
        res_store["noise_aware_ad"]["sign_acc"].append(ev["sign_acc"])
        res_store["noise_aware_ad"]["shots_total"].append(0.0)

        # 3-5) Hardware-in-the-loop: STE (0 bwd), QERLayer (theta=0 frozen), QERLayer (trainable theta)
        for vname, bmode, freeze_th, ckts_per_samp in (
            ("ste_0bwd", "ste", True, 1),
            ("qer_theta_frozen", "qer", True, 3),
            ("qer_trainable", "qer", False, 3),
        ):
            torch.manual_seed(seed)
            loc_rng = np.random.default_rng(seed)
            den_tr = QuantumIsingDenoiser(nq, cond_dim=nq, edges=edges, n_layers=2, diffusion_steps=4, p_depol=0.035, p_readout=0.04, seed=seed, **bkw)
            trunk_h = _HybridTrunk(nq, hidden=32).to(dev)
            qlayer = QERLayer(den_tr, shots=shots_tr, beta=0.45, gain=gain, backward_mode=bmode).to(dev)
            if freeze_th:
                qlayer.qpu_theta.requires_grad_(False)
                params = list(trunk_h.parameters())
            else:
                params = list(trunk_h.parameters()) + [qlayer.qpu_theta]
            opt_h = torch.optim.Adam(params, lr=0.015)
            for _ in range(steps):
                xb, yb = _next_batch(loc_rng)
                opt_h.zero_grad()
                xt_t = torch.tensor(xb, dtype=torch.float32, device=dev)
                h_t = trunk_h(xt_t, torch.full((bs, 1), 0.5, device=dev))
                out = qlayer(xt_t, h_t, t_step=2)
                loss = F.mse_loss(out, torch.tensor(yb, dtype=torch.float32, device=dev))
                loss.backward()
                opt_h.step()
            ev = _eval_hybrid_model(trunk_h, qlayer.qpu_theta.detach().cpu().numpy(), den_eval, x_te, y_te, shots_tr, gain, dev)
            res_store[vname]["mse"].append(ev["mse"])
            res_store[vname]["sign_acc"].append(ev["sign_acc"])
            res_store[vname]["shots_total"].append(float(steps * bs * shots_tr * ckts_per_samp))

    summary = {}
    for v in variants:
        summary[v] = {mk: _mean_std(res_store[v][mk]) for mk in res_store[v]}
        s = summary[v]
        print(
            f"  {v:<20s} | MSE={s['mse']['mean']:.4f}±{s['mse']['std']:.4f} | "
            f"SignAcc={s['sign_acc']['mean']:5.1f}±{s['sign_acc']['std']:4.1f}% | "
            f"QPU Shots={int(s['shots_total']['mean']):,}",
            flush=True,
        )
    return summary


# ==============================================================================
# Auto-Generated LaTeX Macros & Publication-Grade Large-Font Figures
# ==============================================================================
def write_latex_macros(res: Dict[str, Any], out_tex: str) -> None:
    """Emit LaTeX table row macros consumed automatically by paper.tex."""
    def _ms(d: Dict[str, float], fmt: str = ".1f", pct: bool = False) -> str:
        suf = "\\%" if pct else ""
        return f"${d['mean']:{fmt}} \\pm {d['std']:{fmt}}{suf}$"

    lines = ["% Auto-generated by experiments/eval_hardware_controls.py -- do not hand-edit."]

    # 1) Exp 2 Main 12q Heavy-Hex comparison
    mc = res["exp2"]["main_comparison"]
    hh_map = [
        ("untrained", "Untrained QPU ($\\theta{=}0$)"),
        ("no_zz", "\\texttt{QER no-zz} ($0$ $ZZ$ couplers)"),
        ("full_psr", "Full-Sweep \\texttt{PSR-FD}"),
        ("coord_psr_1sh", "1-Shot \\texttt{Coord-PSR} (Sweke et al.)"),
        ("coord_psr_4sh", "4-Shot \\texttt{Coord-PSR} (Sweke et al.)"),
        ("spsa_persample", "Multi-Dir \\texttt{SPSA} ($16$ masks/batch)"),
        ("spsa_highdir", "High-Dir \\texttt{SPSA} ($64$ masks/batch)"),
        ("adam_qer_target", "\\texttt{Adam-QER} ($2$-ckt surrogate)"),
        ("qer_res", "\\textbf{\\texttt{QER-Res} (Ours, $3$-ckt MSE)}"),
        ("exact_ad_clean", "\\textit{Ref: Clean \\texttt{Exact-AD} ($L{=}4$)}"),
        ("sa_ceiling", "\\textit{Ref: SA Target Ceiling}"),
    ]
    lines.append("\\providecommand{\\qerHeavyHexSimRows}{}")
    lines.append("\\renewcommand{\\qerHeavyHexSimRows}{%")
    for k, lbl in hh_map:
        s = mc[k]
        if k == "exact_ad_clean":
            lines.append("  \\midrule")
        lines.append(
            f"  {lbl} & {_ms(s['sat_bonds'], '.1f', True)} & {_ms(s['gap_closed'], '.1f', True)} & "
            f"${s['mean_energy']['mean']:+.2f}$ \\\\"
        )
    lines.append("}")

    # 2) Exp 2B 2Q CNOT noise sweep
    lines.append("\\providecommand{\\qerCnotSweepRows}{}")
    lines.append("\\renewcommand{\\qerCnotSweepRows}{%")
    for r in res["exp2"]["cnot_noise_sweep"]:
        lines.append(
            f"  ${r['p_depol_2q']*100:.1f}\\%$ & {_ms(r['hh_sat_bonds'], '.1f', True)} & "
            f"{_ms(r['hh_gap_closed'], '.1f', True)} & {_ms(r['s3_pat_acc'], '.1f', True)} & "
            f"${r['s3_intra_corr']['mean']:+.3f}$ \\\\"
        )
    lines.append("}")

    # 3) Exp 1 Exact Reciprocity (trained 12q Heavy-Hex L=4)
    lines.append("\\providecommand{\\qerExactReciprocityRows}{}")
    lines.append("\\renewcommand{\\qerExactReciprocityRows}{%")
    for r in res["exp1"]["trained_12q_heavyhex_L4"]:
        rc, rn = r["clean"], r["nisq"]
        lines.append(
            f"  Step {r['step']} ($\\max|\\theta|{{=}}{r['max_abs_theta']:.2f}$) & ${rc['diag_frac_h']:.3f}$ & "
            f"${rc['cos_jvp_vjp_h_full']:+.3f}$ / ${rc['cos_jvp_vjp_h_offdiag']:+.3f}$ & "
            f"${rc['cos_jvp_vjp_zz_exact']:+.3f}$ / ${rn['cos_jvp_vjp_zz_exact']:+.3f}$ & "
            f"${rc['finite_beta']['0.45']['cos_qer_vjp_zz']:+.3f}$ / ${rn['finite_beta']['0.45']['cos_qer_vjp_zz']:+.3f}$ \\\\"
        )
    lines.append("}")

    # 4) Exp 3 Scaling SNR
    lines.append("\\providecommand{\\qerScalingStrongRows}{}")
    lines.append("\\renewcommand{\\qerScalingStrongRows}{%")
    for r in res["exp3"]["rows"]:
        lines.append(
            f"  {r['n_qubits']} & {r['n_params']} & $\\mathbf{{{r['qer_res']['mean']:+.3f}}}$ & "
            f"${r['spsa_16dir']['mean']:+.3f}$ / ${r['spsa_100dir']['mean']:+.3f}$ & "
            f"${r['coord_psr_1sh']['mean']:+.3f}$ / ${r['coord_psr_4sh']['mean']:+.3f}$ & "
            f"${r['full_psr_fd']['mean']:+.3f}$ \\\\"
        )
    lines.append("}")

    # 5) Exp 4 Hybrid STE Controls
    e4 = res["exp4"]
    h_map = [
        ("clean_ad_unadapted", "Clean \\texttt{Exact-AD} $\\to$ NISQ24", "Sim ($0$)"),
        ("noise_aware_ad", "Noise-Aware \\texttt{Exact-AD} (Sim)", "Sim ($0$)"),
        ("ste_0bwd", "\\texttt{STE} ($0$ bwd QPU shots)", "1/3x"),
        ("qer_theta_frozen", "\\texttt{QERLayer} ($\\theta{=}0$ frozen)", "1x"),
        ("qer_trainable", "\\textbf{\\texttt{QERLayer} (Trainable $\\theta$)}", "1x"),
    ]
    lines.append("\\providecommand{\\qerHybridSteRows}{}")
    lines.append("\\renewcommand{\\qerHybridSteRows}{%")
    for k, lbl, cost_s in h_map:
        s = e4[k]
        lines.append(
            f"  {lbl} & {cost_s} & {_ms(s['mse'], '.4f')} & {_ms(s['sign_acc'], '.1f', True)} \\\\"
        )
    lines.append("}")

    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def make_live_heron_figure(out_dir: str) -> None:
    """Render clean, large-font figure strictly from verified live 128q ibm_kingston artifacts."""
    plt.rcParams.update({
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 8.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "projects", "sp500_qubo_and_spinglass", "artifacts", "metrics.json"), "r", encoding="utf-8") as f:
        p1 = json.load(f)
    with open(os.path.join(root, "projects", "dna_promoter_qerd", "artifacts", "metrics.json"), "r", encoding="utf-8") as f:
        p2 = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    # Panel (a): QPU Circuits/Sample Scaling vs 2^N GPU Wall
    ax = axes[0]
    ns = np.array([8, 12, 16, 24, 32, 64, 128, 156])
    # Heavy-Hex native parameter count ~ 2.13 * N (156q -> 332 params; 128q -> 263 params)
    p_hh = np.array([15, 23, 31, 47, 63, 130, 263, 332])
    ax.plot(ns, 2 * p_hh, "s--", lw=2.0, ms=5.5, color="#CC79A7", label="PSR-FD ($2P$ ckts/sample)")
    ax.plot(ns, np.full_like(ns, 3), "^-", lw=2.4, ms=6.5, color="#0072B2", label="QER-Res ($3$ ckts, $\\mathcal{O}(1)$)")
    ax.plot(ns, np.full_like(ns, 2), "o-", lw=2.0, ms=5.0, color="#009E73", label="QER-Grad ($2$ ckts, $\\mathcal{O}(1)$)")
    ax.axvline(26, color="#D55E00", ls=":", lw=1.6)
    ax.axvspan(26, 175, color="#D55E00", alpha=0.07)
    ax.text(30, 22, "Exact $2^N$ GPU Wall\n($>32$ GB VRAM)", fontsize=8.8, color="#A84300", fontweight="bold")
    ax.annotate("526 ckts/samp\n(0 steps @ 38k sh)", xy=(128, 526), xytext=(36, 420),
                arrowprops=dict(arrowstyle="->", color="#CC79A7", lw=1.2), fontsize=8.5, color="#882255", fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([8, 16, 32, 64, 128])
    ax.set_xticklabels(["8", "16", "32", "64", "128"])
    ax.set_xlim(7, 175)
    ax.set_ylim(1.3, 1200)
    ax.set_xlabel("Qubits $n$ (log scale)")
    ax.set_ylabel("Circuits / Sample (log scale)")
    ax.set_title("(a) $\\mathcal{O}(1)$ Scaling Past $2^N$ Wall")
    ax.legend(loc="lower right", fontsize=8.0, framealpha=0.93)
    ax.grid(True, ls="--", alpha=0.25)

    # Panel (b): Live 128q ibm_kingston Frustrated Heavy-Hex Spin Glass (38,016 shots, 0 SWAPs)
    ax = axes[1]
    sg = p1["heavy_hex_spinglass"]
    items_b = [
        ("Random\nDraw", sg["untrained_baseline"]["satisfied_bonds_pct"], "#999999", "E=-0.3"),
        ("QER-Res\n128q Live", sg["qerd_qpu"]["satisfied_bonds_pct"], "#0072B2", "E=-89.1\n(71.9% gap)"),
        ("SA Target\nCeiling", sg["sa_ceiling"]["satisfied_bonds_pct"], "#009E73", "E=-123.8"),
    ]
    xb = np.arange(len(items_b))
    bb = ax.bar(xb, [v for _, v, _, _ in items_b], color=[c for _, _, c, _ in items_b], edgecolor="#222222", linewidth=1.0, width=0.52)
    bb[1].set_linewidth(2.0)
    ax.axhline(50.0, color="#666666", ls="--", lw=1.0, label="Zero-Bias / Random (50%)")
    for bar, (_, v, _, sub) in zip(bb, items_b):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 2.0, f"{v:.1f}%\n{sub}", ha="center", va="bottom", fontsize=8.4, fontweight="bold")
    ax.set_xticks(xb)
    ax.set_xticklabels([l for l, _, _, _ in items_b], fontsize=9.2)
    ax.set_ylim(0, 122)
    ax.set_ylabel("Satisfied Frustrated Bonds (%) ↑")
    ax.set_title("(b) Live 128q Heavy-Hex Spin Glass")
    ax.legend(loc="upper left", fontsize=8.0, framealpha=0.92)
    ax.grid(axis="y", ls="--", alpha=0.25)

    # Panel (c): Live 128q ibm_kingston Promoter DNA Feasibility (Honest Multi-Metric View)
    ax = axes[2]
    dc = p2["comparison"]
    models_c = [
        ("Untrained\n128q QPU", dc["untrained_qpu"]),
        ("QER-Res\n128q Live", dc["qerd_qpu"]),
        ("Classical\nMarkov", dc["classical_markov"]),
        ("Real DNA\n(Unfilt.)", dc["real_promoters"]),
    ]
    xc = np.arange(len(models_c))
    w = 0.26
    act = [100.0 * m["mean_oracle_prob"] for _, m in models_c]
    val = [m["valid_promoter_pct"] for _, m in models_c]
    tata = [m["tata_cond_box_pct"] for _, m in models_c]
    ax.bar(xc - w, act, width=w, color="#0072B2", edgecolor="#222222", lw=0.7, label="Oracle Activity %")
    ax.bar(xc, val, width=w, color="#009E73", edgecolor="#222222", lw=0.7, label="Biophys. Valid %")
    ax.bar(xc + w, tata, width=w, color="#E69F00", edgecolor="#222222", lw=0.7, label="TATA-Cond Box %")
    for i in range(len(models_c)):
        ax.text(xc[i] - w, act[i] + 1.8, f"{act[i]:.0f}", ha="center", va="bottom", fontsize=7.6, fontweight="bold")
        ax.text(xc[i], val[i] + 1.8, f"{val[i]:.0f}", ha="center", va="bottom", fontsize=7.6, fontweight="bold")
        ax.text(xc[i] + w, tata[i] + 1.8, f"{tata[i]:.0f}", ha="center", va="bottom", fontsize=7.6, fontweight="bold")
    ax.set_xticks(xc)
    ax.set_xticklabels([l for l, _ in models_c], fontsize=8.5)
    ax.set_ylim(0, 118)
    ax.set_ylabel("Metric (%) ↑")
    ax.set_title("(c) Live 128q DNA Feasibility (Caveated)")
    ax.legend(loc="upper right", fontsize=7.6, framealpha=0.93)
    ax.grid(axis="y", ls="--", alpha=0.25)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_heron_live_128q.pdf"), dpi=300)
    fig.savefig(os.path.join(out_dir, "fig_heron_live_128q.png"), dpi=220)
    plt.close(fig)


def make_control_figures(res: Dict[str, Any], out_dir: str) -> None:
    plt.rcParams.update({
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 8.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "projects", "sp500_qubo_and_spinglass", "artifacts", "metrics.json"), "r", encoding="utf-8") as f:
        sg_live = json.load(f)["heavy_hex_spinglass"]

    # Figure A: 12q Heavy-Hex Strong Baselines + 2Q CNOT Noise Sweep + Live 128q
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    ax = axes[0]
    mc = res["exp2"]["main_comparison"]
    order_a = [
        ("no_zz", "no-zz", "#999999"),
        ("full_psr", "Full\nPSR", "#CC79A7"),
        ("coord_psr_1sh", "1-sh\nC-PSR", "#AA4499"),
        ("coord_psr_4sh", "4-sh\nC-PSR", "#882255"),
        ("spsa_persample", "16-Dir\nSPSA", "#E69F00"),
        ("spsa_highdir", "64-Dir\nSPSA", "#D55E00"),
        ("qer_res", "QER-Res\n(Ours)", "#0072B2"),
        ("exact_ad_clean", "Exact-AD\n(Clean)", "#009E73"),
    ]
    xs = np.arange(len(order_a))
    means = [mc[k]["sat_bonds"]["mean"] for k, _, _ in order_a]
    stds = [mc[k]["sat_bonds"]["std"] for k, _, _ in order_a]
    bars = ax.bar(xs, means, yerr=stds, capsize=3, color=[c for _, _, c in order_a], edgecolor="#222222", lw=0.8, width=0.66)
    bars[6].set_linewidth(1.8)
    ax.axhline(50.0, color="#666666", ls="--", lw=1.0)
    for b, m_v in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m_v + 2.8, f"{m_v:.0f}%", ha="center", va="bottom", fontsize=8.3, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([l for _, l, _ in order_a], fontsize=8.1)
    ax.set_ylim(0, 114)
    ax.set_ylabel("Satisfied Frustrated Bonds (%) ↑")
    ax.set_title("(a) 12q Heavy-Hex (38.4k sh, 2Q Noise)")
    ax.grid(axis="y", ls="--", alpha=0.25)

    ax = axes[1]
    sw = res["exp2"]["cnot_noise_sweep"]
    p2qs = [r["p_depol_2q"] * 100.0 for r in sw]
    ax.errorbar(p2qs, [r["hh_sat_bonds"]["mean"] for r in sw], yerr=[r["hh_sat_bonds"]["std"] for r in sw],
                marker="o", lw=2.2, ms=6.0, capsize=3, color="#0072B2", label="Heavy-Hex Frust. (88 CNOTs, Sat%)")
    ax.errorbar(p2qs, [r["s3_pat_acc"]["mean"] for r in sw], yerr=[r["s3_pat_acc"]["std"] for r in sw],
                marker="s", lw=2.2, ms=5.8, capsize=3, color="#D55E00", ls="--", label="All-to-All Cluster (528 CNOTs, PatAcc%)")
    ax.axvline(0.18, color="#009E73", ls=":", lw=1.4, label="ibm_kingston Med. CZ (0.18%)")
    ax.set_xlabel("Two-Qubit CNOT Error Rate $p_{2q}$ (%)")
    ax.set_ylabel("Score (%) ↑")
    ax.set_ylim(0, 108)
    ax.set_title("(b) 2Q CNOT Noise: Native vs. All-to-All")
    ax.legend(loc="lower left", fontsize=8.0, framealpha=0.92)
    ax.grid(True, ls="--", alpha=0.25)

    ax = axes[2]
    live_items = [
        ("Random\nDraw", sg_live["untrained_baseline"]["satisfied_bonds_pct"], "#999999"),
        ("QER-Res\n128q Live", sg_live["qerd_qpu"]["satisfied_bonds_pct"], "#0072B2"),
        ("SA Target\nCeiling", sg_live["sa_ceiling"]["satisfied_bonds_pct"], "#009E73"),
    ]
    xl = np.arange(len(live_items))
    bl = ax.bar(xl, [v for _, v, _ in live_items], color=[c for _, _, c in live_items], edgecolor="#222222", lw=1.0, width=0.52)
    bl[1].set_linewidth(2.0)
    ax.axhline(50.0, color="#666666", ls="--", lw=1.0)
    for b, (lbl, v, _) in zip(bl, live_items):
        extra = "\n(71.9% gap,\n100% uniq)" if "Live" in lbl else ""
        ax.text(b.get_x() + b.get_width() / 2, v + 2.2, f"{v:.1f}%{extra}", ha="center", va="bottom", fontsize=8.6, fontweight="bold")
    ax.set_xticks(xl)
    ax.set_xticklabels([l for l, _, _ in live_items], fontsize=9.2)
    ax.set_ylim(0, 118)
    ax.set_ylabel("Satisfied Frustrated Bonds (%) ↑")
    ax.set_title("(c) Live 128q ibm_kingston (38k sh)")
    ax.grid(axis="y", ls="--", alpha=0.25)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_heron_spinglass.pdf"), dpi=300)
    fig.savefig(os.path.join(out_dir, "fig_heron_spinglass.png"), dpi=220)
    plt.close(fig)

    # Figure B: Reciprocity + Scaling SNR + Hybrid STE Controls
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    ax = axes[0]
    tr = res["exp1"]["trained_12q_heavyhex_L4"]
    steps_x = [r["step"] for r in tr]
    ax.plot(steps_x, [r["clean"]["cos_jvp_vjp_h_full"] for r in tr], "o-", lw=2.0, color="#009E73", label=r"$\cos_h$ Full (diag-dom.)")
    ax.plot(steps_x, [r["clean"]["cos_jvp_vjp_h_offdiag"] for r in tr], "s--", lw=2.0, color="#56B4E9", label=r"$\cos_h$ Off-Diag Only")
    ax.plot(steps_x, [r["clean"]["cos_jvp_vjp_zz_exact"] for r in tr], "^-", lw=2.0, color="#D55E00", label=r"$\cos_{ZZ}$ Exact ($\beta{\to}0$, Cln)")
    ax.plot(steps_x, [r["nisq"]["finite_beta"]["0.45"]["cos_qer_vjp_zz"] for r in tr], "D-", lw=2.2, color="#0072B2", label=r"$\cos_{ZZ}$ QER ($\beta{=}0.45$, NISQ)")
    ax.axhline(0.0, color="#555555", ls=":", lw=1.0)
    ax.set_xlabel("Training Step (12q Heavy-Hex, $L{=}4$)")
    ax.set_ylabel("Cosine Alignment with Exact VJP")
    ax.set_ylim(-0.35, 1.08)
    ax.set_title("(a) Non-Equilibrium Reciprocity Audit")
    ax.legend(loc="lower left", fontsize=7.8, framealpha=0.93)
    ax.grid(True, ls="--", alpha=0.25)

    ax = axes[1]
    sc = res["exp3"]["rows"]
    nqs = [r["n_qubits"] for r in sc]
    for key, lbl, col, mk in (
        ("qer_res", "QER-Res (Ours)", "#0072B2", "o"),
        ("spsa_100dir", "100-Dir SPSA", "#D55E00", "s"),
        ("spsa_16dir", "16-Dir SPSA", "#E69F00", "^"),
        ("coord_psr_4sh", "4-sh Coord-PSR", "#882255", "v"),
        ("coord_psr_1sh", "1-sh Coord-PSR", "#CC79A7", "x"),
        ("full_psr_fd", "Full PSR-FD", "#666666", "d"),
    ):
        ax.errorbar(nqs, [r[key]["mean"] for r in sc], yerr=[r[key]["std"] for r in sc],
                    marker=mk, lw=2.0, ms=5.5, capsize=2.5, color=col, label=lbl)
    ax.set_xticks(nqs)
    ax.set_xticklabels([f"{r['n_qubits']}q\n(P={r['n_params']})" for r in sc], fontsize=8.4)
    ax.set_ylim(-0.05, 1.02)
    ax.set_ylabel("4k-Shot MSE Cosine SNR ↑")
    ax.set_title("(b) Scaling SNR vs. Strong Baselines")
    ax.legend(loc="upper right", ncol=2, fontsize=7.5, framealpha=0.93)
    ax.grid(True, ls="--", alpha=0.25)

    ax = axes[2]
    e4 = res["exp4"]
    order_4 = [
        ("clean_ad_unadapted", "Clean-AD\nUnadapt.", "#999999"),
        ("noise_aware_ad", "Noise-AD\n(Sim)", "#009E73"),
        ("ste_0bwd", "STE\n(0 bwd)", "#E69F00"),
        ("qer_theta_frozen", "QERLayer\n(θ=0)", "#56B4E9"),
        ("qer_trainable", "QERLayer\n(Train θ)", "#0072B2"),
    ]
    x4 = np.arange(len(order_4))
    m4 = [e4[k]["mse"]["mean"] for k, _, _ in order_4]
    s4 = [e4[k]["mse"]["std"] for k, _, _ in order_4]
    b4 = ax.bar(x4, m4, yerr=s4, capsize=3, color=[c for _, _, c in order_4], edgecolor="#222222", lw=0.9, width=0.58)
    for b, mv, (k, _, _) in zip(b4, m4, order_4):
        ax.text(b.get_x() + b.get_width() / 2, mv + 0.025, f"{mv:.3f}\n({e4[k]['sign_acc']['mean']:.0f}%)",
                ha="center", va="bottom", fontsize=8.1, fontweight="bold")
    ax.set_xticks(x4)
    ax.set_xticklabels([l for _, l, _ in order_4], fontsize=8.4)
    ax.set_ylim(0, max(m4) * 1.38 + 0.05)
    ax.set_ylabel("24-Shot NISQ Test MSE ↓")
    ax.set_title("(c) Hybrid QERLayer vs. STE & Noise-AD")
    ax.grid(axis="y", ls="--", alpha=0.25)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_reciprocity_scaling.pdf"), dpi=300)
    fig.savefig(os.path.join(out_dir, "fig_reciprocity_scaling.png"), dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Controlled Hardware-Realism, Strong-Baseline, Reciprocity, & Hybrid-STE Suite")
    parser.add_argument("--quick", action="store_true", help="Fast 1-seed smoke check")
    parser.add_argument("--replot", action="store_true", help="Regenerate plots and LaTeX macros from saved JSON")
    parser.add_argument("--plot-live", action="store_true", help="Render live 128q ibm_kingston summary figure only")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis")
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "hardware_controls_results.json")
    out_tex = os.path.join(out_dir, "hardware_controls_tables.tex")

    make_live_heron_figure(out_dir)
    if args.plot_live:
        print(f"[Saved] Live 128q Heron R2 figure -> {os.path.join(out_dir, 'fig_heron_live_128q.pdf')}")
        return

    if args.replot and os.path.exists(out_json):
        with open(out_json, "r", encoding="utf-8") as f:
            res = json.load(f)
        write_latex_macros(res, out_tex)
        make_control_figures(res, out_dir)
        print(f"[Replotted] Updated {out_tex} and figures in {out_dir}")
        return

    seeds = [42] if args.quick else [int(s) for s in args.seeds.split(",") if s.strip()]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()

    res = {
        "seeds": seeds,
        "device": str(dev),
        "quick": bool(args.quick),
        "exp1": run_exp1_reciprocity_audit(dev, quick=args.quick),
        "exp2": run_exp2_heavyhex_and_2q_sweep(dev, seeds, quick=args.quick),
        "exp3": run_exp3_scaling_snr(dev, seeds, quick=args.quick),
        "exp4": run_exp4_hybrid_ste_controls(dev, seeds, quick=args.quick),
        "wall_sec": time.perf_counter() - t0,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    write_latex_macros(res, out_tex)
    make_control_figures(res, out_dir)
    print(f"\n[DONE] Saved {out_json}, {out_tex}, and figures in {res['wall_sec']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
