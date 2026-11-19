#!/usr/bin/env python3
"""
Autoencoder-Free 12-Qubit Cluster Spin-Glass Factorial Attribution Suite.

Evaluates 12-qubit stride-3 correlated cluster spin glass (c=0, E[x_i]=0, 8 valid modes)
across equal-shot NISQ optimizers (40k shots), hardware coupling topologies, Trotter depth
(L=2 vs L=4), discrete Forward-OLS calibration, QER-Res, and noiseless Exact-AD / Ising-EP references.

Outputs:
  ./analysis/spinglass_diagnosis_results.json
  ./analysis/fig_spinglass_diagnosis.pdf / .png
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
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import benchmark_mnist as bm
from qerd import QERConfig, QEROptimizer, QERDTrainer, QuantumIsingDenoiser


def _build_spinglass_problem() -> Tuple[np.ndarray, Dict[Tuple[int, ...], int], List[Tuple[int, int]], np.ndarray]:
    modes = []
    for s0 in (-1.0, 1.0):
        for s1 in (-1.0, 1.0):
            for s2 in (-1.0, 1.0):
                sig = [s0, s1, s2]
                modes.append([sig[q % 3] for q in range(12)])
    modes_arr = np.array(modes, dtype=np.float64)
    valid_set = {tuple(r.astype(np.int64).tolist()): idx for idx, r in enumerate(modes_arr)}
    intra_pairs = [(u, v) for u in range(12) for v in range(u + 1, 12) if (u % 3) == (v % 3)]
    c_ref = (modes_arr.T @ modes_arr) / len(modes_arr)
    return modes_arr, valid_set, intra_pairs, c_ref


def _eval_spin_samples(
    gens: np.ndarray,
    valid_set: Dict[Tuple[int, ...], int],
    intra_pairs: List[Tuple[int, int]],
    c_ref: np.ndarray,
    n_modes: int,
) -> Dict[str, float]:
    bits = np.where(gens.reshape(-1, 12) >= 0.0, 1.0, -1.0)
    hit_modes = [valid_set[tuple(r.astype(np.int64).tolist())] for r in bits if tuple(r.astype(np.int64).tolist()) in valid_set]
    pat_acc = float(len(hit_modes) / len(bits))
    mode_cov = float(len(set(hit_modes)) / n_modes)
    bc = bits - bits.mean(axis=0, keepdims=True)
    c_gen = (bc.T @ bc) / len(bits)
    off = ~np.eye(12, dtype=bool)
    corr_rmse = float(np.sqrt(np.mean((c_gen[off] - c_ref[off]) ** 2)))
    intra_corr = float(np.mean([c_gen[u, v] for u, v in intra_pairs]))
    bit_std = float(np.mean(np.std(bits, axis=0)))
    return {
        "intra_corr": intra_corr,
        "corr_rmse": corr_rmse,
        "pattern_acc": pat_acc,
        "mode_cov": mode_cov,
        "bit_std": bit_std,
    }


def _train_spsa_uncond(
    model: QuantumIsingDenoiser,
    x_tr: np.ndarray,
    max_shots: int = 40_000,
    shots_per_run: int = 20,
    batch_size: int = 10,
    lr: float = 0.08,
    c0: float = 0.18,
    clip_bound: float = 0.90,
    objective: str = "mse",
) -> None:
    """Equal-shot Adam-SPSA for cond_dim=0 Ising circuits (MSE or matched contrastive surrogate)."""
    eff_batch = min(batch_size, len(x_tr))
    total_batches = max(1, max_shots // (2 * shots_per_run * eff_batch))
    m_th = np.zeros_like(model.theta)
    v_th = np.zeros_like(model.theta)
    for step in range(1, total_batches + 1):
        c_k = c0 / (step ** 0.101)
        lr_k = 0.015 + 0.5 * (lr - 0.015) * (1.0 + math.cos(math.pi * (step - 1) / max(1, total_batches - 1)))
        ids = model.rng.choice(len(x_tr), size=eff_batch, replace=False)
        tc = (np.arange(eff_batch) % model.diffusion_steps + 1).astype(np.int64)
        model.rng.shuffle(tc)
        tg = x_tr[ids]
        ab = model.alpha_bar[tc][:, None]
        xt = np.clip(np.sqrt(ab) * tg + np.sqrt(1.0 - ab) * model.rng.normal(0.0, 0.70, size=tg.shape), -1.0, 1.0)
        d_th = model.rng.choice([-1.0, 1.0], size=(eff_batch,) + model.theta.shape)
        _, z2 = model.measure_observables_batch(
            np.concatenate([xt, xt], axis=0),
            np.concatenate([tc, tc], axis=0),
            shots=shots_per_run,
            theta_batch=np.concatenate([model.theta[None, ...] + c_k * d_th, model.theta[None, ...] - c_k * d_th], axis=0),
        )
        zp, zm = z2[:eff_batch], z2[eff_batch:]
        z_mid = 0.5 * (zp + zm)
        for k in range(eff_batch):
            model.calibrator.update(int(tc[k]), z_mid[k], tg[k])
        if objective == "surrogate":
            # Minimize -sum(z * tg), matching QER's exact contrastive surrogate alignment objective
            lp = -np.sum(zp * tg, axis=1)
            lm = -np.sum(zm * tg, axis=1)
        else:
            gain_b = np.stack([model.calibrator.gain[int(t)] for t in tc], axis=0)
            if gain_b.ndim == 1:
                gain_b = gain_b[:, None]
            lp = np.mean((np.clip(zp / gain_b, -1.0, 1.0) - tg) ** 2, axis=1)
            lm = np.mean((np.clip(zm / gain_b, -1.0, 1.0) - tg) ** 2, axis=1)
        diff = (lp - lm) / (2.0 * c_k)
        g_th = -np.mean(diff[:, None] * d_th, axis=0)
        m_th[:] = 0.75 * m_th + 0.25 * g_th
        v_th[:] = 0.98 * v_th + 0.02 * (g_th ** 2)
        m_hat = m_th / (1.0 - (0.75 ** step))
        v_hat = v_th / (1.0 - (0.98 ** step))
        model.theta[:] = np.clip(model.theta + lr_k * m_hat / (np.sqrt(v_hat) + 1e-6), -clip_bound, clip_bound)


def _train_qer_residual_uncond(
    model: QuantumIsingDenoiser,
    x_tr: np.ndarray,
    max_shots: int = 40_000,
    shots_per_run: int = 20,
    batch_size: int = 10,
    clip_bound: float = 2.0,
) -> None:
    """3-run residual QER optimizing true MSE directly via y = tg - x0_hat at equal total shots."""
    eff_batch = min(batch_size, len(x_tr))
    total_batches = max(1, max_shots // (3 * shots_per_run * eff_batch))
    opt = QEROptimizer(model.parameters_dict(), QERConfig(lr_max=0.12, lr_min=0.015, clip_bound=clip_bound))
    for b_i in range(1, total_batches + 1):
        ids = model.rng.choice(len(x_tr), size=eff_batch, replace=False)
        tc = (np.arange(eff_batch) % model.diffusion_steps + 1).astype(np.int64)
        model.rng.shuffle(tc)
        tg = x_tr[ids]
        ab = model.alpha_bar[tc][:, None]
        xt = np.clip(np.sqrt(ab) * tg + np.sqrt(1.0 - ab) * model.rng.normal(0.0, 0.70, size=tg.shape), -1.0, 1.0)

        # Free readout (beta=0) for residual y = (tg - z_free / gain) / gain
        _, z_free = model.measure_observables_batch(xt, tc, shots=shots_per_run)
        y_res = np.zeros_like(tg)
        for k in range(eff_batch):
            t_k = int(tc[k])
            model.calibrator.update(t_k, z_free[k], tg[k])
            x0_hat = model.calibrator.calibrate(t_k, z_free[k])
            g_t = np.maximum(model.calibrator.gain[t_k], 0.20)
            y_res[k] = np.clip((tg[k] - x0_hat) / g_t, -2.0, 2.0)

        beta = opt.beta
        obs_both, z_both = model.measure_observables_batch(
            np.concatenate([xt, xt], axis=0),
            np.concatenate([tc, tc], axis=0),
            shots=shots_per_run,
            nudge_target_batch=np.concatenate([y_res, y_res], axis=0),
            nudge_beta_batch=np.concatenate([np.full(eff_batch, beta), np.full(eff_batch, -beta)]),
        )
        opt.step_contrast(
            model.parameters_dict(),
            obs_both[:eff_batch],
            obs_both[eff_batch:],
            z_plus_list=z_both[:eff_batch],
            z_minus_list=z_both[eff_batch:],
            nudge_dirs=y_res,
            progress=(b_i - 1) / max(1, total_batches - 1),
            shots_in_batch=3 * shots_per_run * eff_batch,
        )


def _train_psr_uncond(
    model: QuantumIsingDenoiser,
    x_tr: np.ndarray,
    max_shots: int = 40_000,
    shots_per_run: int = 20,
    lr: float = 0.08,
    delta: float = 0.25,
    clip_bound: float = 0.90,
) -> None:
    """Equal-shot coordinate finite-difference Adam-PSR-FD for cond_dim=0 Ising circuits."""
    n_th = model.theta.size
    step_shots = 2 * n_th * shots_per_run
    est_steps = max(1, max_shots // step_shots)
    m_th = np.zeros_like(model.theta)
    v_th = np.zeros_like(model.theta)
    shots_used = 0
    step = 0
    idx_th = np.arange(n_th)

    while shots_used + step_shots <= max_shots:
        step += 1
        lr_k = 0.015 + 0.5 * (lr - 0.015) * (1.0 + math.cos(math.pi * (step - 1) / max(1, est_steps - 1)))
        tgt = x_tr[int(model.rng.integers(0, len(x_tr)))]
        t = int(model.rng.integers(1, model.diffusion_steps + 1))
        ab = model.alpha_bar[t]
        xt = np.clip(math.sqrt(ab) * tgt + math.sqrt(1.0 - ab) * model.rng.normal(0.0, 0.70, size=model.n_visible), -1.0, 1.0)

        th_p = np.broadcast_to(model.theta, (n_th, n_th)).copy()
        th_m = th_p.copy()
        th_p[idx_th, idx_th] += delta
        th_m[idx_th, idx_th] -= delta

        _, z_all = model.measure_observables_batch(
            np.broadcast_to(xt, (2 * n_th, model.n_visible)),
            np.full((2 * n_th,), t, dtype=np.int64),
            shots=shots_per_run,
            theta_batch=np.concatenate([th_p, th_m], axis=0),
        )
        shots_used += step_shots
        model.calibrator.update(t, np.mean(z_all, axis=0), tgt)
        gain_t = model.calibrator.gain[t]
        lp = np.mean((np.clip(z_all[:n_th] / gain_t, -1.0, 1.0) - tgt[None, :]) ** 2, axis=1)
        lm = np.mean((np.clip(z_all[n_th:] / gain_t, -1.0, 1.0) - tgt[None, :]) ** 2, axis=1)
        g_th = -(lp - lm) / (2.0 * delta)
        m_th[:] = 0.75 * m_th + 0.25 * g_th
        v_th[:] = 0.98 * v_th + 0.02 * (g_th ** 2)
        m_hat = m_th / (1.0 - (0.75 ** step))
        v_hat = v_th / (1.0 - (0.98 ** step))
        model.theta[:] = np.clip(model.theta + lr_k * m_hat / (np.sqrt(v_hat) + 1e-6), -clip_bound, clip_bound)


def _train_exact_ad_uncond(
    x_tr: np.ndarray,
    n_layers: int,
    clip_bound: float,
    dev: torch.device,
    seed: int,
    n_eval: int = 320,
    n_steps: int = 200,
    objective: str = "mse",
    use_fwdols: bool = False,
) -> Tuple[np.ndarray, float]:
    """Noiseless exact PyTorch autograd control on the identical Ising unitary circuit."""
    from qerd import ReadoutGainCalibrator

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 11)
    edges = bm.build_ising_edges((3, 4), topology="all")
    circ = bm._DiffIsingCircuit(12, cond_dim=0, edges=edges, n_hidden=0, n_layers=n_layers, device=dev)
    th = torch.zeros(len(edges) + 12, dtype=torch.float32, device=dev, requires_grad=True)
    opt = torch.optim.Adam([th], lr=0.08)
    ab = bm.make_alpha_bar(4, "cosine")
    x_t_all = torch.tensor(x_tr, dtype=torch.float32, device=dev)
    cal = ReadoutGainCalibrator(4, 12, init_gain=0.75, ema=0.10) if use_fwdols else None

    for _ in range(n_steps):
        ids = rng.integers(0, len(x_tr), size=32)
        tc = rng.integers(1, 5, size=32)
        tg = x_t_all[ids]
        a = torch.tensor(ab[tc], dtype=torch.float32, device=dev)[:, None]
        eps = torch.tensor(rng.normal(0.0, 0.70, size=(32, 12)), dtype=torch.float32, device=dev)
        xt = torch.clamp(torch.sqrt(a) * tg + torch.sqrt(1.0 - a) * eps, -1.0, 1.0)
        _, z = circ.expectation(xt, torch.tensor(tc, dtype=torch.long, device=dev), None, th, None)
        if cal is not None:
            with torch.no_grad():
                z_np = z.detach().cpu().numpy()
                tg_np = tg.detach().cpu().numpy()
                for k in range(32):
                    cal.update(int(tc[k]), z_np[k], tg_np[k])
        if objective == "surrogate":
            loss = -torch.sum(z * tg, dim=1).mean()
        elif cal is not None:
            g_b = torch.tensor(
                np.stack([cal.gain[int(t)] for t in tc], axis=0), dtype=torch.float32, device=dev
            )
            if g_b.ndim == 1:
                g_b = g_b[:, None]
            loss = F.mse_loss(torch.clamp(z / g_b, -1.0, 1.0), tg)
        else:
            loss = F.mse_loss(z, tg)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            th.clamp_(-clip_bound, clip_bound)

    max_th = float(torch.max(torch.abs(th)).item())
    with torch.no_grad():
        xt = np.clip(rng.normal(0.0, 0.90, size=(n_eval, 12)), -1.0, 1.0)
        for t in range(4, 0, -1):
            _, z = circ.expectation(
                torch.tensor(xt, dtype=torch.float32, device=dev),
                torch.full((n_eval,), t, dtype=torch.long, device=dev),
                None, th, None,
            )
            z_np = z.cpu().numpy()
            x0_hat = cal.calibrate(t, z_np) if cal is not None else z_np
            if t > 1:
                xt = np.clip(math.sqrt(ab[t - 1]) * x0_hat + math.sqrt(1.0 - ab[t - 1]) * rng.normal(0.0, 0.30, x0_hat.shape), -1.0, 1.0)
            else:
                xt = x0_hat
    return xt, max_th


def _agg_metrics(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for k in rows[0]:
        arr = np.array([r[k] for r in rows], dtype=np.float64)
        out[k] = {"mean": float(np.mean(arr)), "std": float(np.std(arr))}
    return out


def run_experiments(
    out_dir: str,
    seeds: List[int],
    max_shots: int = 40_000,
    n_eval: int = 320,
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend = "torch"
    modes_arr, valid_set, intra_pairs, c_ref = _build_spinglass_problem()

    configs = [
        # Group A: Matched-NISQ Intra-QPU Controls (p_depol=0.035, p_readout=0.04, L=2, clip=0.90, OptA)
        ("qerd_nozz", "QERD no-zz (L=2, NISQ OptA)", dict(kind="qer", topo="none", nl=2, clip=0.90, pd=0.035, pr=0.04, g0=1.0, ema=0.0)),
        ("qerd_chain", "QERD chain (L=2, NISQ OptA)", dict(kind="qer", topo="chain", nl=2, clip=0.90, pd=0.035, pr=0.04, g0=1.0, ema=0.0)),
        ("qerd_grid4", "QERD grid4 (L=2, NISQ OptA)", dict(kind="qer", topo="grid4", nl=2, clip=0.90, pd=0.035, pr=0.04, g0=1.0, ema=0.0)),
        ("qerd_grid8", "QERD grid8 (L=2, NISQ OptA)", dict(kind="qer", topo="grid8", nl=2, clip=0.90, pd=0.035, pr=0.04, g0=1.0, ema=0.0)),
        ("qerd_all_optA_nisq", "QERD all (L=2, NISQ OptA, Table 14)", dict(kind="qer", topo="all", nl=2, clip=0.90, pd=0.035, pr=0.04, g0=1.0, ema=0.0)),
        ("spsa_all_nisq", "Adam-SPSA MSE all (L=2, NISQ 40k)", dict(kind="spsa", topo="all", nl=2, clip=0.90, pd=0.035, pr=0.04, g0=0.75, ema=0.10, obj="mse")),
        ("spsa_all_l2_surr_nisq", "Adam-SPSA Surrogate all (L=2, NISQ 40k)", dict(kind="spsa", topo="all", nl=2, clip=0.90, pd=0.035, pr=0.04, g0=1.0, ema=0.0, obj="surrogate")),
        ("psr_all_nisq", "Adam-PSR-FD all (L=2, NISQ 40k)", dict(kind="psr", topo="all", nl=2, clip=0.90, pd=0.035, pr=0.04, g0=0.75, ema=0.10)),
        # Group B: Equal-Objective & Matched-Box Controls (Calibrator -> Noise -> Depth/Clip -> Classical Oracle)
        ("exact_ad_l2_clean", "Exact-AD MSE (L=2, clip=0.90, Clean)", dict(kind="ad", nl=2, clip=0.90, obj="mse", fwdols=False)),
        ("exact_ad_l2_surr_clean", "Exact-AD Surrogate (L=2, clip=0.90, Clean)", dict(kind="ad", nl=2, clip=0.90, obj="surrogate", fwdols=False)),
        ("qerd_all_l2_fwdols_nisq", "QERD all (L=2, clip=0.90, NISQ FwdOLS)", dict(kind="qer", topo="all", nl=2, clip=0.90, pd=0.035, pr=0.04, g0=0.75, ema=0.10)),
        ("qerd_all_l2_fwdols_clean", "QERD all (L=2, clip=0.90, Clean FwdOLS)", dict(kind="qer", topo="all", nl=2, clip=0.90, pd=0.0, pr=0.0, g0=0.75, ema=0.10)),
        ("qerd_all_l4_b090_fwdols_nisq", "QERD all (L=4, clip=0.90, NISQ FwdOLS)", dict(kind="qer", topo="all", nl=4, clip=0.90, pd=0.035, pr=0.04, g0=0.75, ema=0.10)),
        ("qerd_all_l4_optA_nisq", "QERD all (L=4, clip=2.00, NISQ OptA)", dict(kind="qer", topo="all", nl=4, clip=2.0, pd=0.035, pr=0.04, g0=1.0, ema=0.0)),
        ("qerd_nozz_l4_fwdols_nisq", "QERD no-zz (L=4, clip=2.00, NISQ FwdOLS)", dict(kind="qer", topo="none", nl=4, clip=2.0, pd=0.035, pr=0.04, g0=0.75, ema=0.10)),
        ("qerd_all_l4_fwdols_nisq", "QERD all (L=4, clip=2.00, NISQ FwdOLS)", dict(kind="qer", topo="all", nl=4, clip=2.0, pd=0.035, pr=0.04, g0=0.75, ema=0.10)),
        ("qerd_all_l4_fwdols_clean", "QERD all (L=4, clip=2.00, Clean FwdOLS)", dict(kind="qer", topo="all", nl=4, clip=2.0, pd=0.0, pr=0.0, g0=0.75, ema=0.10)),
        ("spsa_all_l4_surr_nisq", "Adam-SPSA Surrogate all (L=4, clip=2.00, NISQ 40k)", dict(kind="spsa", topo="all", nl=4, clip=2.0, pd=0.035, pr=0.04, g0=0.75, ema=0.10, obj="surrogate")),
        ("qer_all_l4_res_mse_nisq", "QER-Residual MSE all (L=4, clip=2.00, NISQ 40k)", dict(kind="qer_res", topo="all", nl=4, clip=2.0, pd=0.035, pr=0.04, g0=0.75, ema=0.10)),
        ("exact_ad_l4_b20_clean", "Exact-AD MSE (L=4, clip=2.00, Clean)", dict(kind="ad", nl=4, clip=2.0, obj="mse", fwdols=False)),
        ("exact_ad_l4_surr_clean", "Exact-AD Surrogate (L=4, clip=2.00, Clean)", dict(kind="ad", nl=4, clip=2.0, obj="surrogate", fwdols=False)),
        ("exact_ad_l4_surr_fwdols_clean", "Exact-AD Surrogate+FwdOLS (L=4, clip=2.00, Clean)", dict(kind="ad", nl=4, clip=2.0, obj="surrogate", fwdols=True)),
        ("classical_ising_ep", "ClassicalIsing-EP (Noiseless Float64 Oracle)", dict(kind="cl_ep")),
    ]

    buckets: Dict[str, List[Dict[str, float]]] = {k: [] for k, _, _ in configs}
    print(f"[Setup] 12q Cluster Spin Glass | Device={dev} | Seeds={seeds} | max_shots={max_shots}", flush=True)

    for seed in seeds:
        t0 = time.time()
        rng = np.random.default_rng(seed)
        x_tr = modes_arr[rng.integers(0, len(modes_arr), size=800)]
        print(f"\n=== Seed {seed} ===", flush=True)

        for key, label, cfg in configs:
            kind = cfg["kind"]
            if kind == "qer":
                m = QuantumIsingDenoiser(
                    (3, 4), cond_dim=0, topology=cfg["topo"], n_layers=cfg["nl"],
                    p_depol=cfg["pd"], p_readout=cfg["pr"], init_gain=cfg["g0"], gain_ema=cfg["ema"],
                    backend=backend, torch_device=str(dev), seed=seed,
                )
                tr = QERDTrainer(m, QEROptimizer(m.parameters_dict(), QERConfig(lr_max=0.12, lr_min=0.015, clip_bound=cfg["clip"])))
                tr.train(x_tr, conds=None, max_shots=max_shots, shots_per_run=20, batch_size=10, eval_every_batches=999999, verbose=False)
                gens = m.sample_batch(n_samples=n_eval, shots_per_step=160, step_noise_scale=0.30)
                max_th = float(np.max(np.abs(m.theta)))
            elif kind == "qer_res":
                m = QuantumIsingDenoiser(
                    (3, 4), cond_dim=0, topology=cfg["topo"], n_layers=cfg["nl"],
                    p_depol=cfg["pd"], p_readout=cfg["pr"], init_gain=cfg["g0"], gain_ema=cfg["ema"],
                    backend=backend, torch_device=str(dev), seed=seed,
                )
                _train_qer_residual_uncond(m, x_tr, max_shots=max_shots, shots_per_run=20, batch_size=10, clip_bound=cfg["clip"])
                gens = m.sample_batch(n_samples=n_eval, shots_per_step=160, step_noise_scale=0.30)
                max_th = float(np.max(np.abs(m.theta)))
            elif kind == "spsa":
                m = QuantumIsingDenoiser(
                    (3, 4), cond_dim=0, topology=cfg["topo"], n_layers=cfg["nl"],
                    p_depol=cfg["pd"], p_readout=cfg["pr"], init_gain=cfg["g0"], gain_ema=cfg["ema"],
                    backend=backend, torch_device=str(dev), seed=seed,
                )
                _train_spsa_uncond(m, x_tr, max_shots=max_shots, shots_per_run=20, batch_size=10, clip_bound=cfg["clip"], objective=cfg["obj"])
                gens = m.sample_batch(n_samples=n_eval, shots_per_step=160, step_noise_scale=0.30)
                max_th = float(np.max(np.abs(m.theta)))
            elif kind == "psr":
                m = QuantumIsingDenoiser(
                    (3, 4), cond_dim=0, topology=cfg["topo"], n_layers=cfg["nl"],
                    p_depol=cfg["pd"], p_readout=cfg["pr"], init_gain=cfg["g0"], gain_ema=cfg["ema"],
                    backend=backend, torch_device=str(dev), seed=seed,
                )
                _train_psr_uncond(m, x_tr, max_shots=max_shots, shots_per_run=20, clip_bound=cfg["clip"])
                gens = m.sample_batch(n_samples=n_eval, shots_per_step=160, step_noise_scale=0.30)
                max_th = float(np.max(np.abs(m.theta)))
            elif kind == "ad":
                gens, max_th = _train_exact_ad_uncond(
                    x_tr, n_layers=cfg["nl"], clip_bound=cfg["clip"], dev=dev, seed=seed,
                    n_eval=n_eval, objective=cfg.get("obj", "mse"), use_fwdols=cfg.get("fwdols", False),
                )
            else:
                mdl = bm.ClassicalIsingEP((3, 4), cond_dim=0, topology="all", seed=seed)
                opt_c = QEROptimizer(mdl.parameters_dict(), QERConfig(lr_max=0.12, lr_min=0.015, clip_bound=0.90))
                n_st = max(1, max_shots // (2 * 20 * 10))
                for b_i in range(1, n_st + 1):
                    ids = mdl.rng.choice(len(x_tr), size=10, replace=False)
                    tc = (np.arange(10) % 4) + 1
                    mdl.rng.shuffle(tc)
                    tg = x_tr[ids]
                    ab = mdl.alpha_bar[tc][:, None]
                    xt = np.clip(np.sqrt(ab) * tg + np.sqrt(1.0 - ab) * mdl.rng.normal(0.0, 0.70, size=tg.shape), -1.0, 1.0)
                    o2, _ = mdl._relax_batch(
                        np.concatenate([xt, xt], axis=0), np.concatenate([tc, tc], axis=0), None,
                        np.concatenate([tg, tg], axis=0), np.concatenate([np.full(10, opt_c.beta), np.full(10, -opt_c.beta)]),
                    )
                    opt_c.step_contrast(mdl.parameters_dict(), o2[:10], o2[10:], progress=(b_i - 1) / max(1, n_st - 1))
                gens = mdl.sample_batch(n_samples=n_eval, step_noise_scale=0.30)
                max_th = float(np.max(np.abs(mdl.theta)))

            met = _eval_spin_samples(gens, valid_set, intra_pairs, c_ref, len(modes_arr))
            met["max_abs_theta"] = max_th
            buckets[key].append(met)
            print(f"  [{key:28s}] IntraCorr={met['intra_corr']:+.3f} | PatAcc={met['pattern_acc']*100:5.1f}% | ModeCov={met['mode_cov']*100:5.1f}% | CorrRMSE={met['corr_rmse']:.3f} | Max|th|={max_th:.2f}", flush=True)
        print(f"  Seed {seed} finished in {time.time() - t0:.1f}s", flush=True)

    summary = []
    for key, label, _ in configs:
        summary.append({
            "key": key,
            "label": label,
            "aggregate": _agg_metrics(buckets[key]),
            "per_seed": buckets[key],
        })

    results = {"seeds": seeds, "max_shots": max_shots, "n_eval": n_eval, "variants": summary}
    json_p = os.path.join(out_dir, "spinglass_diagnosis_results.json")
    with open(json_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[Saved] {json_p}", flush=True)
    return results


def make_proof_figure(results: Dict[str, Any], out_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.size": 8.6, "axes.titlesize": 9.6,
        "axes.titleweight": "bold", "axes.labelsize": 8.6, "xtick.labelsize": 7.6,
        "ytick.labelsize": 7.8, "legend.fontsize": 7.2,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
    })

    by_k = {r["key"]: r["aggregate"] for r in results["variants"]}
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.0))

    # Panel (a): Matched-NISQ Topology & QPU Optimizers (40k shots, p_depol=0.035, p_readout=0.04)
    ax = axes[0, 0]
    keys_a = ["qerd_nozz", "qerd_chain", "qerd_grid4", "qerd_grid8", "qerd_all_optA_nisq", "spsa_all_l2_surr_nisq", "spsa_all_nisq", "psr_all_nisq"]
    lbl_a = ["QER\nno-zz", "QER\nchain", "QER\ngrid4", "QER\ngrid8", "QER\nall", "SPSA-Surr\nall", "SPSA-MSE\nall", "PSR-FD\nall"]
    col_a = ["#CC79A7", "#999999", "#999999", "#56B4E9", "#0072B2", "#E69F00", "#D55E00", "#B07C00"]
    means_a = [by_k[k]["intra_corr"]["mean"] for k in keys_a]
    stds_a = [by_k[k]["intra_corr"]["std"] for k in keys_a]
    x = np.arange(len(keys_a))
    ax.axhline(0.0, color="#444444", linewidth=0.8)
    bars = ax.bar(x, means_a, yerr=stds_a, capsize=3.2, width=0.58, color=col_a, edgecolor="#222222", linewidth=0.7)
    for bar, m_v, s_v in zip(bars, means_a, stds_a):
        ax.text(bar.get_x() + bar.get_width() / 2.0, max(m_v, 0.0) + s_v + 0.006, f"{m_v:+.3f}", ha="center", va="bottom", fontsize=6.8)
    ax.set_xticks(x)
    ax.set_xticklabels(lbl_a, fontsize=7.0)
    y_lo = min(-0.04, min(m - s for m, s in zip(means_a, stds_a)) - 0.03)
    y_hi = max(0.22, max(m + s for m, s in zip(means_a, stds_a)) + 0.04)
    ax.set_ylim(y_lo, y_hi)
    ax.set_ylabel("Intra-Cluster Spin Correlation $\\uparrow$")
    shots_k = int(round(results.get("max_shots", 40_000) / 1000.0))
    ax.set_title(f"(a) Equal-Shot NISQ QPU Controls (${shots_k}\\mathrm{{k}}$ shots, $L=2$, $p_{{\\mathrm{{dep}}}}=3.5\\%$)")
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    # Panel (b): Equal-Objective & Matched-Box Factorial Attribution (Depth, ZZ, Calibration, Noise)
    ax = axes[0, 1]
    keys_b = [
        "qerd_all_l2_fwdols_nisq",
        "qerd_nozz_l4_fwdols_nisq",
        "spsa_all_l4_surr_nisq",
        "qer_all_l4_res_mse_nisq",
        "qerd_all_l4_fwdols_nisq",
        "qerd_all_l4_fwdols_clean",
        "exact_ad_l4_b20_clean",
        "exact_ad_l4_surr_fwdols_clean",
    ]
    lbl_b = [
        "QER L=2\nFwdOLS NISQ",
        "QER L=4\nno-zz NISQ",
        "SPSA-Surr\nL=4 NISQ",
        "QER-Res\nL=4 MSE",
        "QER L=4\nFwdOLS NISQ",
        "QER L=4\nFwdOLS Clean",
        "Exact-AD\nL=4 MSE",
        "Exact-AD\nSurr+FwdOLS",
    ]
    x_b = np.arange(len(keys_b))
    w = 0.37
    corr_m = [by_k[k]["intra_corr"]["mean"] for k in keys_b]
    corr_s = [by_k[k]["intra_corr"]["std"] for k in keys_b]
    acc_m = [by_k[k]["pattern_acc"]["mean"] for k in keys_b]
    acc_s = [by_k[k]["pattern_acc"]["std"] for k in keys_b]
    ax.bar(x_b - w / 2, corr_m, yerr=corr_s, width=w, capsize=2.8, color="#0072B2", edgecolor="#222222", linewidth=0.7, label="Intra-Cluster Corr")
    ax.bar(x_b + w / 2, acc_m, yerr=acc_s, width=w, capsize=2.8, color="#009E73", edgecolor="#222222", linewidth=0.7, label="Exact Pattern Acc / 1.0")
    for xi, cm, cs, am, am_s in zip(x_b, corr_m, corr_s, acc_m, acc_s):
        ax.text(xi - w / 2, max(cm, 0.0) + cs + 0.016, f"{cm:.2f}", ha="center", va="bottom", fontsize=6.0, color="#005587")
        ax.text(xi + w / 2, am + am_s + 0.016, f"{am*100:.0f}%", ha="center", va="bottom", fontsize=6.0, color="#006E50")
    ax.set_xticks(x_b)
    ax.set_xticklabels(lbl_b, fontsize=6.1)
    ax.set_ylim(-0.04, 1.14)
    ax.set_ylabel("Metric Score [0, 1]")
    ax.set_title("(b) Equal-Objective Controls ($L=2/4$, Surrogate vs. MSE, NISQ vs. Clean)")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=True, facecolor="white", framealpha=0.92, loc="upper left")

    # Panel (c): Disentangling Calibrator + Objective + Noise vs Classical Oracle Reference
    ax = axes[1, 0]
    keys_c = [
        "exact_ad_l2_clean",
        "exact_ad_l2_surr_clean",
        "qerd_all_optA_nisq",
        "qerd_all_l2_fwdols_nisq",
        "spsa_all_l4_surr_nisq",
        "qer_all_l4_res_mse_nisq",
        "qerd_all_l4_fwdols_nisq",
        "qerd_all_l4_fwdols_clean",
        "exact_ad_l4_b20_clean",
        "exact_ad_l4_surr_fwdols_clean",
        "classical_ising_ep",
    ]
    lbl_c = [
        "Exact-AD L=2 MSE (Clean)",
        "Exact-AD L=2 Surrogate (Clean)",
        "QER L=2 OptA (NISQ)",
        "QER L=2 FwdOLS (NISQ)",
        "SPSA L=4 Surrogate (NISQ)",
        "QER-Res L=4 MSE (NISQ)",
        "QER L=4 FwdOLS (NISQ)",
        "QER L=4 FwdOLS (Clean)",
        "Exact-AD L=4 MSE (Clean)",
        "Exact-AD L=4 Surr+FwdOLS (Clean)",
        "ClassicalIsing-EP (Float64 Oracle)",
    ]
    col_c = ["#999999", "#BBBBBB", "#D55E00", "#E69F00", "#CC79A7", "#88CCEE", "#56B4E9", "#0072B2", "#6699CC", "#3366AA", "#009E73"]
    y_pos = np.arange(len(keys_c))
    acc_c = [by_k[k]["pattern_acc"]["mean"] * 100.0 for k in keys_c]
    acc_cs = [by_k[k]["pattern_acc"]["std"] * 100.0 for k in keys_c]
    bars_c = ax.barh(y_pos, acc_c, xerr=acc_cs, height=0.62, capsize=2.8, color=col_c, edgecolor="#222222", linewidth=0.7)
    for bar, val, stdv in zip(bars_c, acc_c, acc_cs):
        ax.text(val + stdv + 1.4, bar.get_y() + bar.get_height() / 2.0, f"{val:.1f}%", va="center", ha="left", fontsize=7.0, fontweight="bold")
    ax.axvline(100.0 * 8.0 / 4096.0, color="#666666", linestyle=":", linewidth=1.2, label="Random uniform 12-bit chance (0.20%)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(lbl_c, fontsize=6.6)
    ax.set_xlim(0, 115)
    ax.set_xlabel("Exact Ground-State Pattern Accuracy (%) $\\uparrow$")
    ax.set_title("(c) Exact-Pattern Attribution Across Equal-Objective Controls")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.legend(frameon=False, loc="lower right", fontsize=7.0)

    # Panel (d): Joint Correlation vs Ground-Mode Coverage (No 1-Mode Collapse)
    ax = axes[1, 1]
    plot_d = [
        ("qerd_nozz", "QER L=2 no-zz (NISQ)", "#CC79A7", "X"),
        ("spsa_all_nisq", "SPSA-MSE all (NISQ)", "#D55E00", "v"),
        ("spsa_all_l4_surr_nisq", "SPSA-Surr L=4 (NISQ)", "#E69F00", "d"),
        ("exact_ad_l2_clean", "Exact-AD L=2 MSE (Clean)", "#888888", "s"),
        ("qerd_all_l2_fwdols_nisq", "QER L=2 FwdOLS (NISQ)", "#56B4E9", "P"),
        ("qer_all_l4_res_mse_nisq", "QER-Res L=4 MSE (NISQ)", "#88CCEE", "p"),
        ("qerd_all_l4_fwdols_nisq", "QER L=4 FwdOLS (NISQ)", "#0072B2", "D"),
        ("qerd_all_l4_fwdols_clean", "QER L=4 FwdOLS (Clean)", "#004D7A", "*"),
        ("exact_ad_l4_b20_clean", "Exact-AD L=4 MSE (Clean)", "#6699CC", "h"),
        ("exact_ad_l4_surr_fwdols_clean", "Exact-AD L=4 Surr+FwdOLS (Clean)", "#3366AA", "H"),
        ("classical_ising_ep", "ClassicalIsing-EP (Oracle)", "#009E73", "^"),
    ]
    for k, lbl, col, mk in plot_d:
        cx = by_k[k]["intra_corr"]["mean"]
        my = by_k[k]["mode_cov"]["mean"] * 100.0
        sz = 190 if mk == "*" else 90
        ax.scatter([cx], [my], color=col, marker=mk, s=sz, edgecolor="#111111", linewidth=0.8, label=lbl, zorder=5)
    ax.axvline(0.0, color="#444444", linewidth=0.7, linestyle="--")
    ax.set_xlim(-0.08, 1.05)
    ax.set_ylim(0, 108)
    ax.set_xlabel("Intra-Cluster Spin Correlation $\\uparrow$")
    ax.set_ylabel("Valid Ground Modes Covered (%) $\\uparrow$")
    ax.set_title("(d) Multi-Mode Recovery (Cluster Correlation vs. Mode Coverage)")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(frameon=True, facecolor="white", framealpha=0.92, loc="lower right", fontsize=6.4, ncol=1)

    n_seeds = len(results["seeds"])
    seed_lbl = "Seed" if n_seeds == 1 else "Seeds"
    fig.suptitle(
        f"Attribution of Zero-AE 12-Qubit Stride-3 Correlated Cluster Performance ({n_seeds} {seed_lbl})\n"
        "(Matched-Box Factorial Controls Isolate $ZZ$ Coupling, Depth $L$, Forward-OLS Calibration, & Hardware Noise)",
        fontsize=10.4, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.945))
    pdf_p = os.path.join(out_dir, "fig_spinglass_diagnosis.pdf")
    png_p = os.path.join(out_dir, "fig_spinglass_diagnosis.png")
    fig.savefig(pdf_p, bbox_inches="tight")
    fig.savefig(png_p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {pdf_p} & {png_p}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default="./analysis")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--max-shots", type=int, default=40_000)
    ap.add_argument("--n-eval", type=int, default=320)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--replot", action="store_true")
    args = ap.parse_args()

    json_p = os.path.join(args.out_dir, "spinglass_diagnosis_results.json")
    if args.replot:
        with open(json_p, "r", encoding="utf-8") as f:
            res = json.load(f)
    else:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        shots = 20_000 if args.quick else args.max_shots
        neval = 160 if args.quick else args.n_eval
        res = run_experiments(args.out_dir, seeds=seeds, max_shots=shots, n_eval=neval)
    make_proof_figure(res, args.out_dir)


if __name__ == "__main__":
    main()
