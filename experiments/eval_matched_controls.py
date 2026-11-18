#!/usr/bin/env python3
"""
Matched Same-Circuit Exact-AD, L=4 Spin-Glass Reciprocity, and Hybrid Field-PSR Control Suite:
  [Control 1] Same-Circuit Noiseless Exact-AD References on identical QPU circuits:
              - MNIST (12+0q, P=198, L=2): step200_mse, asymp600_mse, step200_surr
              - Fashion-MNIST (12+2q, P=224, L=2): step200_mse, asymp600_mse, step200_surr
              - Zero-W_cond Label-Qubit Circuits (13q {0,1} L=3, 14q {0,1,4} L=3)
  [Control 2] Matched Spin-Glass L=4 Exact-AD Control (MSE + FwdOLS + rand08 + Cosine LR).
  [Control 3] Trained L=4 Spin-Glass Checkpoint Reciprocity (Clean vs NISQ) + 4k-Shot SNR.
  [Control 4] Equal-Shot PSR-on-Input-Fields Hybrid Bridge Control (HybridResMLPPSR, h=48).

Outputs:
  ./analysis/matched_controls_results.json
  ./analysis/fig_matched_controls.pdf / .png

Usage:
  python -u experiments/eval_matched_controls.py --quick
  python -u experiments/eval_matched_controls.py
  python -u experiments/eval_matched_controls.py --replot
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
import eval_reciprocity_and_stepper as prb
import eval_conditioning_and_opt as prf
from qerd import (
    QERConfig,
    QEROptimizer,
    QERDTrainer,
    QuantumIsingDenoiser,
    ReadoutGainCalibrator,
)


def _cosine_lr(step: int, total_steps: int, lr_max: float, lr_min: float = 0.015) -> float:
    prog = (step - 1) / max(1, total_steps - 1)
    return lr_min + 0.5 * max(0.0, lr_max - lr_min) * (1.0 + math.cos(math.pi * prog))


def _load_or_cache_helpers(
    dataset: str, data_dir: str, out_dir: str, dev: torch.device, allow_download: bool = True
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, nn.Module, bm.EvalCNN, np.ndarray, np.ndarray]]:
    """Load dataset + cached frozen AE & EvalCNN (reusing paper_suite caches if available)."""
    if dataset == "fashion" and not allow_download and not os.path.exists(os.path.join(data_dir, "FashionMNIST")):
        return None
    x_tr, y_tr, x_te, y_te, _ = bm.load_dataset(dataset, data_dir=data_dir)
    candidates = [
        os.path.join(out_dir, f"{dataset}_d12_ae8_clf5_helpers.pt"),
        os.path.join("./paper_suite", f"{dataset}_main", f"{dataset}_d12_ae8_clf5_helpers.pt"),
        os.path.join("./benchmark_out", f"{dataset}_d12_ae8_clf5_helpers.pt"),
    ]
    cache_p = next((p for p in candidates if os.path.exists(p)), candidates[0])
    ae, clf, z_tr, z_te = bm.train_classical_helpers(
        x_tr, y_tr, x_te, y_te, latent_dim=12, ae_epochs=8, clf_epochs=5,
        device=dev, cache_path=cache_p, use_bn=True,
    )
    return x_tr, y_tr, x_te, y_te, ae, clf, z_tr, z_te


def _train_same_circuit_exact_ad(
    z_tr: np.ndarray, y_tr: np.ndarray, z_te: np.ndarray, x_te: np.ndarray, y_te: np.ndarray,
    ae: nn.Module, clf: bm.EvalCNN, dev: torch.device,
    n_hidden: int, n_steps: int, batch_size: int, objective: str,
    seeds: List[int], eval_per_cls: int, lr_max: float = 0.12, clip_bound: float = 0.90,
) -> Dict[str, Any]:
    """Train Exact-AD directly on the identical QuantumIsingDenoiser unitary circuit."""
    n_vis = 12
    edges = bm.build_ising_edges(n_vis, topology="all", n_hidden=n_hidden)
    n_edges = len(edges)
    n_qpu = n_edges + n_vis + n_hidden
    ab = bm.make_alpha_bar(4, "cosine")
    eye_c = np.eye(10, dtype=np.float32)
    c_tr = eye_c[y_tr]
    pools = [np.flatnonzero(y_tr == c) for c in range(10)]
    ev_lbl = np.repeat(np.arange(10, dtype=np.int64), eval_per_cls)
    ev_c = torch.from_numpy(eye_c[ev_lbl]).to(dev)
    circ = bm._DiffIsingCircuit(n_vis, cond_dim=10, edges=edges, n_hidden=n_hidden, n_layers=2, device=dev)

    rows = []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        th_init = np.zeros(n_qpu, dtype=np.float32)
        if n_hidden > 0:
            for e_i, (u, v) in enumerate(edges):
                if u >= n_vis or v >= n_vis:
                    th_init[e_i] = rng.normal(0.0, 0.08)
            th_init[n_edges + n_vis :] = rng.normal(0.0, 0.04, size=n_hidden)

        th = torch.tensor(th_init, dtype=torch.float32, device=dev, requires_grad=True)
        cp = torch.zeros((n_vis, 10), dtype=torch.float32, device=dev, requires_grad=True)
        opt = torch.optim.Adam([th, cp], lr=lr_max, betas=(0.75, 0.98), eps=1e-6)

        for st in range(1, n_steps + 1):
            lr_k = _cosine_lr(st, n_steps, lr_max, 0.015)
            decay = 1.0 - 0.01 * (lr_k / max(lr_max, 1e-8))
            for pg in opt.param_groups:
                pg["lr"] = lr_k
            ids = bm._sample_stratified_ids(rng, pools, batch_size)
            tb = (np.arange(batch_size) % 4 + 1).astype(np.int64); rng.shuffle(tb)
            tg_np = z_tr[ids].astype(np.float32)
            a_b = ab[tb][:, None]
            xt_np = np.clip(np.sqrt(a_b) * tg_np + np.sqrt(1.0 - a_b) * rng.normal(0.0, 0.70, size=tg_np.shape), -1.0, 1.0).astype(np.float32)

            xt_t = torch.from_numpy(xt_np).to(dev)
            tg_t = torch.from_numpy(tg_np).to(dev)
            cb_t = torch.from_numpy(c_tr[ids]).to(dev)
            t_t = torch.from_numpy(tb).to(dev)

            _, z_pred = circ.expectation(xt_t, t_t, cb_t, th, cp)
            if objective == "surrogate":
                loss = -torch.sum(z_pred * tg_t, dim=1).mean()
            else:
                loss = F.mse_loss(z_pred, tg_t)
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                th.mul_(decay).clamp_(-clip_bound, clip_bound)
                cp.mul_(decay).clamp_(-clip_bound, clip_bound)

        # Clean analytic 4-step reverse diffusion sampling (chunked for 14q memory safety)
        n_ev = len(ev_lbl)
        x_gen = np.clip(rng.normal(0.0, 0.90, size=(n_ev, n_vis)), -1.0, 1.0).astype(np.float32)
        chunk = 250
        with torch.no_grad():
            for t in range(4, 0, -1):
                x0_list = []
                for s_i in range(0, n_ev, chunk):
                    e_i = min(n_ev, s_i + chunk)
                    _, zp = circ.expectation(
                        torch.from_numpy(x_gen[s_i:e_i]).to(dev),
                        torch.full((e_i - s_i,), t, dtype=torch.long, device=dev),
                        ev_c[s_i:e_i], th, cp,
                    )
                    x0_list.append(zp.cpu().numpy())
                x0_h = np.concatenate(x0_list, axis=0)
                if t > 1:
                    an = ab[t - 1]
                    x_gen = np.clip(math.sqrt(an) * x0_h + math.sqrt(1.0 - an) * rng.normal(0.0, 0.70, size=x0_h.shape), -1.0, 1.0).astype(np.float32)
                else:
                    x_gen = x0_h
        met, _ = bm.evaluate_generated(f"ExactAD-nh{n_hidden}-{objective}-st{n_steps}", x_gen.astype(np.float64), ev_lbl, ae, clf, z_te, x_te, y_te, dev)
        rows.append(met)
    return prf._agg_dicts(rows)


def _train_label_qubit_exact_ad(
    z_tr: np.ndarray, y_tr: np.ndarray, cls_sub: List[int], codes: np.ndarray,
    ae: nn.Module, clf: bm.EvalCNN, dev: torch.device,
    seeds: List[int], n_steps: int, batch_size: int = 10, n_eval_per_cls: int = 120,
    lr_max: float = 0.16, clip_bound: float = 1.50, field_amp: float = 1.15,
) -> Dict[str, Any]:
    """Same-circuit noiseless Exact-AD control for Zero-W_cond Label-Qubit Ising (13q/14q, L=3)."""
    mask = np.isin(y_tr, cls_sub)
    z_sub = z_tr[mask].astype(np.float32)
    cmap = {c: i for i, c in enumerate(cls_sub)}
    y_idx = np.array([cmap[int(y)] for y in y_tr[mask]], dtype=np.int64)
    n_cls = len(cls_sub)
    n_lbl = codes.shape[1]
    nq = 12 + n_lbl
    edges = bm.build_ising_edges(nq, topology="all", n_hidden=0)
    ab = bm.make_alpha_bar(4, "cosine")
    circ = bm._DiffIsingCircuit(nq, cond_dim=0, edges=edges, n_hidden=0, n_layers=3, device=dev)
    pools = [np.flatnonzero(y_idx == i) for i in range(n_cls)]

    rows = []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        th = torch.zeros(len(edges) + nq, dtype=torch.float32, device=dev, requires_grad=True)
        opt = torch.optim.Adam([th], lr=lr_max, betas=(0.75, 0.98), eps=1e-6)
        for st in range(1, n_steps + 1):
            lr_k = _cosine_lr(st, n_steps, lr_max, 0.015)
            decay = 1.0 - 0.01 * (lr_k / max(lr_max, 1e-8))
            for pg in opt.param_groups:
                pg["lr"] = lr_k
            c_ord = rng.permutation(n_cls)
            ids = np.array([rng.choice(pools[int(c_ord[k % n_cls])]) for k in range(batch_size)], dtype=np.int64)
            tc = (np.arange(batch_size) % 4 + 1).astype(np.int64); rng.shuffle(tc)
            zt = z_sub[ids]
            a_b = ab[tc][:, None]
            xv = np.clip(np.sqrt(a_b) * zt + np.sqrt(1.0 - a_b) * rng.normal(0.0, 0.70, size=zt.shape), -1.0, 1.0).astype(np.float32)
            in_pin = circ.in_pinning_base + circ.in_pinning_slope * (1.0 - tc / 4.0)
            lbl_x = ((field_amp / in_pin)[:, None] * codes[y_idx[ids]]).astype(np.float32)
            xf = torch.from_numpy(np.concatenate([xv, lbl_x], axis=1)).to(dev)
            tg_t = torch.from_numpy(zt).to(dev)

            _, zp = circ.expectation(xf, torch.from_numpy(tc).to(dev), None, th, None)
            loss = F.mse_loss(zp[:, :12], tg_t)
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                th.mul_(decay).clamp_(-clip_bound, clip_bound)

        ev_idx = np.repeat(np.arange(n_cls, dtype=np.int64), n_eval_per_cls)
        ev_orig = np.array([cls_sub[int(i)] for i in ev_idx], dtype=np.int64)
        xt_v = np.clip(rng.normal(0.0, 0.90, size=(len(ev_idx), 12)), -1.0, 1.0).astype(np.float32)
        with torch.no_grad():
            for ts in range(4, 0, -1):
                in_pin_t = circ.in_pinning_base + circ.in_pinning_slope * (1.0 - ts / 4.0)
                lbl_e = ((field_amp / in_pin_t) * codes[ev_idx]).astype(np.float32)
                chunk = 180
                out_chunks = []
                for s_i in range(0, len(ev_idx), chunk):
                    e_i = min(len(ev_idx), s_i + chunk)
                    _, zo = circ.expectation(
                        torch.from_numpy(np.concatenate([xt_v[s_i:e_i], lbl_e[s_i:e_i]], axis=1)).to(dev),
                        torch.full((e_i - s_i,), ts, dtype=torch.long, device=dev),
                        None, th, None,
                    )
                    out_chunks.append(zo[:, :12].cpu().numpy())
                x0h = np.concatenate(out_chunks, axis=0)
                if ts > 1:
                    xt_v = np.clip(math.sqrt(ab[ts - 1]) * x0h + math.sqrt(1.0 - ab[ts - 1]) * rng.normal(0.0, 0.35, size=x0h.shape), -1.0, 1.0).astype(np.float32)
                else:
                    xt_v = x0h
        rows.append(prf._eval_subset_logits(ae, clf, xt_v.astype(np.float64), cls_sub, ev_orig, dev))
    return prf._agg_dicts(rows)


def run_control1_same_circuit_exact_ad(
    mn_data: Tuple, fa_data: Optional[Tuple], dev: torch.device, seeds: List[int], quick: bool
) -> Dict[str, Any]:
    """Control 1: Same-Circuit Noiseless Exact-AD on MNIST (12+0q), Fashion (12+2q), and Label-Qubits."""
    print("\n=== [Control 1] Same-Circuit Noiseless Exact-AD (MNIST, Fashion, Label-Qubit) ===", flush=True)
    _, y_tr_m, x_te_m, y_te_m, ae_m, clf_m, z_tr_m, z_te_m = mn_data
    ev_n = 40 if quick else 200
    st_200 = 25 if quick else 200
    st_asymp = 40 if quick else 600
    bs_asymp = 12 if quick else 32

    mnist_res = {}
    for tag, n_st, bs_s, obj in [
        ("step200_mse", st_200, 10, "mse"),
        ("asymp600_mse", st_asymp, bs_asymp, "mse"),
        ("step200_surr", st_200, 10, "surrogate"),
    ]:
        ag = _train_same_circuit_exact_ad(
            z_tr_m, y_tr_m, z_te_m, x_te_m, y_te_m, ae_m, clf_m, dev,
            n_hidden=0, n_steps=n_st, batch_size=bs_s, objective=obj, seeds=seeds, eval_per_cls=ev_n,
        )
        mnist_res[tag] = ag
        print(f"  [MNIST 12+0 Exact-AD {tag:14s}] ClfAcc={ag['clf_acc']['mean']*100:5.1f}±{ag['clf_acc']['std']*100:4.1f}% | FeatFID={ag['feature_fid']['mean']:6.1f} | Prec={ag['precision']['mean']:.3f} Rec={ag['recall']['mean']:.3f}", flush=True)

    fashion_res = {}
    if fa_data is not None:
        _, y_tr_f, x_te_f, y_te_f, ae_f, clf_f, z_tr_f, z_te_f = fa_data
        for tag, n_st, bs_s, obj in [
            ("step200_mse", st_200, 10, "mse"),
            ("asymp600_mse", st_asymp, bs_asymp, "mse"),
            ("step200_surr", st_200, 10, "surrogate"),
        ]:
            ag = _train_same_circuit_exact_ad(
                z_tr_f, y_tr_f, z_te_f, x_te_f, y_te_f, ae_f, clf_f, dev,
                n_hidden=2, n_steps=n_st, batch_size=bs_s, objective=obj, seeds=seeds, eval_per_cls=ev_n,
            )
            fashion_res[tag] = ag
            print(f"  [Fashion 12+2 Exact-AD {tag:14s}] ClfAcc={ag['clf_acc']['mean']*100:5.1f}±{ag['clf_acc']['std']*100:4.1f}% | FeatFID={ag['feature_fid']['mean']:6.1f} | Prec={ag['precision']['mean']:.3f} Rec={ag['recall']['mean']:.3f}", flush=True)
    else:
        print("  [Fashion 12+2 Exact-AD] Skipped in offline --quick test (no FashionMNIST cache locally)", flush=True)

    lbl_ev = 40 if quick else 120
    lbl_res = {}
    for ltag, cls_sub, codes in [
        ("13q_01", [0, 1], np.array([[-1.0], [1.0]], dtype=np.float32)),
        ("14q_014", [0, 1, 4], np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0]], dtype=np.float32)),
    ]:
        ag_200 = _train_label_qubit_exact_ad(
            z_tr_m, y_tr_m, cls_sub, codes, ae_m, clf_m, dev,
            seeds=seeds, n_steps=(25 if quick else 200), n_eval_per_cls=lbl_ev,
        )
        ag_400 = _train_label_qubit_exact_ad(
            z_tr_m, y_tr_m, cls_sub, codes, ae_m, clf_m, dev,
            seeds=seeds, n_steps=(40 if quick else 400), n_eval_per_cls=lbl_ev,
        )
        lbl_res[ltag] = {"step200": ag_200, "step400": ag_400}
        print(
            f"  [LabelQubit {ltag:7s} Exact-AD] 200st SubAcc={ag_200['sub_acc']['mean']*100:5.1f}% (10way={ag_200['ten_way_acc']['mean']*100:4.1f}%) | "
            f"400st SubAcc={ag_400['sub_acc']['mean']*100:5.1f}% (10way={ag_400['ten_way_acc']['mean']*100:4.1f}%)",
            flush=True,
        )

    return {"mnist": mnist_res, "fashion": fashion_res, "label_qubit": lbl_res}


def run_control2_spinglass_matched_exact_ad(dev: torch.device, seeds: List[int], quick: bool) -> Dict[str, Any]:
    """Control 2: Matched Spin-Glass L=4 Exact-AD (MSE + FwdOLS + rand08 + Cosine LR)."""
    print("\n=== [Control 2] Matched Spin-Glass L=4 Exact-AD (MSE + FwdOLS + rand08 + Cosine) ===", flush=True)
    modes, valid_set, intra_pairs, c_ref = psp._build_spinglass_problem()
    edges_sg = bm.build_ising_edges((3, 4), topology="all")
    ab = bm.make_alpha_bar(4, "cosine")
    n_eval = 80 if quick else 320
    st200 = 25 if quick else 200
    st400 = 40 if quick else 400

    configs = [
        ("exact_ad_mse_fwdols_zero_const", 0.0, False, st200, 2.0, 0.08),
        ("exact_ad_mse_fwdols_rand08_cos200", 0.08, True, st200, 2.0, 0.12),
        ("exact_ad_mse_fwdols_rand08_cos400", 0.08, True, st400, 2.0, 0.12),
        ("exact_ad_mse_fwdols_rand08_cos200_b090", 0.08, True, st200, 0.90, 0.12),
    ]
    out: Dict[str, Any] = {}
    for tag, init_std, use_cos, n_st, clip_b, lr_max in configs:
        tag_rows = []
        for sd in seeds:
            torch.manual_seed(sd)
            rng_sg = np.random.default_rng(sd + 11)
            circ_sg = bm._DiffIsingCircuit(12, cond_dim=0, edges=edges_sg, n_hidden=0, n_layers=4, device=dev)
            if init_std > 0.0:
                th = (init_std * torch.randn(len(edges_sg) + 12, dtype=torch.float32, device=dev)).detach().requires_grad_(True)
            else:
                th = torch.zeros(len(edges_sg) + 12, dtype=torch.float32, device=dev, requires_grad=True)
            opt = torch.optim.Adam([th], lr=lr_max)
            cal = ReadoutGainCalibrator(4, 12, init_gain=0.75, ema=0.10)
            x_all = torch.tensor(modes, dtype=torch.float32, device=dev)
            losses = []
            for st_i in range(1, n_st + 1):
                if use_cos:
                    lr_c = _cosine_lr(st_i, n_st, lr_max, 0.015)
                    for pg in opt.param_groups:
                        pg["lr"] = lr_c
                ids = rng_sg.integers(0, len(modes), size=32)
                tc = rng_sg.integers(1, 5, size=32)
                tg = x_all[ids]
                a = torch.tensor(ab[tc], dtype=torch.float32, device=dev)[:, None]
                xt = torch.clamp(torch.sqrt(a) * tg + torch.sqrt(1.0 - a) * torch.tensor(rng_sg.normal(0.0, 0.70, size=(32, 12)), dtype=torch.float32, device=dev), -1.0, 1.0)
                _, z = circ_sg.expectation(xt, torch.tensor(tc, dtype=torch.long, device=dev), None, th, None)
                with torch.no_grad():
                    z_np, tg_np = z.detach().cpu().numpy(), tg.detach().cpu().numpy()
                    for k in range(32):
                        cal.update(int(tc[k]), z_np[k], tg_np[k])
                g_b = torch.tensor(np.stack([cal.gain[int(t)] for t in tc], axis=0), dtype=torch.float32, device=dev)
                if g_b.ndim == 1:
                    g_b = g_b[:, None]
                loss = F.mse_loss(torch.clamp(z / g_b, -1.0, 1.0), tg)
                opt.zero_grad(); loss.backward(); opt.step()
                with torch.no_grad():
                    th.clamp_(-clip_b, clip_b)
                losses.append(float(loss.item()))

            with torch.no_grad():
                xt = np.clip(rng_sg.normal(0.0, 0.90, size=(n_eval, 12)), -1.0, 1.0)
                for ts in range(4, 0, -1):
                    _, z = circ_sg.expectation(torch.tensor(xt, dtype=torch.float32, device=dev), torch.full((n_eval,), ts, dtype=torch.long, device=dev), None, th, None)
                    x0_h = cal.calibrate(ts, z.cpu().numpy())
                    xt = np.clip(math.sqrt(ab[ts - 1]) * x0_h + math.sqrt(1.0 - ab[ts - 1]) * rng_sg.normal(0.0, 0.30, x0_h.shape), -1.0, 1.0) if ts > 1 else x0_h
            met = psp._eval_spin_samples(xt, valid_set, intra_pairs, c_ref, len(modes))
            met["final_loss"] = float(np.mean(losses[-10:]))
            met["max_abs_th"] = float(torch.max(torch.abs(th)).item())
            tag_rows.append(met)
        ag = prf._agg_dicts(tag_rows)
        out[tag] = ag
        print(
            f"  [{tag:40s}] Loss={ag['final_loss']['mean']:.4f} | Corr={ag['intra_corr']['mean']:+.3f}±{ag['intra_corr']['std']:.3f} | "
            f"Acc={ag['pattern_acc']['mean']*100:5.1f}±{ag['pattern_acc']['std']*100:4.1f}% | ModeCov={ag['mode_cov']['mean']*100:5.1f}%",
            flush=True,
        )
    return out


def run_control3_spinglass_l4_reciprocity(dev: torch.device, tdev_str: str, quick: bool) -> Dict[str, Any]:
    """Control 3: Trained L=4 Spin-Glass Checkpoint Reciprocity (Clean & NISQ) + 4k-Shot SNR."""
    print("\n=== [Control 3] Trained L=4 Spin-Glass Checkpoint Reciprocity (Clean vs NISQ) & 4k-Shot SNR ===", flush=True)
    modes, valid_set, intra_pairs, c_ref = psp._build_spinglass_problem()
    edges_sg = bm.build_ising_edges((3, 4), topology="all")
    sg_l4 = QuantumIsingDenoiser(
        (3, 4), cond_dim=0, n_hidden=0, topology="all", n_layers=4,
        p_depol=0.035, p_readout=0.04, init_gain=0.75, gain_ema=0.10,
        backend="torch", torch_device=tdev_str, seed=0,
    )
    opt = QEROptimizer(sg_l4.parameters_dict(), QERConfig(clip_bound=2.0))
    n_steps = 25 if quick else 125  # 40,000 shots / (2*20*8) = 125 steps
    ckpt_steps = [0, 12, 25] if quick else [0, 30, 60, 125]
    ckpts: Dict[int, np.ndarray] = {0: sg_l4.theta.copy()}
    for st in range(1, n_steps + 1):
        prog = (st - 1) / max(1, n_steps - 1)
        ids = sg_l4.rng.choice(len(modes), size=8, replace=False)
        tb = (np.arange(8) % 4 + 1).astype(np.int64); sg_l4.rng.shuffle(tb)
        tg = modes[ids]
        a_b = sg_l4.alpha_bar[tb][:, None]
        xt = np.clip(np.sqrt(a_b) * tg + np.sqrt(1.0 - a_b) * sg_l4.rng.normal(0.0, 0.70, size=tg.shape), -1.0, 1.0)
        ob2, zb2 = sg_l4.measure_observables_batch(
            np.concatenate([xt, xt]), np.concatenate([tb, tb]), shots=20,
            nudge_target_batch=np.concatenate([tg, tg]),
            nudge_beta_batch=np.concatenate([np.full(8, opt.beta), np.full(8, -opt.beta)]),
        )
        for k_idx in range(8):
            sg_l4.calibrator.update(int(tb[k_idx]), 0.5 * (zb2[k_idx] + zb2[k_idx + 8]), tg[k_idx])
        opt.step_contrast(sg_l4.parameters_dict(), ob2[:8], ob2[8:], zb2[:8], zb2[8:], None, tg, progress=prog, shots_in_batch=320)
        if st in ckpt_steps:
            ckpts[st] = sg_l4.theta.copy()

    # Evaluate L=4 trained Spin-Glass pattern accuracy so trajectory corresponds to 80% regime
    gens_l4 = sg_l4.sample_batch(n_samples=(80 if quick else 320), shots_per_step=160, step_noise_scale=0.30)
    sg_met = psp._eval_spin_samples(gens_l4, valid_set, intra_pairs, c_ref, len(modes))
    print(f"  [L=4 Surr Ckpt Eval] IntraCorr={sg_met['intra_corr']:+.3f} | PatAcc={sg_met['pattern_acc']*100:.1f}% | ModeCov={sg_met['mode_cov']*100:.1f}%", flush=True)

    rng_p = np.random.default_rng(101)
    bs = 8 if quick else 16
    k_traj = 4 if quick else 12
    n_mc = 4 if quick else 12
    p_ids = rng_p.integers(0, len(modes), size=bs)
    p_x0 = modes[p_ids].astype(np.float32)
    p_t = ((np.arange(bs) % 4) + 1).astype(np.int64)
    p_ab = sg_l4.alpha_bar[p_t][:, None]
    p_xt = np.clip(np.sqrt(p_ab) * p_x0 + np.sqrt(1.0 - p_ab) * rng_p.normal(0.0, 0.70, size=p_x0.shape), -1.0, 1.0).astype(np.float32)

    xt_t = torch.from_numpy(p_xt).to(dev)
    x0_t = torch.from_numpy(p_x0).to(dev)
    t_t = torch.from_numpy(p_t).to(dev)
    circ_l4 = bm._DiffIsingCircuit(12, cond_dim=0, edges=edges_sg, n_hidden=0, n_layers=4, device=dev)
    fm = prb._sample_depol_masks(k_traj, 4, bs, 12, 0.035, dev, rng_p)

    rows = []
    for st in ckpt_steps:
        th_c = ckpts[st]
        cl_m = prb._eval_reciprocity_at_weights(circ_l4, th_c, None, xt_t, t_t, None, x0_t, dev, fault_masks=None, p_readout=0.0)
        nq_m = prb._eval_reciprocity_at_weights(circ_l4, th_c, None, xt_t, t_t, None, x0_t, dev, fault_masks=fm, p_readout=0.04)
        rows.append({"step": st, "max_abs_th": float(np.max(np.abs(th_c))), "clean": cl_m, "nisq": nq_m})
        print(
            f"  [SpinGlass L=4 ckpt step={st:3d} max|th|={rows[-1]['max_abs_th']:.2f}] Clean Asym={cl_m['asym_ratio']:.3f} ZZ45={cl_m['beta_045']['cos_qer_vjp_zz']:+.3f} | "
            f"NISQ Asym={nq_m['asym_ratio']:.3f} JVP/VJP={nq_m['cos_jvp_vjp_h']:+.3f} ZZ45={nq_m['beta_045']['cos_qer_vjp_zz']:+.3f}",
            flush=True,
        )

    m_snr = QuantumIsingDenoiser(12, cond_dim=0, n_hidden=0, topology="all", n_layers=4, p_depol=0.035, p_readout=0.04, init_gain=1.0, gain_ema=0.0, backend="torch", torch_device=tdev_str, seed=0)
    snr_trained = prb._measure_4k_snr_at_theta(m_snr, circ_l4, ckpts[ckpt_steps[-1]], p_xt[:8], p_x0[:8], p_t[:8], dev, n_mc, rng_p)
    print(f"  [L=4 Trained 4k-shot SNR] QER_Surr={snr_trained['qer_surr_snr']:+.3f} QER_Res={snr_trained['qer_res_mse_snr']:+.3f} ePSR={snr_trained['epsr_surr_snr']:+.3f} SPSA={snr_trained['spsa_surr_snr']:+.3f}", flush=True)
    return {"sg_eval": sg_met, "l4_reciprocity_ckpts": rows, "l4_snr_4k": snr_trained}


class _PSRSurrogateFunction(torch.autograd.Function):
    """Exact Layered Parameter-Shift Rule on the 12 visible input fields h_cond (49 queries/sample)."""

    @staticmethod
    def forward(ctx, x_t, h_cond, qpu_theta, denoiser, t_step, sh_fwd, sh_psr, gain, t_dev):
        x_np = x_t.detach().cpu().numpy().astype(np.float64)
        h_np = h_cond.detach().cpu().numpy().astype(np.float64)
        th_np = qpu_theta.detach().cpu().numpy().astype(np.float64)
        bs = x_np.shape[0]
        eye_p = np.eye(denoiser.n_visible, dtype=np.float64)
        _, z_raw = denoiser.measure_observables_batch(
            x_np, np.full(bs, t_step, dtype=np.int64), shots=sh_fwd,
            cond_batch=h_np, theta_batch=np.broadcast_to(th_np, (bs, denoiser.n_qpu_params)),
            cond_proj_batch=np.broadcast_to(eye_p, (bs, denoiser.n_visible, denoiser.n_visible)),
        )
        outs = np.clip(z_raw / gain, -1.0, 1.0)
        ctx.denoiser = denoiser
        ctx.t_step = int(t_step)
        ctx.sh_psr = int(sh_psr)
        ctx.gain = float(gain)
        ctx.t_dev = t_dev
        ctx.save_for_backward(x_t.detach(), h_cond.detach(), qpu_theta.detach())
        return torch.tensor(outs, dtype=x_t.dtype, device=x_t.device)

    @staticmethod
    def backward(ctx, grad_output):
        x_t, h_cond, qpu_theta = ctx.saved_tensors
        denoiser: QuantumIsingDenoiser = ctx.denoiser
        x_np = x_t.cpu().numpy().astype(np.float64)
        h_np = h_cond.cpu().numpy().astype(np.float64)
        th_np = qpu_theta.cpu().numpy().astype(np.float64)
        bs, nv = h_np.shape
        nl, ne = denoiser.n_layers, denoiser.n_edges
        t_norm = float(ctx.t_step) / denoiser.diffusion_steps
        in_pin = denoiser.in_pinning_base + denoiser.in_pinning_slope * (1.0 - t_norm)
        h_base = th_np[ne:][None, :] + in_pin * x_np + h_np
        n_q = 2 * nl * nv
        h_stk = np.broadcast_to(h_base[:, None, None, :], (bs, n_q, nl, nv)).copy()
        j_stk = np.broadcast_to(th_np[:ne][None, None, None, :], (bs, n_q, nl, ne)).copy()
        mold_gains = [2.0 * denoiser.dt * denoiser.temp_beta * ((l + 1) / nl) for l in range(nl)]
        ptr = 0
        meta = []
        for l_i, mg in enumerate(mold_gains):
            ang = (0.5 * math.pi) / mg
            for qi in range(nv):
                h_stk[:, ptr, l_i, qi] += ang
                h_stk[:, ptr + 1, l_i, qi] -= ang
                meta.append((qi, mg, ptr, ptr + 1))
                ptr += 2
        z_psr = bm._sample_layer_perturbed_z(
            denoiser, h_stk.reshape(bs * n_q, nl, nv), j_stk.reshape(bs * n_q, nl, ne),
            t_norm, ctx.sh_psr, ctx.t_dev,
        ).reshape(bs, n_q, nv)

        dz_dh = np.zeros((bs, nv, nv), dtype=np.float64)  # [b, z_out_k, h_in_q]
        for qi, mg, p_i, m_i in meta:
            dz_dh[:, :, qi] += 0.5 * mg * (z_psr[:, p_i, :] - z_psr[:, m_i, :])

        g_np = grad_output.detach().cpu().numpy().astype(np.float64)
        grad_h = np.einsum("bk,bkq->bq", g_np / ctx.gain, dz_dh)
        grad_h_t = torch.tensor(grad_h, dtype=h_cond.dtype, device=h_cond.device)
        return grad_h_t * in_pin, grad_h_t, None, None, None, None, None, None, None


class HybridResMLPPSR(prf._HybridResBase):
    """Identical ResMLP trunk + 12-qubit NISQ QPU trained via Exact Layered PSR on h_cond fields."""

    def __init__(self, denoiser: QuantumIsingDenoiser, hidden: int = 48, sh_fwd: int = 24, sh_psr: int = 4, gain: float = 0.70, dev: torch.device = torch.device("cpu")):
        super().__init__(hidden=hidden, diffusion_steps=denoiser.diffusion_steps, gain=gain)
        self.denoiser = denoiser
        self.alpha_bar = denoiser.alpha_bar
        self.sh_fwd = int(sh_fwd)
        self.sh_psr = int(sh_psr)
        self.t_dev = dev
        self.register_buffer("qpu_theta", torch.tensor(denoiser.theta.astype(np.float32), device=dev))

    def forward_t(self, x_t: torch.Tensor, t_step: int, cond: torch.Tensor) -> torch.Tensor:
        h_cond = self._trunk(x_t, t_step, cond)
        return _PSRSurrogateFunction.apply(x_t, h_cond, self.qpu_theta, self.denoiser, t_step, self.sh_fwd, self.sh_psr, self.gain, self.t_dev)


def _verify_psr_grad_unit(dev: torch.device) -> float:
    """Quick unit check confirming _PSRSurrogateFunction backward matches Exact-AD grad_h."""
    edges = bm.build_ising_edges(12, "all")
    circ = bm._DiffIsingCircuit(12, cond_dim=0, edges=edges, n_layers=2, device=dev)
    den = QuantumIsingDenoiser(12, cond_dim=0, n_hidden=0, topology="all", n_layers=2, p_depol=0.0, p_readout=0.0, backend="torch", torch_device=str(dev), seed=0)
    rng = np.random.default_rng(0)
    den.theta[:] = rng.normal(0.0, 0.25, size=den.theta.shape)
    xt = torch.randn(4, 12, device=dev) * 0.3
    h_ad = (torch.randn(4, 12, device=dev) * 0.4).detach().requires_grad_(True)
    th = torch.tensor(den.theta, dtype=torch.float32, device=dev)
    g_out = torch.randn(4, 12, device=dev)
    _, z_ad = circ.expectation(xt, torch.full((4,), 2, dtype=torch.long, device=dev), h_ad, th, torch.eye(12, device=dev))
    (gh_ad,) = torch.autograd.grad(((z_ad / 0.70) * g_out).sum(), [h_ad])

    h_psr = h_ad.detach().clone().requires_grad_(True)
    y_psr = _PSRSurrogateFunction.apply(xt, h_psr, th, den, 2, 256, 256, 0.70, dev)
    (gh_psr,) = torch.autograd.grad((y_psr * g_out).sum(), [h_psr])
    cos_v = prb._cos(gh_psr.detach().cpu().numpy().ravel(), gh_ad.detach().cpu().numpy().ravel())
    print(f"  [UnitCheck PSR-vs-ExactAD grad_h] cosine={cos_v:+.4f}", flush=True)
    return float(cos_v)


def run_control4_hybrid_psr_fields(
    z_tr: np.ndarray, y_tr: np.ndarray, z_te: np.ndarray, x_te: np.ndarray, y_te: np.ndarray,
    ae: nn.Module, clf: bm.EvalCNN, dev: torch.device, tdev_str: str, seeds: List[int], quick: bool,
) -> Dict[str, Any]:
    """Control 4: Equal-Shot (1.38M shots) PSR-on-Input-Fields Hybrid Bridge (h=48) vs HybridQER."""
    print("\n=== [Control 4] Equal-Shot PSR-on-Input-Fields Hybrid Control (h=48, <=1.38M Shots) ===", flush=True)
    grad_cos = _verify_psr_grad_unit(dev)

    eye_c = np.eye(10, dtype=np.float64)
    c_tr = eye_c[y_tr]
    ev_lbl = np.repeat(np.arange(10, dtype=np.int64), 20 if quick else 200)
    ev_c = eye_c[ev_lbl]
    pools = [np.flatnonzero(y_tr == c) for c in range(10)]
    cached_real = {sd: pcd.precompute_real_knn_ref(clf, x_te, dev, seed=sd) for sd in seeds}

    bs_hyb = 8 if quick else 32
    if quick:
        scheds = [
            ("hybrid_psr_24sh_4st", 24, 24, 4),
            ("hybrid_psr_4sh_12st", 24, 4, 12),
        ]
    else:
        # HybridQER budget: 300 steps * 2 calls * 32 bs * 72 sh/sample = 1,382,400 total shots
        # PSR-4sh-100st:    100 steps * 2 calls * 32 bs * (24 + 48*4 = 216) sh = 1,382,400 shots exact
        # PSR-24sh-18st:     18 steps * 2 calls * 32 bs * (24 + 48*24 = 1176) sh = 1,354,752 <= 1,382,400 shots
        scheds = [
            ("hybrid_psr_24sh_18st", 24, 24, 18),
            ("hybrid_psr_4sh_100st", 24, 4, 100),
        ]

    out: Dict[str, Any] = {"psr_grad_h_unit_cos": grad_cos, "runs": {}}
    for tag, sh_f, sh_p, n_st in scheds:
        per_sn: Dict[str, List[Dict[str, float]]] = {"0.35": [], "0.85": []}
        eff_sh_p = max(4, (sh_p // 4) * 4)
        shots_used = n_st * 2 * bs_hyb * (sh_f + 48 * eff_sh_p)
        for sd in seeds:
            torch.manual_seed(sd)
            rng_s = np.random.default_rng(sd)
            qpu_den = QuantumIsingDenoiser(
                12, cond_dim=12, n_hidden=0, topology="all", n_layers=2,
                p_depol=0.035, p_readout=0.04, init_gain=0.70, gain_ema=0.0,
                backend="torch", torch_device=tdev_str, seed=sd,
            )
            model = HybridResMLPPSR(qpu_den, hidden=48, sh_fwd=sh_f, sh_psr=sh_p, gain=0.70, dev=dev).to(dev)
            opt = torch.optim.Adam(model.parameters(), lr=3.5e-3)
            ab = model.alpha_bar
            model.train()
            for st in range(1, n_st + 1):
                lr_k = 0.08 * 3.5e-3 + 0.92 * 3.5e-3 * 0.5 * (1.0 + math.cos(math.pi * (st - 1) / max(1, n_st - 1)))
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
            for sn in (0.35, 0.85):
                # Evaluate cleanly with same 24-shot NISQ QPU inference as HybridQER
                zg = model.sample_batch(ev_c, dev, seed=100 + sd, step_noise_scale=sn)
                met, _ = bm.evaluate_generated(f"{tag}-sn{int(sn*100)}", zg, ev_lbl, ae, clf, z_te, x_te, y_te, dev)
                met.update(pcd.measure_knn_radius_geometry(ae, clf, zg, x_te, dev, seed=sd, cached_real=cached_real[sd]))
                per_sn[str(sn)].append(met)

        out["runs"][tag] = {
            "total_shots": int(shots_used),
            "steps": int(n_st),
            "sh_fwd": int(sh_f),
            "sh_psr": int(eff_sh_p),
            "0.35": prf._agg_dicts(per_sn["0.35"]),
            "0.85": prf._agg_dicts(per_sn["0.85"]),
        }
        r35, r85 = out["runs"][tag]["0.35"], out["runs"][tag]["0.85"]
        print(
            f"  [{tag:22s} shots={shots_used:7d}] sn=0.35 Acc={r35['clf_acc']['mean']*100:5.1f}% FID={r35['feature_fid']['mean']:6.1f} Rec={r35['recall']['mean']:.3f} | "
            f"sn=0.85 Acc={r85['clf_acc']['mean']*100:5.1f}% FID={r85['feature_fid']['mean']:6.1f} Rec={r85['recall']['mean']:.3f}",
            flush=True,
        )
    return out


def make_figure(res: Dict[str, Any], out_dir: str) -> None:
    """Plot 4-panel summary covering Same-Circuit Exact-AD, L=4 Spin Glass, and Hybrid PSR controls."""
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8.6, "axes.titlesize": 9.6,
        "axes.titleweight": "bold", "axes.labelsize": 8.6, "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.6, "legend.fontsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.8))

    # (a) Control 1: Same-Circuit Exact-AD MNIST & Fashion vs Adam-QER reference points
    ax = axes[0, 0]
    c1 = res["control1"]
    labels, vals, errs, cols = ["MNIST\nQER (80k)", "MNIST AD\n200st", "MNIST AD\n600st", "MNIST AD\nSurr200"], [100.0], [0.0], ["#0072B2"]
    for k in ("step200_mse", "asymp600_mse", "step200_surr"):
        vals.append(c1["mnist"][k]["clf_acc"]["mean"] * 100.0)
        errs.append(c1["mnist"][k]["clf_acc"]["std"] * 100.0)
        cols.append("#56B4E9")
    if c1["fashion"]:
        labels.extend(["Fashion\nQER (80k)", "Fashion AD\n200st", "Fashion AD\n600st", "Fashion AD\nSurr200"])
        vals.append(71.8); errs.append(3.2); cols.append("#009E73")
        for k in ("step200_mse", "asymp600_mse", "step200_surr"):
            vals.append(c1["fashion"][k]["clf_acc"]["mean"] * 100.0)
            errs.append(c1["fashion"][k]["clf_acc"]["std"] * 100.0)
            cols.append("#E69F00")
    xp = np.arange(len(labels))
    ax.bar(xp, vals, yerr=errs, capsize=3.0, color=cols, edgecolor="#222222", lw=0.7)
    ax.set_xticks(xp); ax.set_xticklabels(labels, fontsize=7.6, rotation=18, ha="right")
    ax.set_ylabel("ClfAcc (%) $\\uparrow$"); ax.set_ylim(0, 114)
    ax.set_title("(a) Same-Circuit Noiseless Exact-AD vs. NISQ Adam-QER")
    ax.grid(axis="y", ls="--", alpha=0.25)

    # (b) Control 2: Matched Spin-Glass L=4 Exact-AD (MSE + FwdOLS)
    ax = axes[0, 1]
    c2 = res["control2"]
    skeys = list(c2.keys())
    xp2 = np.arange(len(skeys)); w = 0.27
    ax.bar(xp2 - w, [c2[k]["intra_corr"]["mean"] * 100.0 for k in skeys], width=w, color="#0072B2", edgecolor="#222", lw=0.6, label="IntraCorr x100")
    ax.bar(xp2, [c2[k]["pattern_acc"]["mean"] * 100.0 for k in skeys], width=w, color="#009E73", edgecolor="#222", lw=0.6, label="PatternAcc %")
    ax.bar(xp2 + w, [c2[k]["mode_cov"]["mean"] * 100.0 for k in skeys], width=w, color="#E69F00", edgecolor="#222", lw=0.6, label="ModeCov %")
    ax.set_xticks(xp2)
    ax.set_xticklabels(["Zero/Const\n($B_c=2.0$)", "Rand08/Cos200\n($B_c=2.0$)", "Rand08/Cos400\n($B_c=2.0$)", "Rand08/Cos200\n($B_c=0.9$)"], fontsize=7.6)
    ax.set_ylabel("Metric / Score (%) $\\uparrow$"); ax.set_ylim(0, 116)
    ax.set_title("(b) Matched $L=4$ Spin-Glass Exact-AD (MSE + FwdOLS)")
    ax.grid(axis="y", ls="--", alpha=0.25); ax.legend(frameon=False, loc="upper left")

    # (c) Control 3: Trained L=4 Spin-Glass Reciprocity (Clean vs NISQ)
    ax = axes[1, 0]
    ck = res["control3"]["l4_reciprocity_ckpts"]
    xs = [r["step"] for r in ck]
    ax.plot(xs, [r["clean"]["beta_045"]["cos_qer_vjp_zz"] for r in ck], "o-", color="#0072B2", lw=2.0, label="Clean $\\cos(\\mathrm{QER}, \\mathrm{VJP})_{ZZ}$")
    ax.plot(xs, [r["nisq"]["beta_045"]["cos_qer_vjp_zz"] for r in ck], "s--", color="#D55E00", lw=2.0, label="NISQ (3.5%/4%) $ZZ$")
    ax.plot(xs, [r["nisq"]["asym_ratio"] for r in ck], "^:", color="#009E73", lw=1.8, label="NISQ Asym. Ratio")
    ax.set_xlabel("Spin-Glass $L=4$ Training Step $t$"); ax.set_ylabel("Cosine / Asymmetry Ratio")
    ax.set_title("(c) Trained $L=4$ Spin-Glass Reciprocity (Clean vs. NISQ)")
    ax.set_ylim(-0.05, 1.08); ax.grid(True, ls="--", alpha=0.25)
    ax.legend(frameon=False, loc="best")

    # (d) Control 4: Equal-Shot Hybrid PSR-on-Input-Fields vs HybridQER (h=48)
    ax = axes[1, 1]
    runs = res["control4"]["runs"]
    h_names = ["QERLayer\n($\\sigma{=}0.35$)", "QERLayer\n($\\sigma{=}0.85$)"]
    h_acc = [95.5, 89.7]
    h_rec = [32.6, 42.8]
    h_cols = ["#0072B2", "#0072B2"]
    for t_k in runs:
        for sn_s in ("0.35", "0.85"):
            short = t_k.replace("hybrid_psr_", "PSR ").replace("_", "/")
            h_names.append(f"{short}\n($\\sigma{{=}}{sn_s}$)")
            h_acc.append(runs[t_k][sn_s]["clf_acc"]["mean"] * 100.0)
            h_rec.append(runs[t_k][sn_s]["recall"]["mean"] * 100.0)
            h_cols.append("#CC79A7")
    xp4 = np.arange(len(h_names))
    ax.bar(xp4 - 0.18, h_acc, width=0.36, color=h_cols, edgecolor="#222", lw=0.6, label="ClfAcc (%) $\\uparrow$")
    ax.bar(xp4 + 0.18, h_rec, width=0.36, color="#E69F00", edgecolor="#222", lw=0.6, label="Recall x100 $\\uparrow$")
    ax.set_xticks(xp4); ax.set_xticklabels(h_names, fontsize=7.3, rotation=18, ha="right")
    ax.set_ylabel("Percent / Score $\\uparrow$"); ax.set_ylim(0, 114)
    ax.set_title("(d) Equal-Shot Hybrid QERLayer vs. PSR-on-Fields ($h=48$)")
    ax.grid(axis="y", ls="--", alpha=0.25); ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_matched_controls.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig_matched_controls.png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="./data")
    ap.add_argument("--out-dir", type=str, default="./analysis")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--replot", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    json_p = os.path.join(args.out_dir, "matched_controls_results.json")
    if args.replot:
        with open(json_p, "r", encoding="utf-8") as f:
            res = json.load(f)
        make_figure(res, args.out_dir)
        print(f"[Replotted] {json_p} -> fig_matched_controls.pdf", flush=True)
        return

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tdev_str = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [0] if args.quick else [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    print(f"[Setup] Matched Controls | Device={dev} | Seeds={seeds} | quick={args.quick}", flush=True)

    mn_data = _load_or_cache_helpers("mnist", args.data_dir, args.out_dir, dev, allow_download=True)
    assert mn_data is not None
    fa_data = _load_or_cache_helpers("fashion", args.data_dir, args.out_dir, dev, allow_download=not args.quick)

    t0 = time.time()
    c1 = run_control1_same_circuit_exact_ad(mn_data, fa_data, dev, seeds, args.quick)
    c2 = run_control2_spinglass_matched_exact_ad(dev, seeds, args.quick)
    c3 = run_control3_spinglass_l4_reciprocity(dev, tdev_str, args.quick)
    c4 = run_control4_hybrid_psr_fields(mn_data[6], mn_data[1], mn_data[7], mn_data[2], mn_data[3], mn_data[4], mn_data[5], dev, tdev_str, seeds, args.quick)

    res = {
        "device": str(dev),
        "seeds": seeds,
        "quick": args.quick,
        "wall_sec": round(time.time() - t0, 1),
        "control1": c1,
        "control2": c2,
        "control3": c3,
        "control4": c4,
    }
    with open(json_p, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    make_figure(res, args.out_dir)
    print(f"[Done] Saved {json_p} & fig_matched_controls.pdf in {res['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
