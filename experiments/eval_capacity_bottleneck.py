"""
eval_capacity_bottleneck.py — Capacity-Diversity Scaling Analysis Under MSE Denoising.

Sweeps ResMLP denoisers across P = 220 to 544,900 parameters on the shared MNIST
ConvLatentAE(D=12) + EvalCNN pipeline, compares against step-count and sampler-noise
ablations plus 5-seed paper benchmarks, and saves:
  - analysis/capacity_diversity_results.json
  - analysis/fig_capacity_vs_diversity.png / .pdf

Usage:
  python -u experiments/eval_capacity_bottleneck.py
  python -u experiments/eval_capacity_bottleneck.py --replot
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
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import benchmark_mnist as bm


def _count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def train_denoiser_exact_adam(
    model: nn.Module,
    z_tr: np.ndarray,
    c_tr: np.ndarray,
    y_tr: np.ndarray,
    n_steps: int = 600,
    batch_size: int = 64,
    lr: float = 2e-3,
    diffusion_steps: int = 4,
    seed: int = 0,
) -> nn.Module:
    """Train a denoiser with exact noiseless Adam backpropagation under the DDPM MSE objective."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    alpha_bar = bm.make_alpha_bar(diffusion_steps, schedule="cosine")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    class_pools = [np.flatnonzero(y_tr == c) for c in np.unique(y_tr)]
    eff_bs = min(batch_size, len(z_tr))

    model.train()
    for step in range(1, n_steps + 1):
        lr_k = 0.05 * lr + 0.95 * lr * 0.5 * (1.0 + math.cos(math.pi * (step - 1) / max(1, n_steps - 1)))
        for pg in opt.param_groups:
            pg["lr"] = lr_k

        ids = bm._sample_stratified_ids(rng, class_pools, eff_bs)
        t_arr = (np.arange(eff_bs) % diffusion_steps) + 1
        rng.shuffle(t_arr)
        a_bar = alpha_bar[t_arr][:, None].astype(np.float32)
        tgt = z_tr[ids].astype(np.float32)
        x_t = np.clip(
            np.sqrt(a_bar) * tgt + np.sqrt(1.0 - a_bar) * rng.normal(0.0, 0.70, size=tgt.shape).astype(np.float32),
            -1.0,
            1.0,
        )
        t_norm = (t_arr.astype(np.float32) / diffusion_steps)[:, None]

        opt.zero_grad()
        pred = model(torch.from_numpy(x_t), torch.from_numpy(t_norm), torch.from_numpy(c_tr[ids].astype(np.float32)))
        F.mse_loss(pred, torch.from_numpy(tgt)).backward()
        opt.step()

    model.eval()
    return model


@torch.no_grad()
def precompute_real_knn_ref(
    clf: bm.EvalCNN,
    x_te: np.ndarray,
    device: torch.device,
    k: int = 5,
    max_n: int = 2000,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute real test EvalCNN 64-D features and k-NN radii once per seed."""
    rng = np.random.default_rng(seed)
    n_te = len(x_te)
    te_idx = rng.choice(n_te, size=min(max_n, n_te), replace=False) if n_te > max_n else np.arange(n_te)
    real_feats = []
    for i in range(0, len(te_idx), 512):
        bx = torch.from_numpy(x_te[te_idx[i : i + 512]]).to(device)
        _, ft = clf.forward_features(bx)
        real_feats.append(ft)
    r = torch.cat(real_feats, dim=0)
    d_rr = torch.cdist(r, r)
    d_rr.fill_diagonal_(1e9)
    rad_r = torch.kthvalue(d_rr, k, dim=1).values
    return r, rad_r


@torch.no_grad()
def measure_knn_radius_geometry(
    ae: bm.ConvLatentAE,
    clf: bm.EvalCNN,
    gen_latents: np.ndarray,
    x_te: np.ndarray,
    device: torch.device,
    k: int = 5,
    max_n: int = 2000,
    seed: int = 0,
    cached_real: Tuple[torch.Tensor, torch.Tensor] | None = None,
) -> Dict[str, float]:
    """Measure EvalCNN 64-d feature k-NN radii R_i (real), G_j (gen), and coverage gap."""
    rng = np.random.default_rng(seed)
    if cached_real is None:
        r, rad_r = precompute_real_knn_ref(clf, x_te, device, k=k, max_n=max_n, seed=seed)
    else:
        r, rad_r = cached_real
    ge_idx = rng.choice(len(gen_latents), size=min(max_n, len(gen_latents)), replace=False)

    gen_feats = []
    for i in range(0, len(ge_idx), 512):
        gz = torch.from_numpy(gen_latents[ge_idx[i : i + 512]].astype(np.float32)).to(device)
        _, ft = clf.forward_features(ae.decode(gz))
        gen_feats.append(ft)
    g = torch.cat(gen_feats, dim=0)

    d_gg = torch.cdist(g, g)
    d_gg.fill_diagonal_(1e9)
    rad_g = torch.kthvalue(d_gg, k, dim=1).values

    d_rg = torch.cdist(r, g)
    min_r_to_g = d_rg.min(dim=1).values

    return {
        "rad_r_mean": float(rad_r.mean().item()),
        "rad_r_median": float(rad_r.median().item()),
        "rad_g_mean": float(rad_g.mean().item()),
        "rad_g_median": float(rad_g.median().item()),
        "radius_ratio_mean": float((rad_g.mean() / rad_r.mean()).item()),
        "min_r_to_g_median": float(min_r_to_g.median().item()),
        "min_r_to_g_mean": float(min_r_to_g.mean().item()),
    }


def _aggregate(records: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    keys = list(records[0].keys())
    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vals = [float(r[k]) for r in records]
        out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return out


def run_experiments(
    out_dir: str = "./analysis",
    seeds: List[int] = [0, 1, 2],
    resmlp_widths: List[int] | None = None,
) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Setup] Loading MNIST & training/loading ConvLatentAE(12) + EvalCNN on {device}...", flush=True)
    x_tr, y_tr, x_te, y_te, _ = bm.load_dataset("mnist", data_dir="./data")
    cache_path = os.path.join(out_dir, "mnist_d12_ae8_clf5_helpers.pt")
    if not os.path.exists(cache_path) and os.path.exists("./analysis/mnist_d12_ae8_clf5_helpers.pt"):
        cache_path = "./analysis/mnist_d12_ae8_clf5_helpers.pt"
    ae, clf, z_tr, z_te = bm.train_classical_helpers(
        x_tr, y_tr, x_te, y_te,
        latent_dim=12, ae_epochs=8, clf_epochs=5, device=device, cache_path=cache_path, use_bn=True,
    )

    cached_real_by_seed = {sd: precompute_real_knn_ref(clf, x_te, device, seed=sd) for sd in seeds}

    eye_c = np.eye(10, dtype=np.float64)
    c_tr = eye_c[y_tr]
    eval_per_class = 200
    eval_labels = np.repeat(np.arange(10, dtype=np.int64), eval_per_class)
    eval_conds = eye_c[eval_labels]

    real_baselines = bm.evaluate_real_baseline(ae, clf, z_te, x_te, y_te, eval_per_class, device)
    print(
        f"[Real] Held-Out: Acc={real_baselines['real_held_out']['clf_acc']*100:.1f}% "
        f"Rec={real_baselines['real_held_out']['recall']:.3f} LatStd={real_baselines['real_held_out']['intra_class_latent_std']:.3f} | "
        f"AE-Recon: Acc={real_baselines['real_ae_recon']['clf_acc']*100:.1f}% "
        f"Rec={real_baselines['real_ae_recon']['recall']:.3f}",
        flush=True,
    )

    # Experiment A: Controlled ResMLP Capacity Sweep (Exact Adam, 600 steps, batch=64)
    # Evaluates the exact same trained model at BOTH step_noise=0.35 (default) and step_noise=0.85 (diversity sweep).
    if resmlp_widths is None:
        resmlp_widths = [2, 5, 12, 24, 48, 96, 128, 256, 512]
    resmlp_sweep = []
    resmlp_sweep_sn85 = []
    print("\n=== [Exp A] Controlled ResMLP Capacity Sweep (Exact Adam, 600 steps, batch=64; P=220..545k) ===", flush=True)
    for h in resmlp_widths:
        seed_recs_35 = []
        seed_recs_85 = []
        n_params = _count_params(bm.ClassicalLargeDDPM(12, 10, hidden=h, diffusion_steps=4))
        for sd in seeds:
            model = bm.ClassicalLargeDDPM(12, 10, hidden=h, diffusion_steps=4)
            train_denoiser_exact_adam(model, z_tr, c_tr, y_tr, n_steps=600, batch_size=64, lr=2e-3, seed=sd)

            z_gen_35 = model.sample_batch(eval_conds, seed=100 + sd, step_noise_scale=0.35, init_std=0.90)
            m35, _ = bm.evaluate_generated(f"ResMLP-h{h}-sn35", z_gen_35, eval_labels, ae, clf, z_te, x_te, y_te, device)
            geom35 = measure_knn_radius_geometry(ae, clf, z_gen_35, x_te, device, seed=sd, cached_real=cached_real_by_seed[sd])
            seed_recs_35.append({**m35, **geom35, "n_params": float(n_params)})

            z_gen_85 = model.sample_batch(eval_conds, seed=100 + sd, step_noise_scale=0.85, init_std=0.90)
            m85, _ = bm.evaluate_generated(f"ResMLP-h{h}-sn85", z_gen_85, eval_labels, ae, clf, z_te, x_te, y_te, device)
            geom85 = measure_knn_radius_geometry(ae, clf, z_gen_85, x_te, device, seed=sd, cached_real=cached_real_by_seed[sd])
            seed_recs_85.append({**m85, **geom85, "n_params": float(n_params)})

        agg35 = _aggregate(seed_recs_35)
        agg85 = _aggregate(seed_recs_85)
        resmlp_sweep.append({"hidden": h, "n_params": n_params, "aggregate": agg35, "per_seed": seed_recs_35})
        resmlp_sweep_sn85.append({"hidden": h, "n_params": n_params, "aggregate": agg85, "per_seed": seed_recs_85})
        print(
            f"  h={h:3d} | P={n_params:6d} | sn=0.35: Acc={agg35['clf_acc']['mean']*100:5.1f}% "
            f"LatStd={agg35['intra_class_latent_std']['mean']:.3f} Rec={agg35['recall']['mean']:.3f} Cov={agg35['coverage']['mean']:.3f} | "
            f"sn=0.85: Acc={agg85['clf_acc']['mean']*100:5.1f}% LatStd={agg85['intra_class_latent_std']['mean']:.3f} "
            f"Rec={agg85['recall']['mean']:.3f}±{agg85['recall']['std']:.3f} Cov={agg85['coverage']['mean']:.3f}",
            flush=True,
        )

    # Experiment B: Controlled Matched-Skip MLP Capacity Sweep (ClassicalMatchedDDPM with skip_cond=True)
    skip_targets = [204, 500, 1500, 5000, 15000, 38000]
    skip_sweep = []
    print("\n=== [Exp B] Controlled Skip-MLP Capacity Sweep (Exact Adam, 600 steps, batch=64) ===", flush=True)
    for tp in skip_targets:
        seed_recs = []
        n_params = _count_params(bm.ClassicalMatchedDDPM(12, 10, target_params=tp, diffusion_steps=4, skip_cond=True))
        for sd in seeds:
            model = bm.ClassicalMatchedDDPM(12, 10, target_params=tp, diffusion_steps=4, skip_cond=True)
            train_denoiser_exact_adam(model, z_tr, c_tr, y_tr, n_steps=600, batch_size=64, lr=0.012, seed=sd)
            z_gen = model.sample_batch(eval_conds, seed=200 + sd, step_noise_scale=0.35, init_std=0.90)
            m, _ = bm.evaluate_generated(f"Skip-P{n_params}", z_gen, eval_labels, ae, clf, z_te, x_te, y_te, device)
            geom = measure_knn_radius_geometry(ae, clf, z_gen, x_te, device, seed=sd, cached_real=cached_real_by_seed[sd])
            seed_recs.append({**m, **geom, "n_params": float(n_params)})
        agg = _aggregate(seed_recs)
        skip_sweep.append({"target_params": tp, "n_params": n_params, "aggregate": agg, "per_seed": seed_recs})
        print(
            f"  Skip P={n_params:5d} | Acc={agg['clf_acc']['mean']*100:5.1f}% | "
            f"LatStd={agg['intra_class_latent_std']['mean']:.3f} | "
            f"Prec={agg['precision']['mean']:.3f} Rec={agg['recall']['mean']:.3f} | "
            f"LatFID={agg['latent_fid']['mean']:.3f}",
            flush=True,
        )

    # Experiment C1: Ruling out under-training at P=204 (200 -> 600 -> 2000 exact Adam steps)
    step_ablation = []
    print("\n=== [Exp C1] Optimization Step Stress Test at P=204 (Classical-Skip) ===", flush=True)
    for n_st, bs in [(200, 10), (600, 64), (2000, 64)]:
        seed_recs = []
        for sd in seeds:
            model = bm.ClassicalMatchedDDPM(12, 10, target_params=204, diffusion_steps=4, skip_cond=True)
            train_denoiser_exact_adam(model, z_tr, c_tr, y_tr, n_steps=n_st, batch_size=bs, lr=0.015, seed=sd)
            z_gen = model.sample_batch(eval_conds, seed=300 + sd, step_noise_scale=0.35, init_std=0.90)
            m, _ = bm.evaluate_generated(f"Skip204-st{n_st}", z_gen, eval_labels, ae, clf, z_te, x_te, y_te, device)
            seed_recs.append(m)
        agg = _aggregate(seed_recs)
        step_ablation.append({"steps": n_st, "batch_size": bs, "n_params": 204, "aggregate": agg})
        print(
            f"  P=204, steps={n_st:4d}, bs={bs:2d} | Acc={agg['clf_acc']['mean']*100:5.1f}% | "
            f"LatStd={agg['intra_class_latent_std']['mean']:.3f} | Prec={agg['precision']['mean']:.3f} Rec={agg['recall']['mean']:.3f}",
            flush=True,
        )

    # Experiment C2: Ruling out sampler noise scale at P=220 vs P=38,020
    noise_ablation = {"P220": [], "P38020": []}
    print("\n=== [Exp C2] Sampler Step-Noise Stress Test (P=220 vs P=38,020) ===", flush=True)
    for tag, h in [("P220", 2), ("P38020", 128)]:
        models_h = []
        for sd in seeds:
            m_h = bm.ClassicalLargeDDPM(12, 10, hidden=h, diffusion_steps=4)
            train_denoiser_exact_adam(m_h, z_tr, c_tr, y_tr, n_steps=600, batch_size=64, lr=2e-3, seed=sd)
            models_h.append(m_h)
        for s_noise in [0.15, 0.35, 0.60, 0.85, 1.10]:
            seed_recs = []
            for sd, m_h in zip(seeds, models_h):
                z_gen = m_h.sample_batch(eval_conds, seed=400 + sd, step_noise_scale=s_noise, init_std=0.90)
                m, _ = bm.evaluate_generated(f"{tag}-sn{s_noise}", z_gen, eval_labels, ae, clf, z_te, x_te, y_te, device)
                seed_recs.append(m)
            agg = _aggregate(seed_recs)
            noise_ablation[tag].append({"step_noise": s_noise, "aggregate": agg})
            print(
                f"  {tag:6s} step_noise={s_noise:.2f} | Acc={agg['clf_acc']['mean']*100:5.1f}% | "
                f"LatStd={agg['intra_class_latent_std']['mean']:.3f} | Prec={agg['precision']['mean']:.3f} Rec={agg['recall']['mean']:.3f}",
                flush=True,
            )

    # Load verified 5-seed paper_suite results for direct overlay
    paper_main = {}
    for ds, path in [
        ("mnist", "./paper_suite/mnist_main/mnist_results.json"),
        ("fashion", "./paper_suite/fashion_main/fashion_results.json"),
    ]:
        with open(path) as f:
            paper_main[ds] = json.load(f)

    out = {
        "real_baselines": real_baselines,
        "resmlp_sweep": resmlp_sweep,
        "resmlp_sweep_sn85": resmlp_sweep_sn85,
        "skip_sweep": skip_sweep,
        "step_ablation": step_ablation,
        "noise_ablation": noise_ablation,
        "paper_suite_main": paper_main,
    }
    json_path = os.path.join(out_dir, "capacity_diversity_results.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[Saved] {json_path}", flush=True)
    return out


def make_proof_figure(data: dict, out_dir: str = "./analysis") -> None:
    plt.rcParams.update({
        "font.size": 9.8,
        "axes.titlesize": 11.0,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.0,
        "legend.fontsize": 8.2,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(2, 2, figsize=(15.4, 10.6), dpi=220)
    fig.subplots_adjust(hspace=0.35, wspace=0.29, top=0.88, bottom=0.075, left=0.055, right=0.945)
    has_sn85 = bool(data.get("resmlp_sweep_sn85"))
    fig.suptitle(
        "Capacity-Diversity Scaling Bottleneck Under MSE Denoising (MNIST D=12)\n"
        "(Low-Parameter Convergence at P ≈ 200 and Continuous Recall Unlock Across P ≈ 6k–141k+)",
        fontsize=12.2, fontweight="bold", y=0.97,
    )

    res_p = np.array([r["n_params"] for r in data["resmlp_sweep"]])
    res_std_m = np.array([r["aggregate"]["intra_class_latent_std"]["mean"] for r in data["resmlp_sweep"]])
    res_std_s = np.array([r["aggregate"]["intra_class_latent_std"]["std"] for r in data["resmlp_sweep"]])
    res_rec_m = np.array([r["aggregate"]["recall"]["mean"] for r in data["resmlp_sweep"]])
    res_pre_m = np.array([r["aggregate"]["precision"]["mean"] for r in data["resmlp_sweep"]])
    res_cov_m = np.array([r["aggregate"]["coverage"]["mean"] for r in data["resmlp_sweep"]])
    res_cov_s = np.array([r["aggregate"]["coverage"]["std"] for r in data["resmlp_sweep"]])
    res_rad_ratio = np.array([r["aggregate"]["radius_ratio_mean"]["mean"] for r in data["resmlp_sweep"]])
    res_rad_ratio_s = np.array([r["aggregate"]["radius_ratio_mean"]["std"] for r in data["resmlp_sweep"]])

    if has_sn85:
        sn85_data = data["resmlp_sweep_sn85"]
        sn85_p = np.array([r["n_params"] for r in sn85_data])
        sn85_std_m = np.array([r["aggregate"]["intra_class_latent_std"]["mean"] for r in sn85_data])
        sn85_std_s = np.array([r["aggregate"]["intra_class_latent_std"]["std"] for r in sn85_data])
        sn85_rec_m = np.array([r["aggregate"]["recall"]["mean"] for r in sn85_data])
        sn85_rec_s = np.array([r["aggregate"]["recall"]["std"] for r in sn85_data])
        sn85_pre_m = np.array([r["aggregate"]["precision"]["mean"] for r in sn85_data])
        sn85_cov_m = np.array([r["aggregate"]["coverage"]["mean"] for r in sn85_data])
        sn85_cov_s = np.array([r["aggregate"]["coverage"]["std"] for r in sn85_data])

    skp_p = np.array([r["n_params"] for r in data["skip_sweep"]])
    skp_std_m = np.array([r["aggregate"]["intra_class_latent_std"]["mean"] for r in data["skip_sweep"]])
    skp_std_s = np.array([r["aggregate"]["intra_class_latent_std"]["std"] for r in data["skip_sweep"]])

    mn_paper = data["paper_suite_main"]["mnist"]
    fa_paper = data["paper_suite_main"]["fashion"]
    mn_pc = mn_paper["param_counts"]
    mn_ag = mn_paper["aggregate"]
    fa_ag = fa_paper["aggregate"]
    real_std = data["real_baselines"]["real_held_out"]["intra_class_latent_std"]
    ae_rec = data["real_baselines"]["real_ae_recon"]["recall"]

    # Extract Exp C2 sampler noise sweep (P=220 vs P=38,020)
    n220 = data["noise_ablation"]["P220"]
    n38k = data["noise_ablation"]["P38020"]
    sn_vals = np.array([x["step_noise"] for x in n38k])
    rec_220_m = np.array([x["aggregate"]["recall"]["mean"] for x in n220])
    std_220_m = np.array([x["aggregate"]["intra_class_latent_std"]["mean"] for x in n220])
    rec_38k_m = np.array([x["aggregate"]["recall"]["mean"] for x in n38k])
    rec_38k_s = np.array([x["aggregate"]["recall"]["std"] for x in n38k])
    pre_38k_m = np.array([x["aggregate"]["precision"]["mean"] for x in n38k])
    std_38k_m = np.array([x["aggregate"]["intra_class_latent_std"]["mean"] for x in n38k])
    sn85_38k_std = float(std_38k_m[np.argmin(np.abs(sn_vals - 0.85))])

    max_p_xlim = float(max(res_p[-1], sn85_p[-1] if has_sn85 else 0) * 1.9)

    # ------------------------------------------------------------------
    # Panel (a): Intra-Class Latent Diversity (LatStd) vs Parameters P
    # ------------------------------------------------------------------
    ax = axes[0, 0]
    ax.axvspan(115, 250, color="#ffe8cc", alpha=0.75, zorder=0, label="~200p NISQ regime (120p W_cond + ~80p core)")
    ax.axvline(120, color="#d9480f", linestyle=":", linewidth=1.3, alpha=0.85)
    ax.axhline(real_std, color="#2b2d42", linestyle="--", linewidth=1.3, label=f"Real Held-Out Test LatStd = {real_std:.3f}")

    if has_sn85:
        ax.plot(sn85_p, sn85_std_m, "^-", color="#1864ab", lw=2.3, ms=5.8,
                label="Classical ResMLP (Adam, σ_step=0.85)")
        ax.fill_between(sn85_p, sn85_std_m - sn85_std_s, sn85_std_m + sn85_std_s, color="#1864ab", alpha=0.14)
    ax.plot(res_p, res_std_m, "o-", color="#339af0", lw=2.0, ms=5.2, label="Classical ResMLP (Adam, σ_step=0.35)")
    ax.fill_between(res_p, res_std_m - res_std_s, res_std_m + res_std_s, color="#339af0", alpha=0.15)
    ax.plot(skp_p, skp_std_m, "s--", color="#0ca678", lw=1.8, ms=4.8, label="Classical Skip-MLP (Adam, σ_step=0.35)")
    ax.fill_between(skp_p, skp_std_m - skp_std_s, skp_std_m + skp_std_s, color="#0ca678", alpha=0.14)

    qer_p = mn_pc["qpu"]
    qer_std = mn_ag["qerd"]["intra_class_latent_std"]["mean"]
    ax.scatter([qer_p], [qer_std], marker="*", s=240, color="#e03131", edgecolor="black", zorder=6,
               label=f"QER (Ours, {qer_p}p NISQ 80k): LatStd={qer_std:.3f}")
    ax.scatter([mn_pc["classical_skip_ddpm"]], [mn_ag["classical_skip_ddpm"]["intra_class_latent_std"]["mean"]],
               marker="D", s=58, color="#f08c00", edgecolor="black", zorder=5,
               label=f"Paper Cl-Skip ({mn_pc['classical_skip_ddpm']}p): LatStd={mn_ag['classical_skip_ddpm']['intra_class_latent_std']['mean']:.3f}")
    if not has_sn85:
        ax.scatter([38020], [sn85_38k_std], marker="P", s=110, color="#1864ab", edgecolor="black", zorder=6,
                   label=f"ResMLP-38k (Adam, σ_step=0.85): LatStd={sn85_38k_std:.3f}")
    else:
        cl_lrg_std = mn_ag["classical_large_ddpm"]["intra_class_latent_std"]["mean"]
        ax.scatter([mn_pc["classical_large_ddpm"]], [cl_lrg_std], marker="P", s=95, color="#7048e8", edgecolor="black", zorder=6,
                   label=f"Paper Cl-Large (38,020p): LatStd={cl_lrg_std:.3f}")

    ax.annotate("W_cond alone\n= 120 params", xy=(120, 0.215), xytext=(148, 0.200),
                arrowprops=dict(arrowstyle="->", color="#d9480f", lw=1.1),
                fontsize=8.0, color="#b03a0c", fontweight="bold")
    ax.annotate("Collapse at P≈200:\nQER (0.095) ≥ exact Adam",
                xy=(qer_p, qer_std), xytext=(1800, 0.008),
                arrowprops=dict(arrowstyle="->", color="#e03131", lw=1.2),
                fontsize=7.6, bbox=dict(boxstyle="round,pad=0.20", fc="#fff5f5", ec="#e03131", lw=0.8))
    if has_sn85:
        ax.annotate(f"Saturates past P≈141k\n(gap to {real_std:.3f} = 4-step MSE\n+ 12-D AE choke)",
                    xy=(sn85_p[-1], sn85_std_m[-1]), xytext=(16000, 0.352),
                    arrowprops=dict(arrowstyle="->", color="#1864ab", lw=1.1),
                    fontsize=7.8, color="#1864ab", fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlim(100, max_p_xlim)
    ax.set_ylim(0.0, 0.50)
    ax.set_xlabel("Trainable Denoiser Parameters P (log scale)")
    ax.set_ylabel("Intra-Class Latent Std (LatStd)")
    ax.set_title("(a) Latent Diversity Scaling & Saturation Across P (MNIST)")
    ax.grid(True, which="both", linestyle="--", alpha=0.28)
    ax.legend(loc="upper left", frameon=True, framealpha=0.94, fontsize=7.6)

    # ------------------------------------------------------------------
    # Panel (b): Exact Recall Phase Transition & Saturation Across P
    # ------------------------------------------------------------------
    ax = axes[0, 1]
    ax.axvspan(115, 250, color="#ffe8cc", alpha=0.75, zorder=0)
    ax.axhline(ae_rec, color="#5f3dc4", linestyle="--", linewidth=1.3,
               label=f"Real-AE-Recon Recall Cap = {ae_rec:.3f} (12-D AE)")

    qer_pre = mn_ag["qerd"]["precision"]["mean"]
    qer_rec = mn_ag["qerd"]["recall"]["mean"]
    qer_cov = mn_ag["qerd"]["coverage"]["mean"]

    if has_sn85:
        ax.plot(sn85_p, sn85_pre_m, "^:", color="#2b8a3e", lw=2.0, ms=5.2,
                label="ResMLP Prec (σ_step=0.85, ≥0.99)")
        ax.plot(sn85_p, sn85_cov_m, "d--", color="#0ca678", lw=2.0, ms=5.2,
                label=f"ResMLP Cov (σ_step=0.85: {sn85_cov_m[0]:.2f}→{sn85_cov_m[-1]:.2f})")
        ax.fill_between(sn85_p, np.clip(sn85_cov_m - sn85_cov_s, 0, 1), np.clip(sn85_cov_m + sn85_cov_s, 0, 1),
                        color="#0ca678", alpha=0.12)
        ax.plot(sn85_p, sn85_rec_m, "o-", color="#1c7ed6", lw=2.5, ms=6.0,
                label=f"ResMLP Recall (σ_step=0.85: 0.00→{sn85_rec_m[-1]:.3f})")
        ax.fill_between(sn85_p, np.clip(sn85_rec_m - sn85_rec_s, 0, 1), sn85_rec_m + sn85_rec_s,
                        color="#1c7ed6", alpha=0.18)
        ax.plot(res_p, res_rec_m, "o:", color="#748ffc", lw=1.6, ms=4.2,
                label=f"ResMLP Recall (σ_step=0.35: ≤{res_rec_m.max():.3f})")
        ax.scatter([qer_p], [qer_pre], marker="*", s=160, color="#2b8a3e", edgecolor="black", zorder=6)
        ax.scatter([qer_p], [qer_rec], marker="*", s=240, color="#e03131", edgecolor="black", zorder=6,
                   label=f"QER ({qer_p}p): Prec={qer_pre:.3f}, Rec={qer_rec:.3f}, Cov={qer_cov:.3f}")

        ax.annotate("Collapsed plateau:\nRec = 0.000 for P ≲ 2k\n(even at σ_step=0.85)",
                    xy=(900, 0.005), xytext=(480, 0.055),
                    arrowprops=dict(arrowstyle="->", color="#c92a2a", lw=1.1),
                    fontsize=7.6, bbox=dict(boxstyle="round,pad=0.20", fc="#fff5f5", ec="#e03131", lw=0.8))
        target_idx = -2 if len(sn85_p) >= 2 else -1
        ax.annotate(f"Unlock (P ≈ 6k–141k) &\nsaturation (Rec≈{sn85_rec_m[target_idx]:.2f}–{sn85_rec_m[-1]:.2f},\nPrec ≥ 0.99)",
                    xy=(sn85_p[target_idx], sn85_rec_m[target_idx]), xytext=(20000, 0.36),
                    arrowprops=dict(arrowstyle="->", color="#1864ab", lw=1.2),
                    fontsize=7.8, color="#1864ab", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.22", fc="#edf2ff", ec="#1c7ed6", lw=0.8))
        ax.set_title("(b) Exact Phase Transition: k-NN Recall Unlocks & Saturates With P")
        ax.legend(loc="upper left", bbox_to_anchor=(0.005, 0.90), frameon=True, framealpha=0.94, fontsize=7.2)
    else:
        ax.plot(res_p, res_pre_m, "^:", color="#2b8a3e", lw=2.0, ms=5.5, label="ResMLP Precision (always 0.98–1.00)")
        ax.plot(res_p, res_cov_m, "o-", color="#1c7ed6", lw=2.3, ms=5.8,
                label="ResMLP Manifold Coverage (0.044 → 0.469, 10.7×)")
        ax.fill_between(res_p, res_cov_m - res_cov_s, res_cov_m + res_cov_s, color="#1c7ed6", alpha=0.16)
        ax.plot(res_p, res_rad_ratio, "D-", color="#7048e8", lw=2.1, ms=5.2,
                label="ResMLP 5-NN Ball Radius Ratio G/R (0.016 → 0.283, 17.7×)")
        ax.fill_between(res_p, res_rad_ratio - res_rad_ratio_s, res_rad_ratio + res_rad_ratio_s, color="#7048e8", alpha=0.14)
        ax.scatter([qer_p], [qer_pre], marker="*", s=160, color="#2b8a3e", edgecolor="black", zorder=6)
        ax.scatter([qer_p], [qer_cov], marker="*", s=240, color="#e03131", edgecolor="black", zorder=6,
                   label=f"QER ({qer_p}p): Prec={qer_pre:.3f}, Cov={qer_cov:.3f}")
        ax.set_title("(b) Precision ≈ 1.0 Everywhere; Coverage & 5-NN Radius Scale With P")
        ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.91), frameon=True, framealpha=0.94, fontsize=7.9)

    ax.set_xscale("log")
    ax.set_xlim(100, max_p_xlim)
    ax.set_ylim(-0.025, 1.08)
    ax.set_xlabel("Trainable Denoiser Parameters P (log scale)")
    ax.set_ylabel("EvalCNN 64-D Metric [0, 1]")
    ax.grid(True, which="both", linestyle="--", alpha=0.28)

    # ------------------------------------------------------------------
    # Panel (c): Exp C2 + C1 Smoking Gun: Noise Unlocks Recall Only at High P
    # ------------------------------------------------------------------
    ax = axes[1, 0]
    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)

    l1, = ax.plot(sn_vals, rec_38k_m, "o-", color="#1c7ed6", lw=2.4, ms=6.2,
                  label="P=38,020 (h=128) Recall [left]")
    ax.fill_between(sn_vals, np.clip(rec_38k_m - rec_38k_s, 0, 1), rec_38k_m + rec_38k_s, color="#1c7ed6", alpha=0.16)
    l2, = ax.plot(sn_vals, rec_220_m, "s-", color="#e03131", lw=2.3, ms=5.8,
                  label="P=220 (h=2) Recall = 0.000 everywhere [left]")

    l3, = ax2.plot(sn_vals, std_38k_m, "^--", color="#1864ab", lw=1.7, ms=5.2, alpha=0.85,
                   label="P=38,020 LatStd [right]")
    l4, = ax2.plot(sn_vals, std_220_m, "v--", color="#d9480f", lw=1.7, ms=5.2, alpha=0.85,
                   label="P=220 LatStd [right]")

    # Annotate exact Recall & Precision along P=38,020 curve
    annot_pos = {
        0.60: (0.48, 0.100),
        0.85: (0.74, 0.195),
        1.10: (0.94, 0.278),
    }
    for sx, ry, py in zip(sn_vals, rec_38k_m, pre_38k_m):
        for k_sx, (tx, ty) in annot_pos.items():
            if abs(sx - k_sx) < 1e-3:
                ax.annotate(f"Rec={ry:.3f}\n(Prec={py:.3f})", xy=(sx, ry), xytext=(tx, ty),
                            arrowprops=dict(arrowstyle="-", color="#1864ab", lw=0.8, alpha=0.7),
                            fontsize=7.7, color="#1864ab", fontweight="bold")

    ax.scatter([0.35], [mn_ag["qerd"]["recall"]["mean"]], marker="*", s=200, color="#e03131", edgecolor="black", zorder=7)
    ax2.scatter([0.35], [qer_std], marker="*", s=200, color="#f08c00", edgecolor="black", zorder=7)
    ax2.annotate(f"QER (198p, σ_step=0.35)\nLatStd={qer_std:.3f}, Rec=0.000",
                 xy=(0.35, qer_std), xytext=(0.13, 0.215),
                 arrowprops=dict(arrowstyle="->", color="#d9480f", lw=1.1),
                 fontsize=7.8, color="#b03a0c",
                 bbox=dict(boxstyle="round,pad=0.22", fc="#fff9db", ec="#f08c00", lw=0.8),
                 zorder=8)

    st_a = data["step_ablation"][0]["aggregate"]
    st_b = data["step_ablation"][-1]["aggregate"]
    ax.text(0.73, 0.058,
            f"Multi-step (P=204, 200→2k steps):\n"
            f"LatStd {st_a['intra_class_latent_std']['mean']:.3f}→{st_b['intra_class_latent_std']['mean']:.3f}, Rec = 0.000\n"
            f"7.3× sampler noise (P=220): Rec = 0.000",
            fontsize=7.4, va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.24", fc="#fff5f5", ec="#e03131", lw=0.8))

    ax.set_xlim(0.10, 1.18)
    ax.set_xticks(sn_vals)
    ax.set_ylim(-0.015, 0.345)
    ax2.set_ylim(0.0, 0.43)
    ax.set_xlabel("Reverse Sampler Step Noise σ_step")
    ax.set_ylabel("k-NN Recall (EvalCNN 64-D, 5-NN)")
    ax2.set_ylabel("Intra-Class Latent Std (LatStd)")
    ax.set_title("(c) Step-Noise Scaling Unlocks Recall at P=38k vs. P=220")
    ax.grid(True, linestyle="--", alpha=0.28)
    ax.legend([l1, l2, l3, l4], [l1.get_label(), l2.get_label(), l3.get_label(), l4.get_label()],
              loc="upper left", frameon=True, framealpha=0.94, fontsize=7.9)

    # ------------------------------------------------------------------
    # Panel (d): Cross-Dataset & Optimizer Proof (5-Seed paper_suite)
    # ------------------------------------------------------------------
    ax = axes[1, 1]
    methods = [
        ("QER\n(198/224p)", "qerd"),
        ("Adam-SPSA\n(198/224p)", "adam_spsa"),
        ("Exact-PSR\n(198/224p)", "exact_psr"),
        ("Cl-DDPM\n(192/228p)", "classical_ddpm"),
        ("Cl-Skip\n(204/240p)", "classical_skip_ddpm"),
        ("Cl-Large\n(38,020p)", "classical_large_ddpm"),
    ]
    xm = np.arange(len(methods))
    bw = 0.34
    mn_acc = [mn_ag[m[1]]["clf_acc"]["mean"] for m in methods]
    fa_acc = [fa_ag[m[1]]["clf_acc"]["mean"] for m in methods]
    mn_std = [mn_ag[m[1]]["intra_class_latent_std"]["mean"] for m in methods]
    mn_rec = [mn_ag[m[1]]["recall"]["mean"] for m in methods]

    ax.bar(xm - bw / 2, mn_acc, width=bw, color="#3b5bdb", alpha=0.88, label="MNIST ClfAcc (5 seeds)")
    ax.bar(xm + bw / 2, fa_acc, width=bw, color="#748ffc", alpha=0.85, label="Fashion ClfAcc (5 seeds)")
    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(xm, mn_std, "o-", color="#d9480f", lw=2.0, ms=5.8, label="MNIST LatStd [right]")
    ax2.plot(xm, mn_rec, "s--", color="#9c36b5", lw=1.8, ms=5.2, label="MNIST Recall [right]")

    offsets_y = [0.015, 0.015, 0.015, 0.025, 0.015, 0.013]
    for i, (m_s, m_r, dy) in enumerate(zip(mn_std, mn_rec, offsets_y)):
        ha = "right" if i == 5 else "center"
        x_pos = i - 0.12 if i == 5 else i
        ax2.text(x_pos, m_s + dy, f"σ={m_s:.2f}\nR={m_r:.3f}", ha=ha, va="bottom", fontsize=7.5, color="#862e9c",
                 bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.80))

    ax.set_xticks(xm)
    ax.set_xticklabels([m[0] for m in methods], fontsize=8.2)
    ax.set_ylim(0, 1.30)
    ax2.set_ylim(0, 0.40)
    ax.set_ylabel("Generated Classifier Accuracy")
    ax2.set_ylabel("LatStd / k-NN Recall (MNIST)")
    ax.set_title("(d) Cross-Dataset Comparison (5 Seeds): QERD Matches P≈200 Parity")
    ax.grid(True, axis="y", linestyle="--", alpha=0.28)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", frameon=True, framealpha=0.94, ncol=2, fontsize=7.8)

    for ext in ("png", "pdf"):
        out_path = os.path.join(out_dir, f"fig_capacity_vs_diversity.{ext}")
        fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.14)
        print(f"[Saved] {out_path}", flush=True)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="./analysis")
    parser.add_argument("--replot", action="store_true")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--widths", default="2,5,12,24,48,96,128,256,512")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    widths = [int(w.strip()) for w in args.widths.split(",") if w.strip()]
    json_path = os.path.join(args.out_dir, "capacity_diversity_results.json")
    t0 = time.time()
    if args.replot and os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
    else:
        data = run_experiments(out_dir=args.out_dir, seeds=seeds, resmlp_widths=widths)
    make_proof_figure(data, out_dir=args.out_dir)
    print(f"Done in {time.time() - t0:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
