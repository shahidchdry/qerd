"""
Generate publication-grade Live 128-156q Generative Distribution Audit figure
(`analysis/fig_live_generative_audit.{pdf,png}`) and finalize the IBM Job ID ledger
from saved live ibm_kingston bitstring extractions (0 QPU seconds).
"""
from __future__ import annotations

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    with open("analysis/live_generative_distribution_audit.json", "r", encoding="utf-8") as f:
        audit = json.load(f)
    with open("analysis/phase_b_oracle_and_debias_128q.json", "r", encoding="utf-8") as f:
        pb = json.load(f)
    with open("analysis/ibm_kingston_job_ledger.json", "r", encoding="utf-8") as f:
        ledger = json.load(f)

    # Append Phase B live jobs to ledger cleanly
    ledger["phase_b_128q_seed43_oracle_and_debias"] = {
        "backend": pb["backend"],
        "n_qubits": pb["n_qubits"],
        "n_edges": pb["n_edges"],
        "seed": pb["seed"],
        "n_jobs": len(pb["job_ids"]),
        "qaoa_p4_oracle_job_id": pb["job_ids"][0]["job_id"],
        "revdiff_t4_to_t1_job_ids": [x["job_id"] for x in pb["job_ids"][1:]],
        "all_job_ids": [x["job_id"] for x in pb["job_ids"]],
        "quota_start_sec": pb["quota_start_sec"],
        "quota_end_sec": pb["quota_end_sec"],
    }
    with open("analysis/ibm_kingston_job_ledger.json", "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2)

    runs = audit["live_qpu_runs"]
    refs = audit["references"]

    # Extract seed=43 arrays for crisp head-to-head distributional figure
    s43_qer = runs["qer_res_128q_seed43"]
    s43_spsa = runs["spsa_128q_seed43"]
    s43_sa = refs["inst_128q_seed43"]["sa_heldout_32_z2"]
    s43_rand = refs["inst_128q_seed43"]["random_32"]
    s43_oracle = pb["oracle_template_revdiff_128q_seed43"]
    s43_debias = pb["qer_res_z2_debiased_128q_seed43"]
    s43_qaoa = pb["qaoa_p4_oracle_single_shot"]["qaoa_p4_c0.70_t0.25"]
    s43_qaoa_c1 = pb["qaoa_p4_oracle_single_shot"]["qaoa_p4_c1.00_t0.25"]
    s42_untr = runs["untrained_128q_seed42"]

    plt.rcParams.update({
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.8,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "legend.fontsize": 7.1,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.85), dpi=220)

    # ------------------------------------------------------------------
    # Panel (a): Live 128q Energy Distribution (seed=43, N=32..128)
    # ------------------------------------------------------------------
    ax = axes[0]
    items = [
        ("SPSA\n(32k sh)", s43_spsa["energies_list"], "#E69F00"),
        ("Fixed-Mix.*\n(c=0.70)", s43_qaoa["energies_list"], "#CC79A7"),
        ("QER+$\\mathbb{Z}_2$\n(32sh eval)", s43_debias["energies_list"], "#56B4E9"),
        ("Oracle*\n($b\\equiv 0$, 32sh)", s43_oracle["energies_list"], "#888888"),
        ("QER-Res\n(96sh eval)", s43_qer["energies_list"], "#0072B2"),
        ("Held-Out\nSA Target", s43_sa["energies_list"], "#009E73"),
    ]
    positions = np.arange(len(items))
    bp = ax.boxplot(
        [it[1] for it in items],
        positions=positions,
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 1.4},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
    )
    for patch, (_, _, col) in zip(bp["boxes"], items):
        patch.set_facecolor(col)
        patch.set_alpha(0.82)
        patch.set_edgecolor("#222222")
        patch.set_linewidth(0.8)

    # Overlay jittered raw points + mean markers + crisp non-overlapping labels above upper whiskers
    rng_j = np.random.default_rng(7)
    for idx, (_, vals, col) in enumerate(items):
        arr = np.asarray(vals, dtype=float)
        pts = arr[:32] if len(arr) > 32 else arr
        xj = idx + rng_j.uniform(-0.13, 0.13, size=len(pts))
        ax.scatter(xj, pts, s=9, color="#111111", alpha=0.28, zorder=3, linewidths=0)
        mean_v = float(np.mean(arr))
        ax.scatter([idx], [mean_v], s=26, marker="D", color="white", edgecolor="#111111", linewidth=0.9, zorder=4)
        q1, q3 = np.percentile(arr, [25, 75])
        hi_w = float(np.max(arr[arr <= q3 + 1.5 * (q3 - q1)]))
        lo_w = float(np.min(arr[arr >= q1 - 1.5 * (q3 - q1)]))
        y_lbl = (lo_w - 8.5) if idx == 0 else (hi_w + 7.5)
        ax.text(
            idx, y_lbl, f"{mean_v:+.1f}" if idx == 0 else f"{mean_v:.1f}",
            ha="center", va="center", fontsize=7.1, fontweight="bold", color="#111111",
            bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.85),
            zorder=5,
        )

    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels([it[0] for it in items], fontsize=7.0)
    ax.set_ylabel("Ising Spin-Glass Energy $E$ $\\downarrow$")
    ax.set_title("(a) Live 128q Energy Distributions (seed=43)")
    ax.set_ylim(-140, 26)
    ax.grid(axis="y", linestyle=":", alpha=0.45)

    # ------------------------------------------------------------------
    # Panel (b): Two-Point Edge Correlation Scatter (<s_u s_v> vs SA)
    # ------------------------------------------------------------------
    ax = axes[1]
    c_sa = np.asarray(s43_qer["edge_corrs_sa_ho"], dtype=float)
    c_qer = np.asarray(s43_qer["edge_corrs_gen"], dtype=float)
    c_deb = np.asarray(s43_debias["edge_corrs_gen"], dtype=float)
    c_spsa = np.asarray(s43_spsa["edge_corrs_gen"], dtype=float)

    ax.plot([-1.05, 1.05], [-1.05, 1.05], "--", color="#222222", linewidth=1.0, alpha=0.75, label="Ideal $y=x$")
    ax.axhline(0.0, color="#aaaaaa", linewidth=0.6, alpha=0.6)
    ax.axvline(0.0, color="#aaaaaa", linewidth=0.6, alpha=0.6)
    ax.scatter(
        c_sa, c_spsa, s=16, alpha=0.42, color="#E69F00", marker="^", linewidths=0,
        label=f"Live SPSA ($r={s43_spsa['edge_corr_pearson_r']:+.2f}$, RMSE={s43_spsa['edge_corr_rmse_vs_sa_ho']:.2f})",
    )
    ax.scatter(
        c_sa, c_qer, s=18, alpha=0.68, color="#0072B2", marker="o", linewidths=0.3, edgecolor="white",
        label=f"Live QER-Res ($r={s43_qer['edge_corr_pearson_r']:+.2f}$, RMSE={s43_qer['edge_corr_rmse_vs_sa_ho']:.2f})",
    )
    ax.scatter(
        c_sa, c_deb, s=18, alpha=0.72, color="#009E73", marker="s", linewidths=0.3, edgecolor="white",
        label=f"QER+$\\mathbb{{Z}}_2$ Deb. ($r={s43_debias['edge_corr_pearson_r']:+.2f}$, 99.3% sign)",
    )
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.set_xlabel("Held-Out SA Target $\\langle s_u s_v \\rangle_{\\mathrm{SA}}$ (135 edges)")
    ax.set_ylabel("Live QPU Generated $\\langle s_u s_v \\rangle_{\\mathrm{QPU}}$")
    ax.set_title("(b) Two-Point Edge Correlation Fidelity (128q)")
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.90, edgecolor="#cccccc")
    ax.grid(True, linestyle=":", alpha=0.45)

    # ------------------------------------------------------------------
    # Panel (c): Edwards-Anderson Pairwise Spin Overlap P(q)
    # ------------------------------------------------------------------
    ax = axes[2]
    bins = np.linspace(-1.0, 1.0, 33)
    ax.hist(
        s43_sa["overlaps_q"], bins=bins, density=True, histtype="stepfilled",
        color="#009E73", alpha=0.22, edgecolor="#009E73", linewidth=1.4,
        label=f"Held-Out SA ($\\overline{{|q|}}={s43_sa['overlap_abs_q_mean']:.2f}$, $\\sigma_q={s43_sa['overlap_q_std']:.2f}$)",
    )
    ax.hist(
        s43_debias["overlaps_q"], bins=bins, density=True, histtype="step",
        color="#0072B2", linewidth=2.0,
        label=f"QER+$\\mathbb{{Z}}_2$ ($\\overline{{|q|}}={s43_debias['overlap_abs_q_mean']:.2f}$, $\\sigma_q={s43_debias['overlap_q_std']:.2f}$)",
    )
    ax.hist(
        s43_oracle["overlaps_q"], bins=bins, density=True, histtype="step",
        color="#666666", linestyle="--", linewidth=1.4,
        label=f"Oracle ($b\\equiv 0$) ($\\overline{{|q|}}={s43_oracle['overlap_abs_q_mean']:.2f}$, 0%<0)",
    )
    ax.hist(
        s43_qer["overlaps_q"], bins=bins, density=True, histtype="step",
        color="#56B4E9", linewidth=1.5,
        label=f"QER-Res Raw ($\\overline{{|q|}}={s43_qer['overlap_abs_q_mean']:.2f}$, 0%<0)",
    )
    ax.hist(
        s43_spsa["overlaps_q"], bins=bins, density=True, histtype="stepfilled",
        color="#E69F00", alpha=0.28, edgecolor="#D55E00", linewidth=1.3,
        label=f"Live SPSA ($\\overline{{|q|}}={s43_spsa['overlap_abs_q_mean']:.2f}$, collapsed)",
    )
    ax.axvline(0.0, color="#222222", linestyle=":", linewidth=1.0)
    ax.set_xlim(-0.85, 1.02)
    ax.set_xlabel("Pairwise Spin Overlap $q_{ab} = \\frac{1}{n}\\mathbf{s}^{(a)}\\!\\cdot\\!\\mathbf{s}^{(b)}$")
    ax.set_ylabel("Empirical Density $P(q)$")
    ax.set_title("(c) Overlap $P(q)$: Drift Diagnosis & $\\mathbb{Z}_2$ Fix")
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.90, edgecolor="#cccccc", fontsize=6.6)
    ax.grid(axis="y", linestyle=":", alpha=0.45)

    plt.tight_layout(w_pad=1.6)
    os.makedirs("analysis", exist_ok=True)
    fig.savefig("analysis/fig_live_generative_audit.pdf", bbox_inches="tight")
    fig.savefig("analysis/fig_live_generative_audit.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("[DONE] Saved analysis/fig_live_generative_audit.{pdf,png} and updated analysis/ibm_kingston_job_ledger.json")


if __name__ == "__main__":
    main()
