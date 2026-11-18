"""4-panel showcase plotter, FASTA exporter, and JSON reporter for Project 2 (`dna_promoter_qerd`)."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .biology.biophysics import (
    compute_santalucia_dg37,
    gc_percent,
    has_tss_initiator,
    has_upstream_promoter_box,
)
from .hardware.kingston_clean import SubgraphReport


def make_dna_showcase_figure(
    comp: Dict[str, Dict[str, float]],
    profiles: Dict[str, Dict[str, np.ndarray]],
    thermo_samples: Dict[str, Tuple[np.ndarray, np.ndarray]],
    n_qubits: int,
    max_shots: int,
    out_png: str,
    circuits_per_sample: int = 3,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8.8,
        "axes.titlesize": 10.2,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(2, 2, figsize=(14.8, 9.6))
    seq_len = n_qubits // 2
    fig.suptitle(
        f"Project 2: Conditional Human Promoter DNA Designer ({n_qubits}q / {seq_len} bp, {max_shots:,} Train Shots)",
        fontsize=12.0, fontweight="bold", y=0.985,
    )

    # Panel (a): Oracle Promoter Score & Biophysical Validity % (every bar higher = strictly better!)
    ax = axes[0, 0]
    order = [
        ("untrained_qpu", "Untrained\nQPU", "#999999"),
        ("psr_qpu", "Equal-Shot\nPSR", "#CC79A7"),
        ("spsa_qpu", "Equal-Shot\nSPSA", "#E69F00"),
        ("classical_markov", "Classical\n1st-Order", "#56B4E9"),
        ("qerd_qpu", "QERD QPU\n(Ours)", "#0072B2"),
        ("real_promoters", "Real Human\nPromoters", "#009E73"),
    ]
    rows = [(k, l, c) for k, l, c in order if k in comp]
    x = np.arange(len(rows))
    w = 0.36
    act = [100.0 * comp[k]["mean_oracle_prob"] for k, _, _ in rows]
    val = [comp[k]["valid_promoter_pct"] for k, _, _ in rows]
    b1 = ax.bar(x - w / 2, act, width=w, color=[c for _, _, c in rows], edgecolor="#222222", label="Promoter Activity Score (0-100%) ↑")
    b2 = ax.bar(x + w / 2, val, width=w, color=[c for _, _, c in rows], alpha=0.55, edgecolor="#222222", hatch="//", label="Valid Promoter Rules % ↑")
    for b, v in zip(b1, act):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=7.0, fontweight="bold")
    for b, v in zip(b2, val):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=7.0)
    ax.set_xticks(x)
    ax.set_xticklabels([l for _, l, _ in rows], fontsize=7.8)
    ax.set_ylim(0.0, 114.0)
    ax.set_ylabel("Percentage / Score (%) ↑")
    ax.set_title("(a) Human Promoter Activity & Biophysical Validity")
    ax.legend(fontsize=7.3, loc="upper left", framealpha=0.95)
    ax.grid(axis="y", ls="--", alpha=0.25)

    # Panel (b): Positional GC% Profile across Core Promoter Window (-49..+14 or 8bp Core)
    ax = axes[0, 1]
    if "real_promoters" in profiles and profiles["real_promoters"]["gc"].size > 0:
        gc_r = 100.0 * profiles["real_promoters"]["gc"]
        pos = np.arange(-49, -49 + len(gc_r)) if len(gc_r) >= 32 else np.arange(len(gc_r))
        ax.plot(pos, gc_r, "-", color="#009E73", lw=2.3, label="Real Human Promoters")
        if "qerd_qpu" in profiles and profiles["qerd_qpu"]["gc"].size > 0:
            ax.plot(pos, 100.0 * profiles["qerd_qpu"]["gc"], "-", color="#0072B2", lw=2.1, label="QERD QPU (Ours)")
        if "spsa_qpu" in profiles and profiles["spsa_qpu"]["gc"].size > 0:
            ax.plot(pos, 100.0 * profiles["spsa_qpu"]["gc"], ":", color="#E69F00", lw=1.8, label="Equal-Shot SPSA")
        if "untrained_qpu" in profiles and profiles["untrained_qpu"]["gc"].size > 0:
            ax.plot(pos, 100.0 * profiles["untrained_qpu"]["gc"], "--", color="#888888", lw=1.5, label="Untrained QPU (~50%)")
        if len(gc_r) >= 32:
            ax.axvspan(-33, -25, color="#D55E00", alpha=0.10, label="-30 TATA AT-Dip Zone")
            ax.axvline(0, color="#111111", ls=":", lw=1.2, label="TSS (+1 Initiator)")
            ax.set_xlabel("Position Relative to Transcription Start Site (bp)")
        else:
            ax.set_xlabel("Core Motif Position (0..3: -30 Box | 4..7: TSS Inr)")
    ax.set_ylim(-2.0, 108.0)
    ax.set_ylabel("Position GC Content (%)")
    ax.set_title("(b) Learned Promoter Motif & TSS Architecture")
    ax.legend(fontsize=7.2, loc="lower left", framealpha=0.95)
    ax.grid(True, ls="--", alpha=0.25)

    # Panel (c): Thermodynamic Realism (GC% vs SantaLucia dG37 kcal/mol)
    ax = axes[1, 0]
    plot_order = [
        ("untrained_qpu", "Untrained QPU", "#999999", "o", 26),
        ("spsa_qpu", "Equal-Shot SPSA", "#E69F00", "s", 28),
        ("real_promoters", "Real Human Genome", "#009E73", "^", 30),
        ("qerd_qpu", "QERD QPU (Ours)", "#0072B2", "D", 38),
    ]
    for k, lbl, col, mk, sz in plot_order:
        if k in thermo_samples:
            gcs, dgs = thermo_samples[k]
            ax.scatter(gcs, dgs, s=sz, color=col, marker=mk, alpha=0.68, edgecolors="#222222", linewidths=0.5, label=lbl)
    ax.set_xlabel("GC Content (%)")
    ax.set_ylabel("SantaLucia dG37 (kcal/mol) ↓")
    ax.set_title("(c) Biophysical Stability vs. Real Genome")
    ax.legend(fontsize=7.4, loc="upper right", framealpha=0.95)
    ax.grid(True, ls="--", alpha=0.25)

    # Panel (d): TSS Initiator %, Unique Valid Designs %, and O(1)=2 Circuit Scaling
    ax = axes[1, 1]
    d_rows = [(k, l, c) for k, l, c in order if k in comp and k != "real_promoters"]
    xd = np.arange(len(d_rows))
    uval_vals = [comp[k].get("unique_valid_pct", comp[k]["valid_promoter_pct"]) for k, _, _ in d_rows]
    inr_vals = [comp[k]["tss_inr_pct"] for k, _, _ in d_rows]
    b_inr = ax.bar(xd - w / 2, inr_vals, width=w, color="#0072B2", edgecolor="#222222", label="TSS Initiator Motif % ↑")
    b_unq = ax.bar(xd + w / 2, uval_vals, width=w, color="#009E73", alpha=0.80, edgecolor="#222222", label="Unique Valid Promoters % ↑")
    for b, v in zip(b_inr, inr_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=7.0, fontweight="bold")
    for b, v in zip(b_unq, uval_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=7.0, fontweight="bold")
    if "qerd_qpu" in comp:
        nov = comp["qerd_qpu"].get("novel_pct", 100.0)
        hdiv = 100.0 * comp["qerd_qpu"].get("hamming_diversity", 0.0)
        note = (
            f"QERD Hardware Edge: {circuits_per_sample} circuits/sample vs PSR {2*n_qubits + 2*(n_qubits-4)}\n"
            f"Novel vs Train: {nov:.1f}% | Mean Pairwise Diversity: {hdiv:.1f}% bp"
        )
        ax.text(0.02, 0.97, note, transform=ax.transAxes, va="top", fontsize=7.2,
                bbox=dict(boxstyle="round,pad=0.30", facecolor="#F4F9FF", edgecolor="#0072B2", alpha=0.96))
    ax.set_xticks(xd)
    ax.set_xticklabels([l for _, l, _ in d_rows], fontsize=7.8)
    ax.set_ylim(0.0, 152.0)
    ax.set_ylabel("Percentage (%) ↑")
    ax.set_title("(d) TSS Motif, Unique Valid Designs & Novelty")
    ax.legend(fontsize=7.1, loc="upper right", bbox_to_anchor=(0.99, 0.99), framealpha=0.95)
    ax.grid(axis="y", ls="--", alpha=0.25)

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def export_top_promoters_fasta(
    seqs: Sequence[str],
    probs: np.ndarray,
    cond_tags: Sequence[str],
    out_fasta: str,
    top_k: int = 12,
) -> None:
    """Export top diverse generated human promoter designs in standard FASTA format."""
    os.makedirs(os.path.dirname(out_fasta), exist_ok=True)
    seen = set()
    ranked = sorted(range(len(seqs)), key=lambda i: float(probs[i]), reverse=True)
    lines: List[str] = []
    kept = 0
    for idx in ranked:
        s = seqs[idx]
        if s in seen:
            continue
        seen.add(s)
        p = float(probs[idx])
        dg = compute_santalucia_dg37(s)
        gc = gc_percent(s)
        tag = cond_tags[idx] if idx < len(cond_tags) else "promoter"
        inr = "Inr+" if has_tss_initiator(s) else "Inr-"
        box = "Box+" if has_upstream_promoter_box(s, "any") else "Box-"
        lines.append(f">QERD_DNA_{kept+1:02d} class={tag} oracle={p:.3f} gc={gc:.1f}% dG37={dg:.1f}kcal {box} {inr}")
        lines.append(s)
        kept += 1
        if kept >= top_k:
            break
    with open(out_fasta, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def save_metrics_json(
    out_json: str,
    cfg_dict: Dict[str, Any],
    sub_rep: SubgraphReport,
    oracle_metrics: Dict[str, float],
    scaling_rows: List[Dict[str, Any]],
    comp: Dict[str, Dict[str, float]],
) -> None:
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    payload = {
        "config": cfg_dict,
        "subgraph_report": asdict(sub_rep),
        "oracle_cnn_metrics": oracle_metrics,
        "t4_simulation_wall": scaling_rows,
        "comparison": comp,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
