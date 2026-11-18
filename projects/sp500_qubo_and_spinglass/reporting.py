"""Reporting, JSON artifacts, and publication-grade 3-panel hardware advantage plots."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

from .hardware.heron_r2 import HeronAudit, RuntimeProjection, SubgraphAudit


def format_heron_comparison(audits: List[HeronAudit], proj: RuntimeProjection) -> str:
    lines = [
        "\n[Hardware Audit] IBM 156-Qubit Heron R2 Snapshot Comparison (Native Full-Chip 2-Layer QERD)",
        f"{'Processor':<15} | {'Edges':>5} | {'2Q Med':>7} | {'2Q p90':>7} | {'>2% CZ':>6} | {'RO Med':>7} | {'RO p90':>7} | {'T1/T2 (us)':>11}",
        "-" * 85,
    ]
    for a in audits:
        lines.append(
            f"{a.device:<15} | {a.native_edges:>5} | {a.med_2q_err_pct:>6.3f}% | {a.p90_2q_err_pct:>6.2f}% | "
            f"{a.bad_2q_couplers_gt2pct:>6} | {a.med_ro_err_pct:>6.2f}% | {a.p90_ro_err_pct:>6.2f}% | "
            f"{a.med_t1_us:>4.0f} / {a.med_t2_us:<4.0f}"
        )
    lines.append(
        f"\n[4,000 shots/s Projection @ {proj.n_qubits}q, P={proj.n_params}] "
        f"QERD ({proj.qerd_shots:,} sh) = {proj.qerd_qpu_pulse_sec:.1f}s pure QPU pulse (~{proj.qerd_session_wall_est_min:.1f}m sync Session) "
        f"vs PSR equal-steps ({proj.psr_equal_steps_shots:,} sh) = {proj.psr_equal_steps_pulse_sec:.0f}s ({proj.psr_equal_steps_pulse_sec/3600:.2f} hrs) | "
        f"PSR @80k shots = {proj.psr_equal_shots_steps} steps"
    )
    return "\n".join(lines)


def make_summary_figure(
    scaling_rows: List[Dict[str, Any]],
    minvar_comp: Dict[str, Dict[str, float]],
    spinglass_comp: Dict[str, Dict[str, float]],
    proj_156: RuntimeProjection,
    out_png: str,
    mv_qubits: int | None = None,
    sg_qubits: int | None = None,
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

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.2))
    fig.suptitle(
        "QERD on IBM Heron R2 (4,000 shots/s): O(1) Circuit Scaling, S&P 500 Profit & Heavy-Hex Spin Glass",
        fontsize=11.5, fontweight="bold", y=0.985,
    )

    # Panel (a): Clean single-axis QPU Circuits/Sample + 2x T4 GPU OOM Zone
    ax = axes[0]
    hw_nq = int(proj_156.n_qubits)
    nq_list = sorted({8, 12, 16, 24, 32, 64, 100, hw_nq, 156})
    nq_q = np.array(nq_list, dtype=np.float64)
    psr_circs = 2.0 * (2.128 * nq_q)
    hw_idx = nq_list.index(hw_nq)
    psr_circs[hw_idx] = float(2 * proj_156.n_params)
    ax.plot(nq_q, psr_circs, "s--", color="#CC79A7", lw=2.1, ms=5.5, label=f"PSR Finite-Diff (2P = {2*proj_156.n_params} @{hw_nq}q)")
    ax.plot(nq_q, np.full_like(nq_q, 2.0), "^-", color="#0072B2", lw=2.6, ms=6.8, label="QERD QPU (Ours, O(1) = 2-3 fixed)")
    ax.axvspan(25.5, 195.0, color="#D55E00", alpha=0.09)
    ax.axvline(25.5, color="#D55E00", ls=":", lw=1.6)
    ax.text(27.5, 42.0, "2x T4 GPU OOM (> ~26q)\nExact 2^N Statevector > 32 GB", fontsize=7.8, fontweight="bold", color="#B23A00")
    if scaling_rows:
        r0, r1 = scaling_rows[0], scaling_rows[-1]
        vram_note = (
            f"Exact 2^N Sim VRAM:\n"
            f"  {r0['n_qubits']}q: {r0['sv_mb_noisy_batch']:.2f} MB\n"
            f"  {r1['n_qubits']}q: {r1['sv_mb_noisy_batch']:.2f} MB\n"
            f"  26q: >32 GB (OOM)"
        )
        ax.text(8.1, 4.6, vram_note, fontsize=7.0, color="#333333",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF6F0", edgecolor="#D55E00", alpha=0.94))
    ax.annotate(f"{2*proj_156.n_params} circs/sample\n({proj_156.psr_equal_steps_pulse_sec/3600:.2f}h @ 4k/s)",
                xy=(hw_nq, 2 * proj_156.n_params), xytext=(max(48, hw_nq * 0.48), 820),
                fontsize=7.3, fontweight="bold", color="#9A3B6E",
                arrowprops=dict(arrowstyle="->", color="#CC79A7", lw=1.1))
    ax.annotate(f"O(1) circs/sample\n({proj_156.qerd_qpu_pulse_sec:.1f}s pulse @ {proj_156.qerd_shots//1000}k sh)",
                xy=(hw_nq, 2.0), xytext=(max(48, hw_nq * 0.46), 3.6),
                fontsize=7.3, fontweight="bold", color="#005587",
                arrowprops=dict(arrowstyle="->", color="#0072B2", lw=1.1))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(7.2, 185.0)
    ax.set_ylim(1.2, 2200.0)
    tick_vals = sorted({8, 12, 16, 24, 32, 64, hw_nq, 156})
    ax.set_xticks(tick_vals)
    ax.set_xticklabels([str(x) for x in tick_vals])
    ax.minorticks_off()
    ax.set_xlabel("Qubits N (log scale)")
    ax.set_ylabel("QPU Circuits / Sample (log scale)")
    ax.set_title("(a) O(1) QPU Circuits vs. 2^N GPU Wall")
    ax.legend(fontsize=7.4, loc="upper left", framealpha=0.95)
    ax.grid(True, ls="--", alpha=0.25)

    # Panel (b): S&P 500 Unseen Market Profit ($10k Capital) + Exact QUBO Cost
    ax = axes[1]
    order_mv = [
        ("buy_hold_1N", "Buy&Hold\n(1/N Base)", "#4D4D4D"),
        ("untrained_qpu", "Untrained\nQPU", "#999999"),
        ("untrained_baseline", "Random\nBaseline", "#999999"),
        ("psr_qpu", "Equal-Shot\nPSR", "#CC79A7"),
        ("spsa_qpu", "Equal-Shot\nSPSA", "#E69F00"),
        ("qerd_qpu", "QERD QPU\n(Ours)", "#0072B2"),
        ("sa_ceiling", "SA Target\n(Classical)", "#009E73"),
    ]
    rows_mv = [(k, lbl, col) for k, lbl, col in order_mv if k in minvar_comp]
    profits = [minvar_comp[k]["oos_ens_profit_usd"] for k, _, _ in rows_mv]
    costs = [minvar_comp[k]["mean_qubo_cost"] for k, _, _ in rows_mv]
    xb = np.arange(len(rows_mv))
    bars = ax.bar(xb, profits, width=0.58, color=[c for _, _, c in rows_mv], edgecolor="#222222", linewidth=0.9)
    if "qerd_qpu" in minvar_comp:
        q_idx = [k for k, _, _ in rows_mv].index("qerd_qpu")
        bars[q_idx].set_linewidth(1.8)
    bh_val = minvar_comp.get("buy_hold_1N", {}).get("oos_ens_profit_usd", None)
    if bh_val is not None:
        ax.axhline(bh_val, color="#333333", ls="--", lw=1.2, label=f"Buy&Hold 1/N Base (+${bh_val:,.0f})")
    for b, prof, cost in zip(bars, profits, costs):
        p_txt = f"+${prof/1000:.1f}k" if prof >= 0 else f"-${abs(prof)/1000:.1f}k"
        ax.text(b.get_x() + b.get_width() / 2, max(prof, 0.0) + 220.0,
                f"{p_txt}\n(Cost {cost:+.3f})", ha="center", va="bottom", fontsize=6.6, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor="none", alpha=0.88), zorder=5)
    ax.set_xticks(xb)
    ax.set_xticklabels([l for _, l, _ in rows_mv], fontsize=7.3)
    ax.set_ylim(0.0, max(profits + [10000.0]) * 1.24)
    mv_tag = f"{mv_qubits}q, " if mv_qubits else ""
    ax.set_ylabel("Unseen 2022-25 Profit (USD, 10k Capital) ↑")
    ax.set_title(f"(b) S&P 500 Profit ({mv_tag}+$10k) & QUBO Cost")
    ax.legend(fontsize=7.3, loc="upper left", framealpha=0.95)
    ax.grid(axis="y", ls="--", alpha=0.25)

    # Panel (c): Native Heavy-Hex Spin Glass Satisfied Bonds % ONLY (clean single metric)
    ax = axes[2]
    order_sg = [
        ("untrained_qpu", "Untrained", "#999999"),
        ("untrained_baseline", "Random\nBase", "#999999"),
        ("psr_qpu", "Equal-Shot\nPSR", "#CC79A7"),
        ("spsa_qpu", "Equal-Shot\nSPSA", "#E69F00"),
        ("qerd_qpu", "QERD QPU\n(Ours)", "#0072B2"),
        ("sa_ceiling", "SA Target", "#009E73"),
    ]
    rows_sg = [(k, lbl, col) for k, lbl, col in order_sg if k in spinglass_comp]
    sat = [spinglass_comp[k]["satisfied_bonds_pct"] for k, _, _ in rows_sg]
    xc = np.arange(len(rows_sg))
    b1 = ax.bar(xc, sat, width=0.56, color=[c for _, _, c in rows_sg], edgecolor="#222222", linewidth=0.9)
    if "qerd_qpu" in spinglass_comp:
        q_idx = [k for k, _, _ in rows_sg].index("qerd_qpu")
        b1[q_idx].set_linewidth(1.8)
    ax.axhline(50.0, color="#666666", ls="--", lw=1.0, zorder=1, label="Random Chance (50%)")
    for b, (k, _, _), v in zip(b1, rows_sg, sat):
        lbl_top = f"{v:.1f}%"
        if k == "qerd_qpu":
            uniq = 100.0 * spinglass_comp[k].get("unique_frac", 0.0)
            lbl_top = f"{v:.1f}%\n({uniq:.0f}% uniq)"
        y_txt = max(v + 2.0, 51.8) if 43.0 <= v <= 50.5 else v + 2.0
        ax.text(b.get_x() + b.get_width() / 2, y_txt, lbl_top,
                ha="center", va="bottom", fontsize=7.3, fontweight="bold")
    ax.set_xticks(xc)
    ax.set_xticklabels([l for _, l, _ in rows_sg], fontsize=7.6)
    ax.set_ylim(0.0, 114.0)
    sg_tag = f"{sg_qubits}q, " if sg_qubits else ""
    ax.set_ylabel("Satisfied Frustrated Bonds (%) ↑")
    ax.set_title(f"(c) Heavy-Hex Spin Glass ({sg_tag}0 SWAPs)")
    ax.legend(fontsize=7.3, loc="upper left", framealpha=0.95)
    ax.grid(axis="y", ls="--", alpha=0.25)

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def save_metrics_json(
    out_json: str,
    cfg_dict: Dict[str, Any],
    audits: List[HeronAudit],
    proj_156: RuntimeProjection,
    scaling_rows: List[Dict[str, Any]],
    minvar_comp: Dict[str, Any],
    spinglass_comp: Dict[str, Any],
    subgraph_audits: Optional[Dict[str, SubgraphAudit]] = None,
) -> None:
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    payload = {
        "config": cfg_dict,
        "heron_r2_audits": [asdict(a) for a in audits],
        "heron_156q_projection": asdict(proj_156),
        "t4_simulation_wall": scaling_rows,
        "sp500_minvar_qubo": minvar_comp,
        "heavy_hex_spinglass": spinglass_comp,
    }
    if subgraph_audits is not None:
        payload["subgraph_audits"] = {k: asdict(v) for k, v in subgraph_audits.items()}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
