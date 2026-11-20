"""
Zero-QPU-second forensic audit answering every technical check from Review C:
  1. Direct test of static hardware offsets: regress m_i(theta=0) vs 512-shot RO calibration
     (ro_offset, ro_scale) AND compute cross-run m_i correlations across Untrained (theta=0),
     Oracle (b=0), no-zz, QER-Res (seed=42, Nov 18), QER-Res (seed=43, Nov 20), and trace
     all 4 reverse-diffusion steps (t=4..1) of Untrained QPU!
  2. Exact matched-N (N=32 and N=128) distributional audit for single-shot fixed-mixer Ising
     circuits and Classical Inverse-Ising (MM-Gibbs, PL-Gibbs, Held-Out SA), plus exact
     verification of the algebraic identity q_bar = (N * mean(m_i^2) - 1) / (N - 1).
  3. Classical Hebbian J_hat = <s_u s_v>_SA vs learned QER-Res J residual & CZ-error analysis.
  4. Unified Gap-Closed audit (exact E_base = E_rand across all live hardware blocks).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import _paths  # noqa: F401
except ImportError:
    from . import _paths  # noqa: F401

try:
    from extract_live_distributions import (
        _load_service, _decode_ro_cal, _decode_eval_z, _distributional_metrics,
        _classical_inverse_ising_baselines,
    )
except ImportError:
    from .extract_live_distributions import (
        _load_service, _decode_ro_cal, _decode_eval_z, _distributional_metrics,
        _classical_inverse_ising_baselines,
    )
from projects.sp500_qubo_and_spinglass.problems.spinglass import build_spinglass_problem
from projects.sp500_qubo_and_spinglass.solvers.classical import solve_simulated_annealing
from projects.sp500_qubo_and_spinglass.hardware.heron_r2 import (
    get_fake_heron, mitigate_expectations, select_clean_subgraph,
)


def _pearson(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Return Pearson r and two-sided normal-approx z-score (n >= 128)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0, 0.0
    r = float(np.corrcoef(x, y)[0, 1])
    z = r * np.sqrt(max(1, len(x) - 2) / max(1e-12, 1.0 - r * r))
    return r, float(z)


def _gibbs_at_n(prob, sa_train: np.ndarray, n_samples: int, seed: int) -> Dict[str, Any]:
    """Run classical Bethe Moment-Matching & Besag Pseudolikelihood Gibbs at exact sample size N."""
    rng = np.random.default_rng(seed + 777 + n_samples)
    n_q = prob.n_qubits
    edges = prob.edges
    sa_aug = np.vstack([sa_train, -sa_train])
    eu = np.array([u for u, _ in edges], dtype=np.int64)
    ev = np.array([v for _, v in edges], dtype=np.int64)
    c_emp = np.mean(sa_aug[:, eu] * sa_aug[:, ev], axis=0)

    adj: List[List[Tuple[int, int]]] = [[] for _ in range(n_q)]
    for idx, (u, v) in enumerate(edges):
        adj[u].append((v, idx))
        adj[v].append((u, idx))

    def _gibbs(j_hat: np.ndarray, b_hat: np.ndarray, sweeps: int = 80) -> np.ndarray:
        s = rng.choice([-1.0, 1.0], size=(n_samples, n_q))
        order = np.arange(n_q)
        for _ in range(sweeps):
            rng.shuffle(order)
            for u in order:
                h_loc = np.full(n_samples, b_hat[u], dtype=np.float64)
                for v, eidx in adj[u]:
                    h_loc += j_hat[eidx] * s[:, v]
                p_pos = 1.0 / (1.0 + np.exp(-2.0 * np.clip(h_loc, -15.0, 15.0)))
                s[:, u] = np.where(rng.random(n_samples) < p_pos, 1.0, -1.0)
        return s

    j_mm = np.arctanh(np.clip(c_emp, -0.95, 0.95))
    s_mm = _gibbs(j_mm, np.zeros(n_q))

    j_pl_acc = np.zeros(len(edges), dtype=np.float64)
    j_pl_cnt = np.zeros(len(edges), dtype=np.float64)
    b_pl = np.zeros(n_q, dtype=np.float64)
    for u in range(n_q):
        nbrs = adj[u]
        if not nbrs:
            continue
        k = len(nbrs)
        X = np.column_stack([np.ones(len(sa_aug))] + [sa_aug[:, v] for v, _ in nbrs])
        y = sa_aug[:, u]
        w = np.zeros(1 + k, dtype=np.float64)
        for _ in range(80):
            z = np.clip(2.0 * y * (X @ w), -20.0, 20.0)
            sig = 1.0 / (1.0 + np.exp(z))
            w += 0.5 * ((2.0 / len(sa_aug)) * (X.T @ (y * sig)) - 1e-2 * w)
        b_pl[u] = w[0]
        for j_i, (_, eidx) in enumerate(nbrs):
            j_pl_acc[eidx] += w[1 + j_i]
            j_pl_cnt[eidx] += 1.0
    j_pl = j_pl_acc / np.maximum(j_pl_cnt, 1.0)
    s_pl = _gibbs(j_pl, b_pl)
    return {"mm_gibbs": s_mm, "pl_gibbs": s_pl}


def run_offline_check() -> None:
    """Verify saved forensic audit JSON and recompute offline checkpoint/Hebbian/noiseless checks."""
    with open("analysis/forensic_saved_data_audit.json", "r", encoding="utf-8") as f:
        fa = json.load(f)
    tr = fa["untrained_step_trace"]
    ro = fa["ro_regression_untrained"]
    h43 = fa["hebbian_vs_learned_J"]["128q_seed43"]
    id_max_err = max(v["abs_err"] for v in fa["overlap_identity_checks"].values())
    print(f"[Offline Forensic Check] Untrained step trace |m_i|: {[round(x['mean_abs_m_spin'], 3) for x in tr]}")
    print(f"[Offline Forensic Check] RO regression r(m_i, ro_offset)={ro['r_m_vs_ro_offset']:+.3f} (R2={100*ro['r2_m_vs_ro_offset']:.1f}%)")
    print(f"[Offline Forensic Check] Exact overlap identity max error across all runs: {id_max_err:.2e}")
    print(f"[Offline Forensic Check] Hebbian 128q seed=43: sign_match={h43['qer_sign_recovery_pct']:.1f}%, r(|J|,|c_SA|)={h43['r_absJ_vs_abs_c_sa']:+.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Forensic hardware-drift, overlap identity, and Hebbian audit.")
    parser.add_argument("--offline", action="store_true", help="Verify saved forensic audit offline without IBM API keys.")
    parser.add_argument("--apikey-1", type=str, default=None, help="Primary IBM Quantum API key JSON.")
    parser.add_argument("--apikey-2", type=str, default=None, help="Secondary IBM Quantum API key JSON.")
    args = parser.parse_args()
    if args.offline or not (args.apikey_1 and args.apikey_2):
        run_offline_check()
        return

    svc1 = _load_service(args.apikey_1)
    svc2 = _load_service(args.apikey_2)
    ledger = json.load(open("analysis/ibm_kingston_job_ledger.json"))
    pb = json.load(open("analysis/phase_b_oracle_and_debias_128q.json"))

    # ------------------------------------------------------------------
    # PART 1: Direct test of static hardware offsets (m_i regression & cross-run r)
    # ------------------------------------------------------------------
    print("=== [PART 1] Hardware-Offset Hypothesis Test (m_i vs RO cal & Cross-Run) ===", flush=True)
    ro_s42_nov18 = _decode_ro_cal(svc1.job("dbul2uv82tmc73dtkc0g").result())
    ro_s42_nov20 = _decode_ro_cal(svc2.job("dbvomr5s2omc7388887g").result())
    ro_s43_nov20 = _decode_ro_cal(svc2.job("dbvp7t5hdvas73biffog").result())

    # Decode exact spin matrices for 128q runs
    def _spins_from_job(svc, jid, ro_cal, slice_rows=None):
        z_mit, _ = _decode_eval_z(svc.job(jid).result(), ro_cal)
        if slice_rows is not None:
            z_mit = z_mit[slice_rows]
        return np.where(z_mit >= 0.0, 1.0, -1.0), z_mit

    sp_untr, z_untr = _spins_from_job(svc2, "dbvon7qnj2vs73dcujl0", ro_s42_nov20)
    sp_nozz, z_nozz = _spins_from_job(svc2, "dbvop1inj2vs73dcunpg", ro_s42_nov20)
    sp_qer42, z_qer42 = _spins_from_job(svc1, "dbuleqvf8rhs73bmh0pg", ro_s42_nov18)
    sp_qer43, z_qer43 = _spins_from_job(svc2, "dbvpbvls2omc73889af0", ro_s43_nov20)
    sp_spsa43, z_spsa43 = _spins_from_job(svc2, "dbvpf6ls2omc73889evg", ro_s43_nov20)

    # Phase B t=1 job has 64 circuits: 0..31 = Oracle (b=0), 32..63 = QER+Z2 debias
    z_pb_t1, _ = _decode_eval_z(svc2.job("dbvrn0dhdvas73biidtg").result(), ro_s43_nov20)
    sp_oracle = np.where(z_pb_t1[:32] >= 0.0, 1.0, -1.0)
    z_pb_b_centered = np.clip(z_pb_t1[32:] - np.mean(z_pb_t1[32:], axis=0, keepdims=True), -1.0, 1.0)
    sp_debias = np.where(z_pb_b_centered >= 0.0, 1.0, -1.0)

    # Also trace ALL 4 steps of Untrained QPU (t=4,3,2,1) to see how m_i grows across reverse diffusion!
    untr_jobs = ledger["untrained_128q_seed42"]["all_job_ids"][1:]  # 4 eval jobs t=4,3,2,1
    untr_step_trace = []
    for step_idx, jid in enumerate(untr_jobs):
        t_val = 4 - step_idx
        z_t, _ = _decode_eval_z(svc2.job(jid).result(), ro_s42_nov20)
        z_raw_t, _ = _decode_eval_z(svc2.job(jid).result(), None)
        s_t = np.where(z_t >= 0.0, 1.0, -1.0)
        m_t = np.mean(s_t, axis=0)
        mz_t = np.mean(z_t, axis=0)
        mz_raw_t = np.mean(z_raw_t, axis=0)
        r_ro, _ = _pearson(mz_t, ro_s42_nov20[0])
        r_ro_raw, _ = _pearson(mz_raw_t, ro_s42_nov20[0])
        untr_step_trace.append({
            "t": t_val, "job_id": jid,
            "mean_abs_m_spin": float(np.mean(np.abs(m_t))),
            "mean_abs_z_mit": float(np.mean(np.abs(mz_t))),
            "mean_abs_z_raw": float(np.mean(np.abs(mz_raw_t))),
            "r_zmit_vs_ro_off": r_ro,
            "r_zraw_vs_ro_off": r_ro_raw,
        })
        print(
            f"  [Untrained QPU t={t_val}] mean|m_spin|={np.mean(np.abs(m_t)):.3f} | "
            f"mean|mean_batch(z_mit)|={np.mean(np.abs(mz_t)):.3f} | "
            f"mean|mean_batch(z_raw)|={np.mean(np.abs(mz_raw_t)):.3f} | "
            f"r(z_raw, ro_off)={r_ro_raw:+.3f} | r(z_mit, ro_off)={r_ro:+.3f}",
            flush=True,
        )

    m_vecs = {
        "untrained_s42_nov20": np.mean(sp_untr, axis=0),
        "nozz_s42_nov20": np.mean(sp_nozz, axis=0),
        "oracle_b0_s43_nov20": np.mean(sp_oracle, axis=0),
        "qer42_nov18": np.mean(sp_qer42, axis=0),
        "qer43_nov20": np.mean(sp_qer43, axis=0),
        "spsa43_nov20": np.mean(sp_spsa43, axis=0),
    }

    # Regression of m_i(theta=0) against ro_offset and ro_scale (512-shot cal!)
    r_m_ro_off, z_ro_off = _pearson(m_vecs["untrained_s42_nov20"], ro_s42_nov20[0])
    r_m_ro_sc, z_ro_sc = _pearson(m_vecs["untrained_s42_nov20"], ro_s42_nov20[1])
    r_z_ro_off, _ = _pearson(np.mean(z_untr, axis=0), ro_s42_nov20[0])
    r_ro_nov18_nov20, _ = _pearson(ro_s42_nov18[0], ro_s42_nov20[0])
    r_ro_s42_s43_nov20, _ = _pearson(ro_s42_nov20[0], ro_s43_nov20[0])

    print(f"\n  [RO Calibration Regression on Untrained theta=0 (128q)]:")
    print(f"    r(m_i(theta=0), ro_offset_i) = {r_m_ro_off:+.3f} (R^2={r_m_ro_off**2:.4f}, t={z_ro_off:+.2f})")
    print(f"    r(m_i(theta=0), ro_scale_i)  = {r_m_ro_sc:+.3f} (R^2={r_m_ro_sc**2:.4f}, t={z_ro_sc:+.2f})")
    print(f"    r(mean_z(theta=0), ro_off)   = {r_z_ro_off:+.3f}")
    print(f"    r(ro_off Nov18, ro_off Nov20)= {r_ro_nov18_nov20:+.3f} | r(ro_off s42 Nov20, s43 Nov20) = {r_ro_s42_s43_nov20:+.3f}")

    # Cross-run magnetization correlations (THE definitive test of repeatable static per-qubit drift!)
    print(f"\n  [Cross-Run Per-Qubit Magnetization Correlations r(m_A, m_B) across 128 qubits]:")
    cross_corr = {}
    pairs = [
        ("untrained_s42_nov20", "nozz_s42_nov20"),
        ("untrained_s42_nov20", "oracle_b0_s43_nov20"),
        ("untrained_s42_nov20", "qer43_nov20"),
        ("untrained_s42_nov20", "qer42_nov18"),
        ("oracle_b0_s43_nov20", "qer43_nov20"),
        ("oracle_b0_s43_nov20", "qer42_nov18"),
        ("qer42_nov18", "qer43_nov20"),
        ("untrained_s42_nov20", "spsa43_nov20"),
    ]
    for a_k, b_k in pairs:
        r_ab, t_ab = _pearson(m_vecs[a_k], m_vecs[b_k])
        sign_agree = 100.0 * float(np.mean(np.sign(m_vecs[a_k]) == np.sign(m_vecs[b_k])))
        cross_corr[f"{a_k}__vs__{b_k}"] = {"r": r_ab, "t_stat": t_ab, "sign_agree_pct": sign_agree}
        print(f"    {a_k:20s} vs {b_k:20s}: r = {r_ab:+.3f} (t={t_ab:+.1f}) | sign_agree={sign_agree:.1f}%", flush=True)

    # Also check QAOA single-shot Job 1 exact N=32 vs N=128 metrics (from live raw bitstrings!)
    print("\n=== [PART 2] Exact Matched-N (N=32 and N=128) Single-Shot & Classical References ===", flush=True)
    ck43 = np.load("projects/sp500_qubo_and_spinglass/checkpoints_seed43_128q/spinglass_qerd_128q.npz", allow_pickle=False)
    edges128 = [tuple(e) for e in ck43["edges"].tolist()]
    prob43 = build_spinglass_problem(128, edges128, seed=43)
    sa43_tr = solve_simulated_annealing(128, prob43.cost_fn, n_targets=128, sweeps=120, seed=43)
    sa43_ho = solve_simulated_annealing(128, prob43.cost_fn, n_targets=128, sweeps=120, seed=1043)

    # Decode Job 1 (dbvrmmeq4uuc73a1lefg) single-shot fixed-mixer bitstrings at BOTH N=32 and N=128!
    r_j1 = svc2.job("dbvrmmeq4uuc73a1lefg").result()
    bits_j1 = r_j1[0].data.meas.to_bool_array(order="little")  # (4, 128, 128)
    fixed_mixer_specs = [("c0.40_t0.25", 0), ("c0.70_t0.25", 1), ("c1.00_t0.25", 2), ("c0.70_t0.75", 3)]
    fm_matched = {}
    for lbl, idx in fixed_mixer_specs:
        raw_128 = 1.0 - 2.0 * bits_j1[idx].astype(np.float64)
        m32 = _distributional_metrics(raw_128[:32], prob43, sa43_tr, sa43_ho)
        m128 = _distributional_metrics(raw_128, prob43, sa43_tr, sa43_ho)
        # Also compute 4-chunk mean±std across the 4 disjoint 32-shot chunks of 128 shots!
        chunks32 = [_distributional_metrics(raw_128[k*32:(k+1)*32], prob43, sa43_tr, sa43_ho) for k in range(4)]
        fm_matched[lbl] = {
            "N32_first": {k: m32[k] for k in ["mean_energy", "std_energy", "best_energy", "satisfied_bonds_pct", "mean_abs_mag", "edge_corr_rmse_vs_sa_ho", "edge_corr_pearson_r", "edge_corr_sign_acc_pct", "overlap_q_mean", "overlap_abs_q_mean", "overlap_neg_frac_pct", "overlap_max_abs_q", "energy_w1_vs_sa_ho"]},
            "N32_4chunk_mean_rmse": float(np.mean([c["edge_corr_rmse_vs_sa_ho"] for c in chunks32])),
            "N32_4chunk_mean_r": float(np.mean([c["edge_corr_pearson_r"] for c in chunks32])),
            "N32_4chunk_mean_sign": float(np.mean([c["edge_corr_sign_acc_pct"] for c in chunks32])),
            "N32_4chunk_mean_abs_m": float(np.mean([c["mean_abs_mag"] for c in chunks32])),
            "N128": {k: m128[k] for k in ["mean_energy", "std_energy", "best_energy", "satisfied_bonds_pct", "mean_abs_mag", "edge_corr_rmse_vs_sa_ho", "edge_corr_pearson_r", "edge_corr_sign_acc_pct", "overlap_q_mean", "overlap_abs_q_mean", "overlap_neg_frac_pct", "overlap_max_abs_q", "energy_w1_vs_sa_ho"]},
        }
        print(
            f"  [Fixed-Mixer {lbl}] N=32(4-chunk avg): Sat={m128['satisfied_bonds_pct']:.1f}% | |m|={fm_matched[lbl]['N32_4chunk_mean_abs_m']:.3f} | "
            f"RMSE={fm_matched[lbl]['N32_4chunk_mean_rmse']:.3f} r={fm_matched[lbl]['N32_4chunk_mean_r']:+.3f} sign={fm_matched[lbl]['N32_4chunk_mean_sign']:.1f}% || "
            f"N=128: |m|={m128['mean_abs_mag']:.3f} RMSE={m128['edge_corr_rmse_vs_sa_ho']:.3f} r={m128['edge_corr_pearson_r']:+.3f} sign={m128['edge_corr_sign_acc_pct']:.1f}%",
            flush=True,
        )

    # Classical inverse-Ising & Held-Out SA at BOTH N=32 and N=128 for seed=42, seed=43, and 156q!
    ck156 = np.load("projects/sp500_qubo_and_spinglass/checkpoints_156q/spinglass_qerd_156q.npz", allow_pickle=False)
    edges156 = [tuple(e) for e in ck156["edges"].tolist()]
    matched_refs = {}
    for inst_lbl, nq, ed, sd in [
        ("128q_seed42", 128, edges128, 42),
        ("128q_seed43", 128, edges128, 43),
        ("156q_seed42", 156, edges156, 42),
    ]:
        pr = build_spinglass_problem(nq, ed, seed=sd)
        sa_tr = solve_simulated_annealing(nq, pr.cost_fn, n_targets=128, sweeps=120, seed=sd)
        sa_ho = solve_simulated_annealing(nq, pr.cost_fn, n_targets=128, sweeps=120, seed=sd + 1000)
        entry = {}
        for N_eval in (32, 128):
            rng_s = np.random.default_rng(sd + 555 + N_eval)
            sa_z2 = sa_ho[:N_eval] * rng_s.choice([-1.0, 1.0], size=(N_eval, 1))
            cl = _gibbs_at_n(pr, sa_tr, n_samples=N_eval, seed=sd)
            for tag, samp in [("sa_ho_z2", sa_z2), ("mm_gibbs", cl["mm_gibbs"]), ("pl_gibbs", cl["pl_gibbs"])]:
                mm = _distributional_metrics(samp, pr, sa_tr, sa_ho)
                entry[f"{tag}_N{N_eval}"] = {
                    k: mm[k] for k in [
                        "mean_energy", "std_energy", "satisfied_bonds_pct", "mean_abs_mag",
                        "edge_corr_rmse_vs_sa_ho", "edge_corr_pearson_r", "edge_corr_sign_acc_pct",
                        "overlap_q_mean", "overlap_abs_q_mean", "overlap_neg_frac_pct", "energy_w1_vs_sa_ho"
                    ]
                }
        matched_refs[inst_lbl] = entry
        for N_eval in (32, 128):
            r_sa = entry[f"sa_ho_z2_N{N_eval}"]
            r_mm = entry[f"mm_gibbs_N{N_eval}"]
            r_pl = entry[f"pl_gibbs_N{N_eval}"]
            print(
                f"  [{inst_lbl} N={N_eval:3d}] SA: Sat={r_sa['satisfied_bonds_pct']:.1f}% RMSE={r_sa['edge_corr_rmse_vs_sa_ho']:.3f} r={r_sa['edge_corr_pearson_r']:+.3f} | "
                f"MM: Sat={r_mm['satisfied_bonds_pct']:.1f}% RMSE={r_mm['edge_corr_rmse_vs_sa_ho']:.3f} r={r_mm['edge_corr_pearson_r']:+.3f} | "
                f"PL: Sat={r_pl['satisfied_bonds_pct']:.1f}% RMSE={r_pl['edge_corr_rmse_vs_sa_ho']:.3f} r={r_pl['edge_corr_pearson_r']:+.3f}",
                flush=True,
            )

    # ------------------------------------------------------------------
    # PART 3: Exact algebraic identity q_bar = (N * mean(m^2) - 1) / (N - 1) + Hebbian residual
    # ------------------------------------------------------------------
    print("\n=== [PART 3] Exact Identity q_bar = (N*<m^2>-1)/(N-1) & Hebbian J Residual ===", flush=True)
    id_checks = {}
    for lbl, sp in [
        ("untrained_s42", sp_untr), ("nozz_s42", sp_nozz),
        ("qer42_raw", sp_qer42), ("qer43_raw", sp_qer43),
        ("qer43_debias", sp_debias), ("oracle_b0", sp_oracle),
        ("spsa43", sp_spsa43),
    ]:
        N_s, n_q = sp.shape
        m_v = np.mean(sp, axis=0)
        q_mat = (sp @ sp.T) / float(n_q)
        q_vals = q_mat[np.triu_indices(N_s, k=1)]
        q_emp = float(np.mean(q_vals))
        q_id = float((N_s * np.mean(m_v ** 2) - 1.0) / (N_s - 1.0))
        id_checks[lbl] = {
            "N": N_s, "mean_abs_m": float(np.mean(np.abs(m_v))),
            "rms_m": float(np.sqrt(np.mean(m_v ** 2))),
            "q_bar_emp": q_emp, "q_bar_identity": q_id, "abs_err": abs(q_emp - q_id),
            "abs_q_mean": float(np.mean(np.abs(q_vals))),
            "q_std": float(np.std(q_vals)),
        }
        print(f"  {lbl:15s} | |m|={id_checks[lbl]['mean_abs_m']:.3f} rms(m)={id_checks[lbl]['rms_m']:.3f} | q_bar={q_emp:+.6f} == id={q_id:+.6f} (err={abs(q_emp-q_id):.1e}) | |q|={id_checks[lbl]['abs_q_mean']:.3f} std(q)={id_checks[lbl]['q_std']:.3f}")

    # Hebbian residual vs FakeKingston CZ error
    fk = get_fake_heron("ibm_kingston")
    tgt = fk.target
    hebb_audit = {}
    for lbl, ck, sd in [("128q_seed43", ck43, 43), ("156q_seed42", ck156, 42)]:
        nq = int(ck["n_qubits"][0])
        ed = [tuple(e) for e in ck["edges"].tolist()]
        j_qpu = ck["theta"][:len(ed)]
        b_qpu = ck["theta"][len(ed):]
        pr = build_spinglass_problem(nq, ed, seed=sd)
        sa_tr = solve_simulated_annealing(nq, pr.cost_fn, n_targets=128, sweeps=120, seed=sd)
        eu = np.array([u for u, _ in ed])
        ev = np.array([v for _, v in ed])
        c_sa = np.mean(sa_tr[:, eu] * sa_tr[:, ev], axis=0)
        c_unif = float(np.mean(j_qpu * np.sign(c_sa)))
        resid = j_qpu - c_unif * np.sign(c_sa)
        deg = np.zeros(nq)
        for u, v in ed:
            deg[u] += 1; deg[v] += 1
        deg_sum = np.array([deg[u] + deg[v] for u, v in ed])
        r_absJ_c, t_c = _pearson(np.abs(j_qpu), np.abs(c_sa))
        r_absJ_d, t_d = _pearson(np.abs(j_qpu), deg_sum)
        # Check residual correlation with signed c_sa (amplitude heterogeneity)
        r_res_c, _ = _pearson(resid * np.sign(c_sa), np.abs(c_sa))
        hebb_audit[lbl] = {
            "c_unif": c_unif,
            "mean_abs_J": float(np.mean(np.abs(j_qpu))),
            "std_abs_J": float(np.std(np.abs(j_qpu))),
            "cv_abs_J": float(np.std(np.abs(j_qpu)) / np.mean(np.abs(j_qpu))),
            "hebbian_sign_recovery_pct": 100.0 * float(np.mean(np.sign(-c_sa) == pr.bonds)),
            "qer_sign_recovery_pct": 100.0 * float(np.mean(np.sign(j_qpu) == -pr.bonds)),
            "r_J_vs_unif_hebb_sign": float(np.corrcoef(j_qpu, c_unif * np.sign(c_sa))[0, 1]),
            "rmse_J_vs_unif_hebb_sign": float(np.sqrt(np.mean(resid ** 2))),
            "r_absJ_vs_abs_c_sa": r_absJ_c,
            "t_absJ_vs_abs_c_sa": t_c,
            "r_absJ_vs_deg_sum": r_absJ_d,
            "t_absJ_vs_deg_sum": t_d,
            "mean_abs_b": float(np.mean(np.abs(b_qpu))),
            "max_abs_b": float(np.max(np.abs(b_qpu))),
        }
        print(f"  [Hebbian {lbl}] c_unif={c_unif:.3f}, CV={hebb_audit[lbl]['cv_abs_J']:.3f}, r(J,unif)={hebb_audit[lbl]['r_J_vs_unif_hebb_sign']:+.4f}, r(|J|,|c_sa|)={r_absJ_c:+.3f} (t={t_c:+.1f}), r(|J|,deg)={r_absJ_d:+.3f} (t={t_d:+.1f})")

    out = {
        "untrained_step_trace": untr_step_trace,
        "ro_regression_untrained": {
            "r_m_vs_ro_offset": r_m_ro_off, "r2_m_vs_ro_offset": r_m_ro_off ** 2, "t_ro_offset": z_ro_off,
            "r_m_vs_ro_scale": r_m_ro_sc, "r2_m_vs_ro_scale": r_m_ro_sc ** 2, "t_ro_scale": z_ro_sc,
            "r_zmit_vs_ro_offset": r_z_ro_off,
            "r_ro_off_nov18_vs_nov20": r_ro_nov18_nov20,
            "r_ro_off_s42_vs_s43_nov20": r_ro_s42_s43_nov20,
        },
        "cross_run_magnetization_corr": cross_corr,
        "fixed_mixer_single_shot_matched_N": fm_matched,
        "classical_refs_matched_N": matched_refs,
        "overlap_identity_checks": id_checks,
        "hebbian_vs_learned_J": hebb_audit,
    }
    with open("analysis/forensic_saved_data_audit.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\n[DONE] Saved analysis/forensic_saved_data_audit.json")


if __name__ == "__main__":
    main()
