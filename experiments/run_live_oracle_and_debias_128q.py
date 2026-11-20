"""
Live 128-qubit `ibm_kingston` Oracle-Sign Controls and Batch-Centered Z2 Hardware-Drift
Debiasing Benchmark (produces `analysis/phase_b_oracle_and_debias_128q.json`).

Supports:
  - Offline verification of saved `analysis/phase_b_oracle_and_debias_128q.json`:
      python experiments/run_live_oracle_and_debias_128q.py --offline
  - Live execution on `ibm_kingston` (~10 QPU seconds across 5 packed SamplerV2 jobs):
      python experiments/run_live_oracle_and_debias_128q.py --apikey path/to/apikey.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict

import numpy as np

try:
    import _paths  # noqa: F401
except ImportError:
    from . import _paths  # noqa: F401

from qerd import QuantumIsingDenoiser
from projects.sp500_qubo_and_spinglass.problems.spinglass import build_spinglass_problem
from projects.sp500_qubo_and_spinglass.solvers.classical import solve_simulated_annealing
from projects.sp500_qubo_and_spinglass.hardware.heron_r2 import (
    mitigate_expectations,
    select_clean_subgraph,
)
try:
    from extract_live_distributions import _distributional_metrics
except ImportError:
    from .extract_live_distributions import _distributional_metrics

CKPT_43 = "projects/sp500_qubo_and_spinglass/checkpoints_seed43_128q/spinglass_qerd_128q.npz"
REF_EVAL_JOB_43 = "dbvpbvls2omc73889af0"


def run_offline_check() -> None:
    with open("analysis/phase_b_oracle_and_debias_128q.json", "r", encoding="utf-8") as f:
        pb = json.load(f)
    jids = [j["job_id"] for j in pb["job_ids"]]
    ora = pb["oracle_template_revdiff_128q_seed43"]
    deb = pb["qer_res_z2_debiased_128q_seed43"]
    fm = pb["qaoa_p4_oracle_single_shot"]["qaoa_p4_c1.00_t0.25"]
    print(f"[Offline Phase-B Audit] Verified 5 live ibm_kingston jobs: {jids}")
    print(f"  Single-Shot Fixed-Mixer (c=1.0, N=128): Sat={fm['satisfied_bonds_pct']:.1f}% (N=32: {fm['sat_first32_pct']:.1f}%), |m|={fm['mean_abs_mag']:.3f}")
    print(f"  Oracle-Sign Template (32 sh/step, b=0): Sat={ora['satisfied_bonds_pct']:.1f}%, |m|={ora['mean_abs_mag']:.3f}, r={ora['edge_corr_pearson_r']:+.3f}")
    print(f"  QER-Res + Batch-Z2 Debias (32 sh/step): Sat={deb['satisfied_bonds_pct']:.1f}%, |m|={deb['mean_abs_mag']:.3f}, r={deb['edge_corr_pearson_r']:+.3f}, sign={deb['edge_corr_sign_acc_pct']:.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or verify live 128q oracle controls & batch-Z2 debiasing.")
    parser.add_argument("--offline", action="store_true", help="Verify saved Phase-B JSON offline without IBM API key.")
    parser.add_argument("--apikey", type=str, default=None, help="Path to IBM Quantum API key JSON for live QPU submission.")
    args = parser.parse_args()

    if args.offline or not args.apikey:
        run_offline_check()
        return

    from qiskit_ibm_runtime import QiskitRuntimeService

    with open(args.apikey, "r", encoding="utf-8") as f:
        tok = json.load(f).get("apikey")
    svc = QiskitRuntimeService(channel="ibm_quantum_platform", token=tok)
    usg0 = svc.usage() or {}
    rem0 = usg0.get("usage_remaining_seconds", 0)
    print(f"[Phase B] Starting quota remaining: {rem0}s", flush=True)
    if rem0 is not None and rem0 < 8:
        raise RuntimeError(f"Insufficient remaining quota ({rem0}s < 8s).")

    ck = np.load(CKPT_43, allow_pickle=False)
    n_q = int(ck["n_qubits"][0])
    edges = [tuple(e) for e in ck["edges"].tolist()]
    theta_qer = ck["theta"].copy()
    cal_gain = ck["calibrator_gain"].copy()
    ro_cal = (ck["ro_offset"].copy(), ck["ro_scale"].copy())

    prob = build_spinglass_problem(n_q, edges, seed=43)
    sa_tr = solve_simulated_annealing(n_q, prob.cost_fn, n_targets=128, sweeps=120, seed=43)
    sa_ho = solve_simulated_annealing(n_q, prob.cost_fn, n_targets=128, sweeps=120, seed=1043)

    backend = svc.backend("ibm_kingston")
    edges_live, phys_layout, sub_audit = select_clean_subgraph(backend, n_qubits=n_q)
    if edges_live != edges:
        edges = edges_live
        prob = build_spinglass_problem(n_q, edges, seed=43)
        sa_tr = solve_simulated_annealing(n_q, prob.cost_fn, n_targets=128, sweeps=120, seed=43)
        sa_ho = solve_simulated_annealing(n_q, prob.cost_fn, n_targets=128, sweeps=120, seed=1043)

    model_kw = {
        "backend": "qiskit",
        "real_qpu": True,
        "device_name": "ibm_kingston",
        "device_kwargs": {
            "backend": backend,
            "initial_layout": phys_layout,
            "optimization_level": 1,
            "execution_mode": "job",
            "max_circuits_per_pub": 64,
            "max_pubs_per_job": 8,
        },
    }

    phase_b_out: Dict[str, object] = {
        "backend": "ibm_kingston",
        "n_qubits": n_q,
        "n_edges": len(edges),
        "seed": 43,
        "ref_qer_eval_job_id": REF_EVAL_JOB_43,
        "quota_start_sec": rem0,
        "job_ids": [],
    }

    with QuantumIsingDenoiser(
        data_shape=n_q, cond_dim=0, edges=edges, n_layers=4, diffusion_steps=4,
        cal_mode="inverse", init_gain=0.75, gain_ema=0.10, gain_bounds=(0.40, 1.00),
        seed=43, **model_kw,
    ) as qm:
        qb = qm.backend
        isa_qc, col_perm = qb._get_isa_template(n_layers=4)

        qaoa_specs = [
            ("qaoa_p4_c0.40_t0.25", 0.40, 0.25),
            ("qaoa_p4_c0.70_t0.25", 0.70, 0.25),
            ("qaoa_p4_c1.00_t0.25", 1.00, 0.25),
            ("qaoa_p4_c0.70_t0.75", 0.70, 0.75),
        ]
        h_qaoa = np.zeros((len(qaoa_specs), n_q), dtype=np.float64)
        j_qaoa = np.stack([-c * prob.bonds for _, c, _ in qaoa_specs], axis=0)
        tn_qaoa = np.array([tn for _, _, tn in qaoa_specs], dtype=np.float64)
        pmat_qaoa = qb._build_param_matrix(
            h_qaoa, j_qaoa, tn_qaoa, col_perm,
            n_layers=4, dt=0.42, temp_beta=1.25, shake_scale=2.2, shake_final_frac=0.70,
        )

        t0 = time.perf_counter()
        job1 = qb._sampler.run([(isa_qc, pmat_qaoa, 128)])
        jid1 = job1.job_id()
        res1 = job1.result()
        bits1 = qb._extract_bits_from_pub_result(res1[0])
        phase_b_out["job_ids"].append({"role": "qaoa_p4_oracle_uncond", "job_id": jid1, "wall_sec": round(time.perf_counter() - t0, 1)})

        qaoa_res = {}
        for idx, (lbl, c_val, tn_val) in enumerate(qaoa_specs):
            raw_spins = 1.0 - 2.0 * bits1[idx].astype(np.float64)
            m_all = _distributional_metrics(raw_spins, prob, sa_tr, sa_ho)
            m_32 = _distributional_metrics(raw_spins[:32], prob, sa_tr, sa_ho)
            m_all["c_scale"] = c_val
            m_all["t_norm"] = tn_val
            m_all["job_id"] = jid1
            m_all["sat_first32_pct"] = m_32["satisfied_bonds_pct"]
            qaoa_res[lbl] = m_all
        phase_b_out["qaoa_p4_oracle_single_shot"] = qaoa_res

        n_samp, shots_step = 32, 32
        rng = np.random.default_rng(43)
        x_t_oracle = np.clip(rng.normal(0.0, 0.90, size=(n_samp, n_q)), -1.0, 1.0)
        x_t_debias = x_t_oracle.copy()

        j_oracle = np.broadcast_to((-0.70 * prob.bonds)[None, :], (n_samp, len(edges))).copy()
        b_oracle = np.zeros((n_samp, n_q), dtype=np.float64)
        j_qer = np.broadcast_to(theta_qer[: len(edges)][None, :], (n_samp, len(edges))).copy()
        b_qer = np.broadcast_to(theta_qer[len(edges) :][None, :], (n_samp, n_q)).copy()
        j_both = np.concatenate([j_oracle, j_qer], axis=0)
        qm.calibrator.gain[:] = cal_gain

        completed_steps = 0
        for t in range(4, 0, -1):
            tau = t / 4.0
            lam = 0.28 + 0.52 * (1.0 - tau)
            h_both = np.concatenate([b_oracle + lam * x_t_oracle, b_qer + lam * x_t_debias], axis=0)
            tn_both = np.full(2 * n_samp, tau, dtype=np.float64)
            pmat = qb._build_param_matrix(
                h_both, j_both, tn_both, col_perm,
                n_layers=4, dt=0.42, temp_beta=1.25, shake_scale=2.2, shake_final_frac=0.70,
            )
            t0 = time.perf_counter()
            job_t = qb._sampler.run([(isa_qc, pmat, shots_step)])
            jid_t = job_t.job_id()
            bits_t = qb._extract_bits_from_pub_result(job_t.result()[0])
            z_raw = np.mean(1.0 - 2.0 * bits_t.astype(np.float64), axis=1)
            z_mit = mitigate_expectations(z_raw, ro_cal[0], ro_cal[1])

            x0_a = qm.calibrator.calibrate(t, z_mit[:n_samp])
            z_b = z_mit[n_samp:]
            z_b_centered = np.clip(z_b - np.mean(z_b, axis=0, keepdims=True), -1.0, 1.0)
            x0_b = qm.calibrator.calibrate(t, z_b_centered)

            if t > 1:
                a_next = qm.alpha_bar[t - 1]
                eps = rng.normal(0.0, 0.25, size=(n_samp, n_q))
                x_t_oracle = np.clip(math.sqrt(a_next) * x0_a + math.sqrt(1.0 - a_next) * eps, -1.0, 1.0)
                x_t_debias = np.clip(math.sqrt(a_next) * x0_b + math.sqrt(1.0 - a_next) * eps, -1.0, 1.0)
            else:
                x_t_oracle, x_t_debias = x0_a, x0_b
            completed_steps += 1
            phase_b_out["job_ids"].append({"role": f"revdiff_t{t}", "job_id": jid_t, "wall_sec": round(time.perf_counter() - t0, 1)})

        if completed_steps == 4:
            phase_b_out["oracle_template_revdiff_128q_seed43"] = _distributional_metrics(
                np.where(x_t_oracle >= 0.0, 1.0, -1.0), prob, sa_tr, sa_ho
            )
            phase_b_out["qer_res_z2_debiased_128q_seed43"] = _distributional_metrics(
                np.where(x_t_debias >= 0.0, 1.0, -1.0), prob, sa_tr, sa_ho
            )

    with open("analysis/phase_b_oracle_and_debias_128q.json", "w", encoding="utf-8") as f:
        json.dump(phase_b_out, f, indent=2)


if __name__ == "__main__":
    main()
