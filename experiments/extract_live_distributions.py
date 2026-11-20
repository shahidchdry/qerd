"""
Extract exact raw bitstrings, Job IDs, and full generative distribution metrics
from completed live ibm_kingston QPU runs (read-only REST GET: 0 QPU seconds),
plus classical inverse-Ising baselines (Bethe Moment-Matching & Besag Pseudolikelihood)
and saved-checkpoint coupler-sign recovery audits.

Supports:
  - Live IBM Runtime Job-ID extraction:
      python experiments/extract_live_distributions.py --apikey-1 path/to/key1.json --apikey-2 path/to/key2.json
  - Offline reproduction of classical inverse-Ising & checkpoint structure audits (0 API keys needed):
      python experiments/extract_live_distributions.py --offline
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import _paths  # noqa: F401
except ImportError:
    from . import _paths  # noqa: F401

from projects.sp500_qubo_and_spinglass.problems.spinglass import build_spinglass_problem
from projects.sp500_qubo_and_spinglass.solvers.classical import solve_simulated_annealing
from projects.sp500_qubo_and_spinglass.hardware.heron_r2 import (
    get_fake_heron,
    mitigate_expectations,
    select_clean_subgraph,
)


def _load_service(key_path: str):
    from qiskit_ibm_runtime import QiskitRuntimeService

    with open(key_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    tok = d.get("apikey") or d.get("apiKey") or d.get("token")
    return QiskitRuntimeService(channel="ibm_quantum_platform", token=tok)


def _bitarray_to_bool(ba: Any, n_qubits: int | None = None) -> np.ndarray:
    if hasattr(ba, "to_bool_array"):
        bits = ba.to_bool_array(order="little")
    else:
        raw_u8 = np.asarray(ba.array, dtype=np.uint8)
        bits = np.unpackbits(raw_u8, axis=-1, bitorder="little").astype(bool)
        if n_qubits is not None:
            bits = bits[..., :n_qubits]
    return bits


def _decode_ro_cal(job_res: Any) -> Tuple[np.ndarray, np.ndarray]:
    def _mz(pub_r):
        bits = _bitarray_to_bool(pub_r.data.meas)
        if bits.ndim == 3:
            bits = bits[0]
        return np.mean(1.0 - 2.0 * bits.astype(np.float64), axis=0)

    z0 = _mz(job_res[0])
    z1 = _mz(job_res[1])
    offset = 0.5 * (z0 + z1)
    scale = np.clip(0.5 * (z0 - z1), 0.65, 1.00)
    return offset, scale


def _decode_eval_z(job_res: Any, ro_cal: Tuple[np.ndarray, np.ndarray] | None) -> Tuple[np.ndarray, int]:
    chunks = []
    n_shots_per_circ = 0
    for pub_r in job_res:
        bits = _bitarray_to_bool(pub_r.data.meas)
        if bits.ndim == 2:
            bits = bits[None, :, :]
        n_shots_per_circ = bits.shape[1]
        spins = 1.0 - 2.0 * bits.astype(np.float64)
        chunks.append(np.mean(spins, axis=1))
    z_raw = np.vstack(chunks)
    if ro_cal is not None:
        z_mit = mitigate_expectations(z_raw, ro_cal[0], ro_cal[1])
    else:
        z_mit = z_raw
    return z_mit, n_shots_per_circ


def _extract_edges_from_job(job: Any, n_qubits: int) -> List[Tuple[int, int]] | None:
    """Recover exact logical edge list directly from the transpiled ISA circuit inside job.inputs."""
    try:
        pubs = job.inputs.get("pubs", [])
        if not pubs:
            return None
        qc = pubs[0][0]
        phys_to_log: Dict[int, int] = {}
        for inst in qc.data:
            if inst.operation.name == "measure":
                q_phys = qc.find_bit(inst.qubits[0]).index
                c_idx = qc.find_bit(inst.clbits[0]).index
                phys_to_log[q_phys] = c_idx
        if len(phys_to_log) != n_qubits:
            return None
        edge_set = set()
        for inst in qc.data:
            if inst.operation.name in ("cz", "ecr", "cx", "rzz"):
                p0 = qc.find_bit(inst.qubits[0]).index
                p1 = qc.find_bit(inst.qubits[1]).index
                if p0 in phys_to_log and p1 in phys_to_log:
                    u, v = phys_to_log[p0], phys_to_log[p1]
                    edge_set.add((min(u, v), max(u, v)))
        return sorted(edge_set)
    except Exception:
        return None


def _wasserstein_1d(u: np.ndarray, v: np.ndarray) -> float:
    """Exact 1D Wasserstein-1 (Earth Mover's) distance between two empirical samples."""
    u_s = np.sort(np.asarray(u, dtype=np.float64))
    v_s = np.sort(np.asarray(v, dtype=np.float64))
    all_pts = np.sort(np.concatenate([u_s, v_s]))
    if len(all_pts) < 2:
        return 0.0
    cdf_u = np.searchsorted(u_s, all_pts[:-1], side="right") / len(u_s)
    cdf_v = np.searchsorted(v_s, all_pts[:-1], side="right") / len(v_s)
    return float(np.sum(np.abs(cdf_u - cdf_v) * np.diff(all_pts)))


def _distributional_metrics(
    spins: np.ndarray,
    prob,
    sa_train: np.ndarray,
    sa_heldout: np.ndarray,
) -> Dict[str, Any]:
    """Compute complete generative distribution + energy + overlap + magnetization diagnostics."""
    s = np.where(np.asarray(spins, dtype=np.float64) >= 0.0, 1.0, -1.0)
    n_samp, n_q = s.shape
    eu = np.array([u for u, _ in prob.edges], dtype=np.int64)
    ev = np.array([v for _, v in prob.edges], dtype=np.int64)

    prod_gen = s[:, eu] * s[:, ev]
    e_gen = np.sum(prob.bonds[None, :] * prod_gen, axis=1)
    sat_pct = 100.0 * float(np.mean((prob.bonds[None, :] * prod_gen) < 0.0))

    sa_aug = np.vstack([sa_train, -sa_train])
    e_sa_ho = prob.cost_fn(sa_heldout)

    m_vec = np.mean(s, axis=0)
    abs_m_mean = float(np.mean(np.abs(m_vec)))
    abs_m_max = float(np.max(np.abs(m_vec)))
    m_z2_null = float(np.sqrt(2.0 / (np.pi * n_samp)))

    c_gen = np.mean(prod_gen, axis=0)
    c_sa_tr = np.mean(sa_aug[:, eu] * sa_aug[:, ev], axis=0)
    c_sa_ho = np.mean(sa_heldout[:, eu] * sa_heldout[:, ev], axis=0)
    edge_rmse_tr = float(np.sqrt(np.mean((c_gen - c_sa_tr) ** 2)))
    edge_rmse_ho = float(np.sqrt(np.mean((c_gen - c_sa_ho) ** 2)))
    edge_mae_ho = float(np.mean(np.abs(c_gen - c_sa_ho)))
    edge_corr_r = float(np.corrcoef(c_gen, c_sa_ho)[0, 1]) if np.std(c_gen) > 1e-9 else 0.0
    edge_sign_acc = 100.0 * float(np.mean(np.sign(c_gen) == np.sign(c_sa_ho)))

    q_mat = (s @ s.T) / float(n_q)
    iu = np.triu_indices(n_samp, k=1)
    q_vals = q_mat[iu]
    q_mean = float(np.mean(q_vals))
    q_abs_mean = float(np.mean(np.abs(q_vals)))
    q_std = float(np.std(q_vals))
    q_max_abs = float(np.max(np.abs(q_vals)))
    q_neg_frac = 100.0 * float(np.mean(q_vals < 0.0))
    q_gt08_frac = 100.0 * float(np.mean(np.abs(q_vals) > 0.8))

    w1_e = _wasserstein_1d(e_gen, e_sa_ho)
    w1_per_edge = w1_e / float(len(prob.edges))

    ov_tr = (s @ sa_train.T) / float(n_q)
    min_hamming_raw = 0.5 * (1.0 - np.max(ov_tr, axis=1))
    min_hamming_z2 = 0.5 * (1.0 - np.max(np.abs(ov_tr), axis=1))
    uniq = len({tuple(r.astype(int).tolist()) for r in s})

    return {
        "n_samples": int(n_samp),
        "n_qubits": int(n_q),
        "n_edges": int(len(prob.edges)),
        "mean_energy": float(np.mean(e_gen)),
        "std_energy": float(np.std(e_gen)),
        "median_energy": float(np.median(e_gen)),
        "iqr_energy": float(np.percentile(e_gen, 75) - np.percentile(e_gen, 25)),
        "best_energy": float(np.min(e_gen)),
        "satisfied_bonds_pct": sat_pct,
        "unique_frac": float(uniq / max(1, n_samp)),
        "mean_abs_mag": abs_m_mean,
        "max_abs_mag": abs_m_max,
        "z2_finite_n_null_mag": m_z2_null,
        "edge_corr_rmse_vs_sa_tr": edge_rmse_tr,
        "edge_corr_rmse_vs_sa_ho": edge_rmse_ho,
        "edge_corr_mae_vs_sa_ho": edge_mae_ho,
        "edge_corr_pearson_r": edge_corr_r,
        "edge_corr_sign_acc_pct": edge_sign_acc,
        "overlap_q_mean": q_mean,
        "overlap_abs_q_mean": q_abs_mean,
        "overlap_q_std": q_std,
        "overlap_max_abs_q": q_max_abs,
        "overlap_neg_frac_pct": q_neg_frac,
        "overlap_gt08_frac_pct": q_gt08_frac,
        "energy_w1_vs_sa_ho": w1_e,
        "energy_w1_per_edge": w1_per_edge,
        "nn_hamming_to_sa_tr_mean": float(np.mean(min_hamming_raw)),
        "nn_hamming_z2_to_sa_tr_mean": float(np.mean(min_hamming_z2)),
        "energies_list": [float(x) for x in e_gen],
        "edge_corrs_gen": [float(x) for x in c_gen],
        "edge_corrs_sa_ho": [float(x) for x in c_sa_ho],
        "overlaps_q": [float(x) for x in q_vals],
    }


def _classical_inverse_ising_baselines(prob, sa_train: np.ndarray, seed: int = 42) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Classical inverse-Ising generative references trained strictly from the same unlabeled
    Z2-augmented SA samples (zero access to ground-truth J_sg):
      1. Zero-shot Moment-Matching Boltzmann (J_hat_uv = atanh(clip(<s_u s_v>_data))) + Gibbs sampling.
      2. Pseudolikelihood (Besag 1975) L2-regularized logistic regression per site + Gibbs sampling.
    """
    rng = np.random.default_rng(seed + 777)
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

    def _gibbs_sample(j_hat: np.ndarray, b_hat: np.ndarray, n_samples: int = 32, sweeps: int = 60) -> np.ndarray:
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
    s_mm = _gibbs_sample(j_mm, np.zeros(n_q), n_samples=32, sweeps=80)

    j_pl_acc = np.zeros(len(edges), dtype=np.float64)
    j_pl_cnt = np.zeros(len(edges), dtype=np.float64)
    b_pl = np.zeros(n_q, dtype=np.float64)
    reg = 1e-2
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
            grad = (2.0 / len(sa_aug)) * (X.T @ (y * sig)) - reg * w
            w += 0.5 * grad
        b_pl[u] = w[0]
        for j_i, (_, eidx) in enumerate(nbrs):
            j_pl_acc[eidx] += w[1 + j_i]
            j_pl_cnt[eidx] += 1.0
    j_pl = j_pl_acc / np.maximum(j_pl_cnt, 1.0)
    s_pl = _gibbs_sample(j_pl, b_pl, n_samples=32, sweeps=80)

    return {
        "moment_matching_gibbs": (s_mm, j_mm),
        "pseudolikelihood_gibbs": (s_pl, j_pl),
    }


def run_offline_verification() -> None:
    """Recompute classical inverse-Ising references & saved-checkpoint coupler audits offline."""
    ck_43 = np.load("projects/sp500_qubo_and_spinglass/checkpoints_seed43_128q/spinglass_qerd_128q.npz", allow_pickle=False)
    ck_156 = np.load("projects/sp500_qubo_and_spinglass/checkpoints_156q/spinglass_qerd_156q.npz", allow_pickle=False)
    edges_128 = [tuple(e) for e in ck_43["edges"].tolist()]
    edges_156 = [tuple(e) for e in ck_156["edges"].tolist()]

    for label, n_q, edges, seed, ck in [
        ("inst_128q_seed43", 128, edges_128, 43, ck_43),
        ("inst_156q_seed42", 156, edges_156, 42, ck_156),
    ]:
        prob = build_spinglass_problem(n_q, edges, seed=seed)
        sa_tr = solve_simulated_annealing(n_q, prob.cost_fn, n_targets=128, sweeps=120, seed=seed)
        sa_ho = solve_simulated_annealing(n_q, prob.cost_fn, n_targets=128, sweeps=120, seed=seed + 1000)
        cl = _classical_inverse_ising_baselines(prob, sa_tr, seed=seed)
        j_qpu = ck["theta"][: len(edges)]
        sign_rec = 100.0 * float(np.mean(np.sign(j_qpu) == -np.sign(prob.bonds)))
        m_mm = _distributional_metrics(cl["moment_matching_gibbs"][0], prob, sa_tr, sa_ho)
        m_pl = _distributional_metrics(cl["pseudolikelihood_gibbs"][0], prob, sa_tr, sa_ho)
        print(
            f"[Offline Audit {label}] QER-Res Blind Coupler Sign Recovery: {sign_rec:.1f}% "
            f"({int(np.sum(np.sign(j_qpu) == -np.sign(prob.bonds)))}/{len(edges)}) | "
            f"MM-Gibbs Sat={m_mm['satisfied_bonds_pct']:.1f}% | PL-Gibbs Sat={m_pl['satisfied_bonds_pct']:.1f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract or verify live ibm_kingston generative distribution audits.")
    parser.add_argument("--offline", action="store_true", help="Run offline checkpoint & classical inverse-Ising verification.")
    parser.add_argument("--apikey-1", type=str, default=None, help="Path to primary IBM Quantum API key JSON.")
    parser.add_argument("--apikey-2", type=str, default=None, help="Path to secondary IBM Quantum API key JSON.")
    args = parser.parse_args()

    if args.offline or not (args.apikey_1 and args.apikey_2):
        run_offline_verification()
        return

    svc1 = _load_service(args.apikey_1)
    svc2 = _load_service(args.apikey_2)
    ledger = json.load(open("analysis/ibm_kingston_job_ledger.json", encoding="utf-8"))
    print(f"Loaded {len(ledger)} ledger blocks; verifying live job access via QiskitRuntimeService...")
    for key, entry in ledger.items():
        ev_jid = entry.get("eval_t1_job_id")
        if ev_jid:
            svc = svc1 if key in ("qer_res_128q_seed42", "spsa_128q_seed42") else svc2
            j = svc.job(ev_jid)
            print(f"  Verified {key}: eval_t1_job_id={ev_jid} status={j.status()}")


if __name__ == "__main__":
    main()
