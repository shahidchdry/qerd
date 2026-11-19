"""Command-line entrypoint for Project 1 (`sp500_qubo_and_spinglass`)."""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

if __package__ is None or __package__ == "":
    _pkg_parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _pkg_parent not in sys.path:
        sys.path.insert(0, _pkg_parent)
    __package__ = "projects.sp500_qubo_and_spinglass"

from qerd import QuantumIsingDenoiser, color_ising_edges  # noqa: E402
from .configs import PRESETS, RunConfig, build_run_config  # noqa: E402
from .loaders import ensure_sp500_dataset, fetch_sp500_dataset, load_sp500_split  # noqa: E402
from .hardware import audit_all_heron_r2, prepare_problem_backend, project_heron_timing  # noqa: E402
from .problems import build_minvar_problem, build_spinglass_problem  # noqa: E402
from .reporting import format_heron_comparison, make_summary_figure, save_metrics_json  # noqa: E402
from .solvers import (  # noqa: E402
    benchmark_t4_simulation_wall,
    greedy_minvar_portfolio,
    load_qerd_checkpoint,
    sample_qerd_batch,
    save_qerd_checkpoint,
    solve_simulated_annealing,
    train_coord_psr_equal_shot,
    train_psr_equal_shot,
    train_qerd_explicit,
    train_spsa_equal_shot,
)


def _eval_problem_equal_shot(
    problem_name: str,
    problem,
    x_tr: np.ndarray,
    x_sa_eval: np.ndarray,
    extra_classical: Dict[str, np.ndarray],
    edges: List[tuple[int, int]],
    model_kw: Dict,
    cfg: RunConfig,
    ckpt_path: str,
    stochastic_decode: bool,
    step_noise_scale: float,
    ro_cal: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Dict[str, Dict[str, float]]:
    comp: Dict[str, Dict[str, float]] = {}
    prob_qubits = int(getattr(problem, "n_qubits", cfg.n_qubits))
    is_live_qpu = cfg.mode == "ibm_qpu"

    with QuantumIsingDenoiser(
        data_shape=prob_qubits,
        cond_dim=0,
        edges=edges,
        n_layers=cfg.n_layers,
        diffusion_steps=cfg.diffusion_steps,
        cal_mode=cfg.cal_mode,
        init_gain=0.75,
        gain_ema=0.10,
        gain_bounds=(0.40, 1.00),
        seed=cfg.seed,
        **model_kw,
    ) as q_model:
        if cfg.skip_untrained:
            s_untrained = q_model.rng.choice([-1.0, 1.0], size=(cfg.eval_samples, prob_qubits))
            un_key = "untrained_baseline"
        else:
            if is_live_qpu:
                print(f"[{problem_name}] Sampling untrained QPU baseline ({cfg.eval_samples} samples, {prob_qubits}q)...", flush=True)
            s_untrained = sample_qerd_batch(
                q_model,
                n_samples=cfg.eval_samples,
                shots_per_step=cfg.eval_shots,
                step_noise_scale=step_noise_scale,
                sample_temp=cfg.sample_temp,
                stochastic_decode=stochastic_decode,
                ro_cal=ro_cal,
            )
            un_key = "untrained_qpu"
        un_cost = float(np.mean(problem.cost_fn(np.where(s_untrained >= 0.0, 1.0, -1.0))))
        sa_cost = float(np.mean(problem.cost_fn(x_sa_eval)))

        comp["sa_ceiling"] = problem.evaluate(x_sa_eval, un_cost, sa_cost)
        comp[un_key] = problem.evaluate(s_untrained, un_cost, sa_cost)

        for cname, csamp in extra_classical.items():
            comp[cname] = problem.evaluate(csamp, un_cost, sa_cost)

        if is_live_qpu:
            print(
                f"[{problem_name}] Training QERD ({cfg.nudge_mode}, {cfg.cal_mode}) on {cfg.device} "
                f"({cfg.max_shots:,} shot budget, {prob_qubits}q)...",
                flush=True,
            )
        t0 = time.perf_counter()
        qer_stats = train_qerd_explicit(
            q_model,
            x_tr,
            max_shots=cfg.max_shots,
            shots_per_run=cfg.shots_per_run,
            batch_size=cfg.batch_size,
            nudge_mode=cfg.nudge_mode,
            ro_cal=ro_cal,
            verbose=is_live_qpu,
        )
        t_qerd = time.perf_counter() - t0
        if is_live_qpu:
            print(f"[{problem_name}] Sampling trained QERD model ({cfg.eval_samples} samples, {prob_qubits}q)...", flush=True)
        s_qerd = sample_qerd_batch(
            q_model,
            n_samples=cfg.eval_samples,
            shots_per_step=cfg.eval_shots,
            step_noise_scale=step_noise_scale,
            sample_temp=cfg.sample_temp,
            stochastic_decode=stochastic_decode,
            ro_cal=ro_cal,
        )
        comp["qerd_qpu"] = problem.evaluate(s_qerd, un_cost, sa_cost)
        comp["qerd_qpu"]["wall_sec"] = t_qerd
        comp["qerd_qpu"]["nudge_align"] = qer_stats["mean_nudge_align"]
        comp["qerd_qpu"]["mean_gain"] = q_model.calibrator.mean_gain
        comp["qerd_qpu"]["shots_used"] = float(qer_stats["shots_used"])
        save_qerd_checkpoint(
            ckpt_path,
            q_model,
            edges,
            problem_name,
            tickers=getattr(problem, "tickers", None),
            ro_cal=ro_cal,
        )

    run_spsa = (cfg.mode == "local_sim") or (is_live_qpu and cfg.run_live_spsa)
    if run_spsa:
        with QuantumIsingDenoiser(
            data_shape=prob_qubits,
            cond_dim=0,
            edges=edges,
            n_layers=cfg.n_layers,
            diffusion_steps=cfg.diffusion_steps,
            cal_mode=cfg.cal_mode,
            init_gain=0.75,
            gain_ema=0.10,
            gain_bounds=(0.40, 1.00),
            seed=cfg.seed,
            **model_kw,
        ) as spsa_m:
            if is_live_qpu:
                print(f"[{problem_name}] Training equal-shot SPSA on {cfg.device} ({cfg.max_shots:,} shots, {prob_qubits}q)...", flush=True)
            train_spsa_equal_shot(
                spsa_m,
                x_tr,
                max_shots=cfg.max_shots,
                shots_per_run=cfg.shots_per_run,
                batch_size=cfg.batch_size,
                ro_cal=ro_cal,
                verbose=is_live_qpu,
            )
            s_spsa = sample_qerd_batch(
                spsa_m,
                n_samples=cfg.eval_samples,
                shots_per_step=cfg.eval_shots,
                step_noise_scale=step_noise_scale,
                sample_temp=cfg.sample_temp,
                stochastic_decode=stochastic_decode,
                ro_cal=ro_cal,
            )
            comp["spsa_qpu"] = problem.evaluate(s_spsa, un_cost, sa_cost)

    if cfg.mode == "local_sim":
        with QuantumIsingDenoiser(
            data_shape=prob_qubits,
            cond_dim=0,
            edges=edges,
            n_layers=cfg.n_layers,
            diffusion_steps=cfg.diffusion_steps,
            cal_mode=cfg.cal_mode,
            init_gain=0.75,
            gain_ema=0.10,
            gain_bounds=(0.40, 1.00),
            seed=cfg.seed,
            **model_kw,
        ) as psr_m:
            psr_info = train_psr_equal_shot(
                psr_m,
                x_tr,
                max_shots=cfg.max_shots,
                shots_per_run=cfg.shots_per_run,
                ro_cal=ro_cal,
            )
            s_psr = sample_qerd_batch(
                psr_m,
                n_samples=cfg.eval_samples,
                shots_per_step=cfg.eval_shots,
                step_noise_scale=step_noise_scale,
                sample_temp=cfg.sample_temp,
                stochastic_decode=stochastic_decode,
                ro_cal=ro_cal,
            )
            comp["psr_qpu"] = problem.evaluate(s_psr, un_cost, sa_cost)
            comp["psr_qpu"]["steps_executed"] = float(psr_info["steps"])

        with QuantumIsingDenoiser(
            data_shape=prob_qubits,
            cond_dim=0,
            edges=edges,
            n_layers=cfg.n_layers,
            diffusion_steps=cfg.diffusion_steps,
            cal_mode=cfg.cal_mode,
            init_gain=0.75,
            gain_ema=0.10,
            gain_bounds=(0.40, 1.00),
            seed=cfg.seed,
            **model_kw,
        ) as cpsr_m:
            cpsr_info = train_coord_psr_equal_shot(
                cpsr_m,
                x_tr,
                max_shots=cfg.max_shots,
                shots_per_shift=4,
                coords_per_step=min(8, max(2, cpsr_m.n_qpu_params // 3)),
                ro_cal=ro_cal,
            )
            s_cpsr = sample_qerd_batch(
                cpsr_m,
                n_samples=cfg.eval_samples,
                shots_per_step=cfg.eval_shots,
                step_noise_scale=step_noise_scale,
                sample_temp=cfg.sample_temp,
                stochastic_decode=stochastic_decode,
                ro_cal=ro_cal,
            )
            comp["coord_psr_qpu"] = problem.evaluate(s_cpsr, un_cost, sa_cost)
            comp["coord_psr_qpu"]["steps_executed"] = float(cpsr_info["steps"])

    run_nozz = (cfg.mode == "local_sim") or (is_live_qpu and cfg.run_live_nozz)
    if run_nozz:
        with QuantumIsingDenoiser(
            data_shape=prob_qubits,
            cond_dim=0,
            edges=[],
            n_layers=cfg.n_layers,
            diffusion_steps=cfg.diffusion_steps,
            cal_mode=cfg.cal_mode,
            init_gain=0.75,
            gain_ema=0.10,
            gain_bounds=(0.40, 1.00),
            seed=cfg.seed,
            **model_kw,
        ) as nozz_m:
            if is_live_qpu:
                print(f"[{problem_name}] Training zero-ZZ (no-zz) ablation on {cfg.device} ({cfg.max_shots:,} shots, {prob_qubits}q)...", flush=True)
            train_qerd_explicit(
                nozz_m,
                x_tr,
                max_shots=cfg.max_shots,
                shots_per_run=cfg.shots_per_run,
                batch_size=cfg.batch_size,
                nudge_mode=cfg.nudge_mode,
                ro_cal=ro_cal,
                verbose=is_live_qpu,
            )
            s_nozz = sample_qerd_batch(
                nozz_m,
                n_samples=cfg.eval_samples,
                shots_per_step=cfg.eval_shots,
                step_noise_scale=step_noise_scale,
                sample_temp=cfg.sample_temp,
                stochastic_decode=stochastic_decode,
                ro_cal=ro_cal,
            )
            comp["nozz_qpu"] = problem.evaluate(s_nozz, un_cost, sa_cost)

    return comp


def _count_csv_tickers(csv_path: str) -> int:
    import csv

    with open(csv_path, "r", encoding="utf-8") as f:
        header = next(csv.reader(f))
    return max(1, len(header) - 1)


def _clear_dir_files(folder: str) -> int:
    if not os.path.isdir(folder):
        return 0
    removed = 0
    for name in os.listdir(folder):
        p = os.path.join(folder, name)
        if os.path.isfile(p):
            os.remove(p)
            removed += 1
    return removed


def run_benchmark(cfg: RunConfig, clear_outputs: bool = False) -> None:
    if clear_outputs:
        n1 = _clear_dir_files(cfg.artifact_dir)
        n2 = _clear_dir_files(cfg.ckpt_dir)
        print(f"[Clean] Cleared {n1} artifact(s) and {n2} checkpoint(s).", flush=True)
    os.makedirs(cfg.artifact_dir, exist_ok=True)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    if not cfg.skip_minvar:
        ensure_sp500_dataset(cfg.csv_path)

    audits = audit_all_heron_r2()
    proj_nq = cfg.n_qubits if cfg.n_qubits >= 64 else 128

    scaling_rows: List[Dict] = []
    if not cfg.skip_scaling:
        print(f"\n[Part A] Benchmarking Exact 2^N Statevector Simulation Wall ({cfg.scaling_qubits})...")
        scaling_rows = benchmark_t4_simulation_wall(
            cfg.scaling_qubits, n_layers=cfg.n_layers, batch_size=min(10, cfg.batch_size), shots_per_run=cfg.shots_per_run, seed=cfg.seed
        )
        for r in scaling_rows:
            print(
                f"  N={r['n_qubits']:2d}q (2^N={r['hilbert_dim']:6d}) | {r['batch_ms']:6.1f} ms/batch | "
                f"Batch RAM={r['sv_mb_noisy_batch']:6.2f} MB | QERD=2-3 vs PSR={r['psr_circs_per_sample']} circs/sample"
            )

    sg_edges, sg_mkw, sg_sub, sg_ro_cal, sg_phys = prepare_problem_backend(
        cfg.mode,
        cfg.device,
        cfg.n_qubits,
        instance=cfg.instance,
        execution_mode=cfg.execution_mode,
        seed=cfg.seed,
        api_key_json=cfg.api_key_json,
        readout_mitigation=cfg.readout_mitigation,
    )
    live_qpu = sg_mkw.get("device_kwargs", {}).get("backend") if cfg.mode == "ibm_qpu" else None

    mv_qubits = cfg.n_qubits if cfg.skip_minvar else min(cfg.n_qubits, _count_csv_tickers(cfg.csv_path))
    if mv_qubits == cfg.n_qubits or cfg.skip_minvar:
        mv_edges, mv_mkw, mv_sub, mv_ro_cal = sg_edges, sg_mkw, sg_sub, sg_ro_cal
    else:
        mv_edges, mv_mkw, mv_sub, mv_ro_cal, _ = prepare_problem_backend(
            cfg.mode,
            cfg.device,
            mv_qubits,
            instance=cfg.instance,
            execution_mode=cfg.execution_mode,
            seed=cfg.seed,
            api_key_json=cfg.api_key_json,
            qpu_backend=live_qpu,
            allowed_phys=sg_phys,
            readout_mitigation=cfg.readout_mitigation,
            parent_ro_cal=sg_ro_cal,
            parent_phys=sg_phys,
        )

    proj_hw = project_heron_timing(
        n_qubits=proj_nq,
        n_edges=len(sg_edges) if cfg.n_qubits >= 64 else 139,
        n_steps=cfg.batches,
        batch_size=cfg.batch_size,
        shots_per_run=cfg.shots_per_run,
        shots_per_sec=cfg.qpu_shots_per_sec,
    )
    print(format_heron_comparison(audits, proj_hw))

    print(
        f"\n[Clean Subgraph] Mode={cfg.mode} | Device={cfg.device} | Nudge={cfg.nudge_mode} | Cal={cfg.cal_mode} | L={cfg.n_layers}, D={cfg.diffusion_steps}\n"
        f"  S&P500   : {mv_qubits}q (Edges={len(mv_edges)}, RO>5%={mv_sub.bad_ro_gt5pct if mv_sub else 0}, CZ>2%={mv_sub.bad_cz_gt2pct if mv_sub else 0})\n"
        f"  SpinGlass: {cfg.n_qubits}q (Edges={len(sg_edges)}, Colors={len(color_ising_edges(sg_edges))}, "
        f"RO>5%={sg_sub.bad_ro_gt5pct if sg_sub else 0}, CZ>2%={sg_sub.bad_cz_gt2pct if sg_sub else 0}) | Budget={cfg.max_shots:,} shots/problem",
        flush=True,
    )

    # Problem 1: Real S&P 500 Markowitz QUBO (optional; skipped when --skip-minvar focuses QPU quota on Spin Glass)
    mv_comp: Dict[str, Dict[str, float]] = {}
    mv_ckpt = os.path.join(cfg.ckpt_dir, f"minvar_qerd_{mv_qubits}q.npz")
    if not cfg.skip_minvar:
        split = load_sp500_split(cfg.csv_path, mv_qubits, split_date=cfg.split_date)
        mv_prob = build_minvar_problem(split, risk_aversion=cfg.risk_aversion, card_penalty=cfg.card_penalty)
        mv_sa = solve_simulated_annealing(
            mv_qubits,
            mv_prob.cost_fn,
            n_targets=128 if mv_qubits > 16 else 64,
            sweeps=100 if mv_qubits > 16 else 80,
            seed=cfg.seed,
        )
        mv_greedy = greedy_minvar_portfolio(mv_prob.cov_train_ann, mv_prob.target_k)
        mv_bh = np.ones((1, mv_qubits), dtype=np.float64)
        mv_comp = _eval_problem_equal_shot(
            "minvar_qubo",
            mv_prob,
            x_tr=mv_sa,
            x_sa_eval=mv_sa,
            extra_classical={"buy_hold_1N": mv_bh, "greedy_minvar": mv_greedy},
            edges=mv_edges,
            model_kw=mv_mkw,
            cfg=cfg,
            ckpt_path=mv_ckpt,
            stochastic_decode=True,
            step_noise_scale=0.55,
            ro_cal=mv_ro_cal,
        )

        print(f"\n[Part B1] S&P 500 QUBO ({len(split.tickers)} real stocks, Train {split.train_dates[0]}..{split.train_dates[-1]}, Unseen {split.test_dates[0]}..{split.test_dates[-1]}):")
        print(f"  {'Solver':<18} | {'QUBO Cost ↓':>11} | {'Profit ($10k) ↑':>15} | {'vs Buy&Hold':>12} | {'OOS Vol ↓':>9} | {'Sharpe ↑':>8}")
        print("  " + "-" * 89)
        bh_prof = mv_comp.get("buy_hold_1N", {}).get("oos_ens_profit_usd", 0.0)
        for k, lbl in [("buy_hold_1N", "Buy&Hold 1/N (Base)"), ("untrained_qpu", "Untrained (th=0)"), ("untrained_baseline", "Untrained (rand)"), ("nozz_qpu", "QERD no-zz (0 ZZ)"), ("greedy_minvar", "Greedy Min-Var"), ("psr_qpu", "Equal-Shot PSR"), ("coord_psr_qpu", "Coord-PSR (4sh)"), ("spsa_qpu", "Equal-Shot SPSA"), ("qerd_qpu", "QERD QPU (Ours)"), ("sa_ceiling", "SA Target")]:
            if k in mv_comp:
                m = mv_comp[k]
                prof = m["oos_ens_profit_usd"]
                prof_s = f"+${prof:,.0f}" if prof >= 0 else f"-${abs(prof):,.0f}"
                diff = prof - bh_prof
                diff_s = "$0 (Base)" if k == "buy_hold_1N" else (f"+${diff:,.0f}" if diff >= 0 else f"-${abs(diff):,.0f}")
                print(f"  {lbl:<18} | {m['mean_qubo_cost']:>11.4f} | {prof_s:>15s} | {diff_s:>12s} | {m['oos_ens_vol_pct']:>8.2f}% | {m['oos_ens_sharpe']:>8.2f}")

    # Problem 2: Native Heavy-Hex Frustrated Spin Glass (Z2-symmetrized SA targets eliminate spurious h bias)
    sg_prob = build_spinglass_problem(cfg.n_qubits, sg_edges, seed=cfg.seed)
    sg_sa = solve_simulated_annealing(
        cfg.n_qubits,
        sg_prob.cost_fn,
        n_targets=128 if cfg.n_qubits > 16 else 64,
        sweeps=120 if cfg.n_qubits > 16 else 80,
        seed=cfg.seed,
    )
    sg_train_aug = np.vstack([sg_sa, -sg_sa])
    sg_ckpt = os.path.join(cfg.ckpt_dir, f"spinglass_qerd_{cfg.n_qubits}q.npz")
    sg_comp = _eval_problem_equal_shot(
        "heavy_hex_spinglass",
        sg_prob,
        x_tr=sg_train_aug,
        x_sa_eval=sg_sa,
        extra_classical={},
        edges=sg_edges,
        model_kw=sg_mkw,
        cfg=cfg,
        ckpt_path=sg_ckpt,
        stochastic_decode=False,
        step_noise_scale=0.25,
        ro_cal=sg_ro_cal,
    )

    print(f"\n[Part B2] Native Heavy-Hex Spin Glass ({cfg.n_qubits}q, |E|={len(sg_edges)}):")
    print(f"  {'Solver':<18} | {'Energy ↓':>9} | {'Sat Bonds ↑':>11} | {'Gap Closed ↑':>12} | {'Uniq %':>7}")
    print("  " + "-" * 69)
    for k, lbl in [("sa_ceiling", "SA Ceiling"), ("untrained_qpu", "Untrained (th=0)"), ("untrained_baseline", "Untrained (rand)"), ("nozz_qpu", "QERD no-zz (0 ZZ)"), ("psr_qpu", "Equal-Shot PSR"), ("coord_psr_qpu", "Coord-PSR (4sh)"), ("spsa_qpu", "Equal-Shot SPSA"), ("qerd_qpu", "QERD QPU (Ours)")]:
        if k in sg_comp:
            m = sg_comp[k]
            print(f"  {lbl:<18} | {m['mean_energy']:>9.2f} | {m['satisfied_bonds_pct']:>10.1f}% | {m['gap_closed_pct']:>11.1f}% | {100*m['unique_frac']:>6.1f}%")

    if live_qpu is not None and getattr(live_qpu, "service", None) is not None:
        try:
            rem = (live_qpu.service.usage() or {}).get("usage_remaining_seconds")
            if rem is not None:
                print(f"\n[IBM Runtime] Quota remaining after run: {rem}s")
        except Exception:
            pass

    out_png = os.path.join(cfg.artifact_dir, "fig_hardware_advantage.png")
    out_json = os.path.join(cfg.artifact_dir, "metrics.json")
    make_summary_figure(scaling_rows, mv_comp, sg_comp, proj_hw, out_png, mv_qubits=mv_qubits, sg_qubits=cfg.n_qubits)
    cfg_safe = cfg.to_dict()
    if cfg_safe.get("api_key_json"):
        cfg_safe["api_key_json"] = "<provided>"
    sub_audits = {k: v for k, v in [("minvar", mv_sub), ("spinglass", sg_sub)] if v is not None}
    save_metrics_json(out_json, cfg_safe, audits, proj_hw, scaling_rows, mv_comp, sg_comp, subgraph_audits=sub_audits or None)
    print(f"\n[Saved] Artifacts -> {out_png} & {out_json}")
    print(f"[Saved] Checkpoints -> {mv_ckpt} & {sg_ckpt}")


def run_infer(cfg: RunConfig, problem_kind: str = "minvar") -> None:
    prefix = "minvar_qerd_" if problem_kind == "minvar" else "spinglass_qerd_"
    ckpt_name = f"{prefix}{cfg.n_qubits}q.npz"
    ckpt_path = os.path.join(cfg.ckpt_dir, ckpt_name)
    if not os.path.exists(ckpt_path) and os.path.isdir(cfg.ckpt_dir):
        matches = sorted(f for f in os.listdir(cfg.ckpt_dir) if f.startswith(prefix) and f.endswith(".npz"))
        if matches:
            ckpt_path = os.path.join(cfg.ckpt_dir, matches[-1])
    ck = load_qerd_checkpoint(ckpt_path)
    if ck["n_qubits"] > 18 and cfg.mode != "ibm_qpu":
        raise ValueError(
            f"Checkpoint requires {ck['n_qubits']} qubits (exact 2^{ck['n_qubits']} classical simulation exceeds RAM); "
            "pass `--mode ibm_qpu --api-key-json <path>` to sample this checkpoint on IBM hardware."
        )
    ro_cal: Optional[Tuple[np.ndarray, np.ndarray]] = None
    if cfg.mode == "ibm_qpu":
        _, model_kw, _, ro_cal, _ = prepare_problem_backend(
            cfg.mode,
            cfg.device,
            ck["n_qubits"],
            instance=cfg.instance,
            execution_mode=cfg.execution_mode,
            seed=cfg.seed,
            api_key_json=cfg.api_key_json,
            readout_mitigation=cfg.readout_mitigation,
        )
    else:
        model_kw = {"p_depol": 0.02, "p_readout": 0.02}

    with QuantumIsingDenoiser(
        data_shape=ck["n_qubits"],
        cond_dim=0,
        edges=ck["edges"],
        n_layers=ck["n_layers"],
        diffusion_steps=ck["diffusion_steps"],
        cal_mode=ck["cal_mode"],
        init_gain=0.75,
        gain_ema=0.10,
        gain_bounds=(0.40, 1.00),
        seed=cfg.seed,
        **model_kw,
    ) as m:
        m.theta[:] = ck["theta"]
        m.calibrator.gain[:] = ck["calibrator_gain"]
        samples = sample_qerd_batch(
            m,
            n_samples=cfg.eval_samples,
            shots_per_step=cfg.eval_shots,
            step_noise_scale=0.55 if problem_kind == "minvar" else 0.25,
            sample_temp=cfg.sample_temp,
            stochastic_decode=(problem_kind == "minvar"),
            ro_cal=ro_cal,
        )

    if problem_kind == "minvar":
        split = load_sp500_split(cfg.csv_path, ck["n_qubits"], split_date=cfg.split_date)
        prob = build_minvar_problem(split, risk_aversion=cfg.risk_aversion, card_penalty=cfg.card_penalty)
        sa = solve_simulated_annealing(ck["n_qubits"], prob.cost_fn, n_targets=64, sweeps=80, seed=cfg.seed)
        res = prob.evaluate(samples, float(np.mean(prob.cost_fn(np.random.default_rng(cfg.seed).choice([-1.0, 1.0], size=(64, ck["n_qubits"]))))), float(np.mean(prob.cost_fn(sa))))
        x = 0.5 * (np.where(samples >= 0.0, 1.0, -1.0) + 1.0)
        w = np.mean(x / np.maximum(np.sum(x, axis=1, keepdims=True), 1.0), axis=0)
        top = [(split.tickers[i], round(float(w[i]) * 100.0, 1)) for i in np.argsort(-w)[:5]]
        print(f"[Infer MinVar] Cost={res['mean_qubo_cost']:.4f} | OOS Vol={res['oos_mean_vol_pct']:.2f}% | Top Weights={top}")
    else:
        prob = build_spinglass_problem(ck["n_qubits"], ck["edges"], seed=cfg.seed)
        sa = solve_simulated_annealing(ck["n_qubits"], prob.cost_fn, n_targets=64, sweeps=80, seed=cfg.seed)
        res = prob.evaluate(samples, 0.0, float(np.mean(prob.cost_fn(sa))))
        print(f"[Infer SpinGlass] Energy={res['mean_energy']:.2f} | SatBonds={res['satisfied_bonds_pct']:.1f}% | Unique={100*res['unique_frac']:.1f}%")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Project 1: QPU Hardware Advantage & 128q Heron R2 Suite")
    parser.add_argument("--preset", choices=list(PRESETS), default="standard")
    parser.add_argument("--mode", choices=["local_sim", "qiskit_sim", "ibm_qpu"], default=None)
    parser.add_argument("--device", choices=["ibm_kingston", "ibm_fez", "ibm_marrakesh"], default=None)
    parser.add_argument("--instance", type=str, default=None)
    parser.add_argument("--api-key-json", type=str, default=None)
    parser.add_argument("--clear-outputs", action="store_true", help="Clear existing artifact & checkpoint files before running")
    parser.add_argument("--execution-mode", choices=["job", "batch", "session"], default=None)
    parser.add_argument("--n-qubits", type=int, default=None)
    parser.add_argument("--batches", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--shots-per-run", type=int, default=None)
    parser.add_argument("--eval-samples", type=int, default=None)
    parser.add_argument("--cal-mode", choices=["inverse", "forward"], default=None)
    parser.add_argument("--nudge-mode", choices=["residual", "target"], default=None)
    parser.add_argument("--sample-temp", type=float, default=None)
    parser.add_argument("--no-readout-mitigation", action="store_true")
    parser.add_argument("--run-live-spsa", action="store_true")
    parser.add_argument("--run-live-nozz", action="store_true")
    parser.add_argument("--skip-untrained", action="store_true")
    parser.add_argument("--skip-minvar", action="store_true")
    parser.add_argument("--max-sim-qubits", type=int, default=None)
    parser.add_argument("--skip-scaling", action="store_true")
    parser.add_argument("--risk-aversion", type=float, default=None)
    parser.add_argument("--artifact-dir", type=str, default=None)
    parser.add_argument("--ckpt-dir", type=str, default=None)
    parser.add_argument("--audit-heron", action="store_true", help="Print Heron R2 hardware comparison only")
    parser.add_argument("--fetch-data", action="store_true", help="Download real S&P 500 CSV data")
    parser.add_argument("--infer", choices=["minvar", "spinglass"], default=None)
    args = parser.parse_args(argv)

    cfg = build_run_config(
        preset=args.preset,
        mode=args.mode,
        device=args.device,
        instance=args.instance,
        api_key_json=args.api_key_json,
        execution_mode=args.execution_mode,
        n_qubits=args.n_qubits,
        batches=args.batches,
        batch_size=args.batch_size,
        shots_per_run=args.shots_per_run,
        eval_samples=args.eval_samples,
        cal_mode=args.cal_mode,
        nudge_mode=args.nudge_mode,
        sample_temp=args.sample_temp,
        readout_mitigation=False if args.no_readout_mitigation else None,
        run_live_spsa=True if args.run_live_spsa else None,
        run_live_nozz=True if args.run_live_nozz else None,
        skip_untrained=True if args.skip_untrained else None,
        skip_minvar=True if args.skip_minvar else None,
        max_sim_qubits=args.max_sim_qubits,
        skip_scaling=True if args.skip_scaling else None,
        risk_aversion=args.risk_aversion,
        artifact_dir=args.artifact_dir,
        ckpt_dir=args.ckpt_dir,
    )

    if args.fetch_data:
        saved = fetch_sp500_dataset(cfg.csv_path)
        print(f"Downloaded S&P 500 dataset -> {saved}")
        return

    if args.audit_heron:
        audits = audit_all_heron_r2()
        proj = project_heron_timing(128, 139, cfg.batches, cfg.batch_size, cfg.shots_per_run, cfg.qpu_shots_per_sec)
        print(format_heron_comparison(audits, proj))
        return

    if args.infer:
        run_infer(cfg, problem_kind=args.infer)
        return

    run_benchmark(cfg, clear_outputs=args.clear_outputs)


if __name__ == "__main__":
    main()
