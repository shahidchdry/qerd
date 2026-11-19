"""Command-line entrypoint for Project 2 (`dna_promoter_qerd`)."""

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
    __package__ = "projects.dna_promoter_qerd"

from qerd import QuantumIsingDenoiser, color_ising_edges  # noqa: E402
from .biology import (  # noqa: E402
    PromoterOracleNet,
    base_frequencies,
    compute_santalucia_dg37,
    evaluate_core8_sequences,
    evaluate_promoter_sequences,
    load_or_train_oracle,
    motif_profile_matrix,
)
from .biology.biophysics import (  # noqa: E402
    gc_percent,
    has_tss_initiator,
    has_upstream_promoter_box,
    max_homopolymer_run,
)
from .biology.oracle_cnn import predict_oracle_probs  # noqa: E402
from .configs import PRESETS, RunConfig, build_run_config  # noqa: E402
from .loaders import (  # noqa: E402
    core8_to_spins,
    ensure_promoter_datasets,
    fetch_promoter_datasets,
    load_promoter_datasets,
    seqs_to_spins,
    spins_to_core8,
    spins_to_seqs,
)
from .hardware import prepare_dna_backend  # noqa: E402
from .reporting import export_top_promoters_fasta, make_dna_showcase_figure, save_metrics_json  # noqa: E402
from .solvers import (  # noqa: E402
    benchmark_2n_simulation_wall,
    fit_core8_pssm,
    load_dna_checkpoint,
    sample_classical_dinuc_baseline,
    sample_dna_batch,
    save_dna_checkpoint,
    train_coord_psr_dna,
    train_psr_dna,
    train_qerd_dna,
    train_spsa_dna,
)


def _clear_dir_files(folder: str, keep_names: Tuple[str, ...] = ()) -> int:
    if not os.path.isdir(folder):
        return 0
    removed = 0
    for name in os.listdir(folder):
        if name in keep_names:
            continue
        p = os.path.join(folder, name)
        if os.path.isfile(p):
            os.remove(p)
            removed += 1
    return removed


def _filter_valid_targets(
    seqs: List[str],
    probs: np.ndarray,
    min_prob: float = 0.68,
    cond_mode: str = "any",
) -> List[str]:
    """Keep high-confidence promoters that satisfy class-matched biophysical & motif rules."""
    valid_top: List[str] = []
    prob_top: List[str] = []
    # TATA-box promoters are naturally AT-rich around -30; avoid high-GC thermodynamic bias that erases TATA boxes.
    gc_lo, gc_hi = (32.0, 68.0) if cond_mode == "tata" else (42.0, 80.0)
    dg_max = -74.0 if cond_mode == "tata" else -86.0
    for s, p in sorted(zip(seqs, probs), key=lambda z: -float(z[1])):
        if float(p) < min_prob:
            continue
        prob_top.append(s)
        box_ok = (
            has_upstream_promoter_box(s, "tata")
            if cond_mode == "tata"
            else (has_tss_initiator(s) or has_upstream_promoter_box(s, "nontata"))
        )
        if gc_lo <= gc_percent(s) <= gc_hi and compute_santalucia_dg37(s) <= dg_max and max_homopolymer_run(s) <= 5 and box_ok:
            valid_top.append(s)
    return valid_top if len(valid_top) >= 64 else prob_top


def _prepare_training_targets(
    ds,
    oracle_model: Optional[PromoterOracleNet],
    is_core8: bool,
    max_targets: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Build balanced conditional training set: 50% TATA (`[1,0]`) + 50% Non-TATA/GC (`[0,1]`)."""
    rng = np.random.default_rng(seed)
    tata_seqs = list(ds.train_tata_pos)
    nontata_seqs = list(ds.train_nontata_pos)

    if oracle_model is not None:
        pt = predict_oracle_probs(oracle_model, tata_seqs[:4000])
        pn = predict_oracle_probs(oracle_model, nontata_seqs[:4000])
        top_t = _filter_valid_targets(tata_seqs[:4000], pt, min_prob=0.68, cond_mode="tata")
        top_n = _filter_valid_targets(nontata_seqs[:4000], pn, min_prob=0.68, cond_mode="nontata")
        if len(top_t) >= 64:
            tata_seqs = top_t
        if len(top_n) >= 64:
            nontata_seqs = top_n

    half = max(32, min(max_targets // 2, len(tata_seqs), len(nontata_seqs)))
    pick_t = [tata_seqs[int(i)] for i in rng.choice(len(tata_seqs), size=half, replace=False)]
    pick_n = [nontata_seqs[int(i)] for i in rng.choice(len(nontata_seqs), size=half, replace=False)]

    chosen_seqs = pick_t + pick_n
    cond_tr = np.vstack([
        np.tile(np.array([1.0, 0.0]), (half, 1)),
        np.tile(np.array([0.0, 1.0]), (half, 1)),
    ])
    tags = ["tata"] * half + ["nontata"] * half

    perm = rng.permutation(len(chosen_seqs))
    chosen_seqs = [chosen_seqs[int(i)] for i in perm]
    cond_tr = cond_tr[perm]
    tags = [tags[int(i)] for i in perm]
    x_tr = core8_to_spins(chosen_seqs) if is_core8 else seqs_to_spins(chosen_seqs)
    return x_tr, cond_tr, chosen_seqs, tags


def _eval_cond_vectors(n_samples: int) -> Tuple[np.ndarray, List[str]]:
    half = n_samples // 2
    rem = n_samples - half
    c = np.vstack([
        np.tile(np.array([1.0, 0.0]), (half, 1)),
        np.tile(np.array([0.0, 1.0]), (rem, 1)),
    ])
    return c, ["tata"] * half + ["nontata"] * rem


def run_benchmark(cfg: RunConfig, clear_outputs: bool = False) -> None:
    if clear_outputs:
        n1 = _clear_dir_files(cfg.artifact_dir)
        n2 = _clear_dir_files(cfg.ckpt_dir, keep_names=("promoter_oracle_64bp.pt", "promoter_oracle_78bp.pt"))
        print(f"[Clean] Cleared {n1} artifact(s) and {n2} quantum checkpoint(s).", flush=True)
    os.makedirs(cfg.artifact_dir, exist_ok=True)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    is_core8 = cfg.n_qubits == 16
    oracle_len = 64 if is_core8 else cfg.seq_len
    ensure_promoter_datasets(cfg.data_dir)
    ds = load_promoter_datasets(cfg.data_dir, seq_len=oracle_len)

    oracle_ckpt = os.path.join(cfg.ckpt_dir, f"promoter_oracle_{oracle_len}bp.pt")
    t_or = time.perf_counter()
    oracle_model, oracle_metrics = load_or_train_oracle(
        ds, ckpt_path=oracle_ckpt, epochs=cfg.oracle_epochs, seed=cfg.seed
    )
    print(
        f"[Oracle CNN ({oracle_len}bp)] Ready in {time.perf_counter()-t_or:.1f}s | "
        f"Test AUC={oracle_metrics['test_auc']:.4f} | Test Acc={100*oracle_metrics['test_acc']:.1f}% | "
        f"Random Decoy Mean={oracle_metrics['random_mean_prob']:.3f}",
        flush=True,
    )

    scaling_rows: List[Dict] = []
    if not cfg.skip_scaling:
        print(f"\n[Part A] Exact 2^N Statevector Wall ({cfg.scaling_qubits})...")
        scaling_rows = benchmark_2n_simulation_wall(
            cfg.scaling_qubits, n_layers=cfg.n_layers, batch_size=min(10, cfg.batch_size), shots_per_run=24, seed=cfg.seed
        )
        for r in scaling_rows:
            print(f"  N={r['n_qubits']:2d}q (2^N={r['hilbert_dim']:6d}) | {r['batch_ms']:6.1f} ms/batch | Batch RAM={r['sv_mb_noisy_batch']:6.2f} MB")

    edges, model_kw, sub_rep, ro_cal = prepare_dna_backend(
        cfg.mode, cfg.device, cfg.n_qubits, instance=cfg.instance,
        execution_mode=cfg.execution_mode, seed=cfg.seed, api_key_json=cfg.api_key_json,
    )
    print(
        f"\n[DNA Topology] Mode={cfg.mode} | Device={cfg.device} | Qubits={sub_rep.n_qubits} ({cfg.seq_len} bp) | "
        f"Edges={sub_rep.native_edges} ({sub_rep.intra_base_coupled}/{sub_rep.total_bases} intra-base CZ) | "
        f"RO>5%={sub_rep.bad_ro_gt5pct} (p90={sub_rep.p90_ro_err_pct:.2f}%) | CZ>2%={sub_rep.bad_cz_gt2pct} (p90={sub_rep.p90_cz_err_pct:.2f}%) | "
        f"Train Budget={cfg.max_shots:,} shots ({cfg.batches}x{cfg.batch_size})",
        flush=True,
    )

    x_tr, cond_tr, chosen_train_seqs, _ = _prepare_training_targets(
        ds, oracle_model=oracle_model, is_core8=is_core8, max_targets=cfg.max_train_targets, seed=cfg.seed
    )
    train_set = set(ds.train_all_seqs) | set(ds.train_tata_pos)
    eval_conds, eval_tags = _eval_cond_vectors(cfg.eval_samples)
    is_live_qpu = cfg.mode == "ibm_qpu"

    real_eval_seqs = (ds.test_tata_pos[: cfg.eval_samples // 2] + ds.test_nontata_pos[: cfg.eval_samples - cfg.eval_samples // 2])
    if is_core8:
        real_eval_seqs_show = [s[18:22] + s[47:51] for s in real_eval_seqs]
        pssm = fit_core8_pssm(ds.train_tata_pos + ds.train_nontata_pos, [s for s, y in zip(ds.train_all_seqs, ds.train_all_labels) if y < 0.5])
        real_base_f = base_frequencies(real_eval_seqs_show)
    else:
        real_eval_seqs_show = real_eval_seqs
        pssm = None
        real_base_f = base_frequencies(ds.test_tata_pos[:200] + ds.test_nontata_pos[:200])

    def _eval_seqs(seqs: List[str], tags: Optional[List[str]] = None) -> Dict[str, float]:
        if is_core8 and pssm is not None:
            return evaluate_core8_sequences(seqs, pssm)
        probs = predict_oracle_probs(oracle_model, seqs)
        return evaluate_promoter_sequences(
            seqs, probs, train_seq_set=train_set, real_pos_freqs=real_base_f, cond_tags=tags
        )

    comp: Dict[str, Dict[str, float]] = {}
    profiles: Dict[str, Dict[str, np.ndarray]] = {}
    thermo_samples: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    seqs_store: Dict[str, List[str]] = {}

    def _record(key: str, seqs: List[str], tags: Optional[List[str]] = None) -> None:
        comp[key] = _eval_seqs(seqs, tags=tags)
        profiles[key] = motif_profile_matrix(seqs)
        thermo_samples[key] = (
            np.array([gc_percent(s) for s in seqs], dtype=np.float64),
            np.array([compute_santalucia_dg37(s) for s in seqs], dtype=np.float64),
        )
        seqs_store[key] = seqs

    _record("real_promoters", real_eval_seqs_show, tags=eval_tags[: len(real_eval_seqs_show)])
    if not is_core8:
        pt_te = predict_oracle_probs(oracle_model, ds.test_tata_pos[:400])
        pn_te = predict_oracle_probs(oracle_model, ds.test_nontata_pos[:400])
        ft_te = _filter_valid_targets(ds.test_tata_pos[:400], pt_te, min_prob=0.68, cond_mode="tata")
        fn_te = _filter_valid_targets(ds.test_nontata_pos[:400], pn_te, min_prob=0.68, cond_mode="nontata")
        filt_eval = ft_te[: cfg.eval_samples // 2] + fn_te[: cfg.eval_samples - cfg.eval_samples // 2]
        if len(filt_eval) == cfg.eval_samples:
            _record("real_promoters_filtered", filt_eval, tags=eval_tags)
    cl_seqs = sample_classical_dinuc_baseline(
        [s[18:22] + s[47:51] for s in chosen_train_seqs] if is_core8 else chosen_train_seqs,
        n_samples=cfg.eval_samples,
        seed=cfg.seed,
    )
    _record("classical_markov", cl_seqs, tags=eval_tags)

    q_ckpt = os.path.join(cfg.ckpt_dir, f"dna_qerd_{cfg.n_qubits}q.npz")
    with QuantumIsingDenoiser(
        data_shape=cfg.n_qubits, cond_dim=2, edges=edges, n_layers=cfg.n_layers,
        diffusion_steps=cfg.diffusion_steps, cal_mode=cfg.cal_mode,
        init_gain=0.75, gain_ema=0.10, gain_bounds=(0.40, 1.00), seed=cfg.seed, **model_kw,
    ) as q_model:
        if is_live_qpu:
            print(f"\n[Untrained Baseline] Sampling {cfg.eval_samples} sequences ({cfg.n_qubits}q) on {cfg.device}...", flush=True)
        un_spins = sample_dna_batch(
            q_model, eval_conds, cfg.eval_samples, shots_per_step=cfg.eval_shots,
            sample_temp=cfg.sample_temp, stochastic_decode=cfg.stochastic_decode, ro_cal=ro_cal,
        )
        un_seqs = spins_to_core8(un_spins) if is_core8 else spins_to_seqs(un_spins)
        _record("untrained_qpu", un_seqs, tags=eval_tags)

        if is_live_qpu:
            print(f"[QERD Train ({cfg.nudge_mode})] Running up to {cfg.max_shots:,} shots (sh={cfg.shots_per_run}) on {cfg.device}...", flush=True)
        qer_stats = train_qerd_dna(
            q_model, x_tr, cond_tr, max_shots=cfg.max_shots, shots_per_run=cfg.shots_per_run,
            batch_size=cfg.batch_size, nudge_mode=cfg.nudge_mode, ro_cal=ro_cal, verbose=is_live_qpu,
        )
        if is_live_qpu:
            print(f"[QERD Eval] Sampling {cfg.eval_samples} trained promoter designs on {cfg.device}...", flush=True)
        qd_spins = sample_dna_batch(
            q_model, eval_conds, cfg.eval_samples, shots_per_step=cfg.eval_shots,
            sample_temp=cfg.sample_temp, stochastic_decode=cfg.stochastic_decode, ro_cal=ro_cal,
        )
        qd_seqs = spins_to_core8(qd_spins) if is_core8 else spins_to_seqs(qd_spins)
        _record("qerd_qpu", qd_seqs, tags=eval_tags)
        comp["qerd_qpu"]["wall_sec"] = qer_stats["wall_sec"]
        comp["qerd_qpu"]["nudge_align"] = qer_stats["mean_nudge_align"]
        comp["qerd_qpu"]["mean_gain"] = q_model.calibrator.mean_gain
        save_dna_checkpoint(q_ckpt, q_model, edges, ro_cal=ro_cal)

    run_spsa = (cfg.mode == "local_sim") or (is_live_qpu and cfg.run_live_spsa)
    if run_spsa:
        with QuantumIsingDenoiser(
            data_shape=cfg.n_qubits, cond_dim=2, edges=edges, n_layers=cfg.n_layers,
            diffusion_steps=cfg.diffusion_steps, cal_mode=cfg.cal_mode,
            init_gain=0.75, gain_ema=0.10, gain_bounds=(0.40, 1.00), seed=cfg.seed, **model_kw,
        ) as spsa_m:
            if is_live_qpu:
                print(f"[SPSA Train] Running {cfg.batches} equal-shot jobs ({cfg.max_shots:,} shots) on {cfg.device}...", flush=True)
            train_spsa_dna(
                spsa_m, x_tr, cond_tr, max_shots=cfg.max_shots, shots_per_run=cfg.shots_per_run,
                batch_size=cfg.batch_size, ro_cal=ro_cal, verbose=is_live_qpu,
            )
            sp_spins = sample_dna_batch(
                spsa_m, eval_conds, cfg.eval_samples, shots_per_step=cfg.eval_shots,
                sample_temp=cfg.sample_temp, stochastic_decode=cfg.stochastic_decode, ro_cal=ro_cal,
            )
            sp_seqs = spins_to_core8(sp_spins) if is_core8 else spins_to_seqs(sp_spins)
            _record("spsa_qpu", sp_seqs, tags=eval_tags)

    if cfg.mode == "local_sim":
        with QuantumIsingDenoiser(
            data_shape=cfg.n_qubits, cond_dim=2, edges=edges, n_layers=cfg.n_layers,
            diffusion_steps=cfg.diffusion_steps, cal_mode=cfg.cal_mode,
            init_gain=0.75, gain_ema=0.10, gain_bounds=(0.40, 1.00), seed=cfg.seed, **model_kw,
        ) as psr_m:
            psr_info = train_psr_dna(psr_m, x_tr, cond_tr, max_shots=cfg.max_shots, shots_per_run=cfg.shots_per_run)
            pr_spins = sample_dna_batch(
                psr_m, eval_conds, cfg.eval_samples, shots_per_step=cfg.eval_shots,
                sample_temp=cfg.sample_temp, stochastic_decode=cfg.stochastic_decode, ro_cal=ro_cal,
            )
            pr_seqs = spins_to_core8(pr_spins) if is_core8 else spins_to_seqs(pr_spins)
            _record("psr_qpu", pr_seqs, tags=eval_tags)
            comp["psr_qpu"]["steps_executed"] = float(psr_info["steps"])

        with QuantumIsingDenoiser(
            data_shape=cfg.n_qubits, cond_dim=2, edges=edges, n_layers=cfg.n_layers,
            diffusion_steps=cfg.diffusion_steps, cal_mode=cfg.cal_mode,
            init_gain=0.75, gain_ema=0.10, gain_bounds=(0.40, 1.00), seed=cfg.seed, **model_kw,
        ) as cpsr_m:
            cpsr_info = train_coord_psr_dna(
                cpsr_m, x_tr, cond_tr, max_shots=cfg.max_shots, shots_per_shift=4,
                coords_per_step=min(8, max(2, cpsr_m.n_qpu_params // 4)), ro_cal=ro_cal,
            )
            cpr_spins = sample_dna_batch(
                cpsr_m, eval_conds, cfg.eval_samples, shots_per_step=cfg.eval_shots,
                sample_temp=cfg.sample_temp, stochastic_decode=cfg.stochastic_decode, ro_cal=ro_cal,
            )
            cpr_seqs = spins_to_core8(cpr_spins) if is_core8 else spins_to_seqs(cpr_spins)
            _record("coord_psr_qpu", cpr_seqs, tags=eval_tags)
            comp["coord_psr_qpu"]["steps_executed"] = float(cpsr_info["steps"])

    run_nozz = (cfg.mode == "local_sim") or (is_live_qpu and cfg.run_live_nozz)
    if run_nozz:
        with QuantumIsingDenoiser(
            data_shape=cfg.n_qubits, cond_dim=2, edges=[], n_layers=cfg.n_layers,
            diffusion_steps=cfg.diffusion_steps, cal_mode=cfg.cal_mode,
            init_gain=0.75, gain_ema=0.10, gain_bounds=(0.40, 1.00), seed=cfg.seed, **model_kw,
        ) as nozz_m:
            if is_live_qpu:
                print(f"[No-ZZ Ablation] Training zero-ZZ QPU control ({cfg.max_shots:,} shots) on {cfg.device}...", flush=True)
            train_qerd_dna(
                nozz_m, x_tr, cond_tr, max_shots=cfg.max_shots, shots_per_run=cfg.shots_per_run,
                batch_size=cfg.batch_size, nudge_mode=cfg.nudge_mode, ro_cal=ro_cal, verbose=is_live_qpu,
            )
            nz_spins = sample_dna_batch(
                nozz_m, eval_conds, cfg.eval_samples, shots_per_step=cfg.eval_shots,
                sample_temp=cfg.sample_temp, stochastic_decode=cfg.stochastic_decode, ro_cal=ro_cal,
            )
            nz_seqs = spins_to_core8(nz_spins) if is_core8 else spins_to_seqs(nz_spins)
            _record("nozz_qpu", nz_seqs, tags=eval_tags)

    print(f"\n[Part B] Human Promoter DNA Design Results ({cfg.n_qubits}q / {cfg.seq_len} bp, {cfg.max_shots:,} shots):")
    print(f"  {'Model':<18} | {'Activity ↑':>10} | {'Valid % ↑':>9} | {'Inr TSS ↑':>9} | {'GC %':>6} | {'dG37 ↓':>8} | {'Uniq %':>7} | {'Novel %':>7}")
    print("  " + "-" * 94)
    for k, lbl in [
        ("untrained_qpu", "Untrained QPU"),
        ("nozz_qpu", "QERD no-zz (0 ZZ)"),
        ("psr_qpu", "Equal-Shot PSR"),
        ("coord_psr_qpu", "Coord-PSR (4sh)"),
        ("spsa_qpu", "Equal-Shot SPSA"),
        ("classical_markov", "Classical Markov"),
        ("qerd_qpu", "QERD QPU (Ours)"),
        ("real_promoters", "Real Human DNA"),
        ("real_promoters_filtered", "Real DNA (Filt.)"),
    ]:
        if k in comp:
            m = comp[k]
            print(
                f"  {lbl:<18} | {100*m['mean_oracle_prob']:9.1f}% | {m['valid_promoter_pct']:8.1f}% | "
                f"{m['tss_inr_pct']:8.1f}% | {m['mean_gc_pct']:5.1f}% | {m['mean_dg37_kcal']:8.1f} | "
                f"{m['unique_pct']:6.1f}% | {m.get('novel_pct', 100.0):6.1f}%"
            )

    if is_live_qpu:
        try:
            qpu_obj = model_kw.get("device_kwargs", {}).get("backend")
            if qpu_obj is not None and getattr(qpu_obj, "service", None) is not None:
                rem = (qpu_obj.service.usage() or {}).get("usage_remaining_seconds")
                if rem is not None:
                    print(f"\n[IBM Runtime] Quota remaining after run: {rem}s")
        except Exception:
            pass

    out_png = os.path.join(cfg.artifact_dir, "fig_dna_qpu_showcase.png")
    out_json = os.path.join(cfg.artifact_dir, "metrics.json")
    out_fasta = os.path.join(cfg.artifact_dir, "generated_promoters.fasta")
    make_dna_showcase_figure(
        comp, profiles, thermo_samples, cfg.n_qubits, cfg.max_shots, out_png,
        circuits_per_sample=3 if cfg.nudge_mode == "residual" else 2,
    )
    if not is_core8 and "qerd_qpu" in seqs_store:
        probs_q = predict_oracle_probs(oracle_model, seqs_store["qerd_qpu"])
        export_top_promoters_fasta(seqs_store["qerd_qpu"], probs_q, eval_tags, out_fasta, top_k=12)
    else:
        export_top_promoters_fasta(
            seqs_store.get("qerd_qpu", []),
            np.full(len(seqs_store.get("qerd_qpu", [])), comp.get("qerd_qpu", {}).get("mean_oracle_prob", 0.5)),
            eval_tags,
            out_fasta,
            top_k=12,
        )

    cfg_safe = cfg.to_dict()
    if cfg_safe.get("api_key_json"):
        cfg_safe["api_key_json"] = "<provided>"
    save_metrics_json(out_json, cfg_safe, sub_rep, oracle_metrics, scaling_rows, comp)
    print(f"\n[Saved] Showcase -> {out_png}, {out_fasta}, & {out_json}")
    print(f"[Saved] Checkpoints -> {oracle_ckpt} & {q_ckpt}")


def run_infer(cfg: RunConfig, n_samples: int = 16) -> None:
    prefix = "dna_qerd_"
    ckpt_path = os.path.join(cfg.ckpt_dir, f"{prefix}{cfg.n_qubits}q.npz")
    if not os.path.exists(ckpt_path) and os.path.isdir(cfg.ckpt_dir):
        matches = sorted(f for f in os.listdir(cfg.ckpt_dir) if f.startswith(prefix) and f.endswith(".npz"))
        if matches:
            ckpt_path = os.path.join(cfg.ckpt_dir, matches[-1])
    ck = load_dna_checkpoint(ckpt_path)
    if ck["n_qubits"] > 18 and cfg.mode != "ibm_qpu":
        raise ValueError(
            f"Checkpoint has {ck['n_qubits']} qubits (exceeds local RAM); pass `--mode ibm_qpu --api-key-json <path>`."
        )
    ro_cal = None
    if cfg.mode == "ibm_qpu":
        _, model_kw, _, ro_cal = prepare_dna_backend(
            cfg.mode, cfg.device, ck["n_qubits"], instance=cfg.instance,
            execution_mode=cfg.execution_mode, seed=cfg.seed, api_key_json=cfg.api_key_json,
        )
    else:
        model_kw = {"p_depol": 0.02, "p_readout": 0.02}

    with QuantumIsingDenoiser(
        data_shape=ck["n_qubits"], cond_dim=2, edges=ck["edges"], n_layers=ck["n_layers"],
        diffusion_steps=ck["diffusion_steps"], cal_mode=ck["cal_mode"],
        init_gain=0.75, gain_ema=0.10, gain_bounds=(0.40, 1.00), seed=cfg.seed, **model_kw,
    ) as m:
        m.theta[:] = ck["theta"]
        m.cond_proj[:] = ck["cond_proj"]
        m.calibrator.gain[:] = ck["calibrator_gain"]
        conds, tags = _eval_cond_vectors(n_samples)
        spins = sample_dna_batch(
            m, conds, n_samples, shots_per_step=cfg.eval_shots,
            sample_temp=cfg.sample_temp, stochastic_decode=cfg.stochastic_decode, ro_cal=ro_cal,
        )
    seqs = spins_to_core8(spins) if ck["n_qubits"] == 16 else spins_to_seqs(spins)
    for i, (tag, s) in enumerate(zip(tags[:6], seqs[:6]), 1):
        print(f"  [{i}] ({tag:<7}) dG37={compute_santalucia_dg37(s):6.1f} GC={gc_percent(s):4.1f}% | {s}")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Project 2: Conditional Human Promoter DNA Designer (QERD)")
    parser.add_argument("--preset", choices=list(PRESETS), default="local_cpu")
    parser.add_argument("--mode", choices=["local_sim", "qiskit_sim", "ibm_qpu"], default=None)
    parser.add_argument("--device", choices=["ibm_kingston", "ibm_fez", "ibm_marrakesh"], default=None)
    parser.add_argument("--instance", type=str, default=None)
    parser.add_argument("--api-key-json", type=str, default=None)
    parser.add_argument("--clear-outputs", action="store_true")
    parser.add_argument("--execution-mode", choices=["job", "batch", "session"], default=None)
    parser.add_argument("--n-qubits", type=int, default=None)
    parser.add_argument("--batches", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--shots-per-run", type=int, default=None)
    parser.add_argument("--nudge-mode", choices=["residual", "target"], default=None)
    parser.add_argument("--skip-scaling", action="store_true")
    parser.add_argument("--skip-live-spsa", action="store_true")
    parser.add_argument("--skip-live-nozz", action="store_true")
    parser.add_argument("--artifact-dir", type=str, default=None)
    parser.add_argument("--ckpt-dir", type=str, default=None)
    parser.add_argument("--fetch-data", action="store_true")
    parser.add_argument("--infer", action="store_true")
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
        nudge_mode=args.nudge_mode,
        skip_scaling=True if args.skip_scaling else None,
        run_live_spsa=False if args.skip_live_spsa else None,
        run_live_nozz=False if args.skip_live_nozz else None,
        artifact_dir=args.artifact_dir,
        ckpt_dir=args.ckpt_dir,
    )

    if args.fetch_data:
        saved = fetch_promoter_datasets(cfg.data_dir)
        print(f"Downloaded Human Promoter datasets -> {saved}")
        return

    if args.infer:
        run_infer(cfg)
        return

    run_benchmark(cfg, clear_outputs=args.clear_outputs)


if __name__ == "__main__":
    main()
