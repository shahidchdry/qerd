"""
run_paper_benchmarks.py — Single-command orchestrator for the QERD benchmark & ablation suite.

Executes the full empirical benchmark & rebuttal matrix cleanly while reusing
cached frozen ConvLatentAE + EvalCNN helpers across sweeps:
  1. Main 5-seed benchmarks (MNIST 12+0q & Fashion-MNIST 12+2q, N=2000 eval samples,
     incl. Real-Held-Out / Real-AE-Recon, Adam-SPSA, Adaptive-SPSA, Adam-PSR-FD,
     Exact-PSR, ClassicalIsing-EP, Classical-DDPM, Classical-Skip, Classical-Large,
     PQC-QuDDPM, QCBM-MMD)
  2. 6-row Component & Control Ablations (none, gain-cal, gain-cal-inv, no-zz,
     wcond-only, sgd) on both datasets
  3. Shot-budget scaling sweep (10k -> 160k shots on MNIST)
  4. Gradient fidelity & Barren-Plateau scaling (cos(QER, -grad MSE) across steps,
     beta, depth L, depolarizing noise; gradient variance Var(g) across widths n & L)
  5. Hardware realism sweeps (p_depol sweep; hardware topologies all/grid8/grid4/chain
     & depths L=1..6 under correlated 2Q CNOT noise)
  6. Non-whitened entanglement controls (unwhitened Fashion-MNIST AE without BatchNorm1d
     + pure-QPU zero-AE 12-qubit stride-3 cluster spin-glass parity)

Outputs inside `--out-dir` (default `./paper_suite`):
  - `paper_report.txt`      : Complete Markdown tables, 95% CIs & timings
  - `paper_all_results.json`: Combined structured JSON across every study
  - `paper_tables.tex`      : Auto-populated LaTeX macros (`\\renewcommand`) for `paper.tex`
  - `figures/*.pdf`, `*.png`: Publication charts + copied visual sample grids
  - `run_full_log.txt`      : Full streamed console log

Usage (full multi-seed GPU run):
  python -u experiments/run_paper_benchmarks.py

Quick local CPU smoke verification (sklearn digits + full pipeline exercise):
  python -u experiments/run_paper_benchmarks.py --quick --out-dir ./legacy/runs/smoke_suite
"""

from __future__ import annotations

try:
    import _paths  # noqa: F401
except ImportError:
    from . import _paths  # noqa: F401

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, TextIO, Tuple

import numpy as np
import torch


# ==============================================================================
# 1. GENERAL HELPERS & SUBPROCESS RUNNERS
# ==============================================================================
def _parse_csv_ints(spec: str) -> List[int]:
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def _parse_csv_strs(spec: str) -> List[str]:
    return [x.strip() for x in spec.split(",") if x.strip()]


def _result_json_path(save_dir: str, dataset: str, ablate: str, tag_suffix: str = "") -> str:
    tag = (dataset if ablate == "none" else f"{dataset}_{ablate}") + tag_suffix
    return os.path.join(save_dir, f"{tag}_results.json")


def _helper_filename(
    dataset: str,
    latent_dim: int,
    ae_epochs: int,
    clf_epochs: int,
    use_bn: bool = True,
) -> str:
    bn_tag = "" if use_bn else "_nobn"
    return f"{dataset}_d{latent_dim}_ae{ae_epochs}_clf{clf_epochs}{bn_tag}_helpers.pt"


def _copy_helper_cache(
    src_dir: str,
    dst_dir: str,
    filename: str,
    extra_search_dirs: Sequence[str] = (),
) -> Optional[str]:
    """Copy frozen AE + EvalCNN checkpoint into `dst_dir` so helpers are not retrained."""
    os.makedirs(dst_dir, exist_ok=True)
    dst_path = os.path.join(dst_dir, filename)
    if os.path.exists(dst_path):
        return dst_path
    for cand_dir in (src_dir, *extra_search_dirs):
        if not cand_dir:
            continue
        cand_path = os.path.join(cand_dir, filename)
        if os.path.exists(cand_path):
            shutil.copy2(cand_path, dst_path)
            return dst_path
    return None


def _parse_log_metadata(lines: Sequence[str]) -> Dict[str, Any]:
    """Extract EvalCNN accuracy, backend header, and per-method timing/steps metadata."""
    meta: Dict[str, Any] = {"timings_sec": {}, "steps": {}}
    timings: Dict[str, List[float]] = {}
    steps_map: Dict[str, int] = {}
    for line in lines:
        if m_hdr := re.search(r"=== QERD BENCHMARK \((.+)\) ===", line):
            meta["header"] = m_hdr.group(1)
        elif m_cnn := re.search(r"\[EvalCNN\] Test Accuracy on real held-out split:\s*([\d.]+)%", line):
            meta["eval_cnn_acc"] = float(m_cnn.group(1)) / 100.0
        elif m_ae := re.search(r"\[AE\] Epoch \d+/\d+ \| Recon MSE=([\d.]+)", line):
            meta["ae_recon_mse"] = float(m_ae.group(1))
        elif m_t := re.search(r"(QERD|Adam-SPSA|Adam-PSR(?:-FD)?) finished (\d+) shots \((\d+) steps\) in ([\d.]+)s", line):
            name = "Adam-PSR" if m_t.group(1).startswith("Adam-PSR") else m_t.group(1)
            timings.setdefault(name, []).append(float(m_t.group(4)))
            steps_map[name] = int(m_t.group(3))

    for k, vals in timings.items():
        meta["timings_sec"][k] = {"mean": sum(vals) / len(vals), "per_seed": vals}
    meta["steps"] = steps_map
    return meta


def _run_bench(
    bench_script: str,
    title: str,
    cli_args: List[str],
    save_dir: str,
    dataset: str,
    ablate: str,
    log_fp: TextIO,
    tag_suffix: str = "",
) -> Dict[str, Any]:
    """Execute one `benchmark_mnist.py` invocation, streaming stdout to console and log."""
    os.makedirs(save_dir, exist_ok=True)
    full_args = list(cli_args)
    if tag_suffix:
        full_args.extend(["--tag-suffix", tag_suffix])
    cmd = [sys.executable, "-u", bench_script, *full_args, "--save-dir", save_dir]
    banner = f"\n{'=' * 72}\n[{title}] CMD: {' '.join(cmd)}\n{'=' * 72}\n"
    print(banner, end="", flush=True)
    log_fp.write(banner)
    log_fp.flush()

    t0 = time.time()
    captured: List[str] = []
    sub_env = os.environ.copy()
    sub_env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1, env=sub_env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        log_fp.write(line)
        captured.append(line)
    ret = proc.wait()
    wall_sec = time.time() - t0
    log_fp.flush()
    if ret != 0:
        raise RuntimeError(f"Command failed (exit={ret}) for '{title}'. Check {log_fp.name}")

    json_path = _result_json_path(save_dir, dataset, ablate, tag_suffix=tag_suffix)
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    return {
        "title": title,
        "save_dir": save_dir,
        "json_path": json_path,
        "wall_sec": round(wall_sec, 2),
        "log_meta": _parse_log_metadata(captured),
        "results": payload,
    }


def _resolve_torch_device(pref: str) -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _run_grad_and_bp_study(
    main_cfg: Dict[str, Any],
    helper_dir: str,
    torch_device_str: str,
    quick: bool,
    log_fp: TextIO,
) -> Dict[str, Any]:
    """Run Proposition 1 gradient alignment & Barren-Plateau variance scaling on cached latents."""
    import benchmark_mnist as bm

    dev = _resolve_torch_device(torch_device_str)
    ds = main_cfg["dataset"]
    cache_file = os.path.join(
        helper_dir,
        _helper_filename(ds, main_cfg["latent_dim"], main_cfg["ae_epochs"], main_cfg["clf_epochs"], use_bn=True),
    )
    msg = f"\n{'=' * 72}\n[Gradient Fidelity & Barren-Plateau Scaling] dataset={ds} dev={dev}\n{'=' * 72}\n"
    print(msg, end="", flush=True)
    log_fp.write(msg)
    log_fp.flush()

    t0 = time.time()
    x_tr, y_tr, x_te, y_te, _ = bm.load_dataset(ds)
    _, _, z_tr, _ = bm.train_classical_helpers(
        x_tr, y_tr, x_te, y_te, latent_dim=main_cfg["latent_dim"],
        ae_epochs=main_cfg["ae_epochs"], clf_epochs=main_cfg["clf_epochs"],
        device=dev, cache_path=cache_file, use_bn=True,
    )
    eye10 = np.eye(10, dtype=np.float64)
    c_tr = eye10[y_tr]

    grad_align = bm.compute_circuit_grad_alignment(
        z_tr, c_tr, y_tr, n_hidden=main_cfg["n_hidden"], topology="all",
        device=dev, seed=0, quick=quick,
    )
    bp_widths = [6, 8, 10] if quick else [6, 8, 10, 12, 14]
    bp_trials = 6 if quick else 12
    bp_scaling = bm.measure_gradient_variance_scaling(
        widths=bp_widths, depths=[2, 6], n_trials=bp_trials, device=dev, seed=0,
    )
    wall = round(time.time() - t0, 2)
    summary = f"  Completed grad alignment ({len(grad_align['step_curve'])} step pts) & BP scaling ({len(bp_scaling)} configs) in {wall}s\n"
    print(summary, end="", flush=True)
    log_fp.write(summary)
    log_fp.flush()
    return {"dataset": ds, "wall_sec": wall, "grad_alignment": grad_align, "bp_scaling": bp_scaling}


def _run_pure_qpu_parity_study(
    seeds: List[int],
    shots: int,
    n_eval: int,
    backend: str,
    torch_device_str: str,
    log_fp: TextIO,
) -> Dict[str, Any]:
    """Run zero-AE 12-qubit stride-3 correlated cluster spin-glass benchmark."""
    import benchmark_mnist as bm

    msg = f"\n{'=' * 72}\n[Pure-QPU 12q Stride-3 Spin-Glass Parity] seeds={seeds} shots={shots}\n{'=' * 72}\n"
    print(msg, end="", flush=True)
    log_fp.write(msg)
    log_fp.flush()

    t0 = time.time()
    out = bm.run_pure_qpu_parity_benchmark(
        seeds=seeds, p_depol=0.035, p_readout=0.04, p_depol_2q=0.0,
        max_shots=shots, n_eval=n_eval, backend=backend, torch_device=torch_device_str,
    )
    out["wall_sec"] = round(time.time() - t0, 2)
    for vk, met in out["aggregate"].items():
        line = (
            f"  [Pure-QPU {vk:18s}] IntraCorr={met['intra_corr']['mean']:.3f} | "
            f"CorrRMSE={met['corr_rmse']['mean']:.3f} | PatAcc={met['pattern_acc']['mean']*100:.1f}%\n"
        )
        print(line, end="", flush=True)
        log_fp.write(line)
    log_fp.flush()
    return out


# ==============================================================================
# 2. SHARED METHOD / METRIC REGISTRY & MARKDOWN REPORT GENERATOR
# ==============================================================================
_MAIN_METHOD_SPECS: List[Tuple[str, str, str, str]] = [
    ("qerd", "QER (Ours)", "\\textbf{QER (Ours)}", "#0072B2"),
    ("adam_spsa", "Adam-SPSA", "Adam-SPSA", "#D55E00"),
    ("adaptive_spsa", "Adaptive-SPSA", "Adaptive-SPSA", "#CC79A7"),
    ("adam_psr", "Adam-PSR-FD", "Adam-PSR-FD", "#E69F00"),
    ("exact_psr", "Exact-PSR", "Exact-PSR", "#A6761D"),
    ("classical_ising_ep", "ClassicalIsing-EP", "ClassicalIsing-EP", "#66A61E"),
    ("classical_ddpm", "Classical-DDPM", "Classical-DDPM", "#009E73"),
    ("classical_skip_ddpm", "Classical-Skip", "Classical-Skip", "#56B4E9"),
    ("classical_large_ddpm", "Classical-Large", "Classical-Large", "#332288"),
    ("pqc_quddpm", "PQC-QuDDPM", "PQC-QuDDPM", "#AA4499"),
    ("qcbm_mmd", "QCBM-MMD", "QCBM-MMD", "#999933"),
]

_ABLATION_SPECS: Dict[str, Tuple[str, str, str]] = {
    "none": ("Full QERD (Option A)", "\\textbf{Full QERD (Option A)}", "#0072B2"),
    "gain-cal": ("+Fwd OLS (gain-cal)", "$+$ Forward OLS (\\texttt{gain-cal})", "#56B4E9"),
    "gain-cal-inv": ("+Inv OLS (gain-cal-inv)", "$+$ Inverse OLS (\\texttt{gain-cal-inv})", "#009E73"),
    "no-zz": ("No Ising ZZ (no-zz)", "No Ising $ZZ$ (\\texttt{no-zz})", "#CC79A7"),
    "wcond-only": ("W_cond Only (theta=0)", "$W_{\\text{cond}}$-Only ($\\theta{=}0$)", "#882255"),
    "sgd": ("Fixed-LR SGD (sgd)", "Fixed-LR SGD (\\texttt{sgd})", "#E69F00"),
}


def _param_for_key(results: Dict[str, Any], key: str) -> int:
    pc = results.get("param_counts", {})
    if key in pc:
        return int(pc[key])
    if key in ("qerd", "adam_spsa", "adaptive_spsa", "adam_psr", "exact_psr"):
        return int(results.get("qpu_params", pc.get("qpu", 0)))
    return int(results.get("classical_params", 0))


def _fmt_pm(metric_dict: Dict[str, float], scale: float = 1.0, decimals: int = 2, suffix: str = "") -> str:
    m = metric_dict["mean"] * scale
    s = metric_dict["std"] * scale
    return f"{m:.{decimals}f} ± {s:.{decimals}f}{suffix}"


def _format_method_rows(results: Dict[str, Any], row_label_prefix: str = "", include_real: bool = False) -> List[str]:
    rows: List[str] = []
    if include_real and "real_baselines" in results:
        for rk, rdisp in [("real_held_out", "Real-Held-Out"), ("real_ae_recon", "Real-AE-Recon")]:
            if rk not in results["real_baselines"]:
                continue
            rm = results["real_baselines"][rk]
            name = f"{row_label_prefix}{rdisp}" if row_label_prefix else rdisp
            rows.append(
                f"| {name} | -- | {rm['clf_acc']*100:.1f}% | -- | -- | "
                f"{rm.get('cond_feature_fid', rm['feature_fid']):.1f} | {rm['feature_fid']:.1f} | "
                f"{rm['precision']:.3f} | {rm['recall']:.3f} | -- |"
            )
    agg = results["aggregate"]
    for key, display, _, _ in _MAIN_METHOD_SPECS:
        if key not in agg:
            continue
        m = agg[key]
        name = f"{row_label_prefix}{display}" if row_label_prefix else display
        p_info = _param_for_key(results, key)
        cf_fid = m.get("cond_feature_fid", m["feature_fid"])
        rows.append(
            f"| {name} | {p_info} | {_fmt_pm(m['clf_acc'], 100.0, 1, '%')} | "
            f"{_fmt_pm(m['cond_latent_fid'], 1.0, 2)} | {_fmt_pm(m['latent_fid'], 1.0, 2)} | "
            f"{_fmt_pm(cf_fid, 1.0, 1)} | {_fmt_pm(m['feature_fid'], 1.0, 1)} | "
            f"{_fmt_pm(m['precision'], 1.0, 3)} | {_fmt_pm(m['recall'], 1.0, 3)} | {_fmt_pm(m['intra_class_latent_std'], 1.0, 3)} |"
        )
    return rows


_TABLE_HDR = (
    "| Entry | Params | ClfAcc (%) ↑ | CondLatFID ↓ | LatFID ↓ | CondFeatFID ↓ | FeatFID ↓ | Prec ↑ | Rec ↑ | LatStd |\n"
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
)


def _format_timings(timings_sec: Dict[str, Any]) -> str:
    if not timings_sec:
        return "n/a"
    return ", ".join(f"{k}={v['mean']:.2f}s/seed" for k, v in timings_sec.items())


def build_markdown_report(bundle: Dict[str, Any]) -> str:
    """Render concise shareable Markdown summary tables from every completed study."""
    env = bundle["environment"]
    lines: List[str] = [
        "# QER Paper Benchmark Report",
        "",
        f"- **Generated:** {env['timestamp']}",
        f"- **Device / PyTorch:** `{env['device_name']}` | torch `{env['torch_version']}` | Python `{env['python_version']}`",
        f"- **QPU Simulator Setting:** `--backend {env['backend']} --torch-device {env['torch_device']}` | Wall time: `{bundle['total_wall_sec']:.1f}s`",
        "",
    ]

    if bundle.get("main"):
        lines.extend(["## 1. Main Multi-Seed Comparison", ""])
        for entry in bundle["main"]:
            res = entry["results"]
            meta = entry["log_meta"]
            cnn_str = f"{meta['eval_cnn_acc'] * 100:.2f}%" if "eval_cnn_acc" in meta else "n/a"
            lines.append(
                f"### {entry['title']} (`seeds={res['seeds']}`, `shots={res['shots_used']}`, "
                f"`N={res.get('eval_samples', '?')}`, `AE={res.get('param_counts', {}).get('ae', '?')}`, "
                f"`EvalCNN={cnn_str}`, `{_format_timings(meta.get('timings_sec', {}))}`)"
            )
            lines.extend(["", _TABLE_HDR, *_format_method_rows(res, include_real=True), ""])

    for ds_label, abl_entries in bundle.get("ablations", {}).items():
        if not abl_entries:
            continue
        lines.append(f"## 2. {ds_label} Component Ablations (`seeds={abl_entries[0]['results']['seeds']}`)")
        lines.extend(["", _TABLE_HDR])
        for entry in abl_entries:
            res = entry["results"]
            sub_res = {**res, "aggregate": {"qerd": res["aggregate"]["qerd"]}}
            lines.extend(_format_method_rows(sub_res, row_label_prefix=f"`{res['ablate']}` "))
        ref = next((e["results"] for e in abl_entries if e["results"]["ablate"] == "none"), abl_entries[0]["results"])
        ref_agg = {k: ref["aggregate"][k] for k in ("adam_spsa", "classical_ddpm") if k in ref["aggregate"]}
        lines.extend([*_format_method_rows({**ref, "aggregate": ref_agg}, row_label_prefix="*ref* "), ""])

    if bundle.get("shot_scaling"):
        lines.append(f"## 3. Shot-Budget Scaling (`dataset={bundle['shot_scaling'][0]['results']['dataset']}`)")
        lines.extend(["", _TABLE_HDR])
        for entry in bundle["shot_scaling"]:
            res = entry["results"]
            lines.extend(_format_method_rows(res, row_label_prefix=f"`{res['shots_used']}` "))
        lines.append("")

    if gs := bundle.get("grad_study"):
        lines.extend(["## 4. Gradient Fidelity & Barren-Plateau Scaling", ""])
        ga = gs["grad_alignment"]
        lines.append("| Axis | Setting | cos(QER,-grad MSE) | cos(ZZ) | cos(Bias+Cond) | cos(CorrSurr) | Shot cos |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for r in ga["beta_sweep"]:
            lines.append(f"| beta | {r['beta']:.2f} | {r['cos_mse_all']:.3f} | {r['cos_mse_zz']:.3f} | {r['cos_mse_bias']:.3f} | {r['cos_corr_all']:.3f} | {r['cos_shot_mse_all']:.3f} |")
        for r in ga["depth_sweep"]:
            lines.append(f"| depth L | {int(r['n_layers'])} | {r['cos_mse_all']:.3f} | {r['cos_mse_zz']:.3f} | {r['cos_mse_bias']:.3f} | {r['cos_corr_all']:.3f} | {r['cos_shot_mse_all']:.3f} |")
        lines.append("")
        lines.append("| Qubits n | Depth L | Edges | Var(g_QER, ZZ) | Var(-grad MSE, ZZ) | Var Ratio QER/MSE |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for r in gs["bp_scaling"]:
            ratio = r["var_qer_zz"] / max(r["var_mse_zz"], 1e-12)
            lines.append(f"| {r['n_qubits']} | {r['n_layers']} | {r['n_edges']} | {r['var_qer_zz']:.3e} | {r['var_mse_zz']:.3e} | {ratio:.1f}x |")
        lines.append("")

    for key, hdr in [("noise_sweep", "## 5a. NISQ Depolarizing Noise Sweep"),
                     ("topo_sweep", "## 5b. Hardware Topology Sweep (+2Q CNOT Noise)"),
                     ("depth_sweep", "## 5c. Trotter Depth L Sweep (+2Q CNOT Noise)"),
                     ("unwhitened_ae", "## 6a. Unwhitened Latent AE Control (no BatchNorm1d)")]:
        entries = bundle.get(key, [])
        if entries:
            lines.extend([hdr, "", _TABLE_HDR])
            for e in entries:
                r = e["results"]
                pfx = f"`{e['sweep_tag']}` "
                lines.extend(_format_method_rows({**r, "aggregate": {"qerd": r["aggregate"]["qerd"]}}, row_label_prefix=pfx))
            lines.append("")

    if pq := bundle.get("pure_qpu_parity"):
        lines.extend(["## 6b. Pure-QPU 12q Stride-3 Spin-Glass Parity (Zero AE)", ""])
        lines.append("| Variant | Intra-Cluster Corr ↑ | Corr RMSE ↓ | Exact Pattern Acc ↑ | Mode Cov ↑ | BitStd |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for vk, met in pq["aggregate"].items():
            lines.append(
                f"| `{vk}` | {_fmt_pm(met['intra_corr'], 1.0, 3)} | {_fmt_pm(met['corr_rmse'], 1.0, 3)} | "
                f"{_fmt_pm(met['pattern_acc'], 100.0, 1, '%')} | {_fmt_pm(met['mode_cov'], 100.0, 1, '%')} | {_fmt_pm(met['bit_std'], 1.0, 3)} |"
            )
        lines.append("")

    return "\n".join(lines)


# ==============================================================================
# 3. LATEX TABLE MACRO WRITER (`paper_tables.tex`)
# ==============================================================================
def _tex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def _pm_tex(metric_dict: Dict[str, float], scale: float = 1.0, decimals: int = 2, suffix: str = "", bold: bool = False) -> str:
    m = metric_dict["mean"] * scale
    s = metric_dict["std"] * scale
    core = f"{m:.{decimals}f} \\pm {s:.{decimals}f}{suffix}"
    return f"$\\mathbf{{{core}}}$" if bold else f"${core}$"


def _best_method_keys(agg: Dict[str, Any], method_order: Sequence[str]) -> Dict[str, str]:
    specs = {
        "clf_acc": max, "cond_latent_fid": min, "latent_fid": min,
        "cond_feature_fid": min, "feature_fid": min, "precision": max, "recall": max,
    }
    present = [k for k in method_order if k in agg and all(m in agg[k] for m in specs)]
    if not present:
        present = [k for k in method_order if k in agg]
    if not present:
        return {}
    return {metric: fn(present, key=lambda k: agg[k].get(metric, agg[k]["feature_fid"])["mean"]) for metric, fn in specs.items()}


def _format_tex_method_rows(
    results: Dict[str, Any],
    method_keys: Optional[Sequence[str]] = None,
    prefix: str = "",
    bold_best: bool = True,
    include_real: bool = False,
) -> List[str]:
    """Format 10-col LaTeX rows: Name & Params & ClfAcc & CondLatFID & LatFID & CondFeatFID & FeatFID & Prec & Rec & LatStd."""
    rows: List[str] = []
    if include_real and "real_baselines" in results:
        for rk, rdisp in [("real_held_out", "\\textit{Real Held-Out Split}"), ("real_ae_recon", "\\textit{Real AE Reconstructions}")]:
            if rk in results["real_baselines"]:
                rm = results["real_baselines"][rk]
                cf = rm.get("cond_feature_fid", rm["feature_fid"])
                rows.append(
                    f"{rdisp} & -- & ${rm['clf_acc']*100:.1f}\\%$ & -- & -- & "
                    f"${cf:.1f}$ & ${rm['feature_fid']:.1f}$ & ${rm['precision']:.3f}$ & ${rm['recall']:.3f}$ & -- \\\\"
                )
        rows.append("\\midrule")

    agg = results["aggregate"]
    selected = list(method_keys) if method_keys is not None else [k for k, _, _, _ in _MAIN_METHOD_SPECS]
    present = [k for k in selected if k in agg]
    best = _best_method_keys(agg, present) if bold_best else {}
    tex_disp = {k: tex_d for k, _, tex_d, _ in _MAIN_METHOD_SPECS}
    classical_break = False
    for key in present:
        if method_keys is None and not classical_break and key in ("classical_ising_ep", "classical_ddpm"):
            rows.append("\\midrule")
            classical_break = True
        m = agg[key]
        disp = f"{prefix}{tex_disp.get(key, _tex_escape(key))}" if prefix else tex_disp.get(key, _tex_escape(key))
        p_info = _param_for_key(results, key)
        cf_fid = m.get("cond_feature_fid", m["feature_fid"])
        rows.append(
            f"{disp} & {p_info} & "
            f"{_pm_tex(m['clf_acc'], 100.0, 1, '\\%', bold=(best.get('clf_acc') == key))} & "
            f"{_pm_tex(m['cond_latent_fid'], 1.0, 2, bold=(best.get('cond_latent_fid') == key))} & "
            f"{_pm_tex(m['latent_fid'], 1.0, 2, bold=(best.get('latent_fid') == key))} & "
            f"{_pm_tex(cf_fid, 1.0, 1, bold=(best.get('cond_feature_fid') == key))} & "
            f"{_pm_tex(m['feature_fid'], 1.0, 1, bold=(best.get('feature_fid') == key))} & "
            f"{_pm_tex(m['precision'], 1.0, 3, bold=(best.get('precision') == key))} & "
            f"{_pm_tex(m['recall'], 1.0, 3, bold=(best.get('recall') == key))} & "
            f"{_pm_tex(m['intra_class_latent_std'], 1.0, 3)} \\\\"
        )
    return rows


def _format_tex_ablation_rows(abl_entries: Sequence[Dict[str, Any]]) -> List[str]:
    """Format 8-col LaTeX ablation rows: Variant & ClfAcc & CondLatFID & LatFID & FeatFID & Prec & Rec & LatStd."""
    qerd_map = {e["results"]["ablate"]: e["results"]["aggregate"]["qerd"] for e in abl_entries if "qerd" in e["results"]["aggregate"]}
    best = _best_method_keys(qerd_map, list(qerd_map.keys()))
    rows: List[str] = []
    for entry in abl_entries:
        res = entry["results"]
        abl = res["ablate"]
        m = res["aggregate"]["qerd"]
        disp = _ABLATION_SPECS[abl][1] if abl in _ABLATION_SPECS else _tex_escape(abl)
        rows.append(
            f"{disp} & "
            f"{_pm_tex(m['clf_acc'], 100.0, 1, '\\%', bold=(best.get('clf_acc') == abl))} & "
            f"{_pm_tex(m['cond_latent_fid'], 1.0, 2, bold=(best.get('cond_latent_fid') == abl))} & "
            f"{_pm_tex(m['latent_fid'], 1.0, 2, bold=(best.get('latent_fid') == abl))} & "
            f"{_pm_tex(m['feature_fid'], 1.0, 1, bold=(best.get('feature_fid') == abl))} & "
            f"{_pm_tex(m['precision'], 1.0, 3, bold=(best.get('precision') == abl))} & "
            f"{_pm_tex(m['recall'], 1.0, 3, bold=(best.get('recall') == abl))} & "
            f"{_pm_tex(m['intra_class_latent_std'], 1.0, 3)} \\\\"
        )
    ref = next((e["results"] for e in abl_entries if e["results"]["ablate"] == "none"), abl_entries[0]["results"])
    ref_items = [("adam_spsa", "\\textit{Ref: Adam-SPSA}"), ("classical_ddpm", "\\textit{Ref: Classical-DDPM}")]
    if any(k in ref["aggregate"] for k, _ in ref_items):
        rows.append("\\midrule")
        for key, disp in ref_items:
            if key in ref["aggregate"]:
                m = ref["aggregate"][key]
                rows.append(
                    f"{disp} & {_pm_tex(m['clf_acc'], 100.0, 1, '\\%')} & "
                    f"{_pm_tex(m['cond_latent_fid'], 1.0, 2)} & {_pm_tex(m['latent_fid'], 1.0, 2)} & "
                    f"{_pm_tex(m['feature_fid'], 1.0, 1)} & {_pm_tex(m['precision'], 1.0, 3)} & "
                    f"{_pm_tex(m['recall'], 1.0, 3)} & {_pm_tex(m['intra_class_latent_std'], 1.0, 3)} \\\\"
                )
    return rows


def write_latex_tables(bundle: Dict[str, Any], out_dir: str) -> str:
    """Emit `paper_tables.tex` defining `\\renewcommand` table-body hooks for `paper.tex`."""
    env = bundle["environment"]
    tex_lines: List[str] = [
        "% Auto-generated by run_paper_benchmarks.py -- do not hand-edit.",
        "\\providecommand{\\qerBenchmarkStatus}{}",
        f"\\renewcommand{{\\qerBenchmarkStatus}}{{Live Verified Run: {_tex_escape(env['device_name'])} "
        f"(\\texttt{{torch {env['torch_version']}}}, {env['timestamp']})}}",
    ]

    def _emit_macro(macro_name: str, row_lines: Sequence[str]) -> None:
        if not row_lines:
            return
        tex_lines.append(f"\\providecommand{{\\{macro_name}}}{{}}")
        tex_lines.append(f"\\renewcommand{{\\{macro_name}}}{{%")
        for ln in row_lines:
            tex_lines.append(f"  {ln}")
        tex_lines.append("}")

    ds_suffix_map = {"mnist": "Mnist", "fashion": "Fashion", "digits": "Digits"}
    main_entries = bundle.get("main", [])
    for entry in main_entries:
        res = entry["results"]
        suffix = ds_suffix_map.get(res["dataset"])
        if suffix:
            _emit_macro(f"qerMain{suffix}Rows", _format_tex_method_rows(res, include_real=True))

    # Table 3: Literature Quantum & Classical Baselines (9 cols)
    lit_rows: List[str] = []
    lit_specs = [
        ("pqc_quddpm", "PQC-QuDDPM (Matched Shots/Noise)", "HEA PQC", "$2P_{\\text{pqc}}$"),
        ("qcbm_mmd", "QCBM-MMD (Matched Shots/Noise)", "Born Circuit", "$2P_{\\text{qcbm}}$"),
        ("classical_ising_ep", "ClassicalIsing-EP (Noiseless)", "Ising EP", "$2$ (Exact)"),
        ("classical_skip_ddpm", "Classical-DDPM + Skip", "MLP + $W_{\\text{cond}}c$", "Exact AD"),
        ("classical_large_ddpm", "Classical-Large (ResMLP Skyline)", "ResMLP", "Exact AD"),
        ("qerd", "\\textbf{QERD (Ours, Opt-A)}", "QPU Ising", "\\textbf{$\\mathbf{2}$ ($\\mathcal{O}(1)$)}"),
    ]
    for idx, entry in enumerate(main_entries):
        res = entry["results"]
        ds_lbl = "MNIST" if res["dataset"] == "mnist" else ("Fashion" if res["dataset"] == "fashion" else res["dataset"].title())
        added = 0
        if idx > 0 and lit_rows:
            lit_rows.append("\\midrule")
        for k, disp, dom, cost in lit_specs:
            if k in res["aggregate"]:
                m = res["aggregate"][k]
                p_cnt = _param_for_key(res, k)
                cf_fid = m.get("cond_feature_fid", m["feature_fid"])
                lit_rows.append(
                    f"{ds_lbl}: {disp} & {dom} & {p_cnt} & {cost} & "
                    f"{_pm_tex(m['clf_acc'], 100.0, 1, '\\%', bold=(k=='qerd'))} & "
                    f"{_pm_tex(cf_fid, 1.0, 1, bold=(k=='qerd'))} & "
                    f"{_pm_tex(m['feature_fid'], 1.0, 1)} & "
                    f"{_pm_tex(m['precision'], 1.0, 3)} & {_pm_tex(m['recall'], 1.0, 3)} \\\\"
                )
                added += 1
    _emit_macro("qerLitBaselineRows", lit_rows)

    for _, abl_entries in bundle.get("ablations", {}).items():
        if not abl_entries:
            continue
        suffix = ds_suffix_map.get(abl_entries[0]["results"]["dataset"])
        if suffix:
            _emit_macro(f"qerAbl{suffix}Rows", _format_tex_ablation_rows(abl_entries))

    if bundle.get("shot_scaling"):
        shot_rows: List[str] = []
        for idx, entry in enumerate(bundle["shot_scaling"]):
            res = entry["results"]
            shots_k = res["shots_used"] // 1000
            if idx > 0:
                shot_rows.append("\\midrule")
            shot_rows.extend(_format_tex_method_rows(res, method_keys=["qerd", "adam_spsa", "classical_ddpm"],
                                                     prefix=f"{shots_k}k / ", bold_best=False))
        _emit_macro("qerShotScalingRows", shot_rows)

    if main_entries:
        timing_rows: List[str] = []
        for entry in main_entries:
            res = entry["results"]
            ds_disp = "MNIST ($12+0$q)" if res["dataset"] == "mnist" else (
                "Fashion-MNIST ($12+2$q)" if res["dataset"] == "fashion" else _tex_escape(res["dataset"].title())
            )
            t_map = entry.get("log_meta", {}).get("timings_sec", {})
            s_map = entry.get("log_meta", {}).get("steps", {})
            shots = int(res["shots_used"])
            q_steps = s_map.get("QERD", shots // 400)
            for key, t_key, disp in [
                ("qerd", "QERD", "\\textbf{QER (Ours)}"),
                ("adam_spsa", "Adam-SPSA", "Adam-SPSA"),
                ("adam_psr", "Adam-PSR", "Adam-PSR-FD"),
            ]:
                if key not in res["aggregate"]:
                    continue
                sec_str = f"{t_map[t_key]['mean']:.2f}\\,s" if t_key in t_map else "--"
                steps_v = s_map.get(t_key, q_steps)
                if key == "adam_psr":
                    circ_str = f"${max(2, round(shots / max(1, steps_v * 20)))}$"
                elif key == "adam_spsa":
                    circ_str = "$2$ (batch)"
                else:
                    circ_str = "$2$ ($\\mathcal{O}(1)$)"
                acc_str = _pm_tex(res["aggregate"][key]["clf_acc"], 100.0, 1, "\\%")
                timing_rows.append(f"{ds_disp} & {disp} & {circ_str} & {steps_v} & {sec_str} & {acc_str} \\\\")
        _emit_macro("qerTimingRows", timing_rows)

    # Gradient Alignment & Barren-Plateau Scaling macros
    if gs := bundle.get("grad_study"):
        ga = gs["grad_alignment"]
        ga_rows: List[str] = []
        step_pick = [r for r in ga["step_curve"] if r["step"] in (1, 40, 50, 80, 100, 200)] or ga["step_curve"][:3]
        for r in step_pick:
            ga_rows.append(
                f"Step & $t={int(r['step'])}$ & ${r['cos_mse_all']:.3f}$ & ${r['cos_mse_zz']:.3f}$ & "
                f"${r['cos_mse_bias']:.3f}$ & ${r['cos_corr_all']:.3f}$ & ${r['cos_shot_mse_all']:.3f}$ \\\\"
            )
        ga_rows.append("\\midrule")
        for r in ga["beta_sweep"]:
            ga_rows.append(
                f"$\\beta$ Radius & ${r['beta']:.2f}$ & ${r['cos_mse_all']:.3f}$ & ${r['cos_mse_zz']:.3f}$ & "
                f"${r['cos_mse_bias']:.3f}$ & ${r['cos_corr_all']:.3f}$ & ${r['cos_shot_mse_all']:.3f}$ \\\\"
            )
        ga_rows.append("\\midrule")
        for r in ga["depth_sweep"]:
            ga_rows.append(
                f"Depth $L$ & $L={int(r['n_layers'])}$ & ${r['cos_mse_all']:.3f}$ & ${r['cos_mse_zz']:.3f}$ & "
                f"${r['cos_mse_bias']:.3f}$ & ${r['cos_corr_all']:.3f}$ & ${r['cos_shot_mse_all']:.3f}$ \\\\"
            )
        ga_rows.append("\\midrule")
        for r in ga["noise_sweep"]:
            ga_rows.append(
                f"$p_{{\\text{{depol}}}}$ & ${r['p_depol']:.3f}$ & ${r['cos_mse_all']:.3f}$ & ${r['cos_mse_zz']:.3f}$ & "
                f"${r['cos_mse_bias']:.3f}$ & ${r['cos_corr_all']:.3f}$ & ${r['cos_shot_mse_all']:.3f}$ \\\\"
            )
        _emit_macro("qerGradAlignRows", ga_rows)

        bp_rows: List[str] = []
        for r in gs["bp_scaling"]:
            ratio = r["var_qer_zz"] / max(r["var_mse_zz"], 1e-12)
            bp_rows.append(
                f"{int(r['n_qubits'])} & {int(r['n_layers'])} & {int(r['n_edges'])} & "
                f"${r['var_qer_zz']:.2e}$ & ${r['var_mse_zz']:.2e}$ & ${ratio:.1f}\\times$ \\\\"
            )
        _emit_macro("qerBpScalingRows", bp_rows)

    # Hardware Realism: Noise Sweep (7 cols) & Topology/Depth Sweep (7 cols)
    if bundle.get("noise_sweep"):
        ns_rows = []
        for e in bundle["noise_sweep"]:
            r = e["results"]
            m = r["aggregate"]["qerd"]
            ns_rows.append(
                f"${r['p_depol']:.3f}$ & {_pm_tex(m['clf_acc'], 100.0, 1, '\\%')} & "
                f"{_pm_tex(m['cond_latent_fid'], 1.0, 2)} & {_pm_tex(m['latent_fid'], 1.0, 2)} & "
                f"{_pm_tex(m['feature_fid'], 1.0, 1)} & {_pm_tex(m['precision'], 1.0, 3)} & {_pm_tex(m['recall'], 1.0, 3)} \\\\"
            )
        _emit_macro("qerNoiseSweepRows", ns_rows)

    td_rows: List[str] = []
    for idx_grp, key in enumerate(["topo_sweep", "depth_sweep"]):
        entries = bundle.get(key, [])
        if idx_grp > 0 and entries and td_rows:
            td_rows.append("\\midrule")
        for e in entries:
            r = e["results"]
            m = r["aggregate"]["qerd"]
            cfg_str = f"Topo: \\texttt{{{_tex_escape(r['topology'])}}}" if key == "topo_sweep" else f"Depth: $L={int(r['n_layers'])}$"
            cnots = int(r.get("two_qubit_cnots", 2 * int(r.get("n_layers", 2)) * int(r.get("n_edges", 0))))
            td_rows.append(
                f"{cfg_str} & {cnots} & {_pm_tex(m['clf_acc'], 100.0, 1, '\\%')} & "
                f"{_pm_tex(m['cond_latent_fid'], 1.0, 2)} & {_pm_tex(m['feature_fid'], 1.0, 1)} & "
                f"{_pm_tex(m['precision'], 1.0, 3)} & {_pm_tex(m['recall'], 1.0, 3)} \\\\"
            )
    _emit_macro("qerTopoDepthRows", td_rows)

    # Entanglement Controls: Unwhitened AE (8 cols) & Pure-QPU Cluster Parity (6 cols)
    if bundle.get("unwhitened_ae"):
        uw_rows = []
        for e in bundle["unwhitened_ae"]:
            r = e["results"]
            m = r["aggregate"]["qerd"]
            tag = "Unwhitened Full ($ZZ$, $g{=}1$)" if r["ablate"] == "none" else "Unwhitened No-$ZZ$ (\\texttt{no-zz})"
            uw_rows.append(
                f"{tag} & {int(r['qpu_params'])} & {_pm_tex(m['clf_acc'], 100.0, 1, '\\%')} & "
                f"{_pm_tex(m['cond_latent_fid'], 1.0, 2)} & {_pm_tex(m['latent_fid'], 1.0, 2)} & "
                f"{_pm_tex(m['feature_fid'], 1.0, 1)} & {_pm_tex(m['precision'], 1.0, 3)} & {_pm_tex(m['recall'], 1.0, 3)} \\\\"
            )
        _emit_macro("qerUnwhitenedFashionRows", uw_rows)

    if pq := bundle.get("pure_qpu_parity"):
        pq_names = {
            "qerd_all": "\\textbf{QERD-\\texttt{all}} ($66$ edges)",
            "qerd_grid8": "QERD-\\texttt{grid8} ($29$ edges)",
            "qerd_grid4": "QERD-\\texttt{grid4} ($17$ edges)",
            "qerd_chain": "QERD-\\texttt{chain} ($11$ edges)",
            "qerd_nozz": "QERD-\\texttt{no-zz} ($0$ edges)",
            "classical_ising_ep": "ClassicalIsing-EP (Noiseless)",
        }
        pq_rows = []
        for vk, met in pq["aggregate"].items():
            pq_rows.append(
                f"{pq_names.get(vk, _tex_escape(vk))} & {_pm_tex(met['intra_corr'], 1.0, 3, bold=(vk=='qerd_all'))} & "
                f"{_pm_tex(met['corr_rmse'], 1.0, 3)} & {_pm_tex(met['pattern_acc'], 100.0, 1, '\\%')} & "
                f"{_pm_tex(met['mode_cov'], 100.0, 1, '\\%')} & {_pm_tex(met['bit_std'], 1.0, 3)} \\\\"
            )
        _emit_macro("qerPureQpuRows", pq_rows)

    tex_path = os.path.join(out_dir, "paper_tables.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex_lines) + "\n")
    return tex_path


# ==============================================================================
# 4. PUBLICATION FIGURES (7 VECTOR PDF + 300-DPI PNG ASSETS)
# ==============================================================================
def _apply_pub_style(plt: Any) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9.0,
        "axes.titlesize": 10.0,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _save_fig(plt: Any, fig: Any, fig_dir: str, stem: str) -> str:
    os.makedirs(fig_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        kw: Dict[str, Any] = {"bbox_inches": "tight"}
        if ext == "png":
            kw["dpi"] = 300
        fig.savefig(os.path.join(fig_dir, f"{stem}.{ext}"), **kw)
    plt.close(fig)
    return stem


def _plot_main_comparison(plt: Any, bundle: Dict[str, Any], fig_dir: str) -> Optional[str]:
    entries = bundle.get("main", [])
    if not entries:
        return None
    metrics = [
        ("clf_acc", 100.0, "ClfAcc (%) ↑", "{:.0f}"),
        ("cond_feature_fid", 1.0, "CondFeatFID ↓", "{:.0f}"),
        ("precision", 1.0, "Precision ↑", "{:.2f}"),
        ("recall", 1.0, "Recall ↑", "{:.2f}"),
    ]
    n_rows = len(entries)
    fig, axes = plt.subplots(n_rows, len(metrics), figsize=(14.4, 3.5 * n_rows), squeeze=False)
    style_map = {k: (disp, col) for k, disp, _, col in _MAIN_METHOD_SPECS}
    for r, entry in enumerate(entries):
        res = entry["results"]
        agg = res["aggregate"]
        keys = [k for k, _, _, _ in _MAIN_METHOD_SPECS if k in agg]
        labels = [style_map[k][0] for k in keys]
        colors = [style_map[k][1] for k in keys]
        ds_title = "MNIST (12+0q)" if res["dataset"] == "mnist" else (
            "Fashion-MNIST (12+2q)" if res["dataset"] == "fashion" else res["dataset"].upper()
        )
        for c, (m_key, scale, ylabel, fmt) in enumerate(metrics):
            ax = axes[r, c]
            means = [agg[k].get(m_key, agg[k]["feature_fid"])["mean"] * scale for k in keys]
            stds = [agg[k].get(m_key, agg[k]["feature_fid"])["std"] * scale for k in keys]
            x = np.arange(len(keys))
            bars = ax.bar(x, means, yerr=stds, capsize=2.8, width=0.66, color=colors, edgecolor="#222222", linewidth=0.6)
            ax.grid(axis="y", linestyle="--", alpha=0.25)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=32, ha="right", fontsize=7.5)
            if r == 0:
                ax.set_title(ylabel)
            if c == 0:
                ax.set_ylabel(ds_title, fontweight="bold")
            y_max = max((m + s for m, s in zip(means, stds)), default=1.0)
            ax.set_ylim(0, max(y_max * 1.25, 1e-3))
            for bar, m_val, s_val in zip(bars, means, stds):
                ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + s_val + 0.02 * y_max,
                        fmt.format(m_val), ha="center", va="bottom", fontsize=6.8)
    fig.suptitle("Multi-Seed Benchmark Across Optimizers, Classical & Literature Baselines", fontsize=11.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save_fig(plt, fig, fig_dir, "fig_main_comparison")


def _plot_shot_scaling(plt: Any, bundle: Dict[str, Any], fig_dir: str) -> Optional[str]:
    entries = bundle.get("shot_scaling", [])
    if len(entries) < 2:
        return None
    shots_k = np.array([e["results"]["shots_used"] / 1000.0 for e in entries])
    panels = [
        ("clf_acc", 100.0, "Classifier Accuracy (%) ↑"),
        ("cond_latent_fid", 1.0, "Conditional Latent FID ↓"),
        ("feature_fid", 1.0, "Semantic Feature FID ↓"),
    ]
    styles = {"qerd": ("QER (Ours)", "#0072B2", "o"), "adam_spsa": ("Adam-SPSA", "#D55E00", "s"), "classical_ddpm": ("Classical-DDPM", "#009E73", "D")}
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6))
    for ax, (m_key, scale, title) in zip(axes, panels):
        for key, (name, color, marker) in styles.items():
            xs, ys, es = [], [], []
            for s_val, e in zip(shots_k, entries):
                agg = e["results"]["aggregate"]
                if key in agg:
                    xs.append(s_val)
                    ys.append(agg[key][m_key]["mean"] * scale)
                    es.append(agg[key][m_key]["std"] * scale)
            if xs:
                xs_a, ys_a, es_a = np.array(xs), np.array(ys), np.array(es)
                ax.plot(xs_a, ys_a, marker=marker, linewidth=2.0, markersize=5.5, color=color, label=name)
                ax.fill_between(xs_a, ys_a - es_a, ys_a + es_a, color=color, alpha=0.16)
        ax.set_xscale("log", base=2)
        ax.set_xticks(shots_k)
        ax.set_xticklabels([f"{int(s)}k" for s in shots_k])
        ax.set_xlabel("Total QPU Shot Budget")
        ax.set_title(title)
        ax.grid(True, which="both", linestyle="--", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(f"QPU Shot-Budget Scaling ({entries[0]['results']['dataset'].upper()}, Mean ± 1σ)", fontsize=11.5, fontweight="bold", y=1.01)
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 0.945), fontsize=9.0)
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    return _save_fig(plt, fig, fig_dir, "fig_shot_scaling")


def _plot_ablations(plt: Any, bundle: Dict[str, Any], fig_dir: str) -> Optional[str]:
    abl_dict = {k: v for k, v in bundle.get("ablations", {}).items() if v}
    if not abl_dict:
        return None
    ds_items = list(abl_dict.items())
    fig, axes = plt.subplots(len(ds_items), 3, figsize=(13.6, 3.4 * len(ds_items)), squeeze=False)
    for r, (ds_label, abl_entries) in enumerate(ds_items):
        abls = [e["results"]["ablate"] for e in abl_entries]
        x = np.arange(len(abls))
        ref_none = next((e["results"] for e in abl_entries if e["results"]["ablate"] == "none"), abl_entries[0]["results"])
        ddpm_ref = ref_none["aggregate"].get("classical_ddpm")
        for c, (m_key, scale, title, fmt) in enumerate([
            ("clf_acc", 100.0, "ClfAcc (%) ↑", "{:.1f}"),
            ("cond_latent_fid", 1.0, "CondLatFID ↓", "{:.2f}"),
            ("feature_fid", 1.0, "FeatFID ↓", "{:.0f}"),
        ]):
            ax = axes[r, c]
            means = [e["results"]["aggregate"]["qerd"][m_key]["mean"] * scale for e in abl_entries]
            stds = [e["results"]["aggregate"]["qerd"][m_key]["std"] * scale for e in abl_entries]
            colors = [_ABLATION_SPECS[a][2] if a in _ABLATION_SPECS else "#666666" for a in abls]
            bars = ax.bar(x, means, yerr=stds, capsize=3.0, width=0.58, color=colors, edgecolor="#222222", linewidth=0.7)
            if ddpm_ref is not None:
                ax.axhline(ddpm_ref[m_key]["mean"] * scale, color="#009E73", linestyle="--", linewidth=1.2, label="Classical-DDPM")
            ax.set_xticks(x)
            ax.set_xticklabels([_ABLATION_SPECS[a][0] if a in _ABLATION_SPECS else a for a in abls], rotation=22, ha="right", fontsize=7.6)
            ax.grid(axis="y", linestyle="--", alpha=0.25)
            if r == 0:
                ax.set_title(title)
            if c == 0:
                ax.set_ylabel(ds_label, fontweight="bold")
                if r == 0 and ddpm_ref is not None:
                    ax.legend(frameon=False, loc="upper right", fontsize=7.8)
            ddpm_v = ddpm_ref[m_key]["mean"] * scale if ddpm_ref is not None else 0.0
            y_max = max((*[m + s for m, s in zip(means, stds)], ddpm_v), default=1.0)
            ax.set_ylim(0, max(y_max * 1.34, 1e-3))
            for bar, m_val, s_val in zip(bars, means, stds):
                ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + s_val + 0.02 * y_max,
                        fmt.format(m_val), ha="center", va="bottom", fontsize=7.2)
    fig.suptitle("Component & Calibration Ablation Study", fontsize=11.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save_fig(plt, fig, fig_dir, "fig_ablations")


def _plot_complexity_timing(plt: Any, bundle: Dict[str, Any], fig_dir: str) -> Optional[str]:
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11.4, 4.0))
    p_grid = np.array([20, 40, 90, 116, 198, 224, 384, 512], dtype=np.float64)
    ax_l.plot(p_grid, 2.0 * p_grid, "o-", color="#E69F00", linewidth=2.0, label="PSR ($2P$ evals/sample)")
    ax_l.plot(p_grid, np.full_like(p_grid, 2.0), "s-", color="#0072B2", linewidth=2.2, label="QER Contrastive (2 evals, $O(1)$)")
    ax_l.plot(p_grid, np.full_like(p_grid, 2.0), "--", color="#D55E00", linewidth=1.8, label="Adam-SPSA (2 evals/batch, $O(P)$ var)")
    ax_l.set_yscale("log")
    ax_l.set_ylim(1.2, 1600)
    ax_l.set_xlabel("Active Circuit Parameter Count $P$")
    ax_l.set_ylabel("Circuit Evals / Gradient")
    ax_l.set_title("(a) Gradient Circuit Scaling ($O(1)$ vs. $O(P)$)")
    ax_l.grid(True, which="both", linestyle="--", alpha=0.25)
    ax_l.legend(frameon=False, loc="center right", bbox_to_anchor=(0.99, 0.45))

    entries = bundle.get("main", [])
    ds_names, qer_t, spsa_t, psr_t = [], [], [], []
    for e in entries:
        t_map = e.get("log_meta", {}).get("timings_sec", {})
        ds = e["results"]["dataset"]
        ds_names.append("MNIST\n(12+0q)" if ds == "mnist" else ("Fashion\n(12+2q)" if ds == "fashion" else ds.upper()))
        qer_t.append(t_map.get("QERD", {}).get("mean", 0.0))
        spsa_t.append(t_map.get("Adam-SPSA", {}).get("mean", 0.0))
        psr_t.append(t_map.get("Adam-PSR", {}).get("mean", 0.0))
    if ds_names and any(v > 0 for v in (*qer_t, *spsa_t, *psr_t)):
        x = np.arange(len(ds_names))
        w = 0.25
        b1 = ax_r.bar(x - w, qer_t, width=w, color="#0072B2", edgecolor="#222222", linewidth=0.7, label="QER")
        b2 = ax_r.bar(x, spsa_t, width=w, color="#D55E00", edgecolor="#222222", linewidth=0.7, label="Adam-SPSA")
        b3 = ax_r.bar(x + w, psr_t, width=w, color="#E69F00", edgecolor="#222222", linewidth=0.7, label="Adam-PSR-FD")
        t_max = max((*qer_t, *spsa_t, *psr_t, 1.0))
        ax_r.set_ylim(0, t_max * 1.38)
        for bars in (b1, b2, b3):
            for bar in bars:
                if bar.get_height() > 0:
                    ax_r.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.025 * t_max,
                              f"{bar.get_height():.1f}s", ha="center", va="bottom", fontsize=7.6)
        ax_r.set_xticks(x)
        ax_r.set_xticklabels(ds_names)
        ax_r.set_ylabel("Mean Wall-Clock Time / Seed (s)")
        ax_r.set_title("(b) Measured Simulator Runtime")
        ax_r.grid(axis="y", linestyle="--", alpha=0.25)
        ax_r.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    return _save_fig(plt, fig, fig_dir, "fig_complexity_timing")


def _plot_grad_alignment(plt: Any, bundle: Dict[str, Any], fig_dir: str) -> Optional[str]:
    gs = bundle.get("grad_study")
    if not gs:
        return None
    ga = gs["grad_alignment"]
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 3.4))
    panels = [
        (ga["step_curve"], "step", "(a) Alignment Across Steps", "Optimization Step $t$"),
        (ga["beta_sweep"], "beta", "(b) Perturbation Radius $\\beta$", "Contrastive Radius $\\beta$"),
        (ga["depth_sweep"], "n_layers", "(c) Trotter Depth $L$", "Trotter Layers $L$"),
        (ga["noise_sweep"], "p_depol", "(d) Depolarizing Noise", "Layer Noise $p_{\\text{depol}}$"),
    ]
    curves = [
        ("cos_mse_all", "Exact vs $-\\nabla\\mathcal{L}_{\\mathrm{MSE}}$ (all)", "#0072B2", "o-"),
        ("cos_mse_zz", "$ZZ$ Couplers Only", "#CC79A7", "^--"),
        ("cos_corr_all", "vs Corr. Surrogate (Prop. 1)", "#009E73", "s-"),
        ("cos_shot_mse_all", "Shot-QER vs $-\\nabla\\mathcal{L}_{\\mathrm{MSE}}$", "#D55E00", "d:"),
    ]
    for ax, (rows, x_key, title, xlabel) in zip(axes, panels):
        xs = [r[x_key] for r in rows]
        for y_key, lbl, col, fmt in curves:
            ax.plot(xs, [r[y_key] for r in rows], fmt, color=col, linewidth=1.8, markersize=4.8, label=lbl)
        ax.axhline(0.0, color="#666666", linestyle="-", linewidth=0.7, alpha=0.5)
        ax.set_ylim(-0.1, 1.08)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(True, linestyle="--", alpha=0.25)
    axes[0].set_ylabel("Cosine Similarity $\\cos(\\hat{g}_{\\mathrm{QER}}, \\cdot)$")
    axes[0].legend(frameon=False, loc="lower left", fontsize=7.2)
    fig.suptitle(f"Proposition 1 Empirical Validation: Local-Response Gradient Fidelity ({gs['dataset'].upper()})", fontsize=11.0, fontweight="bold", y=1.03)
    fig.tight_layout()
    return _save_fig(plt, fig, fig_dir, "fig_grad_alignment")


def _plot_noise_topo(plt: Any, bundle: Dict[str, Any], fig_dir: str) -> Optional[str]:
    ns = bundle.get("noise_sweep", [])
    ts = bundle.get("topo_sweep", [])
    ds_e = bundle.get("depth_sweep", [])
    if not (ns or ts or ds_e):
        return None
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.6))

    if ns:
        ax = axes[0]
        xs = [e["results"]["p_depol"] for e in ns]
        ys = [e["results"]["aggregate"]["qerd"]["clf_acc"]["mean"] * 100.0 for e in ns]
        es = [e["results"]["aggregate"]["qerd"]["clf_acc"]["std"] * 100.0 for e in ns]
        fids = [e["results"]["aggregate"]["qerd"]["feature_fid"]["mean"] for e in ns]
        ax.errorbar(xs, ys, yerr=es, marker="o", color="#0072B2", linewidth=2.0, capsize=3.5, label="ClfAcc (%) ↑")
        ax2 = ax.twinx()
        ax2.plot(xs, fids, marker="s", linestyle="--", color="#D55E00", linewidth=1.7, label="FeatFID ↓")
        ax2.spines["top"].set_visible(False)
        ax.set_xlabel("Depolarizing Noise $p_{\\text{depol}}$")
        ax.set_ylabel("ClfAcc (%) ↑", color="#0072B2")
        ax2.set_ylabel("FeatFID ↓", color="#D55E00")
        ax.set_title("(a) NISQ Depolarizing Stress Test")
        ax.grid(True, linestyle="--", alpha=0.25)

    for ax, entries, x_fn, xlabel, title, col in [
        (axes[1], ts, lambda r: r["topology"], "Coupling Graph (+2Q CNOT Noise)", "(b) Hardware Topology", "#009E73"),
        (axes[2], ds_e, lambda r: f"L={int(r['n_layers'])}\n({int(r.get('two_qubit_cnots', 0))}q2)", "Circuit Depth (2Q CNOT Count)", "(c) Trotter Depth $L$ (+2Q Noise)", "#CC79A7"),
    ]:
        if not entries:
            continue
        lbls = [x_fn(e["results"]) for e in entries]
        means = [e["results"]["aggregate"]["qerd"]["clf_acc"]["mean"] * 100.0 for e in entries]
        stds = [e["results"]["aggregate"]["qerd"]["clf_acc"]["std"] * 100.0 for e in entries]
        x = np.arange(len(lbls))
        bars = ax.bar(x, means, yerr=stds, capsize=3.0, width=0.56, color=col, edgecolor="#222222", linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(lbls, fontsize=8.0)
        ax.set_ylim(0, max((m + s for m, s in zip(means, stds)), default=100.0) * 1.25)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("ClfAcc (%) ↑")
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        for bar, m_v, s_v in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + s_v + 1.5, f"{m_v:.1f}%", ha="center", va="bottom", fontsize=7.4)

    fig.suptitle("Hardware Realism: Gate Noise, Coupling Topology, and Depth Tradeoffs", fontsize=11.0, fontweight="bold", y=1.03)
    fig.tight_layout()
    return _save_fig(plt, fig, fig_dir, "fig_noise_topo")


def _plot_bp_entanglement(plt: Any, bundle: Dict[str, Any], fig_dir: str) -> Optional[str]:
    pq = bundle.get("pure_qpu_parity")
    uw = bundle.get("unwhitened_ae", [])
    if not (pq or uw):
        return None
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))

    if pq:
        ax = axes[0]
        order = ["qerd_all", "qerd_grid8", "qerd_grid4", "qerd_chain", "qerd_nozz", "classical_ising_ep"]
        lbl_map = {"qerd_all": "all", "qerd_grid8": "grid8", "qerd_grid4": "grid4", "qerd_chain": "chain", "qerd_nozz": "no-zz", "classical_ising_ep": "Cls-EP"}
        keys = [k for k in order if k in pq["aggregate"]]
        means = [pq["aggregate"][k]["intra_corr"]["mean"] for k in keys]
        stds = [pq["aggregate"][k]["intra_corr"]["std"] for k in keys]
        cols = ["#0072B2" if k != "qerd_nozz" else "#CC79A7" for k in keys]
        x = np.arange(len(keys))
        ax.axhline(0.0, color="#444444", linewidth=0.7)
        bars = ax.bar(x, means, yerr=stds, capsize=3.0, width=0.58, color=cols, edgecolor="#222222", linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([lbl_map[k] for k in keys], rotation=20, ha="right")
        y_lo = min(0.0, min((m - s for m, s in zip(means, stds)), default=0.0) - 0.12)
        y_hi = max((m + s for m, s in zip(means, stds)), default=1.0) * 1.28
        ax.set_ylim(y_lo, max(y_hi, 0.25))
        ax.set_ylabel("Intra-Cluster Correlation ↑")
        ax.set_title("(a) Zero-AE 12q Cluster Spin Glass")
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        for bar, m_v, s_v in zip(bars, means, stds):
            y_pos = (m_v + s_v + 0.02) if m_v >= 0 else 0.02
            ax.text(bar.get_x() + bar.get_width() / 2.0, y_pos, f"{m_v:.2f}", ha="center", va="bottom", fontsize=7.4)

    if uw:
        ax = axes[1]
        lbls = ["Full $ZZ$" if e["results"]["ablate"] == "none" else "No-$ZZ$" for e in uw]
        means = [e["results"]["aggregate"]["qerd"]["clf_acc"]["mean"] * 100.0 for e in uw]
        stds = [e["results"]["aggregate"]["qerd"]["clf_acc"]["std"] * 100.0 for e in uw]
        x = np.arange(len(lbls))
        bars = ax.bar(x, means, yerr=stds, capsize=3.5, width=0.46, color=["#0072B2", "#CC79A7"], edgecolor="#222222", linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(lbls)
        ax.set_ylim(0, max((m + s for m, s in zip(means, stds)), default=100.0) * 1.25)
        ax.set_ylabel("ClfAcc (%) ↑")
        ax.set_title("(b) Unwhitened AE (no BatchNorm1d)")
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        for bar, m_v, s_v in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + s_v + 1.5, f"{m_v:.1f}%", ha="center", va="bottom", fontsize=7.6)

    fig.suptitle("Genuine Two-Qubit $ZZ$ Entanglement Advantage Inside NISQ Circuits", fontsize=11.0, fontweight="bold", y=1.03)
    fig.tight_layout()
    return _save_fig(plt, fig, fig_dir, "fig_bp_entanglement")


def _copy_sample_grids(bundle: Dict[str, Any], fig_dir: str) -> List[str]:
    copied: List[str] = []
    os.makedirs(fig_dir, exist_ok=True)
    tags = [("qerd_samples.png", "qerd"), ("spsa_samples.png", "spsa"), ("psr_samples.png", "psr"),
            ("classical_samples.png", "ddpm"), ("real_test.png", "real")]
    for entry in bundle.get("main", []):
        save_dir = entry.get("save_dir", "")
        ds = entry["results"]["dataset"]
        for src_suffix, dst_tag in tags:
            src_path = os.path.join(save_dir, f"{ds}_{src_suffix}")
            if os.path.exists(src_path):
                dst_path = os.path.join(fig_dir, f"sample_{ds}_{dst_tag}.png")
                shutil.copy2(src_path, dst_path)
                copied.append(dst_path)
    return copied


def generate_paper_assets(bundle: Dict[str, Any], out_dir: str) -> Dict[str, Any]:
    """Generate publication figures (`figures/*.pdf/.png`) and LaTeX `paper_tables.tex`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_pub_style(plt)
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    figs = [
        fn(plt, bundle, fig_dir) for fn in (
            _plot_main_comparison, _plot_shot_scaling, _plot_ablations,
            _plot_complexity_timing, _plot_grad_alignment, _plot_noise_topo, _plot_bp_entanglement,
        )
    ]
    copied_samples = _copy_sample_grids(bundle, fig_dir)
    tex_path = write_latex_tables(bundle, out_dir)
    return {
        "fig_dir": fig_dir,
        "figures": [f for f in figs if f is not None],
        "copied_samples": copied_samples,
        "paper_tables_tex": tex_path,
    }


# ==============================================================================
# 5. ORCHESTRATION MAIN
# ==============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Run full QER paper benchmark suite & emit tables/figures.")
    parser.add_argument("--out-dir", type=str, default="./paper_suite")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "numpy", "torch", "pennylane"])
    parser.add_argument("--torch-device", type=str, default="auto")
    parser.add_argument("--main-seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--sweep-seeds", type=str, default="0,1,2")
    parser.add_argument("--eval-per-class", type=int, default=200)
    parser.add_argument("--shot-budgets", type=str, default="10000,20000,40000,80000,160000")
    parser.add_argument("--ablations", type=str, default="none,gain-cal,gain-cal-inv,no-zz,wcond-only,sgd")
    parser.add_argument("--reuse-cache-dir", type=str, default="")
    parser.add_argument("--skip-main", action="store_true")
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--skip-shots", action="store_true")
    parser.add_argument("--skip-grad-study", action="store_true")
    parser.add_argument("--skip-sweeps", action="store_true")
    parser.add_argument("--skip-correlated", action="store_true")
    parser.add_argument("--skip-controls", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--plots-only", type=str, default="")
    parser.add_argument("--quick", action="store_true", help="Fast offline smoke test on sklearn digits")
    args = parser.parse_args()

    if args.plots_only:
        with open(args.plots_only, "r", encoding="utf-8") as f:
            bundle = json.load(f)
        out_dir = args.out_dir if args.out_dir != "./paper_suite" else os.path.dirname(os.path.abspath(args.plots_only))
        assets = generate_paper_assets(bundle, out_dir)
        print(f"Generated figures in {assets['fig_dir']}: {assets['figures']}", flush=True)
        print(f"Generated LaTeX macros: {assets['paper_tables_tex']}", flush=True)
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    bench_script = os.path.join(script_dir, "benchmark_mnist.py")
    if not os.path.exists(bench_script):
        raise FileNotFoundError(f"Could not find benchmark_mnist.py next to {__file__}")

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "run_full_log.txt")
    t_suite_start = time.time()

    if args.quick:
        datasets_cfg = [{
            "key": "digits", "label": "Digits (Quick Smoke)", "dataset": "digits",
            "latent_dim": 8, "n_hidden": 0, "ae_epochs": 4, "clf_epochs": 3,
            "main_shots": 12_000, "main_seeds": "0", "sweep_seeds": "0", "eval_per_class": 12,
            "shot_budgets": [6_000, 12_000], "ablations": ["none", "gain-cal", "gain-cal-inv", "no-zz", "wcond-only", "sgd"],
        }]
        noise_grid = [0.0, 0.035]
        topo_grid = ["all", "grid4", "chain"]
        depth_grid = [1, 2]
        parity_shots, parity_eval = 12_000, 120
    else:
        shot_budgets = _parse_csv_ints(args.shot_budgets)
        ablations = _parse_csv_strs(args.ablations)
        datasets_cfg = [
            {
                "key": "mnist", "label": "MNIST", "dataset": "mnist",
                "latent_dim": 12, "n_hidden": 0, "ae_epochs": 8, "clf_epochs": 5,
                "main_shots": 80_000, "main_seeds": args.main_seeds, "sweep_seeds": args.sweep_seeds,
                "eval_per_class": args.eval_per_class, "shot_budgets": shot_budgets, "ablations": ablations,
            },
            {
                "key": "fashion", "label": "Fashion-MNIST", "dataset": "fashion",
                "latent_dim": 12, "n_hidden": 2, "ae_epochs": 12, "clf_epochs": 8,
                "main_shots": 80_000, "main_seeds": args.main_seeds, "sweep_seeds": args.sweep_seeds,
                "eval_per_class": args.eval_per_class, "shot_budgets": [], "ablations": ablations,
            },
        ]
        noise_grid = [0.0, 0.01, 0.035, 0.06, 0.08]
        topo_grid = ["all", "grid8", "grid4", "chain"]
        depth_grid = [1, 2, 4, 6]
        parity_shots, parity_eval = 60_000, 320

    main_entries: List[Dict[str, Any]] = []
    ablation_entries: Dict[str, List[Dict[str, Any]]] = {cfg["label"]: [] for cfg in datasets_cfg}
    shot_entries: List[Dict[str, Any]] = []
    grad_study: Optional[Dict[str, Any]] = None
    noise_entries: List[Dict[str, Any]] = []
    topo_entries: List[Dict[str, Any]] = []
    depth_entries: List[Dict[str, Any]] = []
    unwhitened_entries: List[Dict[str, Any]] = []
    pure_qpu_parity: Optional[Dict[str, Any]] = None

    with open(log_path, "w", encoding="utf-8") as log_fp:
        # 1. Main Multi-Seed Tables
        for cfg in datasets_cfg:
            ds = cfg["dataset"]
            main_dir = os.path.join(args.out_dir, f"{cfg['key']}_main")
            helper_name = _helper_filename(ds, cfg["latent_dim"], cfg["ae_epochs"], cfg["clf_epochs"])
            _copy_helper_cache(args.reuse_cache_dir, main_dir, helper_name,
                               extra_search_dirs=(f"./paper_{cfg['key']}", "./paper_fashion", "./paper_mnist"))
            if not args.skip_main:
                cli = [
                    "--dataset", ds, "--latent-dim", str(cfg["latent_dim"]), "--n-hidden", str(cfg["n_hidden"]),
                    "--ae-epochs", str(cfg["ae_epochs"]), "--clf-epochs", str(cfg["clf_epochs"]),
                    "--max-shots", str(cfg["main_shots"]), "--eval-per-class", str(cfg["eval_per_class"]),
                    "--seeds", cfg["main_seeds"], "--backend", args.backend, "--torch-device", args.torch_device,
                ]
                if not args.skip_controls:
                    cli.extend(["--with-controls", "--with-lit-baselines"])
                main_entries.append(_run_bench(bench_script, f"Main {cfg['label']}", cli, main_dir, ds, "none", log_fp))

        # 2. Component Ablation Studies (6-row)
        if not args.skip_ablations:
            for cfg in datasets_cfg:
                ds = cfg["dataset"]
                main_dir = os.path.join(args.out_dir, f"{cfg['key']}_main")
                abl_dir = os.path.join(args.out_dir, f"{cfg['key']}_abl")
                helper_name = _helper_filename(ds, cfg["latent_dim"], cfg["ae_epochs"], cfg["clf_epochs"])
                _copy_helper_cache(main_dir, abl_dir, helper_name, extra_search_dirs=(args.reuse_cache_dir,))
                for abl in cfg["ablations"]:
                    cli = [
                        "--dataset", ds, "--latent-dim", str(cfg["latent_dim"]), "--n-hidden", str(cfg["n_hidden"]),
                        "--ae-epochs", str(cfg["ae_epochs"]), "--clf-epochs", str(cfg["clf_epochs"]),
                        "--max-shots", str(cfg["main_shots"]), "--eval-per-class", str(cfg["eval_per_class"]),
                        "--seeds", cfg["sweep_seeds"], "--ablate", abl, "--skip-psr",
                        "--backend", args.backend, "--torch-device", args.torch_device,
                    ]
                    ablation_entries[cfg["label"]].append(
                        _run_bench(bench_script, f"{cfg['label']} Ablation ({abl})", cli, abl_dir, ds, abl, log_fp)
                    )

        # 3. Shot-Budget Scaling Curve
        if not args.skip_shots:
            for cfg in datasets_cfg:
                if not cfg["shot_budgets"]:
                    continue
                ds = cfg["dataset"]
                main_dir = os.path.join(args.out_dir, f"{cfg['key']}_main")
                helper_name = _helper_filename(ds, cfg["latent_dim"], cfg["ae_epochs"], cfg["clf_epochs"])
                for shots in cfg["shot_budgets"]:
                    shot_dir = os.path.join(args.out_dir, f"{cfg['key']}_shots_{shots}")
                    _copy_helper_cache(main_dir, shot_dir, helper_name, extra_search_dirs=(args.reuse_cache_dir,))
                    cli = [
                        "--dataset", ds, "--latent-dim", str(cfg["latent_dim"]), "--n-hidden", str(cfg["n_hidden"]),
                        "--ae-epochs", str(cfg["ae_epochs"]), "--clf-epochs", str(cfg["clf_epochs"]),
                        "--max-shots", str(shots), "--eval-per-class", str(cfg["eval_per_class"]),
                        "--seeds", cfg["sweep_seeds"], "--skip-psr",
                        "--backend", args.backend, "--torch-device", args.torch_device,
                    ]
                    shot_entries.append(_run_bench(bench_script, f"{cfg['label']} Shots={shots}", cli, shot_dir, ds, "none", log_fp))

        # 4. Proposition 1 Gradient Fidelity & Barren-Plateau Scaling
        primary_cfg = datasets_cfg[0]
        primary_main_dir = os.path.join(args.out_dir, f"{primary_cfg['key']}_main")
        if not args.skip_grad_study:
            grad_study = _run_grad_and_bp_study(primary_cfg, primary_main_dir, args.torch_device, args.quick, log_fp)

        # 5. Hardware Realism Sweeps (Noise, Topology+2Q, Depth+2Q on primary dataset)
        if not args.skip_sweeps:
            ds = primary_cfg["dataset"]
            helper_name = _helper_filename(ds, primary_cfg["latent_dim"], primary_cfg["ae_epochs"], primary_cfg["clf_epochs"])
            sweeps_dir = os.path.join(args.out_dir, f"{primary_cfg['key']}_hw_sweeps")
            _copy_helper_cache(primary_main_dir, sweeps_dir, helper_name, extra_search_dirs=(args.reuse_cache_dir,))

            base_hw_cli = [
                "--dataset", ds, "--latent-dim", str(primary_cfg["latent_dim"]), "--n-hidden", str(primary_cfg["n_hidden"]),
                "--ae-epochs", str(primary_cfg["ae_epochs"]), "--clf-epochs", str(primary_cfg["clf_epochs"]),
                "--max-shots", str(primary_cfg["main_shots"]), "--eval-per-class", str(primary_cfg["eval_per_class"]),
                "--seeds", primary_cfg["sweep_seeds"], "--skip-psr",
                "--backend", args.backend, "--torch-device", args.torch_device,
            ]
            for pd in noise_grid:
                sfx = f"_p1q_{pd:.3f}".replace(".", "p")
                e = _run_bench(bench_script, f"Noise p_depol={pd:.3f}", [*base_hw_cli, "--p-depol", str(pd)],
                               sweeps_dir, ds, "none", log_fp, tag_suffix=sfx)
                e["sweep_tag"] = f"p={pd:.3f}"
                noise_entries.append(e)

            for tp in topo_grid:
                sfx = f"_topo_{tp}"
                e = _run_bench(bench_script, f"Topo {tp} (+2Q noise)",
                               [*base_hw_cli, "--topology", tp, "--p-depol", "0.005", "--p-depol-2q", "0.015"],
                               sweeps_dir, ds, "none", log_fp, tag_suffix=sfx)
                e["sweep_tag"] = f"topo={tp}"
                topo_entries.append(e)

            for nl in depth_grid:
                sfx = f"_depth_L{nl}"
                e = _run_bench(bench_script, f"Depth L={nl} (+2Q noise)",
                               [*base_hw_cli, "--n-layers", str(nl), "--p-depol", "0.005", "--p-depol-2q", "0.015"],
                               sweeps_dir, ds, "none", log_fp, tag_suffix=sfx)
                e["sweep_tag"] = f"L={nl}"
                depth_entries.append(e)

        # 6. Genuine Two-Qubit ZZ Entanglement Controls (Unwhitened AE + Pure-QPU Cluster Parity)
        if not args.skip_correlated:
            uw_cfg = datasets_cfg[-1]
            uw_ds = uw_cfg["dataset"]
            uw_dir = os.path.join(args.out_dir, f"{uw_cfg['key']}_unwhitened")
            uw_base = [
                "--dataset", uw_ds, "--latent-dim", str(uw_cfg["latent_dim"]), "--n-hidden", str(uw_cfg["n_hidden"]),
                "--ae-epochs", str(uw_cfg["ae_epochs"]), "--clf-epochs", str(uw_cfg["clf_epochs"]),
                "--max-shots", str(uw_cfg["main_shots"]), "--eval-per-class", str(uw_cfg["eval_per_class"]),
                "--seeds", uw_cfg["sweep_seeds"], "--no-latent-bn", "--skip-psr",
                "--backend", args.backend, "--torch-device", args.torch_device,
            ]
            for abl in ("none", "no-zz"):
                e = _run_bench(bench_script, f"Unwhitened {uw_cfg['label']} ({abl})", [*uw_base, "--ablate", abl],
                               uw_dir, uw_ds, abl, log_fp, tag_suffix="_nobn")
                e["sweep_tag"] = f"nobn_{abl}"
                unwhitened_entries.append(e)

            pure_qpu_parity = _run_pure_qpu_parity_study(
                seeds=_parse_csv_ints(datasets_cfg[0]["sweep_seeds"]),
                shots=parity_shots, n_eval=parity_eval,
                backend=args.backend, torch_device_str=args.torch_device, log_fp=log_fp,
            )

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    bundle = {
        "environment": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_name": device_name,
            "backend": args.backend,
            "torch_device": args.torch_device,
        },
        "total_wall_sec": round(time.time() - t_suite_start, 2),
        "main": main_entries,
        "ablations": ablation_entries,
        "shot_scaling": shot_entries,
        "grad_study": grad_study,
        "noise_sweep": noise_entries,
        "topo_sweep": topo_entries,
        "depth_sweep": depth_entries,
        "unwhitened_ae": unwhitened_entries,
        "pure_qpu_parity": pure_qpu_parity,
    }

    all_json_path = os.path.join(args.out_dir, "paper_all_results.json")
    with open(all_json_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)

    report_path = os.path.join(args.out_dir, "paper_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(build_markdown_report(bundle))

    assets = generate_paper_assets(bundle, args.out_dir) if not args.skip_plots else None
    print("\n" + "=" * 72, flush=True)
    print(f"ALL BENCHMARKS COMPLETE in {bundle['total_wall_sec']:.1f}s!", flush=True)
    print(f"Shareable report   : {os.path.abspath(report_path)}", flush=True)
    print(f"Full JSON bundle   : {os.path.abspath(all_json_path)}", flush=True)
    if assets is not None:
        print(f"LaTeX table macros : {os.path.abspath(assets['paper_tables_tex'])}", flush=True)
        print(f"Figures folder     : {os.path.abspath(assets['fig_dir'])}", flush=True)
    print(f"Full raw log       : {os.path.abspath(log_path)}", flush=True)
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
