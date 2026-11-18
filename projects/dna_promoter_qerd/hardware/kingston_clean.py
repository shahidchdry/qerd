"""
Noise-pruned 128-qubit Heavy-Hex snake subgraph selection and 1-job live marginal readout mitigation.

Fixes Project 1's 156q tail-noise bottleneck (`16` qubits >5% RO, `24` couplers >2% CZ) by choosing
a noise-pruned connected `128q` subgraph ordered along native Heavy-Hex paths so `(2i, 2i+1)` DNA
nucleotide qubit pairs directly share physical CZ couplers with zero SWAP routing overhead.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.primitives import StatevectorSampler
from qiskit.providers.fake_provider import GenericBackendV2
import qiskit_ibm_runtime.fake_provider as fp

from qerd import build_ising_edges, color_ising_edges


FAKE_HERON_MAP = {
    "ibm_kingston": "FakeKingston",
    "ibm_fez": "FakeFez",
    "ibm_marrakesh": "FakeMarrakesh",
}


@dataclass(frozen=True)
class SubgraphReport:
    device: str
    n_qubits: int
    native_edges: int
    intra_base_coupled: int
    total_bases: int
    med_ro_err_pct: float
    p90_ro_err_pct: float
    bad_ro_gt5pct: int
    med_cz_err_pct: float
    p90_cz_err_pct: float
    bad_cz_gt2pct: int


def get_fake_heron(device: str):
    return getattr(fp, FAKE_HERON_MAP.get(device.lower(), "FakeKingston"))()


def select_clean_dna_subgraph(backend, n_qubits: int = 128) -> Tuple[List[Tuple[int, int]], List[int], SubgraphReport]:
    """
    Greedily grow a clean `n_qubits` connected subgraph weighted heavily against high readout
    error qubits & high 2Q error couplers, then order nodes by Heavy-Hex DFS walk so sequential
    `(2i, 2i+1)` nucleotide qubit pairs preferentially sit directly across a native CZ coupler.
    """
    nq_dev = int(backend.num_qubits)
    tgt = backend.target
    ro = np.array(
        [
            float(tgt["measure"][(q,)].error)
            if "measure" in tgt and (q,) in tgt["measure"] and tgt["measure"][(q,)].error is not None
            else 0.015
            for q in range(nq_dev)
        ],
        dtype=np.float64,
    )
    cz_map: Dict[Tuple[int, int], float] = {}
    for op in ("cz", "ecr", "cx"):
        if op in tgt:
            for pair, prop in tgt[op].items():
                if len(pair) == 2 and pair[0] != pair[1]:
                    e = (min(int(pair[0]), int(pair[1])), max(int(pair[0]), int(pair[1])))
                    err = float(prop.error) if (prop is not None and prop.error is not None) else 0.005
                    cz_map[e] = min(cz_map.get(e, err), err)
    if not cz_map and hasattr(tgt, "build_coupling_map"):
        cmap = tgt.build_coupling_map()
        if cmap is not None:
            for u, v in cmap.get_edges():
                if u != v:
                    cz_map[(min(int(u), int(v)), max(int(u), int(v)))] = 0.005

    adj: Dict[int, List[int]] = {q: [] for q in range(nq_dev)}
    for u, v in cz_map:
        adj[u].append(v)
        adj[v].append(u)

    if n_qubits >= nq_dev:
        order = list(range(nq_dev))
    else:
        anchor = int(np.argmin(ro))
        chosen_set = {anchor}
        chosen_list = [anchor]
        while len(chosen_list) < n_qubits:
            cands = []
            for u in chosen_list:
                for v in adj[u]:
                    if v not in chosen_set:
                        e = (min(u, v), max(u, v))
                        # Strongly penalize tail-noisy qubits (>5% RO, >2% CZ)
                        pen_ro = 3.0 * ro[v] + (0.20 if ro[v] > 0.045 else 0.0)
                        pen_cz = 1.5 * cz_map[e] + (0.10 if cz_map[e] > 0.015 else 0.0)
                        cands.append((pen_ro + pen_cz, v))
            if not cands:
                break
            cands.sort()
            nxt = cands[0][1]
            chosen_set.add(nxt)
            chosen_list.append(nxt)

        local_adj = {
            u: sorted(
                [v for v in adj[u] if v in chosen_set],
                key=lambda v: cz_map[(min(u, v), max(u, v))],
            )
            for u in chosen_set
        }
        order, vis, stack = [], set(), [anchor]
        while stack:
            u = stack.pop()
            if u in vis:
                continue
            vis.add(u)
            order.append(u)
            for v in reversed(local_adj[u]):
                if v not in vis:
                    stack.append(v)
        for q in sorted(range(nq_dev), key=lambda i: ro[i]):
            if len(order) >= n_qubits:
                break
            if q not in vis:
                order.append(q)
                vis.add(q)

    idx_map = {p: i for i, p in enumerate(order)}
    log_edges = sorted(
        (min(idx_map[u], idx_map[v]), max(idx_map[u], idx_map[v]))
        for (u, v) in cz_map
        if u in idx_map and v in idx_map
    )
    edge_set = set(log_edges)
    intra = sum(1 for i in range(len(order) // 2) if (2 * i, 2 * i + 1) in edge_set)
    gro = ro[order]
    gcz = np.array([cz_map[(min(u, v), max(u, v))] for (u, v) in cz_map if u in idx_map and v in idx_map], dtype=np.float64)

    rep = SubgraphReport(
        device=getattr(backend, "name", "heron_r2"),
        n_qubits=len(order),
        native_edges=len(log_edges),
        intra_base_coupled=intra,
        total_bases=len(order) // 2,
        med_ro_err_pct=100.0 * float(np.median(gro)),
        p90_ro_err_pct=100.0 * float(np.percentile(gro, 90)),
        bad_ro_gt5pct=int(np.sum(gro > 0.05)),
        med_cz_err_pct=100.0 * float(np.median(gcz)) if len(gcz) else 0.0,
        p90_cz_err_pct=100.0 * float(np.percentile(gcz, 90)) if len(gcz) else 0.0,
        bad_cz_gt2pct=int(np.sum(gcz > 0.02)) if len(gcz) else 0,
    )
    return log_edges, order, rep


def _load_ibm_token(api_key_json: str = "") -> str:
    if not api_key_json:
        return ""
    import json

    with open(api_key_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    tok = data.get("apikey") or data.get("apiKey") or data.get("token") or ""
    if not tok:
        raise ValueError(f"No apikey/token field found in {api_key_json}.")
    return str(tok)


def _connect_ibm_service(instance: str = "", api_key_json: str = ""):
    from qiskit_ibm_runtime import QiskitRuntimeService

    tok = _load_ibm_token(api_key_json)
    channels = ("ibm_quantum_platform", "ibm_cloud", None) if tok else (None,)
    last_err = None
    for ch in channels:
        try:
            kw: Dict[str, Any] = {}
            if ch:
                kw["channel"] = ch
            if tok:
                kw["token"] = tok
            if instance:
                kw["instance"] = instance
            return QiskitRuntimeService(**kw)
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"IBM Runtime connection failed: {last_err}")


def measure_live_readout_calibration(
    qpu_backend: Any,
    phys_layout: List[int],
    shots: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run 1 lightweight hardware job preparing `|0...0>` and `|1...1>` on `phys_layout` to fit
    marginal symmetric/asymmetric readout correction:
      `z_true_i ~= (z_raw_i - offset_i) / scale_i`.
    """
    from qiskit_ibm_runtime import SamplerV2

    n = len(phys_layout)
    qc0 = QuantumCircuit(n)
    qc0.measure_all()
    qc1 = QuantumCircuit(n)
    for q in range(n):
        qc1.x(q)
    qc1.measure_all()

    tqc0 = transpile(qc0, backend=qpu_backend, initial_layout=phys_layout, optimization_level=1)
    tqc1 = transpile(qc1, backend=qpu_backend, initial_layout=phys_layout, optimization_level=1)
    sampler = SamplerV2(mode=qpu_backend)
    job = sampler.run([(tqc0, None, shots), (tqc1, None, shots)])
    res = job.result()

    def _mean_z(pub_r) -> np.ndarray:
        ba = pub_r.data.meas
        bits = ba.to_bool_array(order="little") if hasattr(ba, "to_bool_array") else (
            np.unpackbits(np.asarray(ba.array, dtype=np.uint8), axis=-1, bitorder="little")[..., :n].astype(bool)
        )
        if bits.ndim == 3:
            bits = bits[0]
        spins = 1.0 - 2.0 * bits.astype(np.float64)
        return np.mean(spins, axis=0)

    z0 = _mean_z(res[0])  # ideal +1
    z1 = _mean_z(res[1])  # ideal -1
    offset = 0.5 * (z0 + z1)
    scale = np.clip(0.5 * (z0 - z1), 0.65, 1.00)
    return offset, scale


def mitigate_expectations(z_batch: np.ndarray, offset: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Invert marginal readout bias cleanly on CPU (`0` QPU overhead)."""
    return np.clip((z_batch - offset[None, :]) / scale[None, :], -1.0, 1.0)


def prepare_dna_backend(
    mode: str,
    device: str,
    n_qubits: int,
    instance: str = "",
    execution_mode: str = "job",
    seed: int = 42,
    api_key_json: str = "",
) -> Tuple[List[Tuple[int, int]], Dict[str, Any], SubgraphReport, Optional[Tuple[np.ndarray, np.ndarray]]]:
    """Create noise-pruned Heavy-Hex DNA subgraph, realistic calibration, and backend kwargs."""
    if mode == "local_sim":
        hw_ref = get_fake_heron(device)
        edges, _, rep = select_clean_dna_subgraph(hw_ref, n_qubits=n_qubits)
        model_kw = {
            "backend": "auto",
            "real_qpu": False,
            "p_depol": round(0.015 + 2.0 * (rep.med_cz_err_pct / 100.0), 4),
            "p_readout": round(0.015 + (rep.med_ro_err_pct / 100.0), 4),
        }
        return edges, model_kw, rep, None

    if mode == "qiskit_sim":
        r = max(2, int(math.ceil(math.sqrt(n_qubits + 2))))
        c = max(2, int(math.ceil((n_qubits + 2) / r)))
        fake_hw = GenericBackendV2(num_qubits=r * c, coupling_map=build_ising_edges((r, c), topology="grid4"), seed=seed)
        edges, phys, rep = select_clean_dna_subgraph(fake_hw, n_qubits=n_qubits)
        model_kw = {
            "backend": "qiskit",
            "real_qpu": True,
            "device_name": "generic_fake",
            "device_kwargs": {
                "sampler": StatevectorSampler(seed=seed),
                "backend": fake_hw,
                "initial_layout": phys,
                "optimization_level": 1,
            },
        }
        return edges, model_kw, rep, None

    if mode == "ibm_qpu":
        svc = _connect_ibm_service(instance=instance, api_key_json=api_key_json)
        eff_mode = execution_mode
        try:
            plans = [str(inst.get("plan", "")).lower() for inst in (svc.instances() or [])]
            if plans and all("open" in p for p in plans) and eff_mode != "job":
                print(f"[IBM Open Plan] Automatically switching execution_mode='{eff_mode}' -> 'job'.")
                eff_mode = "job"
        except Exception:
            pass
        try:
            rem = (svc.usage() or {}).get("usage_remaining_seconds")
            if rem is not None:
                print(f"[IBM Runtime] Quota remaining before run: {rem}s", flush=True)
        except Exception:
            pass
        qpu = svc.backend(device)
        edges, phys, rep = select_clean_dna_subgraph(qpu, n_qubits=n_qubits)
        ro_cal = measure_live_readout_calibration(qpu, phys, shots=256)
        print(
            f"[Live RO Mitigation] Measured 1-job calibration on {len(phys)}q: "
            f"mean scale={ro_cal[1].mean():.3f}, min scale={ro_cal[1].min():.3f}, mean |offset|={np.abs(ro_cal[0]).mean():.3f}",
            flush=True,
        )
        model_kw = {
            "backend": "qiskit",
            "real_qpu": True,
            "device_name": device,
            "device_kwargs": {
                "backend": qpu,
                "initial_layout": phys,
                "execution_mode": eff_mode,
                "dynamical_decoupling": True,
                "twirling": False,
                "max_circuits_per_pub": 500,
                "max_pubs_per_job": 4,
            },
        }
        return edges, model_kw, rep, ro_cal

    raise ValueError(f"Unknown mode '{mode}'")
