"""
IBM Heron R2 calibration audit (ibm_kingston vs ibm_fez vs ibm_marrakesh),
noise-pruned native Heavy-Hex subgraph selection, 4,000 shots/sec pulse-time calculator,
1-job live marginal readout mitigation, and backend factory helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.primitives import StatevectorSampler
from qiskit.providers.fake_provider import GenericBackendV2
import qiskit_ibm_runtime.fake_provider as fp

from qerd import build_ising_edges, color_ising_edges, select_hardware_subgraph


FAKE_HERON_MAP = {
    "ibm_kingston": "FakeKingston",
    "ibm_fez": "FakeFez",
    "ibm_marrakesh": "FakeMarrakesh",
}


@dataclass(frozen=True)
class HeronAudit:
    device: str
    num_qubits: int
    native_edges: int
    edge_colors: int
    n_qerd_params: int
    transpiled_depth: int
    native_2q_gates: int
    transpile_sec: float
    med_2q_err_pct: float
    p90_2q_err_pct: float
    bad_2q_couplers_gt2pct: int
    med_ro_err_pct: float
    p90_ro_err_pct: float
    bad_ro_qubits_gt5pct: int
    med_t1_us: float
    med_t2_us: float


@dataclass(frozen=True)
class SubgraphAudit:
    device: str
    n_qubits: int
    native_edges: int
    edge_colors: int
    med_ro_err_pct: float
    p90_ro_err_pct: float
    bad_ro_gt5pct: int
    med_cz_err_pct: float
    p90_cz_err_pct: float
    bad_cz_gt2pct: int


@dataclass(frozen=True)
class RuntimeProjection:
    n_qubits: int
    n_params: int
    qerd_shots: int
    qerd_qpu_pulse_sec: float
    qerd_session_wall_est_min: float
    psr_equal_steps_shots: int
    psr_equal_steps_pulse_sec: float
    psr_equal_shots_steps: int
    circuit_compression_x: int


def get_fake_heron(device: str):
    cls_name = FAKE_HERON_MAP.get(device.lower(), "FakeKingston")
    return getattr(fp, cls_name)()


def _extract_error_maps(backend) -> Tuple[np.ndarray, Dict[Tuple[int, int], float]]:
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
    return ro, cz_map


def select_clean_subgraph(
    backend,
    n_qubits: int = 128,
    allowed_phys: Optional[Sequence[int]] = None,
) -> Tuple[List[Tuple[int, int]], List[int], SubgraphAudit]:
    """
    Greedily grow a clean connected Heavy-Hex subgraph penalizing noisy readout qubits (>4.5%)
    and bad two-qubit couplers (>1.5%), preserving zero-SWAP native connectivity.
    """
    nq_dev = int(backend.num_qubits)
    ro, cz_map = _extract_error_maps(backend)
    allowed = set(range(nq_dev)) if allowed_phys is None else {int(q) for q in allowed_phys}
    if n_qubits > len(allowed):
        raise ValueError(f"Requested {n_qubits} qubits, but allowed pool only has {len(allowed)}.")

    adj: Dict[int, List[int]] = {q: [] for q in allowed}
    for u, v in cz_map:
        if u in allowed and v in allowed:
            adj[u].append(v)
            adj[v].append(u)

    if n_qubits == nq_dev and allowed_phys is None:
        order = list(range(nq_dev))
    elif n_qubits == len(allowed):
        order = list(allowed)
    else:
        anchor = min(allowed, key=lambda q: ro[q])
        chosen_set = {anchor}
        chosen_list = [anchor]
        while len(chosen_list) < n_qubits:
            cands = []
            for u in chosen_list:
                for v in adj[u]:
                    if v not in chosen_set:
                        e = (min(u, v), max(u, v))
                        pen_ro = 3.0 * ro[v] + (0.20 if ro[v] > 0.045 else 0.0)
                        pen_cz = 1.5 * cz_map[e] + (0.10 if cz_map[e] > 0.015 else 0.0)
                        cands.append((pen_ro + pen_cz, v))
            if not cands:
                break
            cands.sort()
            nxt = cands[0][1]
            chosen_set.add(nxt)
            chosen_list.append(nxt)

        for q in sorted(allowed, key=lambda i: ro[i]):
            if len(chosen_list) >= n_qubits:
                break
            if q not in chosen_set:
                chosen_set.add(q)
                chosen_list.append(q)
        order = chosen_list[:n_qubits]

    idx_map = {p: i for i, p in enumerate(order)}
    log_edges = sorted(
        (min(idx_map[u], idx_map[v]), max(idx_map[u], idx_map[v]))
        for (u, v) in cz_map
        if u in idx_map and v in idx_map
    )
    gro = ro[order]
    gcz = np.array(
        [cz_map[(min(u, v), max(u, v))] for (u, v) in cz_map if u in idx_map and v in idx_map],
        dtype=np.float64,
    )
    cols = color_ising_edges(log_edges)
    audit = SubgraphAudit(
        device=getattr(backend, "name", "heron_r2"),
        n_qubits=len(order),
        native_edges=len(log_edges),
        edge_colors=len(cols),
        med_ro_err_pct=100.0 * float(np.median(gro)),
        p90_ro_err_pct=100.0 * float(np.percentile(gro, 90)),
        bad_ro_gt5pct=int(np.sum(gro > 0.05)),
        med_cz_err_pct=100.0 * float(np.median(gcz)) if len(gcz) else 0.0,
        p90_cz_err_pct=100.0 * float(np.percentile(gcz, 90)) if len(gcz) else 0.0,
        bad_cz_gt2pct=int(np.sum(gcz > 0.02)) if len(gcz) else 0,
    )
    return log_edges, order, audit


def audit_heron_backend(backend, device_label: str = "") -> HeronAudit:
    """Inspect real/fake IBM Heron R2 topology, error tails, and transpiled 2-layer Ising cost."""
    nq = int(backend.num_qubits)
    edges, phys = select_hardware_subgraph(backend, n_qubits=nq)
    cols = color_ising_edges(edges)

    qc = QuantumCircuit(nq)
    for q in range(nq):
        qc.rx(0.4, q)
    for _ in range(2):
        for grp in cols:
            for _, u, v in grp:
                qc.rzz(0.25, u, v)
        for q in range(nq):
            qc.rz(0.15, q)
            qc.rx(0.5, q)
    qc.measure_all()

    t0 = time.perf_counter()
    tqc = transpile(qc, backend=backend, initial_layout=phys, optimization_level=1)
    t_tr = time.perf_counter() - t0
    ops = dict(tqc.count_ops())
    n_2q = int(ops.get("cz", ops.get("ecr", 0)))

    tgt = backend.target
    ro = [
        tgt["measure"][(q,)].error for q in range(nq)
        if "measure" in tgt and (q,) in tgt["measure"] and tgt["measure"][(q,)].error is not None
    ]
    cz = [
        p.error for op in ("cz", "ecr") if op in tgt
        for p in tgt[op].values() if p is not None and p.error is not None
    ]
    t1 = [tgt.qubit_properties[q].t1 * 1e6 for q in range(nq) if tgt.qubit_properties and tgt.qubit_properties[q].t1]
    t2 = [tgt.qubit_properties[q].t2 * 1e6 for q in range(nq) if tgt.qubit_properties and tgt.qubit_properties[q].t2]

    return HeronAudit(
        device=device_label or getattr(backend, "name", "heron_r2"),
        num_qubits=nq,
        native_edges=len(edges),
        edge_colors=len(cols),
        n_qerd_params=nq + len(edges),
        transpiled_depth=int(tqc.depth()),
        native_2q_gates=n_2q,
        transpile_sec=float(t_tr),
        med_2q_err_pct=100.0 * float(np.median(cz)),
        p90_2q_err_pct=100.0 * float(np.percentile(cz, 90)),
        bad_2q_couplers_gt2pct=int(sum(1 for e in cz if e > 0.02)),
        med_ro_err_pct=100.0 * float(np.median(ro)),
        p90_ro_err_pct=100.0 * float(np.percentile(ro, 90)),
        bad_ro_qubits_gt5pct=int(sum(1 for e in ro if e > 0.05)),
        med_t1_us=float(np.median(t1)),
        med_t2_us=float(np.median(t2)),
    )


def audit_all_heron_r2() -> List[HeronAudit]:
    """Audit Kingston, Fez, and Marrakesh calibration snapshots sorted by 2Q median fidelity."""
    audits = [audit_heron_backend(get_fake_heron(name), device_label=name) for name in ("ibm_kingston", "ibm_fez", "ibm_marrakesh")]
    return sorted(audits, key=lambda a: (a.med_2q_err_pct, a.med_ro_err_pct))


def project_heron_timing(
    n_qubits: int = 128,
    n_edges: int = 139,
    n_steps: int = 200,
    batch_size: int = 10,
    shots_per_run: int = 20,
    shots_per_sec: float = 4000.0,
) -> RuntimeProjection:
    """Compute exact pure QPU pulse time @ 4,000 shots/sec + realistic Session latency estimate."""
    n_params = n_qubits + n_edges
    qerd_shots = 2 * batch_size * shots_per_run * n_steps
    qerd_pulse_s = qerd_shots / max(shots_per_sec, 1.0)
    est_session_min = (qerd_pulse_s + n_steps * 5.0) / 60.0

    psr_eq_steps_shots = 2 * n_params * batch_size * shots_per_run * n_steps
    psr_eq_steps_s = psr_eq_steps_shots / max(shots_per_sec, 1.0)
    psr_eq_shots_steps = qerd_shots // max(1, 2 * n_params * shots_per_run)

    return RuntimeProjection(
        n_qubits=n_qubits,
        n_params=n_params,
        qerd_shots=qerd_shots,
        qerd_qpu_pulse_sec=qerd_pulse_s,
        qerd_session_wall_est_min=est_session_min,
        psr_equal_steps_shots=psr_eq_steps_shots,
        psr_equal_steps_pulse_sec=psr_eq_steps_s,
        psr_equal_shots_steps=psr_eq_shots_steps,
        circuit_compression_x=n_params,
    )


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

    token = _load_ibm_token(api_key_json)
    last_err = None
    channels = ("ibm_quantum_platform", "ibm_cloud", None) if token else (None,)
    for ch in channels:
        try:
            kw: Dict[str, Any] = {}
            if ch:
                kw["channel"] = ch
            if token:
                kw["token"] = token
            if instance:
                kw["instance"] = instance
            return QiskitRuntimeService(**kw)
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"Failed to connect QiskitRuntimeService: {last_err}")


def measure_live_readout_calibration(
    qpu_backend: Any,
    phys_layout: List[int],
    shots: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run 1 lightweight hardware job preparing `|0...0>` and `|1...1>` on `phys_layout`
    to fit marginal readout correction: `z_true_i ~= (z_raw_i - offset_i) / scale_i`.
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
        bits = (
            ba.to_bool_array(order="little")
            if hasattr(ba, "to_bool_array")
            else np.unpackbits(np.asarray(ba.array, dtype=np.uint8), axis=-1, bitorder="little")[..., :n].astype(bool)
        )
        if bits.ndim == 3:
            bits = bits[0]
        spins = 1.0 - 2.0 * bits.astype(np.float64)
        return np.mean(spins, axis=0)

    z0 = _mean_z(res[0])
    z1 = _mean_z(res[1])
    offset = 0.5 * (z0 + z1)
    scale = np.clip(0.5 * (z0 - z1), 0.65, 1.00)
    return offset, scale


def mitigate_expectations(z_batch: np.ndarray, offset: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Invert marginal readout bias cleanly on CPU (`0` extra QPU shots)."""
    return np.clip((z_batch - offset[None, :]) / scale[None, :], -1.0, 1.0)


def slice_readout_calibration(
    ro_cal: Optional[Tuple[np.ndarray, np.ndarray]],
    parent_phys: Sequence[int],
    child_phys: Sequence[int],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Slice an already-measured calibration `(offset, scale)` onto a nested physical qubit list."""
    if ro_cal is None:
        return None
    pos = {int(p): i for i, p in enumerate(parent_phys)}
    idx = [pos[int(p)] for p in child_phys]
    return ro_cal[0][idx], ro_cal[1][idx]


def prepare_problem_backend(
    mode: str,
    device: str,
    n_qubits: int,
    instance: str = "",
    execution_mode: str = "job",
    seed: int = 42,
    api_key_json: str = "",
    qpu_backend: Any = None,
    allowed_phys: Optional[Sequence[int]] = None,
    readout_mitigation: bool = True,
    parent_ro_cal: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    parent_phys: Optional[Sequence[int]] = None,
) -> Tuple[List[Tuple[int, int]], Dict[str, Any], Optional[SubgraphAudit], Optional[Tuple[np.ndarray, np.ndarray]], List[int]]:
    """
    Construct noise-pruned Heavy-Hex edges, simulator/QPU backend kwargs, SubgraphAudit,
    optional 1-job live marginal readout calibration, and selected physical qubits.
    """
    if mode == "local_sim":
        hw_ref = get_fake_heron(device)
        edges, phys, sub_rep = select_clean_subgraph(hw_ref, n_qubits=n_qubits, allowed_phys=allowed_phys)
        model_kw: Dict[str, Any] = {
            "backend": "auto",
            "real_qpu": False,
            "p_depol": round(0.015 + 2.0 * (sub_rep.med_cz_err_pct / 100.0), 4),
            "p_readout": round(0.015 + (sub_rep.med_ro_err_pct / 100.0), 4),
        }
        return edges, model_kw, sub_rep, None, phys

    if mode == "qiskit_sim":
        r = max(2, int(math.ceil(math.sqrt(n_qubits + 2))))
        c = max(2, int(math.ceil((n_qubits + 2) / r)))
        fake_hw = GenericBackendV2(num_qubits=r * c, coupling_map=build_ising_edges((r, c), topology="grid4"), seed=seed)
        edges, phys, sub_rep = select_clean_subgraph(fake_hw, n_qubits=n_qubits, allowed_phys=allowed_phys)
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
        return edges, model_kw, sub_rep, None, phys

    if mode == "ibm_qpu":
        eff_mode = execution_mode
        if qpu_backend is None:
            svc = _connect_ibm_service(instance=instance, api_key_json=api_key_json)
            try:
                plans = [str(inst.get("plan", "")).lower() for inst in (svc.instances() or [])]
                if plans and all("open" in p for p in plans) and eff_mode != "job":
                    print(f"[IBM Open Plan] Automatically switching execution_mode='{eff_mode}' -> 'job'.")
                    eff_mode = "job"
            except Exception:
                pass
            try:
                usg = svc.usage() or {}
                rem = usg.get("usage_remaining_seconds")
                if rem is not None:
                    print(f"[IBM Runtime] Quota remaining before run: {rem}s", flush=True)
            except Exception:
                pass
            qpu = svc.backend(device)
        else:
            qpu = qpu_backend
            if getattr(qpu, "service", None) is not None:
                try:
                    plans = [str(inst.get("plan", "")).lower() for inst in (qpu.service.instances() or [])]
                    if plans and all("open" in p for p in plans) and eff_mode != "job":
                        eff_mode = "job"
                except Exception:
                    pass

        edges, phys, sub_rep = select_clean_subgraph(qpu, n_qubits=n_qubits, allowed_phys=allowed_phys)
        ro_cal: Optional[Tuple[np.ndarray, np.ndarray]] = None
        if readout_mitigation:
            if parent_ro_cal is not None and parent_phys is not None and set(phys).issubset(set(parent_phys)):
                ro_cal = slice_readout_calibration(parent_ro_cal, parent_phys, phys)
            else:
                ro_cal = measure_live_readout_calibration(qpu, phys, shots=256)
                print(
                    f"[Live RO Mitigation] 1-job calibration on {len(phys)}q: "
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
        return edges, model_kw, sub_rep, ro_cal, phys

    raise ValueError(f"Unknown mode '{mode}'")


def prepare_quantum_backend(
    mode: str,
    device: str,
    n_qubits: int,
    instance: str = "",
    execution_mode: str = "job",
    seed: int = 42,
    api_key_json: str = "",
    qpu_backend: Any = None,
) -> Tuple[List[Tuple[int, int]], Dict[str, Any], Optional[SubgraphAudit]]:
    """Backward-compatible wrapper returning `(edges, model_kw, sub_audit)`."""
    edges, model_kw, sub_rep, _, _ = prepare_problem_backend(
        mode=mode,
        device=device,
        n_qubits=n_qubits,
        instance=instance,
        execution_mode=execution_mode,
        seed=seed,
        api_key_json=api_key_json,
        qpu_backend=qpu_backend,
        readout_mitigation=False,
    )
    return edges, model_kw, sub_rep
