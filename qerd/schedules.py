"""
Diffusion noise schedules and Ising qubit-coupling graph topologies.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def make_alpha_bar(steps: int, schedule: str = "cosine", s: float = 0.008) -> np.ndarray:
    """Return alpha_bar[1..steps] as 1-indexed length-(steps+1) array (alpha_bar[0] = 1.0)."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    ab = np.ones(steps + 1, dtype=np.float64)
    t_idx = np.arange(1, steps + 1, dtype=np.float64)
    if schedule == "cosine":
        f = np.cos(((t_idx / steps) + s) / (1.0 + s) * (0.5 * math.pi)) ** 2
        f0 = math.cos(s / (1.0 + s) * (0.5 * math.pi)) ** 2
        ab[1:] = np.clip(f / f0, 0.02, 0.98)
    elif schedule == "linear":
        betas = np.linspace(0.08, 0.55, steps)
        ab[1:] = np.clip(np.cumprod(1.0 - betas), 0.02, 0.98)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
    return ab


def build_ising_edges(
    data_shape: Tuple[int, ...],
    topology: str = "grid8",
    n_hidden: int = 0,
) -> List[Tuple[int, int]]:
    """
    Build unique undirected Ising edges for arbitrary data shapes plus visible-to-hidden edges.

    Supported topologies:
      - "grid4" / "grid8": 4- or 8-neighbor 2D grid (1D chain if len(data_shape) == 1)
      - "chain"          : 1D nearest-neighbor chain across flattened visible qubits
      - "all"            : all-to-all pairwise visible couplings (best for unstructured/latent vectors)
      - "none"           : no visible-visible couplings (retains visible-hidden edges if n_hidden > 0)
      - "unentangled"    : strictly zero two-qubit edges (removes both visible-visible and visible-hidden)
    """
    n_visible = int(np.prod(data_shape))
    edges = set()

    if topology in ("grid4", "grid8"):
        height, width = data_shape if len(data_shape) == 2 else (1, n_visible)
        offsets = [(0, 1), (1, 0)] + ([(1, 1), (1, -1)] if topology == "grid8" else [])
        for r in range(height):
            for c in range(width):
                u = r * width + c
                for dr, dc in offsets:
                    for sign in (1, -1):
                        nr, nc = r + sign * dr, c + sign * dc
                        if 0 <= nr < height and 0 <= nc < width:
                            v = nr * width + nc
                            if u != v:
                                edges.add((min(u, v), max(u, v)))
    elif topology == "chain":
        for i in range(n_visible - 1):
            edges.add((i, i + 1))
    elif topology == "all":
        for i in range(n_visible):
            for j in range(i + 1, n_visible):
                edges.add((i, j))
    elif topology in ("none", "unentangled"):
        pass
    else:
        raise ValueError(
            f"Unknown topology '{topology}'; expected 'grid4', 'grid8', 'chain', 'all', 'none', or 'unentangled'."
        )

    if topology != "unentangled":
        for h in range(n_hidden):
            hw = n_visible + h
            for v in range(n_visible):
                edges.add((v, hw))
    return sorted(edges)


def _extract_coupling_edges(
    coupling_source: Any,
) -> List[Tuple[int, int]]:
    """Extract unique undirected hardware coupling edges from a Qiskit backend, Target, CouplingMap, or edge list."""
    raw_edges: Iterable[Sequence[int]]
    if hasattr(coupling_source, "coupling_map") and coupling_source.coupling_map is not None:
        raw_edges = coupling_source.coupling_map.get_edges()
    elif hasattr(coupling_source, "target") and coupling_source.target is not None:
        cmap = coupling_source.target.build_coupling_map()
        raw_edges = cmap.get_edges() if cmap is not None else []
    elif hasattr(coupling_source, "get_edges"):
        raw_edges = coupling_source.get_edges()
    else:
        raw_edges = coupling_source

    undirected = set()
    for pair in raw_edges:
        u, v = int(pair[0]), int(pair[1])
        if u != v:
            undirected.add((min(u, v), max(u, v)))
    return sorted(undirected)


def select_hardware_subgraph(
    coupling_source: Any,
    n_qubits: Optional[int] = None,
    anchor: Optional[int] = None,
) -> Tuple[List[Tuple[int, int]], List[int]]:
    """
    Select a connected subgraph of `n_qubits` physical qubits from a hardware coupling map
    (or Qiskit backend) and remap induced edges to contiguous logical indices `0..n_qubits-1`.

    Returns `(logical_edges, physical_qubits)` so passing both `edges=logical_edges` and
    `initial_layout=physical_qubits` avoids two-qubit SWAP routing overhead on real QPUs.
    """
    undirected = _extract_coupling_edges(coupling_source)
    if not undirected:
        if n_qubits is None or n_qubits <= 0:
            raise ValueError("Cannot infer qubit count from empty coupling map.")
        return [], list(range(n_qubits))

    adj: Dict[int, List[int]] = {}
    for u, v in undirected:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    for k in adj:
        adj[k].sort()

    target_n = len(adj) if n_qubits is None else int(n_qubits)
    if target_n <= 0:
        return [], []
    if target_n > len(adj):
        raise ValueError(f"Requested {target_n} qubits, but coupling graph only has {len(adj)} connected nodes.")

    if anchor is None:
        start = min(adj.keys(), key=lambda q: (-len(adj[q]), q))
    else:
        start = int(anchor)
        if start not in adj:
            raise ValueError(f"Anchor qubit {start} is not in the hardware coupling map.")

    visited = [start]
    seen = {start}
    queue = deque([start])
    while queue and len(visited) < target_n:
        curr = queue.popleft()
        # Prefer neighbors with higher degree so the patch remains compact and well-connected.
        nbrs = sorted(adj[curr], key=lambda q: (-len(adj[q]), q))
        for nb in nbrs:
            if nb not in seen:
                seen.add(nb)
                visited.append(nb)
                queue.append(nb)
                if len(visited) >= target_n:
                    break

    if len(visited) < target_n:
        raise ValueError(f"Could only reach {len(visited)} connected qubits from anchor {start} (wanted {target_n}).")

    phys_to_logical = {p: idx for idx, p in enumerate(visited)}
    logical_edges = sorted(
        (min(phys_to_logical[u], phys_to_logical[v]), max(phys_to_logical[u], phys_to_logical[v]))
        for u, v in undirected
        if u in phys_to_logical and v in phys_to_logical
    )
    return logical_edges, visited


def color_ising_edges(edges: Sequence[Tuple[int, int]]) -> List[List[Tuple[int, int, int]]]:
    """
    Partition Ising edges into vertex-disjoint layers of `(edge_idx, u, v)` tuples.

    All `RZZ` gates within a layer operate on disjoint qubit pairs and can execute in parallel
    on hardware without increasing decoherence depth.
    """
    layers: List[List[Tuple[int, int, int]]] = []
    used_per_layer: List[set] = []
    for edge_idx, (u, v) in enumerate(edges):
        placed = False
        for layer, used in zip(layers, used_per_layer):
            if u not in used and v not in used:
                layer.append((edge_idx, int(u), int(v)))
                used.add(u)
                used.add(v)
                placed = True
                break
        if not placed:
            layers.append([(edge_idx, int(u), int(v))])
            used_per_layer.append({u, v})
    return layers
