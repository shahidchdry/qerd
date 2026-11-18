"""
Real Human Promoter FASTA loader and bijective 2-qubit/base chemical DNA encoding.

Mapping (`2` qubits per nucleotide):
  * Qubit `2i`   = Strong/Weak H-bond (`G/C = +1`, `A/T = -1`)
  * Qubit `2i+1` = Pyrimidine/Purine (`C/T = +1`, `A/G = -1`)
That is: `A=(-1,-1)`, `T=(-1,+1)`, `G=(+1,-1)`, `C=(+1,+1)`. Every bitstring decodes to
a valid DNA sequence with zero invalid syntax tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Dict, List, Sequence, Tuple
import urllib.request

import numpy as np


HF_BASE = "https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks_revised/resolve/main"
DNA_FILES = {
    "promoter_all_train.fna": f"{HF_BASE}/promoter_all/train.fna",
    "promoter_all_test.fna": f"{HF_BASE}/promoter_all/test.fna",
    "promoter_tata_train.fna": f"{HF_BASE}/promoter_tata/train.fna",
    "promoter_tata_test.fna": f"{HF_BASE}/promoter_tata/test.fna",
}

BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
BASE_TO_SPINS = {
    "A": (-1.0, -1.0),
    "T": (-1.0, 1.0),
    "G": (1.0, -1.0),
    "C": (1.0, 1.0),
}
SPINS_TO_BASE = {
    (0, 0): "A",
    (0, 1): "T",
    (1, 0): "G",
    (1, 1): "C",
}


@dataclass(frozen=True)
class PromoterDataset:
    train_all_seqs: List[str]
    train_all_labels: np.ndarray
    test_all_seqs: List[str]
    test_all_labels: np.ndarray
    train_tata_pos: List[str]
    test_tata_pos: List[str]
    train_nontata_pos: List[str]
    test_nontata_pos: List[str]
    seq_len: int


def fetch_promoter_datasets(data_dir: str) -> str:
    """Download HuggingFace Nucleotide Transformer Human Promoter FASTA splits into `data_dir`."""
    os.makedirs(data_dir, exist_ok=True)
    for fname, url in DNA_FILES.items():
        dst = os.path.join(data_dir, fname)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(dst, "wb") as f:
            f.write(resp.read())
    return data_dir


def _read_fna(path: str, seq_len: int = 64) -> Tuple[List[str], np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing DNA dataset {path}. Download via: python -m projects.dna_promoter_qerd --fetch-data"
        )
    seqs: List[str] = []
    labels: List[float] = []
    cur_h: str | None = None
    buf: List[str] = []
    valid_alpha = {"A", "C", "G", "T"}

    def flush():
        if cur_h is not None and buf:
            s = "".join(buf)[:seq_len]
            if len(s) == seq_len and set(s) <= valid_alpha:
                seqs.append(s)
                labels.append(1.0 if cur_h.endswith("|1") else 0.0)

    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln.startswith(">"):
                flush()
                cur_h = ln
                buf = []
            elif ln:
                buf.append(ln.upper())
    flush()
    return seqs, np.asarray(labels, dtype=np.float32)


def load_promoter_datasets(data_dir: str, seq_len: int = 64) -> PromoterDataset:
    """
    Load real TSS-aligned Human Promoter FASTA splits (`[-49 .. +14]` = `64 bp` core promoter,
    or `[-49 .. +28]` = `78 bp` for `156q`) and separate TATA-box vs Non-TATA/GC subclasses.
    """
    tr_s, tr_y = _read_fna(os.path.join(data_dir, "promoter_all_train.fna"), seq_len=seq_len)
    te_s, te_y = _read_fna(os.path.join(data_dir, "promoter_all_test.fna"), seq_len=seq_len)
    t_tr_s, t_tr_y = _read_fna(os.path.join(data_dir, "promoter_tata_train.fna"), seq_len=seq_len)
    t_te_s, t_te_y = _read_fna(os.path.join(data_dir, "promoter_tata_test.fna"), seq_len=seq_len)

    tata_tr_pos = [s for s, y in zip(t_tr_s, t_tr_y) if y > 0.5]
    tata_te_pos = [s for s, y in zip(t_te_s, t_te_y) if y > 0.5]
    tata_set = set(tata_tr_pos) | set(tata_te_pos)

    nontata_tr_pos = [s for s, y in zip(tr_s, tr_y) if y > 0.5 and s not in tata_set]
    nontata_te_pos = [s for s, y in zip(te_s, te_y) if y > 0.5 and s not in tata_set]

    return PromoterDataset(
        train_all_seqs=tr_s,
        train_all_labels=tr_y,
        test_all_seqs=te_s,
        test_all_labels=te_y,
        train_tata_pos=tata_tr_pos,
        test_tata_pos=tata_te_pos,
        train_nontata_pos=nontata_tr_pos,
        test_nontata_pos=nontata_te_pos,
        seq_len=seq_len,
    )


def onehot_from_seqs(seqs: Sequence[str]) -> np.ndarray:
    """Convert equal-length DNA sequences into `(N, 4, L)` float32 one-hot tensor."""
    if not seqs:
        return np.zeros((0, 4, 64), dtype=np.float32)
    L = len(seqs[0])
    arr = np.zeros((len(seqs), 4, L), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j, ch in enumerate(s):
            idx = BASE_TO_IDX.get(ch)
            if idx is not None:
                arr[i, idx, j] = 1.0
    return arr


def seqs_to_spins(seqs: Sequence[str]) -> np.ndarray:
    """Bijectively encode `N` DNA sequences of length `L` into `(N, 2*L)` `{-1, +1}` spin vectors."""
    L = len(seqs[0])
    out = np.empty((len(seqs), 2 * L), dtype=np.float64)
    for i, s in enumerate(seqs):
        for j, ch in enumerate(s):
            b0, b1 = BASE_TO_SPINS[ch]
            out[i, 2 * j] = b0
            out[i, 2 * j + 1] = b1
    return out


def spins_to_seqs(spins: np.ndarray) -> List[str]:
    """Decode continuous or binary `(N, 2*L)` QPU outputs bijectively into `ACGT` strings."""
    bits = (np.asarray(spins, dtype=np.float64) >= 0.0).astype(int)
    n, d = bits.shape
    L = d // 2
    out: List[str] = []
    for i in range(n):
        chars = [SPINS_TO_BASE[(int(bits[i, 2 * j]), int(bits[i, 2 * j + 1]))] for j in range(L)]
        out.append("".join(chars))
    return out


def core8_to_spins(seqs64: Sequence[str]) -> np.ndarray:
    """
    Extract the 8-bp functional core motif (`pos 18..22` -30 TATA/BRE + `pos 47..51` TSS Inr)
    into `16` qubits for compact local CPU simulation benchmarks.
    """
    sub = [s[18:22] + s[47:51] for s in seqs64]
    return seqs_to_spins(sub)


def ensure_promoter_datasets(data_dir: str) -> str:
    """Download missing Human Promoter FASTA splits into `data_dir` if absent locally."""
    if not all(os.path.exists(os.path.join(data_dir, fname)) for fname in DNA_FILES):
        print(f"[Data] Human promoter FASTA files not found in {data_dir}; downloading from HuggingFace...", flush=True)
        return fetch_promoter_datasets(data_dir)
    return data_dir


def spins_to_core8(spins16: np.ndarray) -> List[str]:
    """Decode `16`-qubit spins into `8-bp` (`-30 box` + `TSS Inr`) sequences."""
    return spins_to_seqs(spins16)


if __name__ == "__main__":
    from ..configs import RunConfig

    dst = RunConfig().data_dir
    print(f"Downloaded Human Promoter datasets -> {fetch_promoter_datasets(dst)}")
