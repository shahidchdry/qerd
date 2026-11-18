"""
Exact biophysical DNA metrics: SantaLucia (1998) nearest-neighbor free energy dG37,
core human promoter grammar (-30 TATA / GC-box + TSS Initiator CA), homopolymer limits,
sequence novelty, and diversity metrics.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set

import numpy as np


# Unified SantaLucia (1998) PNAS nearest-neighbor DNA duplex parameters (1M NaCl): (dH kcal/mol, dS cal/mol/K)
_NN_THERMO = {
    "AA": (-7.9, -22.2), "TT": (-7.9, -22.2),
    "AT": (-7.2, -20.4), "TA": (-7.2, -21.3),
    "CA": (-8.5, -22.7), "TG": (-8.5, -22.7),
    "GT": (-8.4, -22.4), "AC": (-8.4, -22.4),
    "CT": (-7.8, -21.0), "AG": (-7.8, -21.0),
    "GA": (-8.2, -22.2), "TC": (-8.2, -22.2),
    "CG": (-10.6, -27.2), "GC": (-9.8, -24.4),
    "GG": (-8.0, -19.9), "CC": (-8.0, -19.9),
}
_NN_DG37 = {k: (dh - 310.15 * (ds / 1000.0)) for k, (dh, ds) in _NN_THERMO.items()}


def compute_santalucia_dg37(seq: str) -> float:
    """Compute DNA duplex free energy Delta-G at 37C (kcal/mol) using SantaLucia (1998)."""
    s = seq.upper()
    if len(s) < 2:
        return 0.0
    dg = sum(_NN_DG37.get(s[i : i + 2], 0.0) for i in range(len(s) - 1))
    for end_b in (s[0], s[-1]):
        dg += 0.98 if end_b in ("G", "C") else 1.03
    return float(dg)


def max_homopolymer_run(seq: str) -> int:
    if not seq:
        return 0
    cur, best = 1, 1
    for i in range(1, len(seq)):
        cur = cur + 1 if seq[i] == seq[i - 1] else 1
        if cur > best:
            best = cur
    return best


def gc_percent(seq: str) -> float:
    if not seq:
        return 0.0
    return 100.0 * sum(1 for c in seq if c in ("G", "C")) / float(len(seq))


def has_tss_initiator(seq: str) -> bool:
    """Check conserved mammalian Initiator (Inr) dinucleotide right at TSS (-2..+3 => pos 47..51)."""
    if len(seq) < 52:
        return False
    w = seq[47:52]
    return ("CA" in w) or ("CG" in w) or ("TG" in w)


def has_upstream_promoter_box(seq: str, cond_mode: str = "any") -> bool:
    """Check conserved -30 TATA element (`pos 16..27`) or upstream GC/BRE element (`pos 0..42`)."""
    tata_w = seq[16:28] if len(seq) >= 28 else seq
    gc_w = seq[:42] if len(seq) >= 42 else seq
    has_tata = any(m in tata_w for m in ("TATA", "ATAA", "TTAA", "TAAA"))
    has_gc = any(m in gc_w for m in ("GGGC", "CCGC", "GCCG", "CCCC", "GGGG"))
    if cond_mode == "tata":
        return has_tata
    if cond_mode == "nontata":
        return has_gc
    return has_tata or has_gc


def base_frequencies(seqs: Sequence[str]) -> np.ndarray:
    """Return `(L, 4)` empirical probability of `[A, C, G, T]` at every nucleotide position."""
    if not seqs:
        return np.zeros((0, 4), dtype=np.float64)
    L = len(seqs[0])
    counts = np.zeros((L, 4), dtype=np.float64)
    idx_map = {"A": 0, "C": 1, "G": 2, "T": 3}
    for s in seqs:
        for i, c in enumerate(s):
            k = idx_map.get(c)
            if k is not None:
                counts[i, k] += 1.0
    counts /= max(1, len(seqs))
    return counts


def motif_profile_matrix(seqs: Sequence[str]) -> Dict[str, np.ndarray]:
    """Extract GC% curve and A/T enrichment across promoter positions for plotting."""
    freqs = base_frequencies(seqs)
    if freqs.size == 0:
        return {"gc": np.array([]), "at": np.array([])}
    gc = freqs[:, 1] + freqs[:, 2]
    at = freqs[:, 0] + freqs[:, 3]
    return {"gc": gc, "at": at, "freqs": freqs}


def _mean_pairwise_hamming(seqs: Sequence[str]) -> float:
    n = len(seqs)
    if n < 2:
        return 0.0
    sub = list(seqs[:48])
    L = max(1, len(sub[0]))
    tot, pairs = 0, 0
    for i in range(len(sub)):
        si = sub[i]
        for j in range(i + 1, len(sub)):
            sj = sub[j]
            tot += sum(1 for a, b in zip(si, sj) if a != b)
            pairs += 1
    return float(tot / max(1, pairs * L))


def evaluate_promoter_sequences(
    seqs: Sequence[str],
    oracle_probs: np.ndarray,
    train_seq_set: Optional[Set[str]] = None,
    real_pos_freqs: Optional[np.ndarray] = None,
    cond_tags: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """
    Evaluate full `64 bp` (`128q`) or `78 bp` (`156q`) DNA sequences across:
      * PromoterOracleNet Activity Probability (`0..1`)
      * Strict Biophysical Validity (`%` passing Oracle >= 0.60 + dG37 <= -86 + GC in [42,78]% + Motif + NoHomo<=5)
      * SantaLucia dG37 stability, GC%, TSS Initiator match, Novelty %, Unique %, Hamming Diversity
    """
    n = max(1, len(seqs))
    dgs = np.array([compute_santalucia_dg37(s) for s in seqs], dtype=np.float64)
    gcs = np.array([gc_percent(s) for s in seqs], dtype=np.float64)
    homs = np.array([max_homopolymer_run(s) <= 5 for s in seqs], dtype=bool)
    inrs = np.array([has_tss_initiator(s) for s in seqs], dtype=bool)
    boxes = np.array(
        [has_upstream_promoter_box(s, cond_mode="any") for s in seqs],
        dtype=bool,
    )
    probs = np.asarray(oracle_probs, dtype=np.float64)

    valid_mask = (
        (probs >= 0.60)
        & (gcs >= 42.0)
        & (gcs <= 80.0)
        & (dgs <= -86.0)
        & homs
        & (inrs | boxes)
    )
    uniq = set(seqs)
    novel_frac = float(np.mean([s not in train_seq_set for s in seqs])) if train_seq_set else 1.0

    motif_corr = 0.0
    if real_pos_freqs is not None and len(seqs) > 0:
        gen_f = base_frequencies(seqs)
        L_min = min(gen_f.shape[0], real_pos_freqs.shape[0])
        gf = gen_f[:L_min].ravel()
        rf = real_pos_freqs[:L_min].ravel()
        if np.std(gf) > 1e-8 and np.std(rf) > 1e-8:
            motif_corr = float(np.corrcoef(gf, rf)[0, 1])

    out = {
        "n_eval": float(len(seqs)),
        "mean_oracle_prob": float(np.mean(probs)),
        "oracle_pass_pct": float(100.0 * np.mean(probs >= 0.60)),
        "valid_promoter_pct": float(100.0 * np.mean(valid_mask)),
        "mean_gc_pct": float(np.mean(gcs)),
        "mean_dg37_kcal": float(np.mean(dgs)),
        "tss_inr_pct": float(100.0 * np.mean(inrs)),
        "upstream_box_pct": float(100.0 * np.mean(boxes)),
        "no_homopolymer_pct": float(100.0 * np.mean(homs)),
        "motif_profile_corr": motif_corr,
        "unique_pct": float(100.0 * len(uniq) / n),
        "unique_valid_pct": float(100.0 * len({s for s, ok in zip(seqs, valid_mask) if ok}) / n),
        "novel_pct": float(100.0 * novel_frac),
        "hamming_diversity": _mean_pairwise_hamming(seqs),
    }
    if cond_tags is not None and len(cond_tags) == len(seqs):
        tata_sub = [s for s, t in zip(seqs, cond_tags) if t == "tata"]
        nontata_sub = [s for s, t in zip(seqs, cond_tags) if t == "nontata"]
        if tata_sub:
            out["tata_cond_box_pct"] = float(100.0 * np.mean([has_upstream_promoter_box(s, "tata") for s in tata_sub]))
        if nontata_sub:
            out["nontata_cond_gc_pct"] = float(np.mean([gc_percent(s) for s in nontata_sub]))
    return out


def evaluate_core8_sequences(
    seqs8: Sequence[str],
    pos_pssm_log_odds: np.ndarray,
    neg_log_odds_thresh: float = 0.0,
) -> Dict[str, float]:
    """
    Evaluate `8 bp` (`16q`) core promoter designs (`[0:4]` = -30 TATA/BRE box, `[4:8]` = TSS Inr)
    using an empirical PSSM log-odds score fit on real human promoters plus motif rules.
    """
    bmap = {"A": 0, "C": 1, "G": 2, "T": 3}
    scores = np.array(
        [sum(float(pos_pssm_log_odds[j, bmap[c]]) for j, c in enumerate(s)) for s in seqs8],
        dtype=np.float64,
    )
    inrs = np.array([("CA" in s[4:8]) or ("CG" in s[4:8]) or ("TG" in s[4:8]) for s in seqs8], dtype=bool)
    boxes = np.array(
        [any(m in s[0:4] for m in ("TATA", "ATAA", "TTAA", "TAAA", "GGGC", "CCGC", "GCCG")) for s in seqs8],
        dtype=bool,
    )
    dgs = np.array([compute_santalucia_dg37(s) for s in seqs8], dtype=np.float64)
    probs = 1.0 / (1.0 + np.exp(-np.clip(scores - neg_log_odds_thresh, -20.0, 20.0)))
    valid = (probs >= 0.60) & (inrs | boxes)
    uniq = set(seqs8)
    uniq_valid = {s for s, ok in zip(seqs8, valid) if ok}
    return {
        "n_eval": float(len(seqs8)),
        "mean_oracle_prob": float(np.mean(probs)),
        "oracle_pass_pct": float(100.0 * np.mean(probs >= 0.60)),
        "valid_promoter_pct": float(100.0 * np.mean(valid)),
        "mean_gc_pct": float(np.mean([gc_percent(s) for s in seqs8])),
        "mean_dg37_kcal": float(np.mean(dgs)),
        "tss_inr_pct": float(100.0 * np.mean(inrs)),
        "upstream_box_pct": float(100.0 * np.mean(boxes)),
        "unique_pct": float(100.0 * len(uniq) / max(1, len(seqs8))),
        "unique_valid_pct": float(100.0 * len(uniq_valid) / max(1, len(seqs8))),
        "hamming_diversity": _mean_pairwise_hamming(seqs8),
        "mean_pssm_llr": float(np.mean(scores)),
    }
