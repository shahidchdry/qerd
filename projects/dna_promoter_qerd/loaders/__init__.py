"""Real Human Promoter FASTA dataset download, TSS windowing, and bijective DNA-spin codecs."""

from .promoters import (
    PromoterDataset,
    core8_to_spins,
    ensure_promoter_datasets,
    fetch_promoter_datasets,
    load_promoter_datasets,
    onehot_from_seqs,
    seqs_to_spins,
    spins_to_core8,
    spins_to_seqs,
)

__all__ = [
    "PromoterDataset",
    "core8_to_spins",
    "ensure_promoter_datasets",
    "fetch_promoter_datasets",
    "load_promoter_datasets",
    "onehot_from_seqs",
    "seqs_to_spins",
    "spins_to_core8",
    "spins_to_seqs",
]
