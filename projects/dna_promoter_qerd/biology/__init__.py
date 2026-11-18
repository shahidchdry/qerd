"""DNA biophysics (SantaLucia dG37, motif PWM/Inr, novelty) and PyTorch 1D-CNN promoter oracle."""

from .biophysics import (
    base_frequencies,
    compute_santalucia_dg37,
    evaluate_core8_sequences,
    evaluate_promoter_sequences,
    motif_profile_matrix,
)
from .oracle_cnn import PromoterOracleNet, load_or_train_oracle

__all__ = [
    "PromoterOracleNet",
    "base_frequencies",
    "compute_santalucia_dg37",
    "evaluate_core8_sequences",
    "evaluate_promoter_sequences",
    "load_or_train_oracle",
    "motif_profile_matrix",
]
