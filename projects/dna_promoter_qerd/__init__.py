"""
Project 2: Conditional Human Promoter DNA Designer (`QERD` + `PromoterOracleNet`).

Combines real Human Promoter FASTA genomics (30,000+ TSS-aligned sequences),
exact SantaLucia nearest-neighbor thermodynamics, a PyTorch 1D-CNN oracle,
and noise-pruned 128q IBM Heron R2 execution (`cal_mode='inverse'` + packed shots).
"""

from .configs import PRESETS, RunConfig, build_run_config

__all__ = ["PRESETS", "RunConfig", "build_run_config"]
