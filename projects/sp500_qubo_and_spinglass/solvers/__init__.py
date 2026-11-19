"""Classical optimization baselines, 2^N GPU wall profiler, and equal-shot QPU solvers."""

from .classical import (
    benchmark_t4_simulation_wall,
    greedy_minvar_portfolio,
    solve_simulated_annealing,
)
from .quantum import (
    load_qerd_checkpoint,
    sample_qerd_batch,
    save_qerd_checkpoint,
    train_coord_psr_equal_shot,
    train_qerd_explicit,
    train_psr_equal_shot,
    train_spsa_equal_shot,
)

__all__ = [
    "benchmark_t4_simulation_wall",
    "greedy_minvar_portfolio",
    "solve_simulated_annealing",
    "load_qerd_checkpoint",
    "sample_qerd_batch",
    "save_qerd_checkpoint",
    "train_coord_psr_equal_shot",
    "train_qerd_explicit",
    "train_psr_equal_shot",
    "train_spsa_equal_shot",
]
