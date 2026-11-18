"""
Real S&P 500 Minimum-Variance Markowitz QUBO Formulation.

Evaluates discrete asset selection ONLY against valid optimization baselines
(SA global ceiling, Untrained Random QPU, Deterministic Greedy Risk Screening,
Equal-Shot PSR, and Equal-Shot SPSA) using Optimization Gap Closed (%) and
unseen 2022-2025 Out-of-Sample Portfolio Volatility Reduction (%), eliminating
single-stock bull-market return lotteries.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Dict, List

import numpy as np

from ..loaders.market import MarketSplit


@dataclass(frozen=True)
class MinVarPortfolioProblem:
    n_qubits: int
    tickers: List[str]
    target_k: int
    cost_fn: Callable[[np.ndarray], np.ndarray]
    mu_train_ann: np.ndarray
    cov_train_ann: np.ndarray
    cov_test_ann: np.ndarray
    r_test: np.ndarray

    def evaluate(
        self,
        samples: np.ndarray,
        baseline_untrained_cost: float,
        ceiling_sa_cost: float,
    ) -> Dict[str, float]:
        spins = np.where(np.asarray(samples, dtype=np.float64) >= 0.0, 1.0, -1.0)
        costs = self.cost_fn(spins)
        mean_cost = float(np.mean(costs))
        best_cost = float(np.min(costs))

        x_sel = 0.5 * (spins + 1.0)
        cnt = np.sum(x_sel, axis=1, keepdims=True)
        w = np.where(cnt > 0.0, x_sel / np.maximum(cnt, 1.0), 1.0 / self.n_qubits)

        pvar_te = np.sum((w @ self.cov_test_ann) * w, axis=1)
        oos_vol_pct = float(np.mean(np.sqrt(np.maximum(pvar_te, 1e-12))) * 100.0)

        w_ens = np.mean(w, axis=0)
        w_ens /= np.sum(w_ens)
        r_ens = self.r_test @ w_ens
        wealth = 10_000.0 * np.cumprod(1.0 + r_ens)
        ens_profit_usd = float(wealth[-1] - 10_000.0)
        ens_ret_pct = float(252.0 * np.mean(r_ens) * 100.0)
        ens_vol_pct = float(math.sqrt(max(float(w_ens @ self.cov_test_ann @ w_ens), 1e-12)) * 100.0)
        ens_sharpe = ens_ret_pct / max(ens_vol_pct, 1e-8)

        denom = max(1e-8, baseline_untrained_cost - ceiling_sa_cost)
        gap_closed_pct = 100.0 * (baseline_untrained_cost - mean_cost) / denom
        uniq = {tuple(r.astype(int).tolist()) for r in spins}

        return {
            "mean_qubo_cost": mean_cost,
            "best_qubo_cost": best_cost,
            "gap_closed_pct": gap_closed_pct,
            "oos_mean_vol_pct": oos_vol_pct,
            "oos_ens_vol_pct": ens_vol_pct,
            "oos_ens_ret_pct": ens_ret_pct,
            "oos_ens_profit_usd": ens_profit_usd,
            "oos_ens_sharpe": ens_sharpe,
            "unique_frac": float(len(uniq) / max(1, len(spins))),
        }


def build_minvar_problem(
    split: MarketSplit,
    risk_aversion: float = 2.5,
    card_penalty: float = 0.35,
) -> MinVarPortfolioProblem:
    n_q = len(split.tickers)
    mu_tr = 252.0 * np.mean(split.r_train, axis=0)
    cov_tr = 252.0 * np.cov(split.r_train, rowvar=False)
    cov_te = 252.0 * np.cov(split.r_test, rowvar=False)
    target_k = max(3, n_q // 2)

    def cost_fn(s_mat: np.ndarray) -> np.ndarray:
        x = 0.5 * (np.where(s_mat >= 0.0, 1.0, -1.0) + 1.0)
        cnt = np.sum(x, axis=1, keepdims=True)
        w = np.where(cnt > 0.0, x / np.maximum(cnt, 1.0), 1.0 / n_q)
        pvar = np.sum((w @ cov_tr) * w, axis=1)
        pret = w @ mu_tr
        card_err = ((cnt[:, 0] - target_k) / max(1.0, float(n_q))) ** 2
        return risk_aversion * pvar - pret + card_penalty * card_err

    return MinVarPortfolioProblem(
        n_qubits=n_q,
        tickers=split.tickers,
        target_k=target_k,
        cost_fn=cost_fn,
        mu_train_ann=mu_tr,
        cov_train_ann=cov_tr,
        cov_test_ann=cov_te,
        r_test=split.r_test,
    )
