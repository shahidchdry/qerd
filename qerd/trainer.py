"""
High-level shot-budget training harness orchestrating forward diffusion and QER updates.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .denoiser import QuantumIsingDenoiser
from .optimizers import QEROptimizer


class QERDTrainer:
    """
    Training harness orchestrating forward Gaussian diffusion, stratified timesteps,
    contrastive QPU calls, calibrator updates, and QEROptimizer steps.
    """

    def __init__(
        self,
        model: QuantumIsingDenoiser,
        optimizer: Optional[QEROptimizer] = None,
    ):
        self.model = model
        self.optimizer = optimizer or QEROptimizer(model.parameters_dict())

    def train(
        self,
        data: np.ndarray,
        conds: Optional[np.ndarray] = None,
        max_shots: int = 24_000,
        shots_per_run: int = 44,
        batch_size: int = 8,
        eval_every_batches: int = 15,
        eval_shots: int = 120,
        stratify_labels: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> List[Dict[str, float]]:
        targets = np.asarray(data, dtype=np.float64).reshape(-1, self.model.n_visible)
        n_data = targets.shape[0]
        if conds is not None:
            if self.model.cond_dim == 0:
                raise ValueError("conds provided but model.cond_dim == 0")
            cond_arr = np.asarray(conds, dtype=np.float64).reshape(n_data, self.model.cond_dim)
        else:
            cond_arr = None

        class_pools = (
            [np.flatnonzero(stratify_labels == c) for c in np.unique(stratify_labels)]
            if stratify_labels is not None
            else None
        )

        eff_batch = min(batch_size, n_data)
        total_batches = max(1, max_shots // (2 * shots_per_run * eff_batch))
        history: List[Dict[str, float]] = []

        for b_idx in range(1, total_batches + 1):
            progress = (b_idx - 1) / max(1, total_batches - 1)
            if class_pools is not None:
                c_order = self.model.rng.permutation(len(class_pools))
                sample_ids = np.array(
                    [self.model.rng.choice(class_pools[c_order[k % len(class_pools)]]) for k in range(eff_batch)],
                    dtype=np.int64,
                )
            else:
                sample_ids = self.model.rng.choice(n_data, size=eff_batch, replace=False)

            t_cycle = np.arange(eff_batch) % self.model.diffusion_steps + 1
            self.model.rng.shuffle(t_cycle)

            tgts_arr = targets[sample_ids]
            c_arr = None if cond_arr is None else cond_arr[sample_ids]
            a_bars = self.model.alpha_bar[t_cycle][:, None]
            x_t_arr = np.clip(
                np.sqrt(a_bars) * tgts_arr
                + np.sqrt(1.0 - a_bars) * self.model.rng.normal(0.0, 0.70, size=(eff_batch, self.model.n_visible)),
                -1.0,
                1.0,
            )

            beta_p = self.optimizer.beta
            beta_m = -self.optimizer.beta if self.optimizer.cfg.centered else 0.0
            x_both = np.concatenate([x_t_arr, x_t_arr], axis=0)
            t_both = np.concatenate([t_cycle, t_cycle], axis=0)
            c_both = None if c_arr is None else np.concatenate([c_arr, c_arr], axis=0)
            nt_both = np.concatenate(
                [tgts_arr, tgts_arr if beta_m != 0.0 else np.zeros_like(tgts_arr)],
                axis=0,
            )
            nb_both = np.concatenate(
                [np.full(eff_batch, beta_p, dtype=np.float64), np.full(eff_batch, beta_m, dtype=np.float64)],
                axis=0,
            )
            obs_both, z_both = self.model.measure_observables_batch(
                x_both, t_both, shots=shots_per_run,
                cond_batch=c_both, nudge_target_batch=nt_both, nudge_beta_batch=nb_both,
            )

            obs_p, obs_m = obs_both[:eff_batch], obs_both[eff_batch:]
            zp_list, zm_list = z_both[:eff_batch], z_both[eff_batch:]
            is_eval_step = verbose and (b_idx % eval_every_batches == 0 or b_idx == total_batches)
            proxy_mses: Optional[List[float]] = [] if is_eval_step else None
            if self.model.calibrator.ema > 0.0 or is_eval_step:
                z_free_batch = 0.5 * (zp_list + zm_list) if self.optimizer.cfg.centered else zm_list
                for k in range(eff_batch):
                    t = int(t_cycle[k])
                    tgt = tgts_arr[k]
                    zf = z_free_batch[k]
                    self.model.calibrator.update(t, zf, tgt)
                    if proxy_mses is not None:
                        x0_hat = self.model.calibrator.calibrate(t, zf)
                        proxy_mses.append(float(np.mean((x0_hat - tgt) ** 2)))

            opt_stats = self.optimizer.step_contrast(
                self.model.parameters_dict(),
                obs_p, obs_m,
                z_plus_list=zp_list, z_minus_list=zm_list,
                conds=c_arr, nudge_dirs=tgts_arr,
                progress=progress,
                shots_in_batch=2 * shots_per_run * eff_batch,
            )

            if is_eval_step:
                ev = self.model.evaluate(
                    targets[: min(n_data, 8)],
                    None if cond_arr is None else cond_arr[: min(n_data, 8)],
                    shots=eval_shots,
                    n_trials=2,
                )
                gain_avg = self.model.calibrator.mean_gain
                row = {
                    "batch": b_idx,
                    "shots": self.optimizer.shots_used,
                    "proxy_mse": float(np.mean(proxy_mses)),
                    "mean_gain": gain_avg,
                    **opt_stats,
                    **ev,
                }
                history.append(row)
                print(
                    f"  [QERD] Batch {b_idx:03d}/{total_batches} | Shots={self.optimizer.shots_used:6d} | "
                    f"lr={opt_stats['lr']:.3f} beta={opt_stats['beta']:.2f} gain={gain_avg:.2f} | "
                    f"GenMSE={ev['mse']:.4f} | SignAcc={ev['sign_acc']*100:5.1f}%",
                    flush=True,
                )

        return history
