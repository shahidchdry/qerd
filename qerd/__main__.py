"""
Module entrypoint (`python -m qerd`) running fast standalone smoke & parity self-tests.
"""

from __future__ import annotations

import time

import numpy as np

from . import (
    QERGradConfig,
    QERGradientEstimator,
    QEROptimizer,
    QERDTrainer,
    QuantumIsingDenoiser,
    StabilizedAdam,
    StabilizedAdamConfig,
)
from .backends import _HAS_TORCH, torch


def _run_self_tests():
    if _HAS_TORCH:
        import torch.nn as nn
        from .torch_layer import QERLayer

    print("=== [TEST 1] 2x4 Conditional Diffusion (Model Owns State, Clean Trainer/Optimizer) ===", flush=True)
    imgs_2x4 = np.array([
        [[+1, +1, +1, +1], [-1, -1, -1, -1]],  # H-Bars
        [[+1, -1, +1, -1], [+1, -1, +1, -1]],  # V-Stripes
        [[+1, -1, +1, -1], [-1, +1, -1, +1]],  # Checker
        [[+1, +1, +1, +1], [+1, -1, -1, +1]],  # Frame
    ], dtype=np.float64)
    cond_4 = np.eye(4, dtype=np.float64)

    t0 = time.time()
    den1 = QuantumIsingDenoiser(data_shape=(2, 4), cond_dim=4, topology="none", backend="numpy", seed=42)
    trainer1 = QERDTrainer(den1)
    trainer1.train(imgs_2x4, cond_4, max_shots=14_000, shots_per_run=40, batch_size=4, eval_every_batches=20)
    ev1 = den1.evaluate(imgs_2x4, cond_4, shots=160, n_trials=6)
    print(f"  -> Final 2x4 Eval ({time.time()-t0:.1f}s): MSE={ev1['mse']:.4f} | SignAcc={ev1['sign_acc']*100:.1f}%", flush=True)

    print("\n=== [TEST 2] 3x4 Grayscale & Spatial Multimodal Diffusion (12 Qubits, Continuous [-1,1]) ===", flush=True)
    imgs_3x4 = np.array([
        [[+0.9, +0.6, +0.2, -0.3], [+0.5, +0.2, -0.4, -0.8], [-0.2, -0.6, -0.9, -1.0]],
        [[+1.0, +0.8, +0.3, -0.2], [+0.6, +0.3, -0.3, -0.7], [-0.1, -0.5, -0.8, -0.9]],
        [[-0.9, +0.8, +0.8, -0.9], [-0.9, +0.9, +0.9, -0.9], [-0.9, +0.8, +0.8, -0.9]],
        [[-0.8, +0.9, +0.7, -0.9], [-0.8, +0.8, +0.9, -0.8], [-0.9, +0.7, +0.9, -0.9]],
        [[+0.9, -0.8, -0.8, +0.9], [-0.8, +0.9, +0.9, -0.8], [+0.9, -0.8, -0.8, +0.9]],
        [[+0.8, -0.9, -0.7, +0.8], [-0.7, +0.8, +0.8, -0.7], [+0.8, -0.7, -0.9, +0.8]],
    ], dtype=np.float64)
    cond_3 = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]], dtype=np.float64)

    t0 = time.time()
    den2 = QuantumIsingDenoiser(data_shape=(3, 4), cond_dim=3, topology="grid4", backend="numpy", seed=7)
    trainer2 = QERDTrainer(den2)
    trainer2.train(imgs_3x4, cond_3, max_shots=18_000, shots_per_run=40, batch_size=6, eval_every_batches=25)
    ev2 = den2.evaluate(imgs_3x4, cond_3, shots=160, n_trials=4)
    print(f"  -> Final 3x4 Grayscale Eval ({time.time()-t0:.1f}s): MSE={ev2['mse']:.4f} | SignAcc={ev2['sign_acc']*100:.1f}%", flush=True)

    # Standalone inference, 1D topology="all" verification, & checkpoint round-trip test
    den_1d = QuantumIsingDenoiser(data_shape=12, cond_dim=3, topology="all", n_hidden=1, backend="numpy", seed=7)
    assert den_1d.data_shape == (12,) and den_1d.n_edges == (12 * 11 // 2) + 12

    sample_img, _ = den2.sample(cond=cond_3[0], shots_per_step=120)
    assert sample_img.shape == (3, 4)

    den_reload = QuantumIsingDenoiser(data_shape=(3, 4), cond_dim=3, topology="grid4", backend="numpy", seed=99)
    den_reload.load_state_dict(den2.state_dict())
    opt_reload = QEROptimizer()
    opt_reload.load_state_dict(trainer2.optimizer.state_dict())
    assert np.allclose(den_reload.theta, den2.theta)
    assert np.allclose(den_reload.cond_proj, den2.cond_proj)
    assert np.allclose(den_reload.calibrator.gain, den2.calibrator.gain)
    assert opt_reload.step_count == trainer2.optimizer.step_count
    print("  -> Standalone sample(), 1D all-to-all topology, & state_dict() checkpoint round-trip passed!", flush=True)

    if _HAS_TORCH:
        print("\n=== [TEST 3] PyTorch Hybrid QERLayer & Decoupled API Check ===", flush=True)
        den_dec = QuantumIsingDenoiser(data_shape=(2, 4), cond_dim=4, topology="grid4", backend="numpy", seed=7)
        est_dec = QERGradientEstimator(QERGradConfig())
        opt_dec = StabilizedAdam(den_dec.parameters_dict(), StabilizedAdamConfig(lr_max=0.05, lr_min=0.015))
        tgts_dec = imgs_2x4.reshape(4, 8)
        t_dec = np.full(4, 2, dtype=np.int64)
        obs_b, z_b = den_dec.measure_observables_batch(
            np.concatenate([tgts_dec * 0.5, tgts_dec * 0.5], axis=0),
            np.concatenate([t_dec, t_dec], axis=0),
            shots=40,
            cond_batch=np.concatenate([cond_4, cond_4], axis=0),
            nudge_target_batch=np.concatenate([tgts_dec, tgts_dec], axis=0),
            nudge_beta_batch=np.concatenate([np.full(4, est_dec.beta), np.full(4, -est_dec.beta)]),
        )
        grads_dec, mean_align = est_dec.estimate(
            den_dec.parameters_dict(),
            obs_b[:4], obs_b[4:],
            z_plus_list=z_b[:4], z_minus_list=z_b[4:],
            conds=cond_4, nudge_dirs=tgts_dec,
        )
        for k in range(4):
            den_dec.calibrator.update(int(t_dec[k]), 0.5 * (z_b[k] + z_b[4 + k]), tgts_dec[k])
        assert set(grads_dec.keys()) == {"theta", "cond_proj"}
        assert mean_align is not None
        assert 0.0 <= den_dec.calibrator.clip_fraction <= 1.0
        assert 0.0 <= den_dec.calibrator.upper_clip_fraction <= 1.0
        lr_dec = opt_dec.step(den_dec.parameters_dict(), grads_dec, progress=0.1, shots_in_batch=320)
        assert opt_dec.step_count == 1 and opt_dec.shots_used == 320 and lr_dec > 0.0
        assert opt_dec.state_dict()["step_count"] == 1

        den_torch = QuantumIsingDenoiser(data_shape=(2, 4), cond_dim=4, topology="grid4", backend="torch", seed=42)
        tr_torch = QERDTrainer(den_torch)
        tr_torch.train(imgs_2x4, cond_4, max_shots=6_400, shots_per_run=40, batch_size=4, eval_every_batches=99, verbose=False)
        ev_torch = den_torch.evaluate(imgs_2x4, cond_4, shots=160, n_trials=4)
        print(f"  -> Torch Backend ({den_torch.backend.device}) 2x4 Eval: MSE={ev_torch['mse']:.4f} | SignAcc={ev_torch['sign_acc']*100:.1f}%", flush=True)

        den_t = QuantumIsingDenoiser(data_shape=(2, 4), cond_dim=0, topology="grid8", backend="numpy", seed=11)
        encoder = nn.Linear(4, 8)
        qpu_layer = QERLayer(den_t, shots=40, beta=0.45, gain=0.75)
        torch_opt = torch.optim.Adam(list(encoder.parameters()) + list(qpu_layer.parameters()), lr=0.05)

        x_in = torch.tensor(imgs_2x4.reshape(4, 8), dtype=torch.float32)
        c_in = torch.eye(4, dtype=torch.float32)
        losses = []
        for step_idx in range(15):
            torch_opt.zero_grad()
            h_c = encoder(c_in)
            x_noisy = 0.5 * x_in + 0.5 * torch.randn_like(x_in) * 0.4
            pred = qpu_layer(x_noisy, h_c) if step_idx % 2 == 0 else qpu_layer(x_noisy, h_c, t_step=2)
            loss = torch.mean((pred - x_in) ** 2)
            loss.backward()
            torch_opt.step()
            losses.append(float(loss.item()))
        print(f"  -> Hybrid PyTorch MSE (QERLayer with/without t_step): start={losses[0]:.4f} -> end={losses[-1]:.4f}", flush=True)


if __name__ == "__main__":
    _run_self_tests()
