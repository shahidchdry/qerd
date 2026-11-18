"""
benchmark_mnist.py — Real-World Latent Diffusion Benchmark for QERD (MNIST / Fashion-MNIST / Digits).

Pipeline:
  1. Loads MNIST (28x28, 60k/10k), Fashion-MNIST (28x28, 60k/10k), or offline sklearn Digits.
  2. Trains a noise-robust Convolutional Autoencoder (PyTorch) compressing 28x28 -> D-dim
     continuous latents in [-1, 1] (normalized via BatchNorm + tanh so all qubits stay active).
  3. Trains an independent 28x28 CNN Classifier on real training images to score generated samples.
  4. Trains up to 4 generative models on the extracted D-dim continuous latents & 10-class labels:
       (a) QER (Ours)     : QuantumIsingDenoiser + QEROptimizer using raw unscaled QPU <Z> expectations
                            (gain=1.0 default on continuous latents; optional OLS gain-cal ablation)
       (b) Adam-SPSA      : Same noisy NISQ circuit & shot budget using 2-run SPSA + Adam
       (c) Adam-PSR       : Same noisy NISQ circuit & shot budget using active Parameter-Shift + Adam
       (d) Classical DDPM : Parameter-matched classical MLP denoiser trained on clean float latents
  5. Generates 28x28 images from pure noise, computes held-out Latent-FID, Classifier-FID,
     Class Accuracy, Nearest-Neighbor Latent MSE, and Intra-Class Diversity, and saves PNG grids.

Usage:
  Kaggle / Full MNIST        : python -u experiments/benchmark_mnist.py --dataset mnist
  Kaggle / Fashion-MNIST     : python -u experiments/benchmark_mnist.py --dataset fashion
  Real QPU hardware run      : python -u experiments/benchmark_mnist.py --dataset mnist --real-qpu --device your.qpu
  Fast local offline check   : python -u experiments/benchmark_mnist.py --dataset digits --latent-dim 8 --max-shots 16000
"""

from __future__ import annotations

try:
    import _paths  # noqa: F401
except ImportError:
    from . import _paths  # noqa: F401

import argparse
import json
import math
import os
import time
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import sqrtm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from qerd import (
    QERConfig,
    QERGradConfig,
    QERGradientEstimator,
    QERLayer,
    QEROptimizer,
    QERDTrainer,
    QuantumIsingDenoiser,
    StabilizedAdam,
    StabilizedAdamConfig,
    build_ising_edges,
    make_alpha_bar,
)


# ==============================================================================
# 1. DATA LOADERS (MNIST, FASHION-MNIST, OFFLINE DIGITS FALLBACK)
# ==============================================================================
def load_dataset(name: str, data_dir: str = "./data") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    name = name.lower()
    if name in ("mnist", "fashion"):
        import torchvision
        import torchvision.transforms as T

        tfm = T.ToTensor()
        if name == "mnist":
            ds_tr = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=tfm)
            ds_te = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=tfm)
            class_names = [str(i) for i in range(10)]
        else:
            ds_tr = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=tfm)
            ds_te = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=tfm)
            class_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]

        x_tr = (ds_tr.data.numpy().astype(np.float32) / 255.0)[:, None, :, :]
        y_tr = ds_tr.targets.numpy().astype(np.int64)
        x_te = (ds_te.data.numpy().astype(np.float32) / 255.0)[:, None, :, :]
        y_te = ds_te.targets.numpy().astype(np.int64)
        return x_tr, y_tr, x_te, y_te, class_names

    if name == "digits":
        from sklearn.datasets import load_digits

        digits = load_digits()
        imgs_8 = torch.tensor(digits.images.astype(np.float32) / 16.0)[:, None, :, :]
        imgs_28 = F.interpolate(imgs_8, size=(28, 28), mode="bilinear", align_corners=False).numpy()
        labels = digits.target.astype(np.int64)
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(labels))
        split = int(0.8 * len(labels))
        tr_idx, te_idx = perm[:split], perm[split:]
        return imgs_28[tr_idx], labels[tr_idx], imgs_28[te_idx], labels[te_idx], [str(i) for i in range(10)]

    raise ValueError(f"Unknown dataset '{name}'. Choose 'mnist', 'fashion', or 'digits'.")


# ==============================================================================
# 2. NOISE-ROBUST 28x28 <-> D-QUBIT AUTOENCODER & EVALUATION CNN CLASSIFIER
# ==============================================================================
class ConvLatentAE(nn.Module):
    """Compresses 28x28 images into balanced D-dimensional latents in [-1, 1]."""

    def __init__(self, latent_dim: int = 12, use_bn: bool = True):
        super().__init__()
        self.use_bn = bool(use_bn)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),   # 14x14
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 7x7
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.GELU(),
            nn.Linear(128, latent_dim),
        )
        # Optional per-channel standardization; disabling preserves cross-channel latent correlations.
        self.bn_latent = nn.BatchNorm1d(latent_dim, affine=False) if self.use_bn else nn.Identity()
        if not self.use_bn:
            nn.init.xavier_uniform_(self.encoder[-1].weight, gain=0.25)
            nn.init.zeros_(self.encoder[-1].bias)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64 * 7 * 7),
            nn.GELU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 14x14
            nn.GELU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),   # 28x28
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(0.85 * self.bn_latent(self.encoder(x)))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor, noise_std: float = 0.08) -> Tuple[torch.Tensor, torch.Tensor]:
        raw = self.bn_latent(self.encoder(x))
        z = torch.tanh(0.85 * raw)
        # Mild latent jitter makes the decoder tolerant to QPU shot/readout noise
        z_in = z + noise_std * torch.randn_like(z) if (self.training and noise_std > 0.0) else z
        self._last_raw_norm = raw.pow(2).mean()
        return self.decode(z_in), z


class EvalCNN(nn.Module):
    """Independent 28x28 classifier used to score generated image quality & FID features."""

    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 64),
            nn.ReLU(),
        )
        self.head = nn.Linear(64, n_classes)

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.conv(x)
        return self.head(feat), feat


def train_classical_helpers(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
    latent_dim: int,
    ae_epochs: int,
    clf_epochs: int,
    device: torch.device,
    cache_path: str | None = None,
    use_bn: bool = True,
) -> Tuple[ConvLatentAE, EvalCNN, np.ndarray, np.ndarray]:
    tr_ds = TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr))
    te_ds = TensorDataset(torch.from_numpy(x_te), torch.from_numpy(y_te))
    te_loader = DataLoader(te_ds, batch_size=512, shuffle=False)

    ae = ConvLatentAE(latent_dim=latent_dim, use_bn=use_bn).to(device)
    clf = EvalCNN().to(device)

    if cache_path is not None and os.path.exists(cache_path):
        ckpt = torch.load(cache_path, map_location=device, weights_only=True)
        ae.load_state_dict(ckpt["ae"])
        clf.load_state_dict(ckpt["clf"])
        print(f"  [Cache] Loaded frozen AE + EvalCNN helpers from {cache_path}", flush=True)
    else:
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        tr_bs = 256 if len(x_tr) >= 10_000 else 32
        tr_loader = DataLoader(tr_ds, batch_size=tr_bs, shuffle=True)

        opt_ae = torch.optim.Adam(ae.parameters(), lr=1.5e-3)
        for ep in range(1, ae_epochs + 1):
            ae.train()
            tot_loss = 0.0
            for bx, _ in tr_loader:
                bx = bx.to(device)
                opt_ae.zero_grad()
                recon, _ = ae(bx, noise_std=0.08)
                loss = F.mse_loss(recon, bx)
                if not use_bn:
                    loss = loss + 0.005 * ae._last_raw_norm
                loss.backward()
                opt_ae.step()
                tot_loss += float(loss.item()) * bx.size(0)
            if ep == 1 or ep == ae_epochs or ep % 5 == 0:
                print(f"  [AE] Epoch {ep:02d}/{ae_epochs} | Recon MSE={tot_loss / len(x_tr):.5f}", flush=True)

        opt_clf = torch.optim.Adam(clf.parameters(), lr=2e-3)
        for ep in range(1, clf_epochs + 1):
            clf.train()
            for bx, by in tr_loader:
                bx, by = bx.to(device), by.to(device)
                opt_clf.zero_grad()
                logits, _ = clf.forward_features(bx)
                F.cross_entropy(logits, by).backward()
                opt_clf.step()

        if cache_path is not None:
            torch.save({"ae": ae.state_dict(), "clf": clf.state_dict()}, cache_path)
            print(f"  [Cache] Saved frozen helpers to {cache_path}", flush=True)

    ae.eval()
    clf.eval()
    with torch.no_grad():
        z_tr = torch.cat([ae.encode(bx.to(device)).cpu() for bx, _ in DataLoader(tr_ds, batch_size=512)], dim=0).numpy()
        z_te = torch.cat([ae.encode(bx.to(device)).cpu() for bx, _ in te_loader], dim=0).numpy()
        correct = 0
        for bx, by in te_loader:
            logits, _ = clf.forward_features(bx.to(device))
            correct += int((logits.argmax(dim=1).cpu() == by).sum().item())
    zc = z_tr - z_tr.mean(axis=0, keepdims=True)
    zn = np.linalg.norm(zc, axis=0, keepdims=True)
    valid = (zn > 1e-8)
    z_unit = np.divide(zc, np.where(valid, zn, 1.0))
    corr_mat = np.abs(z_unit.T @ z_unit)
    off_mask = (~np.eye(latent_dim, dtype=bool)) & (valid.T @ valid)
    mean_offdiag_corr = float(np.mean(corr_mat[off_mask])) if np.any(off_mask) else 0.0
    print(
        f"  [EvalCNN] Test Accuracy on real held-out split: {100.0 * correct / len(x_te):.2f}% | "
        f"AE off-diag |corr|={mean_offdiag_corr:.3f} (bn={use_bn})",
        flush=True,
    )
    return ae, clf, z_tr.astype(np.float64), z_te.astype(np.float64)


# ==============================================================================
# 3. BASELINES (MATCHED-BUDGET ADAM-SPSA & PARAMETER-MATCHED CLASSICAL DDPM)
# ==============================================================================
def _sample_stratified_ids(rng: np.random.Generator, class_pools: List[np.ndarray], eff_batch: int) -> np.ndarray:
    c_order = rng.permutation(len(class_pools))
    return np.array([rng.choice(class_pools[c_order[k % len(class_pools)]]) for k in range(eff_batch)], dtype=np.int64)


def _adam_ascent_update(
    model: QuantumIsingDenoiser,
    g_th: np.ndarray,
    g_cp: np.ndarray,
    m_th: np.ndarray,
    v_th: np.ndarray,
    m_cp: np.ndarray,
    v_cp: np.ndarray,
    lr_k: float,
    step: int,
    clip_bound: float,
) -> None:
    """Shared bias-corrected Adam ascent step for NISQ finite-difference baselines."""
    m_th[:] = 0.75 * m_th + 0.25 * g_th
    v_th[:] = 0.98 * v_th + 0.02 * (g_th ** 2)
    m_cp[:] = 0.75 * m_cp + 0.25 * g_cp
    v_cp[:] = 0.98 * v_cp + 0.02 * (g_cp ** 2)

    m_hat_scale = 1.0 - (0.75 ** step)
    v_hat_scale = 1.0 - (0.98 ** step)
    model.theta[:] = np.clip(
        model.theta + lr_k * (m_th / m_hat_scale) / (np.sqrt(v_th / v_hat_scale) + 1e-6),
        -clip_bound,
        clip_bound,
    )
    model.cond_proj[:] = np.clip(
        model.cond_proj + lr_k * (m_cp / m_hat_scale) / (np.sqrt(v_cp / v_hat_scale) + 1e-6),
        -clip_bound,
        clip_bound,
    )


def train_adam_spsa_baseline(
    model: QuantumIsingDenoiser,
    data: np.ndarray,
    conds: np.ndarray,
    y_labels: np.ndarray,
    max_shots: int = 80_000,
    shots_per_run: int = 20,
    batch_size: int = 10,
    lr: float = 0.08,
    c0: float = 0.18,
    clip_bound: float = 0.90,
    chain_rule_wcond: bool = False,
) -> Dict[str, int]:
    """2-run Adam-SPSA baseline on the identical NISQ circuit, calibrator & shot budget."""
    n_data = data.shape[0]
    eff_batch = min(batch_size, n_data)
    total_batches = max(1, max_shots // (2 * shots_per_run * eff_batch))
    class_pools = [np.flatnonzero(y_labels == c) for c in np.unique(y_labels)]

    m_th = np.zeros_like(model.theta)
    v_th = np.zeros_like(model.theta)
    m_cp = np.zeros_like(model.cond_proj)
    v_cp = np.zeros_like(model.cond_proj)
    shots_used = 0

    for step in range(1, total_batches + 1):
        c_k = c0 / (step ** 0.101)
        lr_k = 0.015 + 0.5 * (lr - 0.015) * (1.0 + math.cos(math.pi * (step - 1) / max(1, total_batches - 1)))
        sample_ids = _sample_stratified_ids(model.rng, class_pools, eff_batch)
        t_batch = (np.arange(eff_batch) % model.diffusion_steps + 1).astype(np.int64)
        model.rng.shuffle(t_batch)

        tgt_batch = data[sample_ids]
        c_batch = conds[sample_ids]
        a_bar = model.alpha_bar[t_batch][:, None]
        x_t_batch = np.clip(
            np.sqrt(a_bar) * tgt_batch
            + np.sqrt(1.0 - a_bar) * model.rng.normal(0.0, 0.70, size=tgt_batch.shape),
            -1.0,
            1.0,
        )

        d_th = model.rng.choice([-1.0, 1.0], size=(eff_batch,) + model.theta.shape)
        th_p = model.theta[None, ...] + c_k * d_th
        th_m = model.theta[None, ...] - c_k * d_th
        if chain_rule_wcond or model.cond_dim == 0:
            d_cp = None
            cp_arg = None
        else:
            d_cp = model.rng.choice([-1.0, 1.0], size=(eff_batch,) + model.cond_proj.shape)
            cp_p = model.cond_proj[None, ...] + c_k * d_cp
            cp_m = model.cond_proj[None, ...] - c_k * d_cp
            cp_arg = np.concatenate([cp_p, cp_m], axis=0)

        _, z_2b = model.measure_observables_batch(
            np.concatenate([x_t_batch, x_t_batch], axis=0),
            np.concatenate([t_batch, t_batch], axis=0),
            shots=shots_per_run,
            cond_batch=np.concatenate([c_batch, c_batch], axis=0),
            theta_batch=np.concatenate([th_p, th_m], axis=0),
            cond_proj_batch=cp_arg,
        )
        shots_used += 2 * eff_batch * shots_per_run
        zp = z_2b[:eff_batch]
        zm = z_2b[eff_batch:]

        z_mid = 0.5 * (zp + zm)
        for k in range(eff_batch):
            model.calibrator.update(int(t_batch[k]), z_mid[k], tgt_batch[k])
        gain_batch = np.stack([model.calibrator.gain[int(t)] for t in t_batch], axis=0)
        if gain_batch.ndim == 1:
            gain_batch = gain_batch[:, None]

        loss_p = np.mean((np.clip(zp / gain_batch, -1.0, 1.0) - tgt_batch) ** 2, axis=1)
        loss_m = np.mean((np.clip(zm / gain_batch, -1.0, 1.0) - tgt_batch) ** 2, axis=1)
        diff = (loss_p - loss_m) / (2.0 * c_k)

        # Negative gradient (ascent step updates param = param + step)
        g_th = -np.mean(diff.reshape(eff_batch, *([1] * model.theta.ndim)) * d_th, axis=0)
        if d_cp is None:
            if chain_rule_wcond and model.cond_dim > 0:
                g_bvis = -(diff[:, None] * d_th[:, model.n_edges : model.n_edges + model.n_visible])
                if model.cond_proj.shape == (model.n_visible, conds.shape[1]):
                    g_cp = np.einsum("bi,bj->ij", g_bvis, c_batch) / eff_batch
                else:
                    g_cp = np.einsum("bj,bi->ij", c_batch, g_bvis) / eff_batch
            else:
                g_cp = np.zeros_like(model.cond_proj)
        else:
            g_cp = -np.mean(diff.reshape(eff_batch, *([1] * model.cond_proj.ndim)) * d_cp, axis=0)

        _adam_ascent_update(model, g_th, g_cp, m_th, v_th, m_cp, v_cp, lr_k, step, clip_bound)

    return {"shots": shots_used, "batches": total_batches}


def train_adam_psr_baseline(
    model: QuantumIsingDenoiser,
    data: np.ndarray,
    conds: np.ndarray,
    y_labels: np.ndarray,
    max_shots: int = 80_000,
    shots_per_run: int = 20,
    lr: float = 0.08,
    delta: float = 0.25,
    clip_bound: float = 0.90,
    chain_rule_wcond: bool = False,
) -> Dict[str, int]:
    """Active-parameter symmetric finite-diff / chain-rule Adam baseline."""
    class_pools = [np.flatnonzero(y_labels == c) for c in np.unique(y_labels)]
    n_classes = len(class_pools)
    use_col_axis = not (
        model.cond_proj.shape[0] == conds.shape[1]
        and model.cond_proj.shape[1] == model.n_visible
        and model.cond_proj.shape[0] != model.cond_proj.shape[1]
    )

    n_th = model.theta.size
    sample_act = int(np.sum(np.abs(conds[0]) > 1e-8)) or (
        model.cond_proj.shape[1] if use_col_axis else model.cond_proj.shape[0]
    )
    est_cp_act = 0 if chain_rule_wcond else (model.cond_proj.shape[0] if use_col_axis else model.cond_proj.shape[1]) * sample_act
    est_step_shots = 2 * (n_th + est_cp_act) * shots_per_run
    est_total_steps = max(1, max_shots // max(1, est_step_shots))

    m_th = np.zeros_like(model.theta)
    v_th = np.zeros_like(model.theta)
    m_cp = np.zeros_like(model.cond_proj)
    v_cp = np.zeros_like(model.cond_proj)
    shots_used = 0
    step = 0

    cls_queue: List[int] = []
    t_queue: List[int] = []

    while True:
        if not cls_queue:
            cls_queue = model.rng.permutation(n_classes).tolist()
        if not t_queue:
            t_queue = (model.rng.permutation(model.diffusion_steps) + 1).tolist()

        cls_idx = cls_queue.pop()
        t = int(t_queue.pop())
        pool = class_pools[cls_idx]
        sid = int(model.rng.choice(pool))

        tgt = data[sid]
        c_vec = conds[sid]
        active_idx = np.flatnonzero(np.abs(c_vec) > 1e-8)
        if active_idx.size == 0:
            active_idx = np.arange(model.cond_proj.shape[1] if use_col_axis else model.cond_proj.shape[0])

        if chain_rule_wcond:
            cp_coords: List[Tuple[int, int]] = []
        elif use_col_axis:
            cp_coords = [(r, int(c)) for c in active_idx for r in range(model.cond_proj.shape[0])]
        else:
            cp_coords = [(int(r), c) for r in active_idx for c in range(model.cond_proj.shape[1])]

        n_active = n_th + len(cp_coords)
        step_shots = 2 * n_active * shots_per_run
        if shots_used + step_shots > max_shots:
            break

        step += 1
        lr_k = 0.015 + 0.5 * (lr - 0.015) * (1.0 + math.cos(math.pi * (step - 1) / max(1, est_total_steps - 1)))

        a_bar = model.alpha_bar[t]
        x_t = np.clip(
            math.sqrt(a_bar) * tgt + math.sqrt(1.0 - a_bar) * model.rng.normal(0.0, 0.70, size=model.n_visible),
            -1.0,
            1.0,
        )

        # Evaluate all +/- parameter perturbations in one batched circuit call
        theta_flat = model.theta.ravel()
        th_p = np.broadcast_to(theta_flat, (n_active, n_th)).copy()
        th_m = th_p.copy()
        idx_th = np.arange(n_th)
        th_p[idx_th, idx_th] += delta
        th_m[idx_th, idx_th] -= delta
        th_p = th_p.reshape((n_active,) + model.theta.shape)
        th_m = th_m.reshape((n_active,) + model.theta.shape)

        if cp_coords:
            cp_p = np.broadcast_to(model.cond_proj, (n_active,) + model.cond_proj.shape).copy()
            cp_m = cp_p.copy()
            for offset, (r, c) in enumerate(cp_coords):
                cp_p[n_th + offset, r, c] += delta
                cp_m[n_th + offset, r, c] -= delta
            cp_arg = np.concatenate([cp_p, cp_m], axis=0)
        else:
            cp_arg = None

        n_eval = 2 * n_active
        _, z_all = model.measure_observables_batch(
            np.broadcast_to(x_t, (n_eval, model.n_visible)),
            np.full((n_eval,), t, dtype=np.int64),
            shots=shots_per_run,
            cond_batch=np.broadcast_to(c_vec, (n_eval, c_vec.size)),
            theta_batch=np.concatenate([th_p, th_m], axis=0),
            cond_proj_batch=cp_arg,
        )
        shots_used += step_shots
        zp = z_all[:n_active]
        zm = z_all[n_active:]

        model.calibrator.update(t, np.mean(z_all, axis=0), tgt)
        gain_t = model.calibrator.gain[t]
        loss_p = np.mean((np.clip(zp / gain_t, -1.0, 1.0) - tgt[None, :]) ** 2, axis=1)
        loss_m = np.mean((np.clip(zm / gain_t, -1.0, 1.0) - tgt[None, :]) ** 2, axis=1)
        grad = (loss_p - loss_m) / (2.0 * delta)

        g_th = np.zeros_like(model.theta)
        g_th.ravel()[:] = -grad[:n_th]
        g_cp = np.zeros_like(model.cond_proj)
        if chain_rule_wcond and model.cond_dim > 0:
            g_bvis = g_th[model.n_edges : model.n_edges + model.n_visible]
            g_cp[:] = np.outer(g_bvis, c_vec) if use_col_axis else np.outer(c_vec, g_bvis)
        else:
            for offset, (r, c) in enumerate(cp_coords):
                g_cp[r, c] = -grad[n_th + offset]

        _adam_ascent_update(model, g_th, g_cp, m_th, v_th, m_cp, v_cp, lr_k, step, clip_bound)

    return {"shots": shots_used, "batches": step}


def train_adaptive_spsa_baseline(
    model: QuantumIsingDenoiser,
    data: np.ndarray,
    conds: np.ndarray,
    y_labels: np.ndarray,
    max_shots: int = 80_000,
    shots_per_run: int = 20,
    batch_size: int = 10,
    lr: float = 0.10,
    c0: float = 0.22,
    clip_bound: float = 0.90,
) -> Dict[str, int]:
    """Tuned two-stage shot-adaptive 2-run SPSA + clipped Adam ascent on identical NISQ budget."""
    n_data = data.shape[0]
    eff_batch = min(batch_size, n_data)
    class_pools = [np.flatnonzero(y_labels == c) for c in np.unique(y_labels)]

    shots_lo = max(4, shots_per_run // 2)
    shots_hi = max(shots_lo, int(round(1.5 * shots_per_run)))
    mid_shots = max_shots // 2

    m_th = np.zeros_like(model.theta)
    v_th = np.zeros_like(model.theta)
    m_cp = np.zeros_like(model.cond_proj)
    v_cp = np.zeros_like(model.cond_proj)
    shots_used = 0
    step = 0
    approx_steps = max(1, max_shots // (2 * shots_per_run * eff_batch))

    while shots_used < max_shots:
        s_run = shots_lo if shots_used < mid_shots else shots_hi
        step_shots = 2 * eff_batch * s_run
        if shots_used + step_shots > max_shots:
            s_rem = (max_shots - shots_used) // (2 * eff_batch)
            if s_rem < 4:
                break
            s_run = s_rem
            step_shots = 2 * eff_batch * s_run

        step += 1
        c_k = max(0.06, c0 / (step ** 0.101))
        progress = min(1.0, shots_used / max(1, max_shots))
        lr_k = 0.015 + 0.5 * (lr - 0.015) * (1.0 + math.cos(math.pi * progress))

        sample_ids = _sample_stratified_ids(model.rng, class_pools, eff_batch)
        t_batch = (np.arange(eff_batch) % model.diffusion_steps + 1).astype(np.int64)
        model.rng.shuffle(t_batch)
        tgt_batch = data[sample_ids]
        c_batch = conds[sample_ids]
        a_bar = model.alpha_bar[t_batch][:, None]
        x_t_batch = np.clip(
            np.sqrt(a_bar) * tgt_batch + np.sqrt(1.0 - a_bar) * model.rng.normal(0.0, 0.70, size=tgt_batch.shape),
            -1.0,
            1.0,
        )

        d_th = model.rng.choice([-1.0, 1.0], size=(eff_batch,) + model.theta.shape)
        d_cp = model.rng.choice([-1.0, 1.0], size=(eff_batch,) + model.cond_proj.shape)

        _, z_2b = model.measure_observables_batch(
            np.concatenate([x_t_batch, x_t_batch], axis=0),
            np.concatenate([t_batch, t_batch], axis=0),
            shots=s_run,
            cond_batch=np.concatenate([c_batch, c_batch], axis=0),
            theta_batch=np.concatenate([model.theta[None, ...] + c_k * d_th, model.theta[None, ...] - c_k * d_th], axis=0),
            cond_proj_batch=np.concatenate([model.cond_proj[None, ...] + c_k * d_cp, model.cond_proj[None, ...] - c_k * d_cp], axis=0),
        )
        shots_used += step_shots
        zp, zm = z_2b[:eff_batch], z_2b[eff_batch:]
        z_mid = 0.5 * (zp + zm)
        for k in range(eff_batch):
            model.calibrator.update(int(t_batch[k]), z_mid[k], tgt_batch[k])
        gain_batch = np.stack([model.calibrator.gain[int(t)] for t in t_batch], axis=0)
        if gain_batch.ndim == 1:
            gain_batch = gain_batch[:, None]

        loss_p = np.mean((np.clip(zp / gain_batch, -1.0, 1.0) - tgt_batch) ** 2, axis=1)
        loss_m = np.mean((np.clip(zm / gain_batch, -1.0, 1.0) - tgt_batch) ** 2, axis=1)
        diff = (loss_p - loss_m) / (2.0 * c_k)

        g_th = np.clip(-np.mean(diff.reshape(eff_batch, *([1] * model.theta.ndim)) * d_th, axis=0), -2.0, 2.0)
        g_cp = np.clip(-np.mean(diff.reshape(eff_batch, *([1] * model.cond_proj.ndim)) * d_cp, axis=0), -2.0, 2.0)
        m_th[:] = 0.85 * m_th + 0.15 * g_th
        v_th[:] = 0.99 * v_th + 0.01 * (g_th ** 2)
        m_cp[:] = 0.85 * m_cp + 0.15 * g_cp
        v_cp[:] = 0.99 * v_cp + 0.01 * (g_cp ** 2)
        mh = 1.0 - (0.85 ** step)
        vh = 1.0 - (0.99 ** step)
        model.theta[:] = np.clip(model.theta + lr_k * (m_th / mh) / (np.sqrt(v_th / vh) + 1e-6), -clip_bound, clip_bound)
        model.cond_proj[:] = np.clip(model.cond_proj + lr_k * (m_cp / mh) / (np.sqrt(v_cp / vh) + 1e-6), -clip_bound, clip_bound)

    return {"shots": shots_used, "batches": step}


def _sample_layer_perturbed_z(
    model: QuantumIsingDenoiser,
    h_layers: np.ndarray,
    j_layers: np.ndarray,
    t_norm: float,
    total_shots: int,
    device: torch.device,
) -> np.ndarray:
    """Vectorized PyTorch trajectory sampler supporting per-layer Ising angle shifts (Exact PSR)."""
    m, n_layers, n_qubits = h_layers.shape
    n_states = 1 << n_qubits
    shots_per_traj = 4
    n_traj = max(1, total_shots // shots_per_traj)

    basis = torch.arange(n_states, dtype=torch.int64, device=device)
    shifts = torch.arange(n_qubits - 1, -1, -1, dtype=torch.int64, device=device)
    bit_table = (basis[:, None] >> shifts[None, :]) & 1
    z_basis = (1.0 - 2.0 * bit_table.to(torch.float32)).t().contiguous()
    if model.n_edges > 0:
        eu = torch.tensor(model._edge_u, dtype=torch.int64, device=device)
        ev = torch.tensor(model._edge_v, dtype=torch.int64, device=device)
        zz_basis = (z_basis[eu] * z_basis[ev]).contiguous()
    else:
        zz_basis = None

    pauli_mats = torch.tensor(
        [[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]],
        dtype=torch.complex64,
        device=device,
    )
    p0 = 1.0 - 0.75 * model.p_depol
    p1 = 0.25 * model.p_depol
    cdf = torch.tensor([p0, p0 + p1, p0 + 2.0 * p1], dtype=torch.float32, device=device)

    h_t = torch.as_tensor(h_layers, dtype=torch.float32, device=device)
    j_t = torch.as_tensor(j_layers, dtype=torch.float32, device=device)
    chunk = max(1, 2_000_000 // max(1, n_traj * n_states))
    z_out = []

    with torch.no_grad():
        for st in range(0, m, chunk):
            en = min(m, st + chunk)
            mc = en - st
            k_tot = mc * n_traj
            psi = torch.full((mc, n_traj, n_states), 1.0 / math.sqrt(n_states), dtype=torch.complex64, device=device)
            m_grid = torch.arange(mc, device=device)[:, None].expand(mc, n_traj)

            for layer in range(n_layers):
                s = (layer + 1) / n_layers
                mold_gain = 2.0 * model.dt * model.temp_beta * s
                diag = h_t[st:en, layer] @ z_basis
                if zz_basis is not None:
                    diag = diag + j_t[st:en, layer] @ zz_basis
                psi.mul_(torch.exp((-0.5j * mold_gain) * diag.to(torch.complex64)).unsqueeze(1))

                shake_angle = model.shake_scale * 2.0 * model.dt * (1.0 - model.shake_final_frac * s) * (0.6 + 0.4 * t_norm)
                ca = torch.full((mc,), math.cos(0.5 * shake_angle), dtype=torch.complex64, device=device)
                sa = torch.full((mc,), -1j * math.sin(0.5 * shake_angle), dtype=torch.complex64, device=device)
                if model.p_depol > 0.0:
                    rx_mat = torch.stack([torch.stack([ca, sa], dim=-1), torch.stack([sa, ca], dim=-1)], dim=-2)
                    u_all = torch.einsum("fij,mjk->mfik", pauli_mats, rx_mat)
                    u_rand = torch.rand((mc, n_traj, n_qubits), device=device)
                    faults = (u_rand.unsqueeze(-1) >= cdf).sum(dim=-1)
                    for q in range(n_qubits):
                        u_q = u_all[m_grid, faults[:, :, q]].reshape(k_tot, 2, 2)
                        v = psi.view(k_tot, 1 << q, 2, 1 << (n_qubits - 1 - q))
                        a0, a1 = v[:, :, 0, :], v[:, :, 1, :]
                        v[:, :, 0, :] = u_q[:, 0, 0].view(k_tot, 1, 1) * a0 + u_q[:, 0, 1].view(k_tot, 1, 1) * a1
                        v[:, :, 1, :] = u_q[:, 1, 0].view(k_tot, 1, 1) * a0 + u_q[:, 1, 1].view(k_tot, 1, 1) * a1
                else:
                    for q in range(n_qubits):
                        v = psi.view(k_tot, 1 << q, 2, 1 << (n_qubits - 1 - q))
                        a0, a1 = v[:, :, 0, :], v[:, :, 1, :]
                        v[:, :, 0, :] = ca[0] * a0 + sa[0] * a1
                        v[:, :, 1, :] = sa[0] * a0 + ca[0] * a1

            probs = (psi.real * psi.real + psi.imag * psi.imag).view(k_tot, n_states)
            probs = probs / torch.clamp(probs.sum(dim=1, keepdim=True), min=1e-12)
            idx = torch.multinomial(probs, num_samples=shots_per_traj, replacement=True)
            bits = bit_table[idx.reshape(-1)].view(mc, n_traj * shots_per_traj, n_qubits)
            if model.p_readout > 0.0:
                bits = bits ^ (torch.rand(bits.shape, device=device) < model.p_readout).to(bits.dtype)
            spins = 1.0 - 2.0 * bits[:, :, : model.n_visible].to(torch.float32)
            z_out.append(spins.mean(dim=1).cpu().numpy().astype(np.float64))

    return np.concatenate(z_out, axis=0)


def train_exact_psr_baseline(
    model: QuantumIsingDenoiser,
    data: np.ndarray,
    conds: np.ndarray,
    y_labels: np.ndarray,
    device: torch.device,
    max_shots: int = 80_000,
    shots_per_run: int = 20,
    lr: float = 0.08,
    clip_bound: float = 0.90,
) -> Dict[str, int]:
    """True unbiased multi-layer Parameter-Shift Rule (pi/2 shift per gate occurrence) + Adam."""
    class_pools = [np.flatnonzero(y_labels == c) for c in np.unique(y_labels)]
    n_classes = len(class_pools)
    n_th = model.theta.size
    n_layers = model.n_layers
    sample_act = int(np.sum(np.abs(conds[0]) > 1e-8)) or model.cond_proj.shape[1]
    est_cp_act = model.cond_proj.shape[0] * sample_act
    est_step_shots = (1 + 2 * n_layers * (n_th + est_cp_act)) * shots_per_run
    est_total_steps = max(1, max_shots // max(1, est_step_shots))

    m_th = np.zeros_like(model.theta)
    v_th = np.zeros_like(model.theta)
    m_cp = np.zeros_like(model.cond_proj)
    v_cp = np.zeros_like(model.cond_proj)
    shots_used = 0
    step = 0
    cls_queue: List[int] = []
    t_queue: List[int] = []

    mold_gains = [2.0 * model.dt * model.temp_beta * ((l + 1) / n_layers) for l in range(n_layers)]

    while True:
        if not cls_queue:
            cls_queue = model.rng.permutation(n_classes).tolist()
        if not t_queue:
            t_queue = (model.rng.permutation(model.diffusion_steps) + 1).tolist()
        cls_idx = cls_queue.pop()
        t = int(t_queue.pop())
        sid = int(model.rng.choice(class_pools[cls_idx]))
        tgt = data[sid]
        c_vec = conds[sid]
        active_idx = np.flatnonzero(np.abs(c_vec) > 1e-8)
        if active_idx.size == 0:
            active_idx = np.arange(model.cond_proj.shape[1])
        cp_coords = [(r, int(c)) for c in active_idx for r in range(model.cond_proj.shape[0])]

        # Only need per-layer shifts on all Ising parameters (n_th) + visible qubit biases (n_visible for cond_proj)
        n_shift_params = n_th
        n_eval = 1 + 2 * n_layers * n_shift_params
        step_shots = n_eval * shots_per_run
        if shots_used + step_shots > max_shots:
            break

        step += 1
        lr_k = 0.015 + 0.5 * (lr - 0.015) * (1.0 + math.cos(math.pi * (step - 1) / max(1, est_total_steps - 1)))
        a_bar = model.alpha_bar[t]
        x_t = np.clip(
            math.sqrt(a_bar) * tgt + math.sqrt(1.0 - a_bar) * model.rng.normal(0.0, 0.70, size=model.n_visible),
            -1.0,
            1.0,
        )
        t_norm = float(t) / model.diffusion_steps
        h0, j0 = model._build_fields(x_t, t_norm, c_vec, None, 0.0)

        h_stack = np.broadcast_to(h0, (n_eval, n_layers, model.n_qubits)).copy()
        j_stack = np.broadcast_to(j0, (n_eval, n_layers, model.n_edges)).copy()
        idx_ptr = 1
        pair_meta: List[Tuple[int, int, float]] = []
        for l_idx in range(n_layers):
            shift_ang = (0.5 * math.pi) / mold_gains[l_idx]
            mg = mold_gains[l_idx]
            for e_idx in range(model.n_edges):
                j_stack[idx_ptr, l_idx, e_idx] += shift_ang
                j_stack[idx_ptr + 1, l_idx, e_idx] -= shift_ang
                pair_meta.append((e_idx, l_idx, mg))
                idx_ptr += 2
            for q_idx in range(model.n_qubits):
                h_stack[idx_ptr, l_idx, q_idx] += shift_ang
                h_stack[idx_ptr + 1, l_idx, q_idx] -= shift_ang
                pair_meta.append((model.n_edges + q_idx, l_idx, mg))
                idx_ptr += 2

        z_all = _sample_layer_perturbed_z(model, h_stack, j_stack, t_norm, shots_per_run, device)
        shots_used += step_shots
        z_base = z_all[0]
        model.calibrator.update(t, z_base, tgt)
        gain_t = model.calibrator.gain[t]
        err_vec = 2.0 / model.n_visible * (np.clip(z_base / gain_t, -1.0, 1.0) - tgt) / gain_t

        dL_dparam = np.zeros(n_shift_params, dtype=np.float64)
        for p_idx, (param_id, _, mg) in enumerate(pair_meta):
            zp = z_all[1 + 2 * p_idx]
            zm = z_all[1 + 2 * p_idx + 1]
            dz = 0.5 * mg * (zp - zm)
            dL_dparam[param_id] += float(np.dot(err_vec, dz))

        g_th = -dL_dparam[:n_th]
        g_cp = np.zeros_like(model.cond_proj)
        dL_dh_vis = dL_dparam[model.n_edges : model.n_edges + model.n_visible]
        for r, c in cp_coords:
            g_cp[r, c] = -dL_dh_vis[r] * c_vec[c]

        _adam_ascent_update(model, g_th, g_cp, m_th, v_th, m_cp, v_cp, lr_k, step, clip_bound)

    return {"shots": shots_used, "batches": step}


def train_qnspsa_baseline(
    model: QuantumIsingDenoiser,
    data: np.ndarray,
    conds: np.ndarray,
    y_labels: np.ndarray,
    max_shots: int = 80_000,
    shots_per_run: int = 20,
    batch_size: int = 10,
    lr: float = 0.08,
    c0: float = 0.18,
    delta_f: float = 0.15,
    reg_beta: float = 1e-3,
    clip_bound: float = 0.90,
    chain_rule_wcond: bool = True,
    n_metric_samples: int = 2,
    device: torch.device | str | None = None,
) -> Dict[str, int]:
    """Quantum Natural SPSA (Spall 2nd-order QFIM + SPSA) on equal NISQ shot budget."""
    n_data = data.shape[0]
    eff_batch = min(batch_size, n_data)
    m_fid = min(eff_batch, max(1, int(n_metric_samples)))
    shots_per_step = (2 * eff_batch + 4 * m_fid) * shots_per_run
    total_steps = max(1, max_shots // max(1, shots_per_step))
    class_pools = [np.flatnonzero(y_labels == c) for c in np.unique(y_labels)]

    tdev = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    circ = _DiffIsingCircuit(
        model.n_visible,
        model.cond_dim,
        model.edges,
        n_hidden=model.n_hidden,
        n_layers=model.n_layers,
        diffusion_steps=model.diffusion_steps,
        dt=model.dt,
        temp_beta=model.temp_beta,
        shake_scale=model.shake_scale,
        shake_final_frac=model.shake_final_frac,
        in_pinning_base=model.in_pinning_base,
        in_pinning_slope=model.in_pinning_slope,
        device=tdev,
    )

    n_th = model.theta.size
    dim_f = n_th if (chain_rule_wcond or model.cond_dim == 0) else (n_th + model.cond_proj.size)
    g_avg = np.eye(dim_f, dtype=np.float64)
    shots_used = 0

    for step in range(1, total_steps + 1):
        c_k = c0 / (step ** 0.101)
        lr_k = 0.015 + 0.5 * (lr - 0.015) * (1.0 + math.cos(math.pi * (step - 1) / max(1, total_steps - 1)))
        sample_ids = _sample_stratified_ids(model.rng, class_pools, eff_batch)
        t_batch = (np.arange(eff_batch) % model.diffusion_steps + 1).astype(np.int64)
        model.rng.shuffle(t_batch)

        tgt_batch = data[sample_ids]
        c_batch = conds[sample_ids]
        a_bar = model.alpha_bar[t_batch][:, None]
        x_t_batch = np.clip(
            np.sqrt(a_bar) * tgt_batch
            + np.sqrt(1.0 - a_bar) * model.rng.normal(0.0, 0.70, size=tgt_batch.shape),
            -1.0,
            1.0,
        )

        # 1) Standard 2-run SPSA loss gradient under full NISQ noise
        d_th = model.rng.choice([-1.0, 1.0], size=(eff_batch,) + model.theta.shape)
        th_p = model.theta[None, ...] + c_k * d_th
        th_m = model.theta[None, ...] - c_k * d_th
        if chain_rule_wcond or model.cond_dim == 0:
            d_cp = None
            cp_arg = None
        else:
            d_cp = model.rng.choice([-1.0, 1.0], size=(eff_batch,) + model.cond_proj.shape)
            cp_arg = np.concatenate(
                [model.cond_proj[None, ...] + c_k * d_cp, model.cond_proj[None, ...] - c_k * d_cp], axis=0
            )

        _, z_2b = model.measure_observables_batch(
            np.concatenate([x_t_batch, x_t_batch], axis=0),
            np.concatenate([t_batch, t_batch], axis=0),
            shots=shots_per_run,
            cond_batch=np.concatenate([c_batch, c_batch], axis=0),
            theta_batch=np.concatenate([th_p, th_m], axis=0),
            cond_proj_batch=cp_arg,
        )
        shots_used += 2 * eff_batch * shots_per_run
        zp, zm = z_2b[:eff_batch], z_2b[eff_batch:]
        z_mid = 0.5 * (zp + zm)
        for k in range(eff_batch):
            model.calibrator.update(int(t_batch[k]), z_mid[k], tgt_batch[k])
        gain_batch = np.stack([model.calibrator.gain[int(t)] for t in t_batch], axis=0)
        if gain_batch.ndim == 1:
            gain_batch = gain_batch[:, None]

        loss_p = np.mean((np.clip(zp / gain_batch, -1.0, 1.0) - tgt_batch) ** 2, axis=1)
        loss_m = np.mean((np.clip(zm / gain_batch, -1.0, 1.0) - tgt_batch) ** 2, axis=1)
        diff = (loss_p - loss_m) / (2.0 * c_k)

        g_th = -np.mean(diff.reshape(eff_batch, *([1] * model.theta.ndim)) * d_th, axis=0)
        if d_cp is None:
            if chain_rule_wcond and model.cond_dim > 0:
                g_bvis = -(diff[:, None] * d_th[:, model.n_edges : model.n_edges + model.n_visible])
                if model.cond_proj.shape == (model.n_visible, conds.shape[1]):
                    g_cp = np.einsum("bi,bj->ij", g_bvis, c_batch) / eff_batch
                else:
                    g_cp = np.einsum("bj,bi->ij", c_batch, g_bvis) / eff_batch
            else:
                g_cp = np.zeros_like(model.cond_proj)
        else:
            g_cp = -np.mean(diff.reshape(eff_batch, *([1] * model.cond_proj.ndim)) * d_cp, axis=0)

        # 2) Spall 4-evaluation finite-shot Fubini-Study metric estimate (compute-uncompute overlaps)
        d1 = model.rng.choice([-1.0, 1.0], size=dim_f)
        d2 = model.rng.choice([-1.0, 1.0], size=dim_f)
        offsets = [
            delta_f * d1 + delta_f * d2,
            delta_f * d1,
            -delta_f * d1 + delta_f * d2,
            -delta_f * d1,
        ]
        with torch.no_grad():
            xt_m = torch.tensor(x_t_batch[:m_fid], dtype=torch.float32, device=tdev)
            tt_m = torch.tensor(t_batch[:m_fid], dtype=torch.int64, device=tdev)
            ct_m = None if model.cond_dim == 0 else torch.tensor(c_batch[:m_fid], dtype=torch.float32, device=tdev)
            th_base = torch.tensor(model.theta, dtype=torch.float32, device=tdev)
            cp_base = None if model.cond_dim == 0 else torch.tensor(model.cond_proj, dtype=torch.float32, device=tdev)
            psi_0 = circ.statevector(xt_m, tt_m, ct_m, th_base, cp_base)
            f_vals = []
            for s_vec in offsets:
                th_p_t = torch.tensor(model.theta + s_vec[:n_th], dtype=torch.float32, device=tdev)
                if not chain_rule_wcond and model.cond_dim > 0:
                    cp_p_t = torch.tensor(
                        model.cond_proj + s_vec[n_th:].reshape(model.cond_proj.shape), dtype=torch.float32, device=tdev
                    )
                else:
                    cp_p_t = cp_base
                psi_s = circ.statevector(xt_m, tt_m, ct_m, th_p_t, cp_p_t)
                ov = torch.abs(torch.sum(psi_0.conj() * psi_s, dim=1)) ** 2
                ov_np = np.clip(ov.detach().cpu().numpy(), 0.0, 1.0)
                f_shot = np.mean(model.rng.binomial(shots_per_run, ov_np) / float(shots_per_run))
                f_vals.append(float(f_shot))
        shots_used += 4 * m_fid * shots_per_run

        df = f_vals[0] - f_vals[1] - f_vals[2] + f_vals[3]
        g_cur = -(df / (8.0 * delta_f * delta_f)) * (np.outer(d1, d2) + np.outer(d2, d1))
        g_avg = (step / float(step + 1)) * g_avg + (1.0 / float(step + 1)) * g_cur
        eigvals, eigvecs = np.linalg.eigh(0.5 * (g_avg + g_avg.T))
        eig_reg = np.maximum(np.sqrt(eigvals * eigvals), reg_beta)
        nat_inv = (eigvecs / eig_reg[None, :]) @ eigvecs.T

        if chain_rule_wcond or model.cond_dim == 0:
            nat_g_th = np.clip(nat_inv @ g_th.ravel(), -1.5, 1.5).reshape(model.theta.shape)
            nat_g_cp = np.clip(g_cp, -1.5, 1.5)
        else:
            g_cat = np.concatenate([g_th.ravel(), g_cp.ravel()], axis=0)
            nat_cat = np.clip(nat_inv @ g_cat, -1.5, 1.5)
            nat_g_th = nat_cat[:n_th].reshape(model.theta.shape)
            nat_g_cp = nat_cat[n_th:].reshape(model.cond_proj.shape)

        model.theta[:] = np.clip(model.theta + lr_k * nat_g_th, -clip_bound, clip_bound)
        model.cond_proj[:] = np.clip(model.cond_proj + lr_k * nat_g_cp, -clip_bound, clip_bound)

    return {"shots": shots_used, "batches": total_steps}


class ClassicalIsingEP:
    """Noiseless parameter-matched classical mean-field Ising / Hopfield EP control."""

    def __init__(
        self,
        data_shape: int | Tuple[int, ...],
        cond_dim: int = 10,
        n_hidden: int = 0,
        topology: str = "all",
        diffusion_steps: int = 4,
        in_pinning_base: float = 0.28,
        in_pinning_slope: float = 0.52,
        n_mf_steps: int = 10,
        seed: int = 42,
    ):
        self.n_visible = int(np.prod(data_shape)) if isinstance(data_shape, tuple) else int(data_shape)
        self.cond_dim = int(cond_dim)
        self.n_hidden = int(n_hidden)
        self.n_qubits = self.n_visible + self.n_hidden
        self.diffusion_steps = int(diffusion_steps)
        self.alpha_bar = make_alpha_bar(diffusion_steps, schedule="cosine")
        self.in_pinning_base = float(in_pinning_base)
        self.in_pinning_slope = float(in_pinning_slope)
        self.n_mf_steps = int(n_mf_steps)
        self.rng = np.random.default_rng(seed)

        self.edges = build_ising_edges(data_shape, topology=topology, n_hidden=self.n_hidden)
        self.n_edges = len(self.edges)
        self._edge_u = np.array([u for u, _ in self.edges], dtype=np.int64)
        self._edge_v = np.array([v for _, v in self.edges], dtype=np.int64)
        self.theta = np.zeros(self.n_edges + self.n_qubits, dtype=np.float64)
        self.cond_proj = np.zeros((self.n_visible, self.cond_dim), dtype=np.float64) if self.cond_dim > 0 else None

    def parameters_dict(self) -> Dict[str, np.ndarray]:
        d = {"theta": self.theta}
        if self.cond_proj is not None:
            d["cond_proj"] = self.cond_proj
        return d

    def _relax_batch(
        self,
        x_t_batch: np.ndarray,
        t_batch: np.ndarray,
        cond_batch: np.ndarray | None,
        nudge_target_batch: np.ndarray | None = None,
        nudge_beta_batch: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        m = x_t_batch.shape[0]
        j_mat = np.zeros((self.n_qubits, self.n_qubits), dtype=np.float64)
        if self.n_edges > 0:
            j_vals = self.theta[: self.n_edges]
            j_mat[self._edge_u, self._edge_v] = j_vals
            j_mat[self._edge_v, self._edge_u] = j_vals

        t_norms = np.asarray(t_batch, dtype=np.float64) / self.diffusion_steps
        in_pin = (self.in_pinning_base + self.in_pinning_slope * (1.0 - t_norms))[:, None]
        h = np.broadcast_to(self.theta[self.n_edges :], (m, self.n_qubits)).copy()
        h[:, : self.n_visible] += in_pin * x_t_batch.reshape(m, self.n_visible)
        if cond_batch is not None and self.cond_proj is not None:
            h[:, : self.n_visible] += cond_batch @ self.cond_proj.T
        if nudge_target_batch is not None and nudge_beta_batch is not None:
            h[:, : self.n_visible] += nudge_beta_batch[:, None] * nudge_target_batch.reshape(m, self.n_visible)

        z = np.zeros((m, self.n_qubits), dtype=np.float64)
        for _ in range(self.n_mf_steps):
            z = 0.5 * z + 0.5 * np.tanh(1.15 * (h + z @ j_mat))
        if self.n_edges > 0:
            zz = z[:, self._edge_u] * z[:, self._edge_v]
            obs = np.concatenate([zz, z], axis=1)
        else:
            obs = z
        return obs, z[:, : self.n_visible]

    def sample_batch(
        self,
        conds: np.ndarray | None = None,
        n_samples: int | None = None,
        step_noise_scale: float = 0.35,
        init_std: float = 0.90,
    ) -> np.ndarray:
        n = int(conds.shape[0]) if conds is not None else int(n_samples or 1)
        x_t = np.clip(self.rng.normal(0.0, init_std, size=(n, self.n_visible)), -1.0, 1.0)
        c_arr = None if conds is None else np.asarray(conds, dtype=np.float64).reshape(n, self.cond_dim)
        for t in range(self.diffusion_steps, 0, -1):
            t_arr = np.full(n, t, dtype=np.int64)
            _, x0_hat = self._relax_batch(x_t, t_arr, c_arr)
            if t > 1:
                a_next = self.alpha_bar[t - 1]
                x_t = np.clip(
                    math.sqrt(a_next) * x0_hat + math.sqrt(1.0 - a_next) * self.rng.normal(0.0, step_noise_scale, size=x0_hat.shape),
                    -1.0,
                    1.0,
                )
            else:
                x_t = x0_hat
        return x_t


def train_classical_ising_ep(
    data: np.ndarray,
    conds: np.ndarray,
    y_labels: np.ndarray,
    n_hidden: int,
    topology: str,
    n_steps: int,
    batch_size: int = 10,
    diffusion_steps: int = 4,
    seed: int = 42,
) -> ClassicalIsingEP:
    model = ClassicalIsingEP(
        data.shape[1], cond_dim=conds.shape[1], n_hidden=n_hidden, topology=topology,
        diffusion_steps=diffusion_steps, seed=seed,
    )
    opt = QEROptimizer(model.parameters_dict(), QERConfig(lr_max=0.12, lr_min=0.015, clip_bound=0.90))
    class_pools = [np.flatnonzero(y_labels == c) for c in np.unique(y_labels)]
    eff_batch = min(batch_size, data.shape[0])

    for b_idx in range(1, n_steps + 1):
        progress = (b_idx - 1) / max(1, n_steps - 1)
        ids = _sample_stratified_ids(model.rng, class_pools, eff_batch)
        t_cycle = (np.arange(eff_batch) % diffusion_steps) + 1
        model.rng.shuffle(t_cycle)
        tgts = data[ids]
        c_arr = conds[ids]
        a_bars = model.alpha_bar[t_cycle][:, None]
        x_t = np.clip(
            np.sqrt(a_bars) * tgts + np.sqrt(1.0 - a_bars) * model.rng.normal(0.0, 0.70, size=tgts.shape),
            -1.0,
            1.0,
        )
        beta = opt.beta
        obs_2b, z_2b = model._relax_batch(
            np.concatenate([x_t, x_t], axis=0),
            np.concatenate([t_cycle, t_cycle], axis=0),
            np.concatenate([c_arr, c_arr], axis=0),
            np.concatenate([tgts, tgts], axis=0),
            np.concatenate([np.full(eff_batch, beta), np.full(eff_batch, -beta)], axis=0),
        )
        opt.step_contrast(
            model.parameters_dict(),
            obs_2b[:eff_batch], obs_2b[eff_batch:],
            z_plus_list=z_2b[:eff_batch], z_minus_list=z_2b[eff_batch:],
            conds=c_arr, nudge_dirs=tgts, progress=progress,
        )
    return model


class ClassicalMatchedDDPM(nn.Module):
    """Parameter-matched classical MLP DDPM baseline operating on clean float latents."""

    def __init__(
        self,
        latent_dim: int,
        cond_dim: int,
        target_params: int,
        diffusion_steps: int = 4,
        skip_cond: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.diffusion_steps = diffusion_steps
        self.skip_cond = bool(skip_cond)
        self.alpha_bar = make_alpha_bar(diffusion_steps, schedule="cosine")
        in_dim = latent_dim + cond_dim + 1
        skip_params = (cond_dim * latent_dim) if self.skip_cond else 0
        rem = max(4, target_params - skip_params - latent_dim)
        h = max(2, round(rem / max(1, in_dim + 1 + latent_dim)))
        self.cond_skip = nn.Linear(cond_dim, latent_dim, bias=False) if self.skip_cond else None
        self.core = nn.Sequential(nn.Linear(in_dim, h), nn.Tanh(), nn.Linear(h, latent_dim))

    def forward(self, x_t: torch.Tensor, t_norm: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.core(torch.cat([x_t, t_norm, cond], dim=1))
        if self.cond_skip is not None:
            out = out + self.cond_skip(cond)
        return torch.tanh(out)

    @torch.no_grad()
    def sample_batch(
        self,
        cond_np: np.ndarray,
        seed: int = 123,
        step_noise_scale: float = 0.35,
        init_std: float = 0.90,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        n = cond_np.shape[0]
        x_t = np.clip(rng.normal(0.0, init_std, size=(n, self.latent_dim)), -1.0, 1.0).astype(np.float32)
        c_t = torch.from_numpy(cond_np.astype(np.float32))
        for t in range(self.diffusion_steps, 0, -1):
            t_in = torch.full((n, 1), float(t) / self.diffusion_steps, dtype=torch.float32)
            x0_hat = self.forward(torch.from_numpy(x_t), t_in, c_t).numpy()
            if t > 1:
                a_next = self.alpha_bar[t - 1]
                eps = rng.normal(0.0, step_noise_scale, size=(n, self.latent_dim)).astype(np.float32)
                x_t = np.clip(math.sqrt(a_next) * x0_hat + math.sqrt(1.0 - a_next) * eps, -1.0, 1.0)
            else:
                x_t = x0_hat
        return x_t.astype(np.float64)


class ClassicalLargeDDPM(nn.Module):
    """Unconstrained well-tuned classical ResMLP DDPM skyline (~21k parameters)."""

    def __init__(self, latent_dim: int, cond_dim: int, hidden: int = 128, diffusion_steps: int = 4):
        super().__init__()
        self.latent_dim = latent_dim
        self.diffusion_steps = diffusion_steps
        self.alpha_bar = make_alpha_bar(diffusion_steps, schedule="cosine")
        in_dim = latent_dim + cond_dim + 1
        self.in_proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU())
        self.res = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.out_proj = nn.Linear(hidden, latent_dim)
        self.cond_skip = nn.Linear(cond_dim, latent_dim, bias=False)

    def forward(self, x_t: torch.Tensor, t_norm: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(torch.cat([x_t, t_norm, cond], dim=1))
        h = h + self.res(h)
        return torch.tanh(self.out_proj(h) + self.cond_skip(cond))

    @torch.no_grad()
    def sample_batch(
        self,
        cond_np: np.ndarray,
        seed: int = 123,
        step_noise_scale: float = 0.35,
        init_std: float = 0.90,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        n = cond_np.shape[0]
        x_t = np.clip(rng.normal(0.0, init_std, size=(n, self.latent_dim)), -1.0, 1.0).astype(np.float32)
        c_t = torch.from_numpy(cond_np.astype(np.float32))
        for t in range(self.diffusion_steps, 0, -1):
            t_in = torch.full((n, 1), float(t) / self.diffusion_steps, dtype=torch.float32)
            x0_hat = self.forward(torch.from_numpy(x_t), t_in, c_t).numpy()
            if t > 1:
                a_next = self.alpha_bar[t - 1]
                eps = rng.normal(0.0, step_noise_scale, size=(n, self.latent_dim)).astype(np.float32)
                x_t = np.clip(math.sqrt(a_next) * x0_hat + math.sqrt(1.0 - a_next) * eps, -1.0, 1.0)
            else:
                x_t = x0_hat
        return x_t.astype(np.float64)


def train_classical_baseline(
    z_tr: np.ndarray,
    c_tr: np.ndarray,
    y_tr: np.ndarray,
    target_params: int,
    n_steps: int,
    batch_size: int = 10,
    diffusion_steps: int = 4,
    seed: int = 7,
    skip_cond: bool = False,
    large: bool = False,
    use_cosine_lr: bool = True,
) -> nn.Module:
    torch.manual_seed(seed)
    if large:
        model: nn.Module = ClassicalLargeDDPM(z_tr.shape[1], c_tr.shape[1], hidden=128, diffusion_steps=diffusion_steps)
        opt = torch.optim.Adam(model.parameters(), lr=2e-3)
        eff_steps = max(400, 2 * n_steps)
        eff_batch = min(64, z_tr.shape[0])
    else:
        model = ClassicalMatchedDDPM(
            z_tr.shape[1], c_tr.shape[1], target_params, diffusion_steps=diffusion_steps, skip_cond=skip_cond
        )
        opt = torch.optim.Adam(model.parameters(), lr=0.02)
        eff_steps = n_steps
        eff_batch = min(batch_size, z_tr.shape[0])

    rng = np.random.default_rng(seed)
    class_pools = [np.flatnonzero(y_tr == c) for c in np.unique(y_tr)]

    for step_idx in range(eff_steps):
        if use_cosine_lr and not large:
            prog = step_idx / max(1, eff_steps - 1)
            lr_now = 0.002 + 0.5 * (0.02 - 0.002) * (1.0 + math.cos(math.pi * prog))
            for pg in opt.param_groups:
                pg["lr"] = lr_now
        ids = _sample_stratified_ids(rng, class_pools, eff_batch)
        t_arr = (np.arange(len(ids)) % diffusion_steps) + 1
        rng.shuffle(t_arr)
        a_bar = np.array([model.alpha_bar[t] for t in t_arr], dtype=np.float32)[:, None]
        tgt = z_tr[ids].astype(np.float32)
        x_t = np.clip(
            np.sqrt(a_bar) * tgt + np.sqrt(1.0 - a_bar) * rng.normal(0.0, 0.70, size=tgt.shape).astype(np.float32),
            -1.0, 1.0,
        )
        t_norm = (t_arr.astype(np.float32) / diffusion_steps)[:, None]

        opt.zero_grad()
        pred = model(torch.from_numpy(x_t), torch.from_numpy(t_norm), torch.from_numpy(c_tr[ids].astype(np.float32)))
        F.mse_loss(pred, torch.from_numpy(tgt)).backward()
        opt.step()

    return model


class _TorchHEACircuit:
    """Fast GPU/CPU hardware-efficient RY-RZ-CNOT PQC simulator for PQC-QuDDPM & QCBM-MMD."""

    def __init__(
        self,
        n_qubits: int,
        p_depol: float = 0.035,
        p_readout: float = 0.04,
        shots_per_traj: int = 4,
        device: torch.device | None = None,
        seed: int = 42,
    ):
        self.n_qubits = int(n_qubits)
        self.p_depol = float(p_depol)
        self.p_readout = float(p_readout)
        self.shots_per_traj = int(shots_per_traj)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(int(seed))

        n_states = 1 << self.n_qubits
        basis = torch.arange(n_states, dtype=torch.int64, device=self.device)
        shifts = torch.arange(self.n_qubits - 1, -1, -1, dtype=torch.int64, device=self.device)
        self.bit_table = (basis[:, None] >> shifts[None, :]) & 1
        self.z_basis = (1.0 - 2.0 * self.bit_table.to(torch.float32)).t().contiguous()

        cur = basis.clone()
        for q in range(self.n_qubits):
            ctrl = (cur >> (self.n_qubits - 1 - q)) & 1
            cur = cur ^ (ctrl << (self.n_qubits - 1 - ((q + 1) % self.n_qubits)))
        self.ring_inv_perm = torch.argsort(cur)

        self._pauli_mats = torch.tensor(
            [[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]],
            dtype=torch.complex64,
            device=self.device,
        )
        p0 = 1.0 - 0.75 * self.p_depol
        p1 = 0.25 * self.p_depol
        self._depol_cdf = torch.tensor([p0, p0 + p1, p0 + 2.0 * p1], dtype=torch.float32, device=self.device)

    def _apply_ry_layer(self, psi: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        b, n_traj, _ = psi.shape
        ca = torch.cos(0.5 * angles).to(torch.complex64)
        sa = torch.sin(0.5 * angles).to(torch.complex64)
        for q in range(self.n_qubits):
            v = psi.view(b, n_traj * (1 << q), 2, 1 << (self.n_qubits - 1 - q))
            a0, a1 = v[:, :, 0, :], v[:, :, 1, :]
            cq = ca[:, q].view(b, 1, 1)
            sq = sa[:, q].view(b, 1, 1)
            w0 = cq * a0 - sq * a1
            w1 = sq * a0 + cq * a1
            v[:, :, 0, :] = w0
            v[:, :, 1, :] = w1
        return psi

    def sample_spins(
        self,
        init_ry: torch.Tensor,
        ry_layers: torch.Tensor,
        rz_layers: torch.Tensor,
        total_shots: int,
    ) -> torch.Tensor:
        """Return (B, S_eff, n_qubits) measurement outcomes in {-1, +1}."""
        b, n_layers, _ = ry_layers.shape
        n_states = 1 << self.n_qubits
        spt = 1 if total_shots < self.shots_per_traj else self.shots_per_traj
        n_traj = max(1, total_shots // spt)
        psi = torch.zeros((b, n_traj, n_states), dtype=torch.complex64, device=self.device)
        psi[:, :, 0] = 1.0
        psi = self._apply_ry_layer(psi, init_ry)

        k_tot = b * n_traj
        for l in range(n_layers):
            psi = self._apply_ry_layer(psi, ry_layers[:, l, :])
            rz_diag = torch.exp((-0.5j) * (rz_layers[:, l, :] @ self.z_basis).to(torch.complex64)).unsqueeze(1)
            psi = psi * rz_diag
            psi = psi.index_select(-1, self.ring_inv_perm)
            if self.p_depol > 0.0:
                u_rand = torch.rand((b, n_traj, self.n_qubits), device=self.device, generator=self.gen)
                faults = (u_rand.unsqueeze(-1) >= self._depol_cdf).sum(dim=-1)
                for q in range(self.n_qubits):
                    f_q = faults[:, :, q].reshape(k_tot)
                    p_q = self._pauli_mats[f_q]
                    v = psi.view(k_tot, 1 << q, 2, 1 << (self.n_qubits - 1 - q))
                    a0, a1 = v[:, :, 0, :], v[:, :, 1, :]
                    w0 = p_q[:, 0, 0].view(k_tot, 1, 1) * a0 + p_q[:, 0, 1].view(k_tot, 1, 1) * a1
                    w1 = p_q[:, 1, 0].view(k_tot, 1, 1) * a0 + p_q[:, 1, 1].view(k_tot, 1, 1) * a1
                    v[:, :, 0, :] = w0
                    v[:, :, 1, :] = w1

        probs = (psi.real * psi.real + psi.imag * psi.imag).view(k_tot, n_states)
        probs = probs / torch.clamp(probs.sum(dim=1, keepdim=True), min=1e-12)
        idx = torch.multinomial(probs, num_samples=spt, replacement=True, generator=self.gen)
        bits = self.bit_table[idx.reshape(-1)].view(b, n_traj * spt, self.n_qubits)
        if self.p_readout > 0.0:
            bits = bits ^ (torch.rand(bits.shape, device=self.device, generator=self.gen) < self.p_readout).to(bits.dtype)
        return 1.0 - 2.0 * bits.to(torch.float32)


class PQCQuDDPM:
    """Shot- and parameter-matched Hardware-Efficient PQC DDPM baseline trained via SPSA."""

    def __init__(
        self,
        n_qubits: int,
        cond_dim: int,
        n_layers: int = 3,
        diffusion_steps: int = 4,
        p_depol: float = 0.035,
        p_readout: float = 0.04,
        device: torch.device | None = None,
        seed: int = 42,
    ):
        self.n_qubits = int(n_qubits)
        self.cond_dim = int(cond_dim)
        self.n_layers = int(n_layers)
        self.diffusion_steps = int(diffusion_steps)
        self.alpha_bar = make_alpha_bar(diffusion_steps, schedule="cosine")
        self.rng = np.random.default_rng(seed)
        self.hea = _TorchHEACircuit(n_qubits, p_depol=p_depol, p_readout=p_readout, device=device, seed=seed)
        self.n_params = self.n_qubits + self.n_qubits + self.n_qubits * self.cond_dim + 2 * self.n_layers * self.n_qubits
        self.params = self.rng.normal(0.0, 0.25, size=self.n_params).astype(np.float32)

    def _unpack_batch(self, p_batch: torch.Tensor, x_t: torch.Tensor, t_norm: torch.Tensor, cond: torch.Tensor):
        q, cd, l = self.n_qubits, self.cond_dim, self.n_layers
        off = 0
        w_in = p_batch[:, off : off + q]; off += q
        w_tm = p_batch[:, off : off + q]; off += q
        w_c = p_batch[:, off : off + q * cd].view(-1, q, cd); off += q * cd
        ry = p_batch[:, off : off + l * q].view(-1, l, q); off += l * q
        rz = p_batch[:, off : off + l * q].view(-1, l, q)
        init_ry = (0.5 * math.pi) + w_in * x_t + w_tm * t_norm + torch.bmm(w_c, cond.unsqueeze(-1)).squeeze(-1)
        return init_ry, ry, rz

    def sample_batch(
        self, conds: np.ndarray, shots_per_step: int = 160, step_noise_scale: float = 0.35, init_std: float = 0.90,
    ) -> np.ndarray:
        n = conds.shape[0]
        dev = self.hea.device
        p_t = torch.tensor(self.params, dtype=torch.float32, device=dev).unsqueeze(0).expand(n, -1)
        c_t = torch.tensor(conds, dtype=torch.float32, device=dev)
        x_t = np.clip(self.rng.normal(0.0, init_std, size=(n, self.n_qubits)), -1.0, 1.0).astype(np.float32)
        with torch.no_grad():
            for t in range(self.diffusion_steps, 0, -1):
                xt_t = torch.tensor(x_t, dtype=torch.float32, device=dev)
                tn_t = torch.full((n, 1), float(t) / self.diffusion_steps, dtype=torch.float32, device=dev)
                init_ry, ry, rz = self._unpack_batch(p_t, xt_t, tn_t, c_t)
                z_hat = self.hea.sample_spins(init_ry, ry, rz, total_shots=shots_per_step).mean(dim=1).cpu().numpy()
                if t > 1:
                    a_next = self.alpha_bar[t - 1]
                    x_t = np.clip(
                        math.sqrt(a_next) * z_hat + math.sqrt(1.0 - a_next) * self.rng.normal(0.0, step_noise_scale, size=z_hat.shape),
                        -1.0, 1.0,
                    ).astype(np.float32)
                else:
                    x_t = z_hat.astype(np.float32)
        return x_t.astype(np.float64)


def train_pqc_quddpm_baseline(
    data: np.ndarray,
    conds: np.ndarray,
    y_labels: np.ndarray,
    p_depol: float,
    p_readout: float,
    device: torch.device,
    max_shots: int = 80_000,
    shots_per_run: int = 20,
    batch_size: int = 10,
    n_layers: int = 3,
    seed: int = 42,
) -> PQCQuDDPM:
    model = PQCQuDDPM(
        data.shape[1], conds.shape[1], n_layers=n_layers, p_depol=p_depol, p_readout=p_readout, device=device, seed=seed,
    )
    eff_batch = min(batch_size, data.shape[0])
    total_batches = max(1, max_shots // (2 * shots_per_run * eff_batch))
    class_pools = [np.flatnonzero(y_labels == c) for c in np.unique(y_labels)]
    m_v = np.zeros_like(model.params, dtype=np.float64)
    v_v = np.zeros_like(model.params, dtype=np.float64)
    dev = model.hea.device

    with torch.no_grad():
        for step in range(1, total_batches + 1):
            c_k = 0.22 / (step ** 0.101)
            lr_k = 0.015 + 0.5 * (0.08 - 0.015) * (1.0 + math.cos(math.pi * (step - 1) / max(1, total_batches - 1)))
            ids = _sample_stratified_ids(model.rng, class_pools, eff_batch)
            t_batch = (np.arange(eff_batch) % model.diffusion_steps) + 1
            model.rng.shuffle(t_batch)
            tgt = data[ids].astype(np.float32)
            a_bar = model.alpha_bar[t_batch][:, None]
            x_t = np.clip(
                np.sqrt(a_bar) * tgt + np.sqrt(1.0 - a_bar) * model.rng.normal(0.0, 0.70, size=tgt.shape),
                -1.0, 1.0,
            ).astype(np.float32)

            delta = model.rng.choice([-1.0, 1.0], size=(eff_batch, model.n_params)).astype(np.float32)
            p_p = torch.tensor(model.params[None, :] + c_k * delta, device=dev)
            p_m = torch.tensor(model.params[None, :] - c_k * delta, device=dev)
            xt_2 = torch.tensor(np.concatenate([x_t, x_t], axis=0), device=dev)
            tn_2 = torch.tensor((np.concatenate([t_batch, t_batch]) / model.diffusion_steps)[:, None], dtype=torch.float32, device=dev)
            c_2 = torch.tensor(np.concatenate([conds[ids], conds[ids]], axis=0).astype(np.float32), device=dev)
            init_ry, ry, rz = model._unpack_batch(torch.cat([p_p, p_m], dim=0), xt_2, tn_2, c_2)
            z_2b = model.hea.sample_spins(init_ry, ry, rz, total_shots=shots_per_run).mean(dim=1).cpu().numpy()

            lp = np.mean((z_2b[:eff_batch] - tgt) ** 2, axis=1)
            lm = np.mean((z_2b[eff_batch:] - tgt) ** 2, axis=1)
            g = np.mean(((lp - lm) / (2.0 * c_k))[:, None] * delta, axis=0)
            m_v[:] = 0.75 * m_v + 0.25 * g
            v_v[:] = 0.98 * v_v + 0.02 * (g ** 2)
            model.params[:] = np.clip(
                model.params - lr_k * (m_v / (1.0 - 0.75 ** step)) / (np.sqrt(v_v / (1.0 - 0.98 ** step)) + 1e-6),
                -math.pi, math.pi,
            ).astype(np.float32)
    return model


class QCBMMMD:
    """Class-conditional Born Machine trained with RBF-MMD + SPSA under matched NISQ budget."""

    def __init__(
        self,
        n_qubits: int,
        cond_dim: int,
        n_layers: int = 4,
        p_depol: float = 0.035,
        p_readout: float = 0.04,
        device: torch.device | None = None,
        seed: int = 42,
    ):
        self.n_qubits = int(n_qubits)
        self.cond_dim = int(cond_dim)
        self.n_layers = int(n_layers)
        self.rng = np.random.default_rng(seed)
        self.hea = _TorchHEACircuit(n_qubits, p_depol=p_depol, p_readout=p_readout, device=device, seed=seed)
        self.n_params = self.n_qubits * self.cond_dim + 2 * self.n_layers * self.n_qubits
        self.params = self.rng.normal(0.0, 0.25, size=self.n_params).astype(np.float32)

    def _unpack_batch(self, p_batch: torch.Tensor, cond: torch.Tensor):
        q, cd, l = self.n_qubits, self.cond_dim, self.n_layers
        off = 0
        w_c = p_batch[:, off : off + q * cd].view(-1, q, cd); off += q * cd
        ry = p_batch[:, off : off + l * q].view(-1, l, q); off += l * q
        rz = p_batch[:, off : off + l * q].view(-1, l, q)
        init_ry = (0.5 * math.pi) + torch.bmm(w_c, cond.unsqueeze(-1)).squeeze(-1)
        return init_ry, ry, rz

    def sample_batch(self, conds: np.ndarray, shots_per_step: int = 160) -> np.ndarray:
        n = conds.shape[0]
        dev = self.hea.device
        p_t = torch.tensor(self.params, dtype=torch.float32, device=dev).unsqueeze(0).expand(n, -1)
        c_t = torch.tensor(conds.astype(np.float32), device=dev)
        with torch.no_grad():
            init_ry, ry, rz = self._unpack_batch(p_t, c_t)
            spins = self.hea.sample_spins(init_ry, ry, rz, total_shots=max(1, int(shots_per_step)))
        return spins.mean(dim=1).cpu().numpy().astype(np.float64)


def _rbf_mmd_batch(gen_spins: torch.Tensor, real_sets: torch.Tensor) -> torch.Tensor:
    d_gg = torch.cdist(gen_spins, gen_spins).pow(2)
    d_rr = torch.cdist(real_sets, real_sets).pow(2)
    d_gr = torch.cdist(gen_spins, real_sets).pow(2)
    mmd = torch.zeros(gen_spins.shape[0], device=gen_spins.device)
    for sigma in (1.5, 4.0, 12.0, 24.0):
        k_gg = torch.exp(-d_gg / sigma).mean(dim=(1, 2))
        k_rr = torch.exp(-d_rr / sigma).mean(dim=(1, 2))
        k_gr = torch.exp(-d_gr / sigma).mean(dim=(1, 2))
        mmd = mmd + k_gg + k_rr - 2.0 * k_gr
    return mmd


def train_qcbm_mmd_baseline(
    data: np.ndarray,
    conds: np.ndarray,
    y_labels: np.ndarray,
    p_depol: float,
    p_readout: float,
    device: torch.device,
    max_shots: int = 80_000,
    shots_per_run: int = 20,
    batch_size: int = 10,
    n_layers: int = 4,
    seed: int = 42,
) -> QCBMMMD:
    model = QCBMMMD(
        data.shape[1], conds.shape[1], n_layers=n_layers, p_depol=p_depol, p_readout=p_readout, device=device, seed=seed,
    )
    eff_batch = min(batch_size, data.shape[0])
    total_batches = max(1, max_shots // (2 * shots_per_run * eff_batch))
    class_pools = [np.flatnonzero(y_labels == c) for c in np.unique(y_labels)]
    eye_c = np.eye(conds.shape[1], dtype=np.float32)
    m_v = np.zeros_like(model.params, dtype=np.float64)
    v_v = np.zeros_like(model.params, dtype=np.float64)
    dev = model.hea.device

    with torch.no_grad():
        for step in range(1, total_batches + 1):
            c_k = 0.22 / (step ** 0.101)
            lr_k = 0.015 + 0.5 * (0.08 - 0.015) * (1.0 + math.cos(math.pi * (step - 1) / max(1, total_batches - 1)))
            classes = (model.rng.permutation(len(class_pools))[:eff_batch] if eff_batch <= len(class_pools)
                       else model.rng.integers(0, len(class_pools), size=eff_batch))
            real_sub = np.stack([data[model.rng.choice(class_pools[c], size=shots_per_run)] for c in classes], axis=0).astype(np.float32)
            c_mat = eye_c[classes]

            delta = model.rng.choice([-1.0, 1.0], size=(eff_batch, model.n_params)).astype(np.float32)
            p_p = torch.tensor(model.params[None, :] + c_k * delta, device=dev)
            p_m = torch.tensor(model.params[None, :] - c_k * delta, device=dev)
            c_2 = torch.tensor(np.concatenate([c_mat, c_mat], axis=0), device=dev)
            init_ry, ry, rz = model._unpack_batch(torch.cat([p_p, p_m], dim=0), c_2)
            spins_2b = model.hea.sample_spins(init_ry, ry, rz, total_shots=shots_per_run)
            r_2b = torch.tensor(np.concatenate([real_sub, real_sub], axis=0), device=dev)
            mmds = _rbf_mmd_batch(spins_2b, r_2b).cpu().numpy()

            lp, lm = mmds[:eff_batch], mmds[eff_batch:]
            g = np.mean(((lp - lm) / (2.0 * c_k))[:, None] * delta, axis=0)
            m_v[:] = 0.75 * m_v + 0.25 * g
            v_v[:] = 0.98 * v_v + 0.02 * (g ** 2)
            model.params[:] = np.clip(
                model.params - lr_k * (m_v / (1.0 - 0.75 ** step)) / (np.sqrt(v_v / (1.0 - 0.98 ** step)) + 1e-6),
                -math.pi, math.pi,
            ).astype(np.float32)
    return model


# ==============================================================================
# 4. METRICS (LATENT FID, IMAGE FID, CLASSIFIER ACC, NN-MSE, DIVERSITY) & PLOTS
# ==============================================================================
def frechet_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-4) -> float:
    mu_a, mu_b = np.mean(a, axis=0), np.mean(b, axis=0)
    d = a.shape[1]
    cov_a = np.atleast_2d(np.cov(a, rowvar=False)) + np.eye(d) * eps
    cov_b = np.atleast_2d(np.cov(b, rowvar=False)) + np.eye(d) * eps
    covmean = sqrtm(cov_a @ cov_b)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu_a - mu_b
    return float(np.dot(diff, diff) + np.trace(cov_a + cov_b - 2.0 * covmean))


def compute_knn_manifold_metrics(
    real_feats: np.ndarray, gen_feats: np.ndarray, k: int = 5, max_n: int = 2000,
) -> Dict[str, float]:
    """Compute k-NN Precision, Recall, Density, and Coverage in classifier feature space."""
    if len(real_feats) < k + 2 or len(gen_feats) < k + 2:
        return {"precision": 0.0, "recall": 0.0, "density": 0.0, "coverage": 0.0}
    r = torch.from_numpy(real_feats[:max_n].astype(np.float32))
    g = torch.from_numpy(gen_feats[:max_n].astype(np.float32))
    d_rr = torch.cdist(r, r)
    d_rr.fill_diagonal_(1e9)
    rad_r = torch.kthvalue(d_rr, k, dim=1).values
    d_gg = torch.cdist(g, g)
    d_gg.fill_diagonal_(1e9)
    rad_g = torch.kthvalue(d_gg, k, dim=1).values
    d_rg = torch.cdist(r, g)
    inside_r = d_rg <= rad_r[:, None]
    precision = float(inside_r.any(dim=0).float().mean().item())
    recall = float((d_rg <= rad_g[None, :]).any(dim=1).float().mean().item())
    density = float(inside_r.sum(dim=0).float().mean().item() / k)
    coverage = float((d_rg.min(dim=1).values <= rad_r).float().mean().item())
    return {"precision": precision, "recall": recall, "density": density, "coverage": coverage}


def _stratified_subset_idx(y: np.ndarray, per_class: int) -> np.ndarray:
    idxs = []
    for c in np.unique(y):
        pool = np.flatnonzero(y == c)
        idxs.append(pool[: min(per_class, len(pool))])
    return np.concatenate(idxs) if idxs else np.empty(0, dtype=np.int64)


def evaluate_real_baseline(
    ae: ConvLatentAE,
    clf: EvalCNN,
    z_te: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
    eval_per_class: int,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    """Compute Real-vs-Real held-out and AE-reconstruction baselines on matching sample sizes."""
    ae.eval()
    clf.eval()
    idx_eval: List[np.ndarray] = []
    idx_ref: List[np.ndarray] = []
    for c in range(10):
        pool = np.flatnonzero(y_te == c)
        take = min(eval_per_class, max(2, len(pool) // 2))
        idx_eval.append(pool[:take])
        idx_ref.append(pool[take : min(len(pool), take + max(take, 200))])
    i_ev = np.concatenate(idx_eval)
    i_rf = np.concatenate(idx_ref)

    with torch.no_grad():
        x_ev = x_te[i_ev]
        z_ev = z_te[i_ev]
        y_ev = y_te[i_ev]
        z_rf = z_te[i_rf]
        x_rf = x_te[i_rf]

        logits_real, feat_real = clf.forward_features(torch.from_numpy(x_ev).to(device))
        _, feat_ref = clf.forward_features(torch.from_numpy(x_rf).to(device))
        recon_ev = ae.decode(torch.from_numpy(z_ev.astype(np.float32)).to(device)).cpu().numpy()
        logits_rec, feat_rec = clf.forward_features(torch.from_numpy(recon_ev).to(device))

    def _score(imgs: np.ndarray, lats: np.ndarray, feats: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
        feat_ref_np = feat_ref.cpu().numpy()
        knn = compute_knn_manifold_metrics(feat_ref_np, feats)
        cond_fids, cond_feat_fids = [], []
        lat_divs, pix_divs = [], []
        y_rf = y_te[i_rf]
        for c in range(10):
            gc = lats[y_ev == c]
            rc = z_rf[y_rf == c]
            fgc = feats[y_ev == c]
            frc = feat_ref_np[y_rf == c]
            ic = imgs[y_ev == c]
            if len(gc) >= 2 and len(rc) >= 2:
                cond_fids.append(frechet_distance(gc, rc, eps=1e-3))
            if len(fgc) >= 2 and len(frc) >= 2:
                cond_feat_fids.append(frechet_distance(fgc, frc, eps=1e-2))
            if len(gc) > 1:
                lat_divs.append(float(np.mean(np.std(gc, axis=0))))
            if len(ic) > 1:
                pix_divs.append(float(np.mean(np.std(ic, axis=0))))
        return {
            "clf_acc": float(np.mean(preds == y_ev)),
            "cond_latent_fid": float(np.mean(cond_fids)) if cond_fids else 0.0,
            "latent_fid": frechet_distance(lats, z_rf),
            "cond_feature_fid": float(np.mean(cond_feat_fids)) if cond_feat_fids else 0.0,
            "feature_fid": frechet_distance(feats, feat_ref_np),
            "nn_latent_mse": 0.0,
            "intra_class_latent_std": float(np.mean(lat_divs)) if lat_divs else 0.0,
            "intra_class_std": float(np.mean(pix_divs)) if pix_divs else 0.0,
            **knn,
        }

    return {
        "real_held_out": _score(x_ev, z_ev, feat_real.cpu().numpy(), logits_real.argmax(dim=1).cpu().numpy()),
        "real_ae_recon": _score(recon_ev, z_ev, feat_rec.cpu().numpy(), logits_rec.argmax(dim=1).cpu().numpy()),
    }


def evaluate_generated(
    name: str,
    gen_latents: np.ndarray,
    gen_labels: np.ndarray,
    ae: ConvLatentAE,
    clf: EvalCNN,
    z_te: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
    device: torch.device,
) -> Tuple[Dict[str, float], np.ndarray]:
    ae.eval()
    clf.eval()
    per_cls_ref = max(100, min(200, len(x_te) // 10))
    ref_idx = _stratified_subset_idx(y_te, per_cls_ref)
    with torch.no_grad():
        gen_imgs = ae.decode(torch.from_numpy(gen_latents.astype(np.float32)).to(device)).cpu().numpy()
        logits_g, feat_g = clf.forward_features(torch.from_numpy(gen_imgs).to(device))
        pred_g = logits_g.argmax(dim=1).cpu().numpy()
        feat_g_np = feat_g.cpu().numpy()

        te_sub = x_te[ref_idx]
        _, feat_r = clf.forward_features(torch.from_numpy(te_sub).to(device))
        feat_r_np = feat_r.cpu().numpy()

    clf_acc = float(np.mean(pred_g == gen_labels))
    lat_fid = frechet_distance(gen_latents, z_te[ref_idx])
    img_fid = frechet_distance(feat_g_np, feat_r_np)
    knn = compute_knn_manifold_metrics(feat_r_np, feat_g_np)

    cond_lat_fids = []
    cond_feat_fids = []
    nn_mses = []
    lat_diversities = []
    diversities = []
    y_ref_sub = y_te[ref_idx]
    for c in range(10):
        g_c = gen_latents[gen_labels == c]
        r_c = z_te[y_te == c]
        fg_c = feat_g_np[gen_labels == c]
        fr_c = feat_r_np[y_ref_sub == c]
        img_c = gen_imgs[gen_labels == c]
        if len(g_c) >= 2 and len(r_c) >= 2:
            cond_lat_fids.append(frechet_distance(g_c, r_c, eps=1e-3))
        if len(fg_c) >= 2 and len(fr_c) >= 2:
            cond_feat_fids.append(frechet_distance(fg_c, fr_c, eps=1e-2))
        if len(g_c) > 0 and len(r_c) > 0:
            dists = np.mean((g_c[:, None, :] - r_c[None, :, :]) ** 2, axis=2)
            nn_mses.append(float(np.mean(np.min(dists, axis=1))))
        if len(g_c) > 1:
            lat_diversities.append(float(np.mean(np.std(g_c, axis=0))))
        if len(img_c) > 1:
            diversities.append(float(np.mean(np.std(img_c, axis=0))))

    metrics = {
        "clf_acc": clf_acc,
        "cond_latent_fid": float(np.mean(cond_lat_fids)) if cond_lat_fids else 0.0,
        "latent_fid": lat_fid,
        "cond_feature_fid": float(np.mean(cond_feat_fids)) if cond_feat_fids else 0.0,
        "feature_fid": img_fid,
        "nn_latent_mse": float(np.mean(nn_mses)) if nn_mses else 0.0,
        "intra_class_latent_std": float(np.mean(lat_diversities)) if lat_diversities else 0.0,
        "intra_class_std": float(np.mean(diversities)) if diversities else 0.0,
        **knn,
    }
    print(
        f"  [{name:16s}] ClfAcc={clf_acc*100:5.1f}% | CondLatFID={metrics['cond_latent_fid']:6.3f} | "
        f"LatFID={lat_fid:6.3f} | CondFeatFID={metrics['cond_feature_fid']:6.2f} | FeatFID={img_fid:6.2f} | "
        f"Prec={metrics['precision']:.3f} Rec={metrics['recall']:.3f} | LatStd={metrics['intra_class_latent_std']:.3f}",
        flush=True,
    )
    return metrics, gen_imgs


# ==============================================================================
# 4B. GRADIENT ALIGNMENT, BARREN PLATEAU SCALING & PURE-QPU PARITY DIAGNOSTICS
# ==============================================================================
class _DiffIsingCircuit:
    """Differentiable PyTorch Ising circuit evaluator for exact autograd & QER alignment analysis."""

    def __init__(
        self,
        n_visible: int,
        cond_dim: int,
        edges: List[Tuple[int, int]],
        n_hidden: int = 0,
        n_layers: int = 2,
        diffusion_steps: int = 4,
        dt: float = 0.42,
        temp_beta: float = 1.25,
        shake_scale: float = 2.2,
        shake_final_frac: float = 0.70,
        in_pinning_base: float = 0.28,
        in_pinning_slope: float = 0.52,
        device: torch.device | None = None,
    ):
        self.n_visible = int(n_visible)
        self.cond_dim = int(cond_dim)
        self.n_hidden = int(n_hidden)
        self.n_qubits = self.n_visible + self.n_hidden
        self.edges = list(edges)
        self.n_edges = len(self.edges)
        self.n_layers = int(n_layers)
        self.diffusion_steps = int(diffusion_steps)
        self.dt = float(dt)
        self.temp_beta = float(temp_beta)
        self.shake_scale = float(shake_scale)
        self.shake_final_frac = float(shake_final_frac)
        self.in_pinning_base = float(in_pinning_base)
        self.in_pinning_slope = float(in_pinning_slope)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        n_states = 1 << self.n_qubits
        basis = torch.arange(n_states, dtype=torch.int64, device=self.device)
        shifts = torch.arange(self.n_qubits - 1, -1, -1, dtype=torch.int64, device=self.device)
        bits = (basis[:, None] >> shifts[None, :]) & 1
        self.z_basis = (1.0 - 2.0 * bits.to(torch.float32)).t().contiguous()
        if self.n_edges > 0:
            self.eu = torch.tensor([u for u, _ in self.edges], dtype=torch.int64, device=self.device)
            self.ev = torch.tensor([v for _, v in self.edges], dtype=torch.int64, device=self.device)
            self.zz_basis = (self.z_basis[self.eu] * self.z_basis[self.ev]).contiguous()
        else:
            self.zz_basis = None
        self.pauli_mats = torch.tensor(
            [[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]],
            dtype=torch.complex64,
            device=self.device,
        )

    def _evolve_psi(
        self,
        x_t: torch.Tensor,
        t_arr: torch.Tensor,
        cond: torch.Tensor | None,
        theta: torch.Tensor,
        cond_proj: torch.Tensor | None,
        nudge_target: torch.Tensor | None = None,
        nudge_beta: float = 0.0,
        fault_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evolve uniform |+...> state through L Trotterized Ising+mixer layers -> psi [k_traj, m, 2^n]."""
        m = x_t.shape[0]
        n_states = 1 << self.n_qubits
        t_norms = t_arr.to(torch.float32) / self.diffusion_steps
        in_pin = (self.in_pinning_base + self.in_pinning_slope * (1.0 - t_norms))[:, None]

        h_vis = theta[self.n_edges : self.n_edges + self.n_visible].unsqueeze(0) + in_pin * x_t
        if cond is not None and cond_proj is not None:
            h_vis = h_vis + cond @ cond_proj.t()
        if nudge_target is not None and nudge_beta != 0.0:
            h_vis = h_vis + float(nudge_beta) * nudge_target
        if self.n_hidden > 0:
            h_hid = theta[self.n_edges + self.n_visible :].unsqueeze(0).expand(m, -1)
            h_tot = torch.cat([h_vis, h_hid], dim=1)
        else:
            h_tot = h_vis

        diag = h_tot @ self.z_basis
        if self.zz_basis is not None:
            diag = diag + theta[: self.n_edges].unsqueeze(0) @ self.zz_basis

        k_traj = 1 if fault_masks is None else fault_masks.shape[0]
        psi = torch.full(
            (k_traj, m, n_states), 1.0 / math.sqrt(n_states), dtype=torch.complex64, device=self.device
        )
        m_idx = torch.arange(m, device=self.device).unsqueeze(0).expand(k_traj, m)

        for layer in range(self.n_layers):
            s = (layer + 1) / self.n_layers
            mold_gain = 2.0 * self.dt * self.temp_beta * s
            phase = torch.exp((-0.5j * mold_gain) * diag.to(torch.complex64)).unsqueeze(0)
            psi = psi * phase

            shake_angle = self.shake_scale * 2.0 * self.dt * (1.0 - self.shake_final_frac * s) * (0.6 + 0.4 * t_norms)
            ca = torch.cos(0.5 * shake_angle).to(torch.complex64)
            sa = (-1j * torch.sin(0.5 * shake_angle)).to(torch.complex64)

            if fault_masks is None:
                for q in range(self.n_qubits):
                    v = psi.reshape(m, 1 << q, 2, 1 << (self.n_qubits - 1 - q))
                    a0, a1 = v[:, :, 0, :], v[:, :, 1, :]
                    cq = ca.view(m, 1, 1)
                    sq = sa.view(m, 1, 1)
                    psi_new = torch.stack([cq * a0 + sq * a1, sq * a0 + cq * a1], dim=2)
                    psi = psi_new.reshape(1, m, n_states)
            else:
                rx_mat = torch.stack([torch.stack([ca, sa], dim=-1), torch.stack([sa, ca], dim=-1)], dim=-2)
                u_all = torch.einsum("fij,mjk->mfik", self.pauli_mats, rx_mat)
                for q in range(self.n_qubits):
                    u_q = u_all[m_idx, fault_masks[:, layer, :, q]].reshape(k_traj * m, 2, 2)
                    v = psi.reshape(k_traj * m, 1 << q, 2, 1 << (self.n_qubits - 1 - q))
                    a0, a1 = v[:, :, 0, :], v[:, :, 1, :]
                    w0 = u_q[:, 0, 0].view(k_traj * m, 1, 1) * a0 + u_q[:, 0, 1].view(k_traj * m, 1, 1) * a1
                    w1 = u_q[:, 1, 0].view(k_traj * m, 1, 1) * a0 + u_q[:, 1, 1].view(k_traj * m, 1, 1) * a1
                    psi = torch.stack([w0, w1], dim=2).reshape(k_traj, m, n_states)
        return psi

    def statevector(
        self,
        x_t: torch.Tensor,
        t_arr: torch.Tensor,
        cond: torch.Tensor | None,
        theta: torch.Tensor,
        cond_proj: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return noiseless statevector [m, 2^n] for overlap / QN-SPSA fidelity checks."""
        return self._evolve_psi(x_t, t_arr, cond, theta, cond_proj, fault_masks=None)[0]

    def expectation(
        self,
        x_t: torch.Tensor,
        t_arr: torch.Tensor,
        cond: torch.Tensor | None,
        theta: torch.Tensor,
        cond_proj: torch.Tensor | None,
        nudge_target: torch.Tensor | None = None,
        nudge_beta: float = 0.0,
        fault_masks: torch.Tensor | None = None,
        p_readout: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (all_obs_exp [m, n_edges+n_qubits], z_vis_exp [m, n_vis])."""
        psi = self._evolve_psi(
            x_t,
            t_arr,
            cond,
            theta,
            cond_proj,
            nudge_target=nudge_target,
            nudge_beta=nudge_beta,
            fault_masks=fault_masks,
        )
        probs = (psi.real * psi.real + psi.imag * psi.imag).mean(dim=0)  # (m, n_states)
        z_all = (probs @ self.z_basis.t()) * (1.0 - 2.0 * p_readout)
        if self.zz_basis is not None:
            zz = (probs @ self.zz_basis.t()) * ((1.0 - 2.0 * p_readout) ** 2)
            obs = torch.cat([zz, z_all], dim=1)
        else:
            obs = z_all
        return obs, z_all[:, : self.n_visible]


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _eval_alignment_point(
    circ: _DiffIsingCircuit,
    shot_model: QuantumIsingDenoiser,
    theta_np: np.ndarray,
    cp_np: np.ndarray,
    x_t_np: np.ndarray,
    t_np: np.ndarray,
    c_np: np.ndarray,
    x0_np: np.ndarray,
    beta: float = 0.45,
    p_depol: float = 0.0,
    p_readout: float = 0.0,
    n_mc_faults: int = 32,
    shots_per_run: int = 20,
    seed: int = 0,
) -> Dict[str, float]:
    dev = circ.device
    th_t = torch.tensor(theta_np, dtype=torch.float32, device=dev, requires_grad=True)
    cp_t = torch.tensor(cp_np, dtype=torch.float32, device=dev, requires_grad=True)
    xt_t = torch.tensor(x_t_np, dtype=torch.float32, device=dev)
    t_t = torch.tensor(t_np, dtype=torch.int64, device=dev)
    c_t = torch.tensor(c_np, dtype=torch.float32, device=dev)
    x0_t = torch.tensor(x0_np, dtype=torch.float32, device=dev)

    faults = None
    if p_depol > 0.0:
        gen = torch.Generator(device=dev)
        gen.manual_seed(int(seed))
        p0, p1 = 1.0 - 0.75 * p_depol, 0.25 * p_depol
        cdf = torch.tensor([p0, p0 + p1, p0 + 2.0 * p1], dtype=torch.float32, device=dev)
        u = torch.rand((n_mc_faults, circ.n_layers, len(x_t_np), circ.n_qubits), device=dev, generator=gen)
        faults = (u.unsqueeze(-1) >= cdf).sum(dim=-1)

    _, z_free = circ.expectation(xt_t, t_t, c_t, th_t, cp_t, fault_masks=faults, p_readout=p_readout)
    loss_mse = F.mse_loss(z_free, x0_t)
    g_th_mse, g_cp_mse = torch.autograd.grad(loss_mse, [th_t, cp_t], retain_graph=True)
    g_true_mse = np.concatenate([-g_th_mse.detach().cpu().numpy(), -g_cp_mse.detach().cpu().numpy().ravel()])

    loss_corr = -torch.mean(z_free * x0_t)
    g_th_corr, g_cp_corr = torch.autograd.grad(loss_corr, [th_t, cp_t])
    g_true_corr = np.concatenate([-g_th_corr.detach().cpu().numpy(), -g_cp_corr.detach().cpu().numpy().ravel()])

    with torch.no_grad():
        obs_p, z_p = circ.expectation(xt_t, t_t, c_t, th_t, cp_t, nudge_target=x0_t, nudge_beta=beta, fault_masks=faults, p_readout=p_readout)
        obs_m, z_m = circ.expectation(xt_t, t_t, c_t, th_t, cp_t, nudge_target=x0_t, nudge_beta=-beta, fault_masks=faults, p_readout=p_readout)
        dq_th = ((obs_p - obs_m) / (2.0 * beta)).mean(dim=0).cpu().numpy()
        dq_cp = torch.einsum("bi,bj->ij", (z_p - z_m) / (2.0 * beta), c_t) / len(x_t_np)
        g_qer_exact = np.concatenate([dq_th, dq_cp.cpu().numpy().ravel()])

    shot_model.theta[:] = theta_np
    shot_model.cond_proj[:] = cp_np
    m = len(x_t_np)
    obs_2b, z_2b = shot_model.measure_observables_batch(
        np.concatenate([x_t_np, x_t_np], axis=0),
        np.concatenate([t_np, t_np], axis=0),
        shots=shots_per_run,
        cond_batch=np.concatenate([c_np, c_np], axis=0),
        nudge_target_batch=np.concatenate([x0_np, x0_np], axis=0),
        nudge_beta_batch=np.concatenate([np.full(m, beta), np.full(m, -beta)]),
    )
    gs_th = np.mean((obs_2b[:m] - obs_2b[m:]) / (2.0 * beta), axis=0)
    gs_cp = np.einsum("bi,bj->ij", (z_2b[:m] - z_2b[m:]) / (2.0 * beta), c_np) / m
    g_qer_shot = np.concatenate([gs_th, gs_cp.ravel()])

    n_e = circ.n_edges
    n_q = circ.n_qubits
    n_th = n_e + n_q
    return {
        "cos_mse_all": _cos_sim(g_qer_exact, g_true_mse),
        "cos_mse_zz": _cos_sim(g_qer_exact[:n_e], g_true_mse[:n_e]),
        "cos_mse_bias": _cos_sim(g_qer_exact[n_e:n_th], g_true_mse[n_e:n_th]),
        "cos_mse_cond": _cos_sim(g_qer_exact[n_th:], g_true_mse[n_th:]),
        "cos_corr_all": _cos_sim(g_qer_exact, g_true_corr),
        "cos_shot_mse_all": _cos_sim(g_qer_shot, g_true_mse),
    }


def compute_circuit_grad_alignment(
    z_tr: np.ndarray,
    c_tr: np.ndarray,
    y_tr: np.ndarray,
    n_hidden: int = 0,
    topology: str = "all",
    device: torch.device | None = None,
    seed: int = 0,
    quick: bool = False,
) -> Dict[str, List[Dict[str, float]]]:
    """Measure QER-to-exact-gradient cosine alignment across steps, beta, depth L, and noise p_depol."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_vis = z_tr.shape[1]
    cdim = c_tr.shape[1]
    edges = build_ising_edges(n_vis, topology=topology, n_hidden=n_hidden)
    rng = np.random.default_rng(seed)
    class_pools = [np.flatnonzero(y_tr == c) for c in np.unique(y_tr)]
    alpha_bar = make_alpha_bar(4, schedule="cosine")

    # Probe minibatch of size 16 for clean variance-reduced reference evaluation
    ids_ref = _sample_stratified_ids(rng, class_pools, 16)
    x0_ref = z_tr[ids_ref]
    c_ref = c_tr[ids_ref]
    t_ref = (np.arange(16) % 4) + 1
    ab_ref = alpha_bar[t_ref][:, None]
    xt_ref = np.clip(np.sqrt(ab_ref) * x0_ref + np.sqrt(1.0 - ab_ref) * rng.normal(0.0, 0.70, size=x0_ref.shape), -1.0, 1.0)

    # 1. Trajectory across training steps (200 steps matching 80k shots)
    model = QuantumIsingDenoiser(
        n_vis, cond_dim=cdim, n_hidden=n_hidden, topology=topology, p_depol=0.035, p_readout=0.04,
        init_gain=1.0, gain_ema=0.0, backend="torch" if dev.type == "cuda" else "numpy",
        torch_device=str(dev), seed=seed,
    )
    opt = QEROptimizer(model.parameters_dict(), QERConfig(lr_max=0.12, lr_min=0.015, clip_bound=0.90))
    circ2 = _DiffIsingCircuit(n_vis, cdim, edges, n_hidden=n_hidden, n_layers=2, device=dev)
    shot_clean = QuantumIsingDenoiser(
        n_vis, cond_dim=cdim, n_hidden=n_hidden, topology=topology, p_depol=0.035, p_readout=0.04,
        init_gain=1.0, gain_ema=0.0, backend="torch" if dev.type == "cuda" else "numpy",
        torch_device=str(dev), seed=seed + 11,
    )

    n_steps = 80 if quick else 200
    eval_steps = {1, 20, 40, 80} if quick else {1, 10, 25, 50, 75, 100, 125, 150, 175, 200}
    step_curve: List[Dict[str, float]] = []
    mid_theta, mid_cp = None, None
    mid_step = 40 if quick else 50

    for st in range(1, n_steps + 1):
        if st in eval_steps or st == mid_step:
            row = _eval_alignment_point(
                circ2, shot_clean, model.theta.copy(), model.cond_proj.copy(),
                xt_ref, t_ref, c_ref, x0_ref, beta=opt.beta, p_depol=0.0, p_readout=0.0, seed=st,
            )
            if st in eval_steps:
                step_curve.append({"step": st, **row})
            if st == mid_step:
                mid_theta, mid_cp = model.theta.copy(), model.cond_proj.copy()

        ids = _sample_stratified_ids(model.rng, class_pools, 10)
        tc = (np.arange(10) % 4) + 1
        model.rng.shuffle(tc)
        tgts = z_tr[ids]
        cs = c_tr[ids]
        ab = alpha_bar[tc][:, None]
        xt = np.clip(np.sqrt(ab) * tgts + np.sqrt(1.0 - ab) * model.rng.normal(0.0, 0.70, size=tgts.shape), -1.0, 1.0)
        b_p, b_m = opt.beta, -opt.beta
        obs_2b, z_2b = model.measure_observables_batch(
            np.concatenate([xt, xt], axis=0), np.concatenate([tc, tc], axis=0), shots=20,
            cond_batch=np.concatenate([cs, cs], axis=0), nudge_target_batch=np.concatenate([tgts, tgts], axis=0),
            nudge_beta_batch=np.concatenate([np.full(10, b_p), np.full(10, b_m)]),
        )
        opt.step_contrast(
            model.parameters_dict(), obs_2b[:10], obs_2b[10:],
            z_plus_list=z_2b[:10], z_minus_list=z_2b[10:], conds=cs, nudge_dirs=tgts,
            progress=(st - 1) / max(1, n_steps - 1), shots_in_batch=400,
        )

    assert mid_theta is not None and mid_cp is not None
    # 2. Beta sweep
    beta_sweep = []
    for beta_v in [0.10, 0.25, 0.45, 0.75, 1.20]:
        r = _eval_alignment_point(circ2, shot_clean, mid_theta, mid_cp, xt_ref, t_ref, c_ref, x0_ref, beta=beta_v, seed=1)
        beta_sweep.append({"beta": beta_v, **r})

    # 3. Depth L sweep
    depth_sweep = []
    for nl in [1, 2, 4, 6]:
        circ_l = _DiffIsingCircuit(n_vis, cdim, edges, n_hidden=n_hidden, n_layers=nl, device=dev)
        shot_l = QuantumIsingDenoiser(
            n_vis, cond_dim=cdim, n_hidden=n_hidden, topology=topology, n_layers=nl,
            p_depol=0.035, p_readout=0.04, init_gain=1.0, gain_ema=0.0,
            backend="torch" if dev.type == "cuda" else "numpy", torch_device=str(dev), seed=nl + 5,
        )
        r = _eval_alignment_point(circ_l, shot_l, mid_theta, mid_cp, xt_ref, t_ref, c_ref, x0_ref, beta=0.45, seed=nl)
        depth_sweep.append({"n_layers": nl, **r})

    # 4. Depolarizing noise p_depol sweep (expected noisy gradient via K=32 MC channels + noisy shot QER)
    noise_sweep = []
    for p_d in [0.0, 0.01, 0.035, 0.06, 0.08]:
        shot_n = QuantumIsingDenoiser(
            n_vis, cond_dim=cdim, n_hidden=n_hidden, topology=topology,
            p_depol=p_d, p_readout=0.04 if p_d > 0.0 else 0.0, init_gain=1.0, gain_ema=0.0,
            backend="torch" if dev.type == "cuda" else "numpy", torch_device=str(dev), seed=99,
        )
        r = _eval_alignment_point(
            circ2, shot_n, mid_theta, mid_cp, xt_ref, t_ref, c_ref, x0_ref,
            beta=0.45, p_depol=p_d, p_readout=0.04 if p_d > 0.0 else 0.0,
            n_mc_faults=16 if quick else 40, seed=int(round(p_d * 1000)) + 1,
        )
        noise_sweep.append({"p_depol": p_d, **r})

    return {
        "step_curve": step_curve,
        "beta_sweep": beta_sweep,
        "depth_sweep": depth_sweep,
        "noise_sweep": noise_sweep,
    }


def measure_gradient_variance_scaling(
    widths: List[int] | None = None,
    depths: List[int] | None = None,
    n_trials: int = 12,
    device: torch.device | None = None,
    seed: int = 0,
) -> List[Dict[str, float]]:
    """Measure QER vs autograd gradient variance across qubit widths n and Trotter depths L."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    widths = widths or [6, 8, 10, 12, 14]
    depths = depths or [2, 6]
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, float]] = []

    for nq in widths:
        edges = build_ising_edges(nq, topology="all")
        x0 = rng.choice([-1.0, 1.0], size=(4, nq)).astype(np.float32)
        xt = np.clip(0.5 * x0 + 0.5 * rng.normal(0.0, 0.70, size=x0.shape), -1.0, 1.0).astype(np.float32)
        t_t = torch.tensor([2, 2, 3, 3], dtype=torch.int64, device=dev)
        xt_t = torch.tensor(xt, dtype=torch.float32, device=dev)
        x0_t = torch.tensor(x0, dtype=torch.float32, device=dev)

        for nl in depths:
            circ = _DiffIsingCircuit(nq, cond_dim=0, edges=edges, n_layers=nl, device=dev)
            g_qer_list = []
            g_mse_list = []
            for _ in range(n_trials):
                th_np = rng.uniform(-0.5, 0.5, size=len(edges) + nq).astype(np.float32)
                th_t = torch.tensor(th_np, dtype=torch.float32, device=dev, requires_grad=True)
                _, z_f = circ.expectation(xt_t, t_t, None, th_t, None)
                loss = 0.5 * torch.sum((z_f - x0_t) ** 2, dim=1).mean()
                (g_th,) = torch.autograd.grad(loss, [th_t])
                g_mse_list.append(-g_th.detach().cpu().numpy())
                with torch.no_grad():
                    obs_p, _ = circ.expectation(xt_t, t_t, None, th_t, None, nudge_target=x0_t, nudge_beta=0.45)
                    obs_m, _ = circ.expectation(xt_t, t_t, None, th_t, None, nudge_target=x0_t, nudge_beta=-0.45)
                    g_qer_list.append(((obs_p - obs_m) / 0.90).mean(dim=0).cpu().numpy())

            gq = np.stack(g_qer_list, axis=0)
            gm = np.stack(g_mse_list, axis=0)
            ne = len(edges)
            rows.append({
                "n_qubits": int(nq),
                "n_layers": int(nl),
                "n_edges": int(ne),
                "var_qer_zz": float(np.mean(np.var(gq[:, :ne], axis=0))),
                "var_qer_bias": float(np.mean(np.var(gq[:, ne:], axis=0))),
                "var_mse_zz": float(np.mean(np.var(gm[:, :ne], axis=0))),
                "var_mse_bias": float(np.mean(np.var(gm[:, ne:], axis=0))),
                "abs_qer_zz": float(np.mean(np.abs(gq[:, :ne]))),
                "abs_mse_zz": float(np.mean(np.abs(gm[:, :ne]))),
            })
    return rows


def run_pure_qpu_parity_benchmark(
    seeds: List[int],
    p_depol: float = 0.035,
    p_readout: float = 0.04,
    p_depol_2q: float = 0.0,
    max_shots: int = 40_000,
    n_eval: int = 320,
    backend: str = "auto",
    torch_device: str = "auto",
) -> Dict:
    """Pure-QPU (zero AE, cond_dim=0) 12-qubit stride-3 correlated cluster spin-glass benchmark."""
    # Build all 8 ground states: 3 clusters of 4 spins {q mod 3 == k} -> zero marginal bias E[x_i]=0
    modes = []
    for s0 in (-1.0, 1.0):
        for s1 in (-1.0, 1.0):
            for s2 in (-1.0, 1.0):
                sig = [s0, s1, s2]
                modes.append([sig[q % 3] for q in range(12)])
    modes_arr = np.array(modes, dtype=np.float64)  # (8, 12)
    valid_set = {tuple(row.astype(np.int64).tolist()): idx for idx, row in enumerate(modes_arr)}
    intra_pairs = [(u, v) for u in range(12) for v in range(u + 1, 12) if (u % 3) == (v % 3)]
    c_ref = (modes_arr.T @ modes_arr) / len(modes_arr)
    off = ~np.eye(12, dtype=bool)

    variants = ["qerd_all", "qerd_grid8", "qerd_grid4", "qerd_chain", "qerd_nozz", "classical_ising_ep"]
    topo_map = {"qerd_all": "all", "qerd_grid8": "grid8", "qerd_grid4": "grid4", "qerd_chain": "chain", "qerd_nozz": "none"}
    per_seed = []
    buckets: Dict[str, List[Dict[str, float]]] = {k: [] for k in variants}

    for seed in seeds:
        rng = np.random.default_rng(seed)
        idx_tr = rng.integers(0, len(modes_arr), size=800)
        x_tr = modes_arr[idx_tr]
        seed_row: Dict = {"seed": seed}

        for vk in variants:
            if vk == "classical_ising_ep":
                mdl = ClassicalIsingEP((3, 4), cond_dim=0, topology="all", seed=seed)
                opt = QEROptimizer(mdl.parameters_dict(), QERConfig(lr_max=0.12, lr_min=0.015, clip_bound=0.90))
                n_steps = max(1, max_shots // (2 * 20 * 10))
                for b_idx in range(1, n_steps + 1):
                    ids = mdl.rng.choice(len(x_tr), size=10, replace=False)
                    tc = (np.arange(10) % 4) + 1
                    mdl.rng.shuffle(tc)
                    tgts = x_tr[ids]
                    ab = mdl.alpha_bar[tc][:, None]
                    xt = np.clip(np.sqrt(ab) * tgts + np.sqrt(1.0 - ab) * mdl.rng.normal(0.0, 0.70, size=tgts.shape), -1.0, 1.0)
                    beta = opt.beta
                    o2, _ = mdl._relax_batch(
                        np.concatenate([xt, xt], axis=0), np.concatenate([tc, tc], axis=0), None,
                        np.concatenate([tgts, tgts], axis=0), np.concatenate([np.full(10, beta), np.full(10, -beta)]),
                    )
                    opt.step_contrast(mdl.parameters_dict(), o2[:10], o2[10:], progress=(b_idx - 1) / max(1, n_steps - 1))
                gens = mdl.sample_batch(n_samples=n_eval, step_noise_scale=0.30)
            else:
                mdl_q = QuantumIsingDenoiser(
                    (3, 4), cond_dim=0, topology=topo_map[vk], p_depol=p_depol, p_readout=p_readout,
                    p_depol_2q=p_depol_2q, init_gain=1.0, gain_ema=0.0, backend=backend, torch_device=torch_device, seed=seed,
                )
                tr = QERDTrainer(mdl_q, QEROptimizer(mdl_q.parameters_dict(), QERConfig(lr_max=0.12, lr_min=0.015, clip_bound=0.90)))
                tr.train(x_tr, conds=None, max_shots=max_shots, shots_per_run=20, batch_size=10, eval_every_batches=999999, verbose=False)
                gens = mdl_q.sample_batch(n_samples=n_eval, shots_per_step=160, step_noise_scale=0.30).reshape(n_eval, 12)

            bits = np.where(gens >= 0.0, 1.0, -1.0)
            hit_modes = [valid_set[tuple(r.astype(np.int64).tolist())] for r in bits if tuple(r.astype(np.int64).tolist()) in valid_set]
            pat_acc = float(len(hit_modes) / len(bits))
            mode_cov = float(len(set(hit_modes)) / len(modes_arr))
            bc = bits - bits.mean(axis=0, keepdims=True)
            c_gen = (bc.T @ bc) / len(bits)
            corr_rmse = float(np.sqrt(np.mean((c_gen[off] - c_ref[off]) ** 2)))
            intra_corr = float(np.mean([c_gen[u, v] for u, v in intra_pairs]))
            bit_div = float(np.mean(np.std(bits, axis=0)))
            m_dict = {
                "pattern_acc": pat_acc,
                "intra_corr": intra_corr,
                "corr_rmse": corr_rmse,
                "mode_cov": mode_cov,
                "bit_std": bit_div,
            }
            seed_row[vk] = m_dict
            buckets[vk].append(m_dict)
        per_seed.append(seed_row)

    agg = {k: _aggregate_seed_metrics(v) for k, v in buckets.items()}
    return {"task": "pure_qpu_stride3_cluster_spin_glass", "n_qubits": 12, "seeds": seeds, "aggregate": agg, "per_seed": per_seed}


def save_grid(imgs: np.ndarray, row_titles: List[str], out_path: str, n_cols: int = 8):
    n_rows = len(row_titles)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.1, n_rows * 1.1))
    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r, c] if n_rows > 1 else axes[c]
            ax.imshow(imgs[r * n_cols + c, 0], cmap="gray", vmin=0.0, vmax=1.0)
            ax.axis("off")
            if c == 0:
                ax.set_ylabel(row_titles[r], fontsize=9, rotation=0, labelpad=24, va="center")
                ax.yaxis.set_visible(True)
                ax.set_yticks([])
    plt.tight_layout(pad=0.3)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ==============================================================================
# 5. MAIN BENCHMARK RUNNER
# ==============================================================================
def _build_qpu_model(args: argparse.Namespace, seed: int, is_qerd: bool = False) -> QuantumIsingDenoiser:
    topo = "none" if (is_qerd and args.ablate == "no-zz") else args.topology
    use_cal = bool(is_qerd and args.ablate in ("gain-cal", "gain-cal-inv"))
    cal_mode = "inverse" if (is_qerd and args.ablate == "gain-cal-inv") else "forward"
    init_gain = 0.75 if use_cal else 1.0
    gain_ema = 0.10 if use_cal else 0.0
    mdl = QuantumIsingDenoiser(
        data_shape=args.latent_dim, cond_dim=10, n_hidden=args.n_hidden, topology=topo,
        n_layers=args.n_layers, p_depol=args.p_depol, p_readout=args.p_readout,
        p_depol_2q=args.p_depol_2q, init_gain=init_gain, gain_ema=gain_ema, cal_mode=cal_mode,
        backend=args.backend, torch_device=args.torch_device, device_name=args.device_name,
        real_qpu=args.real_qpu, seed=seed,
    )
    if is_qerd and args.ablate == "wcond-only":
        mdl.freeze_theta = True
    return mdl


def _aggregate_seed_metrics(per_seed_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    keys = list(per_seed_list[0].keys())
    n = max(1, len(per_seed_list))
    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vals = [d[k] for d in per_seed_list]
        m = float(np.mean(vals))
        s = float(np.std(vals))
        out[k] = {"mean": m, "std": s, "ci95": float(1.96 * s / math.sqrt(n))}
    return out


def main():
    parser = argparse.ArgumentParser(description="MNIST / Fashion-MNIST QPU Latent Diffusion Benchmark")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion", "digits"])
    parser.add_argument("--latent-dim", type=int, default=12)
    parser.add_argument("--n-hidden", type=int, default=0)
    parser.add_argument("--topology", type=str, default="all")
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--ae-epochs", type=int, default=8)
    parser.add_argument("--clf-epochs", type=int, default=5)
    parser.add_argument("--no-latent-bn", action="store_true")
    parser.add_argument("--max-shots", type=int, default=80_000)
    parser.add_argument("--shots-per-run", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--eval-per-class", type=int, default=200)
    parser.add_argument("--eval-shots", type=int, default=160)
    parser.add_argument("--step-noise", type=float, default=0.70)
    parser.add_argument("--p-depol", type=float, default=0.035)
    parser.add_argument("--p-depol-2q", type=float, default=0.0)
    parser.add_argument("--p-readout", type=float, default=0.04)
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "numpy", "torch", "pennylane", "qiskit"])
    parser.add_argument("--torch-device", type=str, default="auto")
    parser.add_argument("--device-name", type=str, default="lightning.qubit")
    parser.add_argument("--real-qpu", action="store_true")
    parser.add_argument("--seeds", type=str, default="42", help="Comma-separated random seeds (e.g. 0,1,2,3,4)")
    parser.add_argument(
        "--ablate",
        type=str,
        default="none",
        choices=["none", "gain-cal", "gain-cal-inv", "no-zz", "wcond-only", "sgd"],
    )
    parser.add_argument("--skip-psr", action="store_true")
    parser.add_argument("--with-controls", action="store_true")
    parser.add_argument("--with-lit-baselines", action="store_true")
    parser.add_argument("--no-cache-helpers", action="store_true")
    parser.add_argument("--tag-suffix", type=str, default="")
    parser.add_argument("--save-dir", type=str, default="./benchmark_out")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bn_suffix = "_nobn" if args.no_latent_bn else ""
    cache_path = (
        None
        if args.no_cache_helpers
        else os.path.join(args.save_dir, f"{args.dataset}_d{args.latent_dim}_ae{args.ae_epochs}_clf{args.clf_epochs}{bn_suffix}_helpers.pt")
    )

    probe = _build_qpu_model(args, seeds[0], is_qerd=True)
    qpu_backend_info = f"{probe.backend_name}" + (f" ({probe.backend.device})" if probe.backend_name == "torch" else "")
    print(
        f"=== QERD BENCHMARK ({args.dataset.upper()} | {args.latent_dim} visible + {args.n_hidden} hidden qubits "
        f"| L={args.n_layers} | topo={args.topology} | p1q={args.p_depol} p2q={args.p_depol_2q} "
        f"| torch={device} | qpu_sim={qpu_backend_info} | ablate={args.ablate} | seeds={seeds}) ===",
        flush=True,
    )

    x_tr, y_tr, x_te, y_te, class_names = load_dataset(args.dataset)
    print(f"Loaded splits: train={x_tr.shape}, test={x_te.shape}", flush=True)
    ae, clf, z_tr, z_te = train_classical_helpers(
        x_tr, y_tr, x_te, y_te, args.latent_dim, args.ae_epochs, args.clf_epochs, device,
        cache_path=cache_path, use_bn=not args.no_latent_bn,
    )
    ae_params = sum(p.numel() for p in ae.parameters())
    clf_params = sum(p.numel() for p in clf.parameters())
    corr_mat = np.abs(np.corrcoef(z_tr, rowvar=False))
    mean_offdiag_corr = float(np.mean(corr_mat[~np.eye(args.latent_dim, dtype=bool)])) if args.latent_dim > 1 else 0.0
    real_refs = evaluate_real_baseline(ae, clf, z_te, x_te, y_te, args.eval_per_class, device)
    print(
        f"  [Real-Held-Out ] ClfAcc={real_refs['real_held_out']['clf_acc']*100:5.1f}% | "
        f"CondFeatFID={real_refs['real_held_out']['cond_feature_fid']:6.2f} | "
        f"FeatFID={real_refs['real_held_out']['feature_fid']:6.2f} | "
        f"Prec={real_refs['real_held_out']['precision']:.3f} Rec={real_refs['real_held_out']['recall']:.3f}",
        flush=True,
    )
    print(
        f"  [Real-AE-Recon ] ClfAcc={real_refs['real_ae_recon']['clf_acc']*100:5.1f}% | "
        f"CondFeatFID={real_refs['real_ae_recon']['cond_feature_fid']:6.2f} | "
        f"FeatFID={real_refs['real_ae_recon']['feature_fid']:6.2f} | "
        f"Prec={real_refs['real_ae_recon']['precision']:.3f} Rec={real_refs['real_ae_recon']['recall']:.3f}",
        flush=True,
    )

    eye10 = np.eye(10, dtype=np.float64)
    c_tr = eye10[y_tr]
    gen_labels = np.repeat(np.arange(10), args.eval_per_class)
    gen_conds = eye10[gen_labels]

    method_keys = ["qerd", "adam_spsa"]
    if not args.skip_psr:
        method_keys.append("adam_psr")
    method_keys.append("classical_ddpm")
    if args.with_controls:
        method_keys.extend(["adaptive_spsa", "exact_psr", "classical_ising_ep", "classical_skip_ddpm", "classical_large_ddpm"])
    if args.with_lit_baselines:
        method_keys.extend(["pqc_quddpm", "qcbm_mmd"])

    per_seed_runs: List[Dict] = []
    agg_buckets: Dict[str, List[Dict[str, float]]] = {k: [] for k in method_keys}
    last_grids: Dict[str, np.ndarray] = {}
    qpu_total_params = probe.n_qpu_params + (probe.cond_proj.size if probe.cond_proj is not None else 0)
    param_counts: Dict[str, int] = {"qpu": qpu_total_params, "ae": ae_params, "clf": clf_params}

    for seed in seeds:
        print(f"\n==================== SEED {seed} ====================", flush=True)
        seed_entry: Dict = {"seed": seed}

        print("--- Training QERD (Quantum Equilibrium-Response Diffusion) ---", flush=True)
        t0 = time.time()
        qerd_model = _build_qpu_model(args, seed, is_qerd=True)
        q_cfg = (
            QERConfig(lr_max=0.05, lr_min=0.05, use_adam=False, auto_beta_guard=False, weight_decay=0.0, clip_bound=0.90)
            if args.ablate == "sgd"
            else QERConfig(lr_max=0.12, lr_min=0.015, clip_bound=0.90)
        )
        q_opt = QEROptimizer(qerd_model.parameters_dict(), q_cfg)
        trainer = QERDTrainer(qerd_model, optimizer=q_opt)
        trainer.train(
            z_tr, c_tr, max_shots=args.max_shots, shots_per_run=args.shots_per_run,
            batch_size=args.batch_size, eval_every_batches=999999, stratify_labels=y_tr, verbose=False,
        )
        qerd_latents = qerd_model.sample_batch(conds=gen_conds, shots_per_step=args.eval_shots, step_noise_scale=args.step_noise)
        print(f"  QERD finished {q_opt.shots_used} shots ({q_opt.step_count} steps) in {time.time()-t0:.1f}s | gain={qerd_model.calibrator.mean_gain:.2f}", flush=True)

        print("--- Training Adam-SPSA Baseline (Identical QPU Circuit & Shot Budget) ---", flush=True)
        t0 = time.time()
        spsa_model = _build_qpu_model(args, seed, is_qerd=False)
        spsa_stats = train_adam_spsa_baseline(
            spsa_model, z_tr, c_tr, y_tr, max_shots=args.max_shots,
            shots_per_run=args.shots_per_run, batch_size=args.batch_size,
        )
        spsa_latents = spsa_model.sample_batch(conds=gen_conds, shots_per_step=args.eval_shots, step_noise_scale=args.step_noise)
        print(f"  Adam-SPSA finished {spsa_stats['shots']} shots ({spsa_stats['batches']} steps) in {time.time()-t0:.1f}s", flush=True)

        psr_latents = None
        if not args.skip_psr:
            print("--- Training Adam-PSR-FD Baseline (Active Finite-Diff Parameter Shift) ---", flush=True)
            t0 = time.time()
            psr_model = _build_qpu_model(args, seed, is_qerd=False)
            psr_stats = train_adam_psr_baseline(
                psr_model, z_tr, c_tr, y_tr, max_shots=args.max_shots, shots_per_run=args.shots_per_run,
            )
            psr_latents = psr_model.sample_batch(conds=gen_conds, shots_per_step=args.eval_shots, step_noise_scale=args.step_noise)
            print(f"  Adam-PSR-FD finished {psr_stats['shots']} shots ({psr_stats['batches']} steps) in {time.time()-t0:.1f}s", flush=True)

        cls_model = train_classical_baseline(
            z_tr, c_tr, y_tr, target_params=qpu_total_params, n_steps=q_opt.step_count, batch_size=args.batch_size, seed=seed,
        )
        param_counts["classical_ddpm"] = sum(p.numel() for p in cls_model.parameters())
        cls_latents = cls_model.sample_batch(gen_conds, seed=seed + 1000, step_noise_scale=args.step_noise)

        extra_latents: Dict[str, Tuple[str, np.ndarray]] = {}
        if args.with_controls:
            aspsa_m = _build_qpu_model(args, seed, is_qerd=False)
            train_adaptive_spsa_baseline(
                aspsa_m, z_tr, c_tr, y_tr, max_shots=args.max_shots, shots_per_run=args.shots_per_run, batch_size=args.batch_size,
            )
            extra_latents["adaptive_spsa"] = ("Adaptive-SPSA", aspsa_m.sample_batch(conds=gen_conds, shots_per_step=args.eval_shots, step_noise_scale=args.step_noise))

            epsr_m = _build_qpu_model(args, seed, is_qerd=False)
            train_exact_psr_baseline(
                epsr_m, z_tr, c_tr, y_tr, device=device, max_shots=args.max_shots, shots_per_run=args.shots_per_run,
            )
            extra_latents["exact_psr"] = ("Exact-PSR", epsr_m.sample_batch(conds=gen_conds, shots_per_step=args.eval_shots, step_noise_scale=args.step_noise))

            cing_m = train_classical_ising_ep(
                z_tr, c_tr, y_tr, n_hidden=args.n_hidden, topology=args.topology, n_steps=q_opt.step_count, batch_size=args.batch_size, seed=seed,
            )
            param_counts["classical_ising_ep"] = cing_m.theta.size + (cing_m.cond_proj.size if cing_m.cond_proj is not None else 0)
            extra_latents["classical_ising_ep"] = ("ClassicalIsing-EP", cing_m.sample_batch(conds=gen_conds, step_noise_scale=args.step_noise))

            cskip_m = train_classical_baseline(
                z_tr, c_tr, y_tr, target_params=qpu_total_params, n_steps=q_opt.step_count,
                batch_size=args.batch_size, seed=seed, skip_cond=True,
            )
            param_counts["classical_skip_ddpm"] = sum(p.numel() for p in cskip_m.parameters())
            extra_latents["classical_skip_ddpm"] = ("Classical-Skip", cskip_m.sample_batch(gen_conds, seed=seed + 2000, step_noise_scale=args.step_noise))

            clrg_m = train_classical_baseline(
                z_tr, c_tr, y_tr, target_params=qpu_total_params, n_steps=q_opt.step_count,
                batch_size=args.batch_size, seed=seed, large=True,
            )
            param_counts["classical_large_ddpm"] = sum(p.numel() for p in clrg_m.parameters())
            extra_latents["classical_large_ddpm"] = ("Classical-Large", clrg_m.sample_batch(gen_conds, seed=seed + 3000, step_noise_scale=args.step_noise))

        if args.with_lit_baselines:
            quddpm_m = train_pqc_quddpm_baseline(
                z_tr, c_tr, y_tr, p_depol=args.p_depol, p_readout=args.p_readout, device=device,
                max_shots=args.max_shots, shots_per_run=args.shots_per_run, batch_size=args.batch_size, seed=seed,
            )
            param_counts["pqc_quddpm"] = quddpm_m.n_params
            extra_latents["pqc_quddpm"] = ("PQC-QuDDPM", quddpm_m.sample_batch(gen_conds, shots_per_step=args.eval_shots, step_noise_scale=args.step_noise))

            qcbm_m = train_qcbm_mmd_baseline(
                z_tr, c_tr, y_tr, p_depol=args.p_depol, p_readout=args.p_readout, device=device,
                max_shots=args.max_shots, shots_per_run=args.shots_per_run, batch_size=args.batch_size, seed=seed,
            )
            param_counts["qcbm_mmd"] = qcbm_m.n_params
            extra_latents["qcbm_mmd"] = ("QCBM-MMD", qcbm_m.sample_batch(gen_conds, shots_per_step=args.eval_shots))

        print(f"\n=== HELD-OUT TEST METRICS (Seed {seed}) ===", flush=True)
        print(f"Param budgets | QPU={qpu_total_params} params (noisy NISQ) | Classical DDPM={param_counts['classical_ddpm']} params (noiseless) | AE={ae_params}", flush=True)
        m_qerd, imgs_qerd = evaluate_generated("QERD (Ours)", qerd_latents, gen_labels, ae, clf, z_te, x_te, y_te, device)
        m_spsa, imgs_spsa = evaluate_generated("Adam-SPSA", spsa_latents, gen_labels, ae, clf, z_te, x_te, y_te, device)
        last_grids = {"qerd": imgs_qerd, "spsa": imgs_spsa}
        seed_entry["qerd"] = m_qerd
        seed_entry["adam_spsa"] = m_spsa
        agg_buckets["qerd"].append(m_qerd)
        agg_buckets["adam_spsa"].append(m_spsa)

        if not args.skip_psr and psr_latents is not None:
            m_psr, imgs_psr = evaluate_generated("Adam-PSR-FD", psr_latents, gen_labels, ae, clf, z_te, x_te, y_te, device)
            last_grids["psr"] = imgs_psr
            seed_entry["adam_psr"] = m_psr
            agg_buckets["adam_psr"].append(m_psr)

        m_cls, imgs_cls = evaluate_generated("Classical-DDPM", cls_latents, gen_labels, ae, clf, z_te, x_te, y_te, device)
        last_grids["classical"] = imgs_cls
        seed_entry["classical_ddpm"] = m_cls
        agg_buckets["classical_ddpm"].append(m_cls)

        for key, (disp_name, lat_arr) in extra_latents.items():
            m_ex, _ = evaluate_generated(disp_name, lat_arr, gen_labels, ae, clf, z_te, x_te, y_te, device)
            seed_entry[key] = m_ex
            agg_buckets[key].append(m_ex)

        per_seed_runs.append(seed_entry)

    tag = (f"{args.dataset}" if args.ablate == "none" else f"{args.dataset}_{args.ablate}") + args.tag_suffix
    n_show = min(8, args.eval_per_class)
    idx_show = np.concatenate([np.where(gen_labels == c)[0][:n_show] for c in range(10)])
    idx_real = np.concatenate([np.where(y_te == c)[0][:n_show] for c in range(10)])
    save_grid(x_te[idx_real], class_names, os.path.join(args.save_dir, f"{tag}_real_test.png"), n_cols=n_show)
    save_grid(last_grids["qerd"][idx_show], class_names, os.path.join(args.save_dir, f"{tag}_qerd_samples.png"), n_cols=n_show)
    save_grid(last_grids["spsa"][idx_show], class_names, os.path.join(args.save_dir, f"{tag}_spsa_samples.png"), n_cols=n_show)
    if "psr" in last_grids:
        save_grid(last_grids["psr"][idx_show], class_names, os.path.join(args.save_dir, f"{tag}_psr_samples.png"), n_cols=n_show)
    save_grid(last_grids["classical"][idx_show], class_names, os.path.join(args.save_dir, f"{tag}_classical_samples.png"), n_cols=n_show)

    aggregate = {k: _aggregate_seed_metrics(v) for k, v in agg_buckets.items()}
    if len(seeds) > 1:
        print(f"\n=== MULTI-SEED SUMMARY (Mean ± Std across N={len(seeds)} seeds) ===", flush=True)
        for mk in method_keys:
            a = aggregate[mk]
            print(
                f"  [{mk:20s}] ClfAcc={a['clf_acc']['mean']*100:5.1f}±{a['clf_acc']['std']*100:4.1f}% | "
                f"CondLatFID={a['cond_latent_fid']['mean']:5.2f}±{a['cond_latent_fid']['std']:4.2f} | "
                f"CondFeatFID={a['cond_feature_fid']['mean']:6.1f}±{a['cond_feature_fid']['std']:5.1f} | "
                f"FeatFID={a['feature_fid']['mean']:6.1f}±{a['feature_fid']['std']:5.1f} | "
                f"Prec={a['precision']['mean']:.3f} Rec={a['recall']['mean']:.3f}",
                flush=True,
            )

    summary = {
        "dataset": args.dataset,
        "ablate": args.ablate,
        "topology": args.topology,
        "n_layers": args.n_layers,
        "n_edges": probe.n_edges,
        "two_qubit_cnots": 2 * args.n_layers * probe.n_edges,
        "p_depol": args.p_depol,
        "p_depol_2q": args.p_depol_2q,
        "p_readout": args.p_readout,
        "latent_bn": not args.no_latent_bn,
        "mean_offdiag_latent_corr": mean_offdiag_corr,
        "eval_samples": int(10 * args.eval_per_class),
        "seeds": seeds,
        "qpu_params": qpu_total_params,
        "classical_params": param_counts["classical_ddpm"],
        "param_counts": param_counts,
        "real_baselines": real_refs,
        "shots_used": args.max_shots,
        "aggregate": aggregate,
        "per_seed": per_seed_runs,
    }
    if len(seeds) == 1:
        for mk in method_keys:
            summary[mk] = per_seed_runs[0][mk]
    with open(os.path.join(args.save_dir, f"{tag}_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved PNG sample grids & results JSON to {args.save_dir}", flush=True)


if __name__ == "__main__":
    main()
