"""
PyTorch 1D-CNN Human Promoter Activity Oracle (`PromoterOracleNet`).

Trains on real `64 bp` (`[-49..+14]`) Human Promoters vs real genomic non-promoters
plus 15% uniform random negative decoys (so uniform random QPU bitstrings cannot masquerade
as high-scoring promoters), auto-supporting CPU and CUDA (`DataParallel` on 2x T4).
"""

from __future__ import annotations

import os
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..loaders.promoters import PromoterDataset, onehot_from_seqs


class PromoterOracleNet(nn.Module):
    """Multi-scale 1D-CNN + TSS-bin pooling classifier (~130k parameters)."""

    def __init__(self, seq_len: int = 64):
        super().__init__()
        self.seq_len = int(seq_len)
        self.conv = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=7, padding=3),
            nn.GELU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.GELU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
        )
        n_bins = max(1, self.seq_len // 4)
        self.head = nn.Sequential(
            nn.Linear(64 * n_bins + 64, 64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        pooled = nn.functional.avg_pool1d(h, 4).flatten(1)
        gmax = h.max(dim=-1).values
        return self.head(torch.cat([pooled, gmax], dim=1)).squeeze(1)


@torch.no_grad()
def predict_oracle_probs(model: PromoterOracleNet, seqs: Sequence[str], batch_size: int = 512) -> np.ndarray:
    if not seqs:
        return np.zeros((0,), dtype=np.float64)
    L = len(seqs[0])
    if L != model.seq_len:
        raise ValueError(f"Oracle expects seq_len={model.seq_len}, got {L}.")
    dev = next(model.parameters()).device
    model.eval()
    arr = onehot_from_seqs(seqs)
    probs = []
    for st in range(0, len(arr), batch_size):
        xb = torch.from_numpy(arr[st : st + batch_size]).to(dev)
        probs.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(probs, axis=0).astype(np.float64)


def load_or_train_oracle(
    dataset: PromoterDataset,
    ckpt_path: str,
    epochs: int = 4,
    seed: int = 42,
) -> Tuple[PromoterOracleNet, Dict[str, float]]:
    """Train or load `PromoterOracleNet` and return held-out test ROC-AUC / accuracy metrics."""
    from sklearn.metrics import accuracy_score, roc_auc_score

    L = dataset.seq_len
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model = PromoterOracleNet(seq_len=int(ck["seq_len"])).to(dev)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        return model, dict(ck["metrics"])

    rng = np.random.default_rng(seed)
    n_decoy = max(500, len(dataset.train_all_seqs) // 6)
    decoys = ["".join(rng.choice(["A", "C", "G", "T"], size=L)) for _ in range(n_decoy)]
    train_seqs = list(dataset.train_all_seqs) + decoys
    train_labels = np.concatenate([dataset.train_all_labels, np.zeros(n_decoy, dtype=np.float32)])

    x_tr = torch.from_numpy(onehot_from_seqs(train_seqs))
    y_tr = torch.from_numpy(train_labels)
    x_te = onehot_from_seqs(dataset.test_all_seqs)
    y_te = dataset.test_all_labels

    torch.manual_seed(seed)
    model = PromoterOracleNet(seq_len=L).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    train_mod: nn.Module = model
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        train_mod = nn.DataParallel(model)

    for _ in range(max(1, epochs)):
        train_mod.train()
        perm = torch.randperm(len(x_tr))
        for st in range(0, len(x_tr), 256):
            idx = perm[st : st + 256]
            xb, yb = x_tr[idx].to(dev), y_tr[idx].to(dev)
            opt.zero_grad()
            loss = loss_fn(train_mod(xb), yb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        pr_te = torch.sigmoid(model(torch.from_numpy(x_te).to(dev))).cpu().numpy()
        rnd_te = ["".join(rng.choice(["A", "C", "G", "T"], size=L)) for _ in range(400)]
        pr_rnd = predict_oracle_probs(model, rnd_te)

    metrics = {
        "test_auc": float(roc_auc_score(y_te, pr_te)),
        "test_acc": float(accuracy_score(y_te, pr_te >= 0.5)),
        "random_mean_prob": float(np.mean(pr_rnd)),
        "random_pass_pct": float(100.0 * np.mean(pr_rnd >= 0.60)),
    }
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({"seq_len": L, "state_dict": model.state_dict(), "metrics": metrics}, ckpt_path)
    return model, metrics
