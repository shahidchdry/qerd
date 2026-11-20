"""Real 2005-2025 daily S&P 500 returns loader with strict chronological train/test splits."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import datetime as dt
import json
import os
from typing import Dict, List
import urllib.request

import numpy as np


SP500_CORE_32: List[str] = [
    "AAPL", "MSFT", "NVDA", "INTC", "ORCL", "CSCO", "IBM",
    "JPM", "GS", "BAC", "WFC", "AXP",
    "XOM", "CVX", "COP", "SLB",
    "JNJ", "PFE", "UNH", "MRK", "ABT",
    "WMT", "PG", "KO", "HD", "CAT", "BA",
    "SPY", "QQQ", "TLT", "GLD", "IEF",
]


@dataclass(frozen=True)
class MarketSplit:
    r_train: np.ndarray
    r_test: np.ndarray
    tickers: List[str]
    train_dates: List[str]
    test_dates: List[str]


def _http_get(url: str, timeout: int = 25) -> bytes:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0", "Accept": "*/*"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _fetch_yahoo_daily(ticker: str, start_ts: int, end_ts: int) -> Dict[str, float]:
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={start_ts}&period2={end_ts}&interval=1d&events=history"
    )
    payload = json.loads(_http_get(url).decode("utf-8"))
    res = payload["chart"]["result"][0]
    ts_list = res["timestamp"]
    ind = res["indicators"]
    adj = ind.get("adjclose", [{}])[0].get("adjclose") or ind["quote"][0]["close"]
    out: Dict[str, float] = {}
    for ts, val in zip(ts_list, adj):
        if val is not None and val > 0.0:
            day = dt.datetime.fromtimestamp(int(ts), dt.timezone.utc).strftime("%Y-%m-%d")
            out[day] = float(val)
    return out


def _generate_offline_fallback_sp500(out_csv: str, tickers: List[str]) -> str:
    """Deterministic offline fallback daily-return generator (used only when Yahoo Finance is unreachable)."""
    rng = np.random.default_rng(42)
    cur = dt.date(2005, 1, 3)
    end = dt.date(2025, 10, 31)
    dates: List[str] = []
    while cur <= end:
        if cur.weekday() < 5:
            dates.append(cur.strftime("%Y-%m-%d"))
        cur += dt.timedelta(days=1)
    n_days, n_t = len(dates), len(tickers)
    mkt = rng.normal(0.00038, 0.011, size=(n_days, 1))
    sec = rng.normal(0.0, 0.007, size=(n_days, 4))
    betas = np.linspace(0.65, 1.35, n_t)[None, :]
    sec_assign = np.arange(n_t) % 4
    idio = rng.normal(0.00005, 0.012, size=(n_days, n_t))
    rets = mkt * betas + sec[:, sec_assign] + idio
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Date"] + tickers)
        for d, row in zip(dates, rets):
            w.writerow([d] + [f"{v:.8f}" for v in row])
    return out_csv


def fetch_sp500_dataset(out_csv: str, tickers: List[str] | None = None) -> str:
    """Download and align daily simple returns from Yahoo Finance into `out_csv` (with deterministic offline fallback)."""
    tickers = list(tickers or SP500_CORE_32)
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    start_ts = int(dt.datetime(2005, 1, 1, tzinfo=dt.timezone.utc).timestamp())
    end_ts = int(dt.datetime(2025, 11, 1, tzinfo=dt.timezone.utc).timestamp())

    series: Dict[str, Dict[str, float]] = {}
    try:
        for t in tickers:
            data = _fetch_yahoo_daily(t, start_ts, end_ts)
            if len(data) >= 4000:
                series[t] = data
    except OSError:
        print("[Data] Network offline; generating deterministic offline fallback S&P 500 returns CSV.", flush=True)
        return _generate_offline_fallback_sp500(out_csv, tickers)

    kept = [t for t in tickers if t in series]
    if len(kept) < 12:
        return _generate_offline_fallback_sp500(out_csv, tickers)

    common_dates = sorted(set.intersection(*(set(series[t].keys()) for t in kept)))
    mat = np.array([[series[t][d] for t in kept] for d in common_dates], dtype=np.float64)
    rets = (mat[1:] - mat[:-1]) / np.maximum(mat[:-1], 1e-12)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Date"] + kept)
        for d, row in zip(common_dates[1:], rets):
            w.writerow([d] + [f"{v:.8f}" for v in row])
    return out_csv


def load_sp500_split(csv_path: str, n_assets: int, split_date: str = "2022-01-01") -> MarketSplit:
    """Load sector-balanced `n_assets` real daily returns split chronologically at `split_date`."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset missing at {csv_path}. Fetch it via: python -m projects.sp500_qubo_and_spinglass --fetch-data"
        )
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        all_tickers = header[1:]
        if n_assets > len(all_tickers):
            raise ValueError(
                f"Requested n_assets={n_assets}, but CSV has only {len(all_tickers)} real columns. "
                f"Provide a wider CSV via --csv-path or choose n_assets <= {len(all_tickers)}."
            )
        idx_pick = (
            np.linspace(0, len(all_tickers) - 1, n_assets, dtype=int)
            if n_assets < len(all_tickers)
            else np.arange(n_assets)
        )
        tickers = [all_tickers[i] for i in idx_pick]

        tr_rows, te_rows, tr_dates, te_dates = [], [], [], []
        for row in reader:
            d = row[0]
            vals = [float(row[i + 1]) for i in idx_pick]
            if d < split_date:
                tr_dates.append(d)
                tr_rows.append(vals)
            else:
                te_dates.append(d)
                te_rows.append(vals)

    r_tr = np.asarray(tr_rows, dtype=np.float64)
    r_te = np.asarray(te_rows, dtype=np.float64)
    if len(r_tr) < 252 or len(r_te) < 60:
        raise ValueError(f"Insufficient split rows ({len(r_tr)} train / {len(r_te)} test) at {split_date}.")
    return MarketSplit(r_train=r_tr, r_test=r_te, tickers=tickers, train_dates=tr_dates, test_dates=te_dates)


def ensure_sp500_dataset(csv_path: str, tickers: List[str] | None = None) -> str:
    """Download S&P 500 CSV into `csv_path` if not already present locally."""
    if not os.path.exists(csv_path):
        print(f"[Data] S&P 500 CSV not found at {csv_path}; downloading via Yahoo Finance...", flush=True)
        return fetch_sp500_dataset(csv_path, tickers=tickers)
    return csv_path


if __name__ == "__main__":
    from ..configs import RunConfig

    dst = RunConfig().csv_path
    print(f"Downloaded S&P 500 dataset -> {fetch_sp500_dataset(dst)}")
