# Project 1: S&P 500 Markowitz QUBO & Heavy-Hex Spin Glass vs. The $2^N$ GPU Wall

A modular quantum benchmarking & optimization suite comparing **QERD (`QER-Res` / `QER-Grad` + `QuantumIsingDenoiser`) on IBM Heron R2 (`ibm_kingston` / `ibm_fez`)** against **`2× T4` exact simulation limits**, **Buy-and-Hold (`1/N`) base profit**, **Simulated Annealing (`SA`)**, and equal-shot calibrated **`PSR` / `SPSA`** QPU optimizers.

---

## Package Structure

```text
projects/sp500_qubo_and_spinglass/
├── __init__.py
├── __main__.py
├── configs.py               # Typed `RunConfig` dataclasses & presets (`quick`, `standard`, `t4_full`, `heron_128q`, `heron_156q`)
├── cli.py                   # CLI entrypoint (`--preset`, `--mode`, `--device`, `--audit-heron`, `--infer`)
├── reporting.py             # Hardware comparison tables, JSON metrics & 3-panel publication plotter
├── loaders/
│   └── market.py            # Real 2005-2025 S&P 500 Yahoo Finance downloader & CSV loader (`data/sp500_real/`)
├── hardware/
│   └── heron_r2.py          # Heron R2 calibration audit, noise-pruned subgraph selection, 1-job live RO mitigation
├── problems/
│   ├── minvar_qubo.py       # S&P 500 Markowitz QUBO (Gap Closed %, Buy&Hold Base Profit, OOS Vol %, Sharpe)
│   └── spinglass.py         # Native Heavy-Hex frustrated spin glass (0 SWAP overhead)
├── solvers/
│   ├── classical.py         # Parallel vectorized Simulated Annealing (SA) ceiling, greedy screening, 2^N GPU profiler
│   └── quantum.py           # `QER-Res` + `cal_mode='inverse'`, calibrated Equal-Shot `SPSA`/`PSR`, sampler, checkpoints
├── tests/
│   └── test_project1.py     # Unit + end-to-end integration tests
├── checkpoints/             # Saved `.npz` QPU weights
└── artifacts/               # Saved `fig_hardware_advantage.png` & `metrics.json`
```

---

## Architecture & Hardware-Aware Design

1. **Heron R2 Sweet Spot (`128q`) & Noise-Pruned Subgraph Routing:** `--preset heron_128q` selects a connected `128q` Spin Glass subgraph (`RO>5% = 0`, `CZ>2% = 3`) and a nested `32q` S&P 500 sub-patch (`RO>5% = 0`, `CZ>2% = 0`), plus **1-job live all-0 / all-1 marginal readout calibration** shared cleanly across both tasks.
2. **Batched `O(1)` Training (`~38k` Shots per Task):** Packs `batch_size=24, shots_per_run=48` into `SamplerV2` PUBs (`38,016` shots per problem across `11` `QER-Res` updates / `53` total QPU jobs when `--skip-untrained` is enabled).
3. **`QER-Res` (`nudge_mode="residual"`) + `cal_mode="inverse"` (`gain_bounds=(0.40, 1.00)`) + Depth `L=4, D=4`:** Uses true MSE residual contrastive probing with Wiener/MMSE inverse readout shrinkage (`L=4` `128q` circuit duration `5.6 µs` vs. `T2 = 144 µs`).
4. **Problem-Matched Symmetry & Decoding:**
   * **Heavy-Hex Spin Glass:** Exact $Z_2$ inversion augmentation (`[sg_sa, -sg_sa]`, `128` parallel SA targets) preserves $h_i \approx 0$ symmetry, paired with deterministic reverse annealing (`step_noise_scale=0.25`).
   * **S&P 500 Markowitz QUBO:** `128` parallel SA targets (`100` sweeps) and stochastic Bernoulli decoding (`sample_temp=1.15`, `step_noise_scale=0.55`) preserve multi-sector portfolio diversity (`unique_frac = 1.00`).

---

## Verified Live IBM Quantum Results (`ibm_kingston`, `38,016` Shots per Task)

Executed live on **`ibm_kingston`** (`--preset heron_128q --batches 17 --batch-size 24 --shots-per-run 48 --skip-untrained`, `158s` total QPU quota across both benchmarks):

### 1. Real S&P 500 Markowitz QUBO (`32q`, OOS `2022–2025`, `$10k` Capital)
| Solver | Train QUBO Cost ↓ | Profit (`$10k`) ↑ | Unseen OOS Vol ↓ | Sharpe ↑ | Unique |
|---|---:|---:|---:|---:|---:|
| Buy & Hold `1/N` (Zero-Algo Base) | +0.0265 | +\$6,829 | 14.98% | 0.99 | 1.00 |
| Untrained Baseline (Random Draw) | -0.0572 | +\$7,088 | 15.36% | 0.99 | 1.00 |
| Greedy Min-Var *(Heuristic)* | -0.0726 | +\$4,579 | **11.37%** | 0.93 | 1.00 |
| **QERD + QER-Res QPU (`32q`, Ours)** | **-0.0945** | +\$6,506 | 13.96% | **1.01** | **1.00** |
| SA Target (`128` chains, `100` sweeps) | -0.0828 | +\$6,852 | 14.29% | 1.03 | 1.00 |

### 2. Native Heavy-Hex Frustrated Spin Glass (`128q`, `|E|=135`, `0` SWAP Overhead)
| Solver | Mean Energy ↓ | Best Energy ↓ | Satisfied Bonds ↑ | Gap Closed ↑ | Unique |
|---|---:|---:|---:|---:|---:|
| SA Target Ceiling (`128` chains) | -123.78 | -129.0 | 95.8% | 100.0% | 100.0% |
| Untrained Baseline (Random Draw) | -0.31 | -11.0 | 50.1% | 0.0% | 100.0% |
| **QERD + QER-Res QPU (`128q`, Ours)** | **-89.06** | **-103.0** | **83.0%** | **71.9%** | **100.0%** |

---

## Verified Local Results (`--preset standard`, `ibm_kingston` `12q`, `36,864` `QER-Res` Shots)

### 1. S&P 500 Markowitz QUBO vs. Buy-and-Hold (`2022–2025`, `$10k` Initial Capital)
*(Train: `2005-01-04 .. 2021-12-31`; Unseen Out-of-Sample: `2022-01-03 .. 2025-10-31`)*

| Solver | QUBO Cost ↓ | Profit (`$10k`) ↑ | vs. Buy&Hold | Unseen OOS Vol ↓ | Sharpe ↑ |
|---|---:|---:|---:|---:|---:|
| Buy & Hold `1/N` (Zero-Algo Base) | +0.0110 | +\$9,257 | \$0 (Base) | 16.45% | **1.13** |
| Untrained QPU (`θ=0`) | -0.0604 | +\$9,628 | +\$371 | 16.75% | **1.14** |
| Greedy Min-Var *(Heuristic)* | -0.0630 | +\$2,673 | -\$6,584 | **11.81%** | 0.58 |
| Equal-Shot PSR (`12q`, calibrated) | -0.0878 | +\$8,814 | -\$443 | 15.62% | **1.14** |
| Equal-Shot Adam-SPSA (`12q`, calibrated) | -0.1187 | **+\$10,477** | **+\$1,220** | 18.57% | 1.10 |
| **QERD + QER-Res QPU (Ours, `32` jobs)** | **-0.1218** | +\$10,243 | +\$986 | 18.45% | 1.09 |
| SA Target | -0.1156 | +\$10,299 | +\$1,041 | 18.14% | 1.11 |

### 2. Native Heavy-Hex Frustrated Spin Glass (`12q`, `|E|=11`, `0` SWAP Overhead)

| Solver | Mean Energy ↓ | Best Energy ↓ | Satisfied Bonds ↑ | Gap Closed ↑ | Unique % |
|---|---:|---:|---:|---:|---:|
| SA Target Ceiling | -10.53 | -11.0 | 97.9% | 100.0% | 9.4% |
| Untrained QPU (`θ=0`) | +0.84 | — | 46.2% | 0.0% | 96.9% |
| Equal-Shot Exact-PSR | -3.25 | — | 64.8% | 36.0% | 42.2% |
| Equal-Shot Adam-SPSA | -2.53 | — | 61.5% | 29.7% | 18.8% |
| **QERD + QER-Res + Z2-Aug (Ours, `32` jobs)** | **-9.41** | **-11.0** | **92.8%** | **90.1%** | **26.6%** |

---

## Commands

```bash
# Download real S&P 500 2005-2025 daily returns into gitignored data/sp500_real/ (also auto-fetched on run)
python -m projects.sp500_qubo_and_spinglass --fetch-data

# Quick test suite (<20s CPU)
python -m unittest discover -s projects/sp500_qubo_and_spinglass/tests -v

# Standard local suite (12q Kingston S&P QUBO & Spin Glass, 38.4k budget)
python -m projects.sp500_qubo_and_spinglass --preset standard

# Kaggle 2x T4 GPU scaling
python -m projects.sp500_qubo_and_spinglass --preset t4_full --max-sim-qubits 16

# Live 128-qubit IBM Quantum execution (bs=24 packed, 38,016 shots/problem)
python -m projects.sp500_qubo_and_spinglass --preset heron_128q --device ibm_kingston --api-key-json /path/to/apikey.json --batches 17 --batch-size 24 --shots-per-run 48 --skip-untrained
```
