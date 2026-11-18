# Project 2: Conditional Human Promoter DNA Designer (`128q` IBM Heron R2 + `PromoterOracleNet`)

Generates novel, biophysically valid **`64 bp` (`[-49..+14]` TSS-aligned) Human Promoter DNA sequences** conditioned on functional subclass (**`TATA-box` vs. `GC-box/Non-TATA`**) directly on `128` superconducting qubits using **`QER-Res` residual QERD** (`3` circuits/sample MSE contrastive nudging, or `--nudge-mode target` for `2`-circuit `QER-Grad`), **stochastic calibrated decoding**, **`cal_mode="inverse"`** MMSE readout shrinkage, **noise-pruned `128q` Heavy-Hex snake routing** (`50/64` intra-base physical `CZ` couplers), and **1-job live marginal readout mitigation**.

## Architecture & Methodology

* **Direct Quantum Nucleotide Encoding (`0` Classical Autoencoder):** Bijective `2`-qubit/base mapping (`A=(-1,-1)`, `T=(-1,+1)`, `G=(+1,-1)`, `C=(+1,+1)`) puts all `64 bp` directly onto `128` hardware qubits with zero neural decoder shortcuts.
* **Rigorous Biological & Thermodynamic Scoring:** Every design is evaluated by **PyTorch `PromoterOracleNet`** (`1D-CNN` trained on real *H. sapiens* promoter FASTA splits downloaded via [`loaders/promoters.py`](loaders/promoters.py) into gitignored `data/dna_real/`, `0.9333` held-out test ROC-AUC) plus **SantaLucia (1998) $\Delta G^\circ_{37}$ thermodynamics**, **GC%**, **TSS Initiator (`Inr`)**, **Homopolymer checks**, and **Novelty/Diversity**.
* **Batched Open-Plan QPU Execution:** Noise-pruned `128q` routing on `ibm_kingston` (`RO>5% = 0`, `CZ>2% = 3`) plus 1 live all-0 / all-1 calibration job executes `38,016` `QER-Res` training shots across `11` packed batches (`bs=24`, `86s` total QPU quota).

---

## Verified Live IBM Quantum Results (`ibm_kingston`, `128q` / `64 bp`, `38,016` Shots)

| Model | Oracle Activity ↑ | Oracle Pass ↑ | Biophysical Valid ↑ | Inr TSS ↑ | Upstream Box ↑ | GC % | dG37 ↓ | Unique / Novel ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Untrained QPU (`θ=0`) | 9.8% | 3.1% | 0.0% | 65.6% | 31.2% | 40.4% | -80.2 | 100% / 100% |
| Classical Markov | 80.5% | 75.0% | **59.4%** | **87.5%** | 93.8% | 64.1% | -96.7 | 100% / 100% |
| Real Human DNA (Held-Out) | 75.8% | 71.9% | 34.4% | 65.6% | 71.9% | 58.7% | -92.4 | 100% / 100% |
| **QERD + QER-Res QPU (`128q`, Ours)** | **92.6%** | **93.8%** | 40.6% | 84.4% | **100.0%** | 73.2% | -104.0 | **100% / 100%** |

---

## Commands

```bash
# Download Human Promoter FASTA splits into gitignored data/dna_real/ (also auto-fetched on run)
python -m projects.dna_promoter_qerd --fetch-data

# Quick unit & end-to-end test suite
python -m unittest discover -s projects/dna_promoter_qerd/tests -v

# Full Local CPU benchmark (`16q` Core TATA/Inr quantum vs PSR/SPSA/Markov)
python -m projects.dna_promoter_qerd --preset local_cpu

# Live 128q (64 bp) IBM Quantum execution (`ibm_kingston`, `38,016` QER-Res shots)
python -m projects.dna_promoter_qerd --preset heron_128q --device ibm_kingston --api-key-json /path/to/apikey.json --batches 17 --batch-size 24 --shots-per-run 48 --skip-live-spsa
```
