# QERD Projects Suite

Modular real-world `128`-qubit IBM Heron R2 quantum optimization and generative modeling applications built on [`qerd`](../qerd/__init__.py).

| Module | Application & Live `ibm_kingston` Benchmark | Status |
|---|---|---|
| [`sp500_qubo_and_spinglass/`](sp500_qubo_and_spinglass/README.md) | **Project 1:** `2^N` Classical GPU Wall, IBM Heron R2 calibration audit, **32q S&P 500 Markowitz QUBO** (`-0.0945` QUBO / `145.4%` gap vs. SA, `+\$6,506` OOS profit, `1.01` Sharpe, `1.00` diversity), and **128q Native Heavy-Hex Spin Glass** (`83.0%` satisfied bonds, `71.9%` gap closed at `38,016` shots). | Verified Live (`ibm_kingston`) |
| [`dna_promoter_qerd/`](dna_promoter_qerd/README.md) | **Project 2:** **128q (`64 bp`) Conditional Human Promoter DNA Designer** via bijective nucleotide encoding, noise-pruned snake routing, 1-job RO calibration, and `QER-Res` (`92.6%` Oracle CNN Activity vs. `80.5%` Markov / `75.8%` Real DNA; `40.6%` full biophysical validity vs. `34.4%` Real DNA; `100%` unique & novel). | Verified Live (`ibm_kingston`) |

## Running Tests & Presets

```bash
# Core QERD library unit tests
python -m unittest discover -s tests -v

# Project 1: download S&P 500 CSV into gitignored ./data/, test, & run local benchmark
python -m projects.sp500_qubo_and_spinglass --fetch-data
python -m unittest discover -s projects/sp500_qubo_and_spinglass/tests -v
python -m projects.sp500_qubo_and_spinglass --preset standard

# Project 2: download Human Promoter FASTAs into gitignored ./data/, test, & run local benchmark
python -m projects.dna_promoter_qerd --fetch-data
python -m unittest discover -s projects/dna_promoter_qerd/tests -v
python -m projects.dna_promoter_qerd --preset local_cpu
```
