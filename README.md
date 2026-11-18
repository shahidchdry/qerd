# Quantum Equilibrium-Response Diffusion (QERD)

Implementation, `128`-qubit IBM Heron R2 hardware applications, verification controls, and manuscript for **QERD** (*Quantum Equilibrium-Response Diffusion: $\mathcal{O}(1)$-Circuit Contrastive Optimization for Ising Generative Models on Noisy NISQ Hardware*).

---

## Repository Structure

```text
optimizer-qpu/
├── qerd/                         # Modular QERD / CFP / StabilizedAdam / QERLayer library
│   ├── __init__.py               # Public API exports & aliases
│   ├── __main__.py               # `python -m qerd` quick self-test suite
│   ├── schedules.py              # Diffusion alpha_bar schedules & Ising graph topologies
│   ├── backends.py               # NumPy, PyTorch (CPU/CUDA), PennyLane, Qiskit/QPU backends
│   ├── calibration.py            # ReadoutGainCalibrator (forward & inverse MMSE OLS mitigation)
│   ├── gradients.py              # QERGradientEstimator (O(1) Contrastive Field Probing)
│   ├── optimizers.py             # StabilizedAdam & QEROptimizer (`Adam-QER`)
│   ├── denoiser.py               # QuantumIsingDenoiser (sampling & observable readout)
│   ├── trainer.py                # QERDTrainer high-level shot-budget training loop
│   └── torch_layer.py            # QERLayer custom PyTorch autograd surrogate VJP bridge
├── projects/                     # Live 128-qubit IBM Heron R2 application benchmarks
│   ├── sp500_qubo_and_spinglass/ # Project 1: 32q S&P 500 QUBO & 128q Heavy-Hex Spin Glass
│   └── dna_promoter_qerd/        # Project 2: 128q Conditional Human Promoter DNA Designer
├── data/                         # Gitignored local cache (auto-downloaded: MNIST, S&P 500 CSV, DNA FASTA)
├── experiments/                  # Paper benchmark runners & controlled verification suites
│   ├── benchmark_mnist.py        # Single-dataset & ablation runner (MNIST/Fashion-MNIST/Digits/QPU)
│   ├── run_paper_benchmarks.py   # Full multi-seed paper suite orchestrator -> paper_suite/
│   ├── eval_gradient_fidelity.py # Surrogate-vs-MSE & shot-SNR verification
│   ├── eval_spinglass_attribution.py
│   ├── eval_capacity_bottleneck.py
│   ├── eval_conditioning_and_opt.py
│   ├── eval_reciprocity_and_stepper.py
│   └── eval_matched_controls.py
├── tests/                        # Fast unit & smoke verification suite
│   ├── test_qerd.py
│   └── test_qiskit_backend.py
├── paper.tex                     # LaTeX manuscript source (27-page paper + Appendices A-F)
├── paper.pdf                     # Compiled paper PDF
├── paper_suite/                  # Paper LaTeX table macros (`paper_tables.tex`) & figures
├── analysis/                     # Verification figures & structured JSON results
└── pyproject.toml
```

---

## Live 128-Qubit IBM Quantum Highlights (`ibm_kingston`)

Both real-world applications in [`projects/`](projects/README.md) scale **beyond exact $2^N$ classical simulation limits (`~26q`)** using noise-pruned subgraphs, 1-job live marginal readout calibration, `cal_mode="inverse"` MMSE shrinkage, and `O(1)` `QER-Res` training (`~38,016` shots per task):

* **Project 1 — [S&P 500 Markowitz QUBO & Heavy-Hex Spin Glass](projects/sp500_qubo_and_spinglass/README.md):**
  * `32q` Real S&P 500 Portfolio (`2022–2025` OOS): **`-0.0945`** QUBO cost (`145.4%` optimization gap vs. SA `-0.0828`), **`+\$6,506`** profit (`1.01` Sharpe vs. Buy&Hold `0.99`), `1.00` portfolio diversity.
  * `128q` Native Heavy-Hex Spin Glass (`135` edges, `0` SWAPs): **`83.0%`** satisfied bonds (`-89.06` mean / `-103.0` best energy, `71.9%` gap closed).
* **Project 2 — [Conditional Human Promoter DNA Designer](projects/dna_promoter_qerd/README.md):**
  * `128q` (`64 bp` bijective encoding, `0` classical autoencoder): **`92.6%`** Oracle CNN promoter activity (`93.8%` pass rate vs. `80.5%` Markov & `75.8%` Real Human DNA), **`40.6%`** full biophysical validity (vs. `34.4%` Real DNA), **`100%` unique & novel** designs.

---

## Installation & Quick Verification

Using an existing Python 3.9+ environment with NumPy (plus optional PyTorch / PennyLane / Qiskit):

```bash
# Editable install (optional — modules also run directly from repo root)
pip install -e .

# Run built-in QERD self-tests (NumPy, Torch backend, decoupled API, & QERLayer)
python -m qerd

# Run unit tests (core library + Project 1 + Project 2)
python -m unittest discover -s tests -v
python -m unittest discover -s projects/sp500_qubo_and_spinglass/tests -v
python -m unittest discover -s projects/dna_promoter_qerd/tests -v

# Download real S&P 500 & Human Promoter datasets locally into gitignored ./data/ (also auto-fetched on run)
python -m projects.sp500_qubo_and_spinglass --fetch-data
python -m projects.dna_promoter_qerd --fetch-data
```

### Minimal Python API Example

```python
from qerd import QuantumIsingDenoiser, QERDTrainer, QEROptimizer, QERConfig

model = QuantumIsingDenoiser(
    data_shape=12,
    cond_dim=10,
    topology="all",
    n_layers=2,
    backend="auto",  # selects "torch" (CUDA if available), "numpy", "pennylane", or "qiskit"
    seed=42,
)
optimizer = QEROptimizer(model.parameters_dict(), QERConfig(lr_max=0.12, beta=0.45, clip_bound=0.90))
trainer = QERDTrainer(model, optimizer)
# trainer.train(latents_train, onehot_conds, max_shots=80_000, shots_per_run=20, batch_size=10)
```

### IBM Quantum Runtime (`qiskit-ibm-runtime`) Batched Hardware Example

```python
from qiskit_ibm_runtime import QiskitRuntimeService
from qerd import QuantumIsingDenoiser, QERDTrainer, select_hardware_subgraph

service = QiskitRuntimeService()
qpu = service.backend("ibm_kingston")  # 156-qubit Heron backend
edges, phys_qubits = select_hardware_subgraph(qpu, n_qubits=32)

with QuantumIsingDenoiser(
    data_shape=32,
    cond_dim=4,
    edges=edges,
    n_layers=2,
    cal_mode="inverse",
    real_qpu=True,
    backend="qiskit",
    device_name="ibm_kingston",
    device_kwargs={
        "backend": qpu,
        "execution_mode": "job",          # "job" (Open Plan), "batch", or "session"
        "initial_layout": phys_qubits,    # zero-SWAP layout matching native edges
        "max_circuits_per_pub": 500,
    },
) as model:
    trainer = QERDTrainer(model)
    # Transpiles once, binds (2*M, P) angle matrices in vectorized NumPy, & submits 1 batched PUB/step
```

---

## Reproducing Paper Benchmarks & Figures

Run all experiment scripts from the **repository root** (`optimizer-qpu/`) so dataset caches (`./data`, auto-downloaded via `torchvision`) and generated tables/figures (`./paper_suite`, `./analysis`) resolve cleanly:

### 1. Fast Offline Smoke Check (CPU, `sklearn` Digits)
```bash
python -u experiments/benchmark_mnist.py --dataset digits --latent-dim 8 --max-shots 16000 --save-dir ./paper_suite/smoke_check
```

### 2. Full Multi-Seed Benchmark Suite (`paper_suite/`)
Generates Tables and Figures across MNIST, Fashion-MNIST, Noise Sweeps, and Unwhitened Controls:
```bash
python -u experiments/run_paper_benchmarks.py
```

### 3. Controlled Verification & Attribution Suites (`analysis/`)
```bash
# Surrogate-vs-MSE decomposition, QER-Res, and batch-window shot SNR
python -u experiments/eval_gradient_fidelity.py

# Zero-AE 12-qubit cluster spin glass factorial
python -u experiments/eval_spinglass_attribution.py

# Capacity bottleneck P=220 -> 545k manifold probes
python -u experiments/eval_capacity_bottleneck.py

# Zero-W_cond, hybrid QERLayer, & optimizer grid controls
python -u experiments/eval_conditioning_and_opt.py
python -u experiments/eval_reciprocity_and_stepper.py
python -u experiments/eval_matched_controls.py
```
*(Analysis scripts accept `--quick` for rapid 1-seed checks and `--replot` to regenerate figures from JSON caches.)*

### 4. Compiling the Paper PDF
```bash
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex
```
