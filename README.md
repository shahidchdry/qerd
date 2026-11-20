# Contrastive Field Probing for Finite-Depth Ising Quantum Circuits (`qerd`)

Implementation, controlled $6$--$14$-qubit reciprocity and shot-SNR verification suites, live $128$--$156$-qubit IBM Heron R2 (`ibm_kingston`) hardware audits, and self-contained manuscript for:

> **Contrastive Field Probing for Finite-Depth Ising Quantum Circuits: Non-Equilibrium Reciprocity, Shot-Budget Scaling, and $128$--$156$-Qubit Hardware Audits**

---

## Repository Structure

```text
qerd/
├── qerd/                                          # Core library: QER-Grad, QER-Res, StabilizedAdam, QERLayer
│   ├── __init__.py                                # Public API exports & aliases
│   ├── __main__.py                                # `python -m qerd` end-to-end self-test suite
│   ├── schedules.py                               # Diffusion alpha_bar schedules & Ising graph topologies
│   ├── backends.py                                # NumPy, PyTorch (CPU/CUDA), PennyLane, Qiskit/QPU backends
│   ├── calibration.py                             # ReadoutGainCalibrator (forward & inverse MMSE OLS)
│   ├── gradients.py                               # QERGradientEstimator (2-ckt QER-Grad & 3-ckt QER-Res)
│   ├── optimizers.py                              # StabilizedAdam (beta_2=0.98) & QEROptimizer
│   ├── denoiser.py                                # QuantumIsingDenoiser (sampling & commuting diagonal readout)
│   ├── trainer.py                                 # QERDTrainer high-level shot-budget training loop
│   └── torch_layer.py                             # QERLayer PyTorch autograd surrogate VJP & STE bridge
├── projects/                                      # Live 128-156-qubit IBM Heron R2 (`ibm_kingston`) benchmarks
│   ├── sp500_qubo_and_spinglass/                  # Project 1: 128q & 156q Native Heavy-Hex Frustrated Spin Glass
│   └── dna_promoter_qerd/                         # Project 2: 128q Fixed-TATA Human Promoter DNA Benchmark
├── experiments/                                   # Reproducible benchmark runners, forensic audits & figure scripts
│   ├── eval_hardware_controls.py                  # Tables 1-3, 5 & Figs 2, 3, 5 (reciprocity, SNR, 12q CNOT, STE)
│   ├── eval_reciprocity_and_stepper.py            # Table 1 bottom (3-seed 12q QER-Grad vs QER-Res MSE cosine)
│   ├── eval_spinglass_attribution.py              # Table 4 (12q Zero-AE Stride-3 Cluster Spin Glass)
│   ├── eval_matched_controls.py                   # Table 4 (13q/14q hidden-label qubits & Binomial noise control)
│   ├── eval_conditioning_and_opt.py               # Fig 4 & StabilizedAdam beta_2 factor isolation
│   ├── extract_live_distributions.py              # Table 7 & 11 (raw QPU bitstrings, inverse-Ising Gibbs, ledger)
│   ├── run_live_oracle_and_debias_128q.py         # Table 7 & Fig 6 (live 128q oracle controls & batch-Z2 debias)
│   ├── audit_saved_hardware_checks.py             # Table 7-8 (hardware-drift regression, overlap identity, Hebbian)
│   ├── plot_live_generative_audit.py              # Fig 6 (`analysis/fig_live_generative_audit.pdf`)
│   ├── verify_paper_consistency.py                # Automated paper consistency, Job-ID & cross-ref verifier
│   └── benchmark_mnist.py                         # Shared vision/autoencoder helper utilities
├── analysis/                                      # Active paper figures (Figs 2-6) & structured JSON audit data
├── tests/                                         # Fast offline unit & transpilation test suite
│   ├── test_qerd.py
│   └── test_qiskit_backend.py
├── data/                                          # Local dataset cache (Human Promoter FASTA & S&P 500 returns)
├── legacy/                                        # Archived prior-draft vision sweeps (`paper_suite/`) & intermediates
├── paper.tex                                      # Self-contained 23-page LaTeX manuscript (all 11 tables inline)
├── paper.pdf                                      # Compiled manuscript PDF
└── pyproject.toml
```

---

## Reproducing & Verifying All Paper Benchmarks (`0` QPU Seconds Required Offline)

All unit tests, offline benchmark verifications, figure regeneration scripts, and automated manuscript checks run deterministically from the repository root:

```bash
# 1. Run built-in QERD self-tests & all unit test suites (100% offline)
python -m unittest discover -s tests -v
python -m unittest discover -s projects/sp500_qubo_and_spinglass/tests -v
python -m unittest discover -s projects/dna_promoter_qerd/tests -v

# 2. Verify live 128q-156q checkpoint coupler-sign recovery (100.0% / 94.9%),
#    classical inverse-Ising baselines (Bethe MM & Besag PL), Phase-B oracle/debias runs,
#    and forensic hardware-drift / overlap-identity audits offline
python experiments/extract_live_distributions.py --offline
python experiments/run_live_oracle_and_debias_128q.py --offline
python experiments/audit_saved_hardware_checks.py --offline

# 3. Regenerate all active manuscript figures from verified JSON artifacts
python experiments/eval_hardware_controls.py --replot
python experiments/plot_live_generative_audit.py

# 4. Run the automated manuscript consistency verifier (checks all 21 Job IDs,
#    35 citations, 38 cross-references, 5 figure files, and unified Gap-Closed arithmetic)
python experiments/verify_paper_consistency.py

# 5. Re-run controlled 6-14q simulation benchmarks from scratch (optional; --quick for 1-seed check)
python -u experiments/eval_hardware_controls.py --quick
python -u experiments/eval_reciprocity_and_stepper.py --quick
python -u experiments/eval_spinglass_attribution.py --quick
python -u experiments/eval_matched_controls.py --quick
python -u experiments/eval_conditioning_and_opt.py --quick

# 6. Compile paper.pdf
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex
```
