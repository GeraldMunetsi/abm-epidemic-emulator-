# ABM Epidemic Emulator — MLP Surrogate for SIR on Barabási-Albert Networks

> **MPhil Population Health Science (Health Data Science) · University of Cambridge**
> Dissertation: *Data-efficient emulation strategies for individual-based epidemic models*
> Author: Gerald Munetsi · 2026

---

## Overview

Agent-Based Models (ABMs) capture realistic epidemic dynamics — household structure, heterogeneous contact networks, stochastic transmission — but they are computationally expensive. Running enough simulations to train a surrogate model or explore intervention scenarios is often infeasible, especially in real-time public health decision support.

This project builds a **physics-constrained MLP emulator** for a stochastic SIR model on a **Barabási-Albert (BA) scale-free network**, and answers the question:

The study aims to answer the following questions.
1.	Can neural networks emulate epidemic agent-based models?  
2.	Which data generation methods provide the best epidemic data training for ABMs  while minimising the number of training simulations?
3.	How can we build a neural network emulator that can generalize well to unseen epidemic scenarios while requiring minimal ABM training simulations?


Three sampling strategies are compared head-to-head. The emulator architecture is held fixed across all three — only the sampling changes.

---

## The Epidemic Model

The underlying ABM is a stochastic SIR process simulated via `EoN.fast_SIR()` on a Barabási-Albert network. Each simulation is defined by three parameters:

| Parameter | Symbol | Range | Description |
|-----------|--------|-------|-------------|
| Transmission rate | τ | [0.0024, 0.05] | Per-contact probability of infection per unit time |
| Recovery rate | γ | [0.07, 0.50] | Rate at which infected nodes recover |
| Initial seed fraction | ρ | [0.001, 0.010] | Fraction of nodes initially infected |

The epidemic threshold on a BA network is:

```
R₀ = (τ / γ) × (⟨k²⟩ / ⟨k⟩)
```

where `⟨k²⟩/⟨k⟩` is the degree-heterogeneity correction factor — hub nodes in a BA graph dramatically lower the epidemic threshold compared to a homogeneous (Erdős-Rényi) network. The network has N = 10,000 nodes and BA attachment parameter m = 5.

Each ABM run produces time series `S(t)`, `I(t)`, `R(t)` across 200 timepoints.

---

## Sampling Strategy Comparison

This is the **core scientific contribution** of the project. All three strategies sample the same 3D parameter space `(τ, γ, ρ)` but differ fundamentally in *where* they place simulations.

### 1. Random Sampling (Baseline)
**Script:** `src/Random_sampling_data_generation.py`
**Notebook (EDA):** `notebooks/step1_random_sampling.ipynb`

Draws parameter sets uniformly at random from the parameter box. Simple to implement and unbiased, but makes no use of any knowledge about where interesting dynamics occur. Most simulations will land in sub-threshold or trivially large-outbreak regions, wasting the simulation budget.

- N = 100,000 nodes, m = 10
- 2,000 initial samples, 2 stochastic replicates per parameter set
- No adaptive component

### 2. Latin Hypercube Sampling (LHS)
**Script:** `src/LHS_sampling.py` + `src/step1_data_generation.py`
**Notebook (EDA):** `notebooks/step1_LHS_sampling.ipynb`

Stratifies the parameter space into a grid and ensures exactly one sample per row and column — guaranteeing uniform marginal coverage even with small budgets. More space-efficient than random sampling, but still does not target the epidemic threshold region specifically.

- N = 10,000 nodes, m = 5
- Uses `scipy.stats.qmc.LatinHypercube`
- R₀ range covered: approximately [0.12, 4.98]
- Discrepancy metric monitored to verify coverage quality

### 3. MCMC Sampling — NUTS near R₀ = 1
**Script:** `src/step1_MCMC_sampling.py`
**Notebook (EDA):** `notebooks/data_generation_mcmc.ipynb`

The most principled strategy. Uses **PyMC's NUTS sampler** (No-U-Turn Sampler, a form of Hamiltonian Monte Carlo) to draw parameter sets `(τ, γ, ρ)` from a designed distribution that concentrates near the epidemic threshold R₀ = 1.

**Priors — Uniform over the full parameter space:**
```
τ ~ Uniform(0.0005, 0.024)
γ ~ Uniform(0.01,   0.50)
ρ ~ Uniform(0.001,  0.010)
```

**Target potential — Gaussian centred at R₀ = 1:**
```
log π(θ) = −0.5 × ((R₀ − 1) / σ)²     σ = 0.30

where  R₀ = (τ / γ) × ⟨k²⟩/⟨k⟩  =  (τ / γ) × 34.0
```

This `pm.Potential` term shapes the posterior so that samples concentrate near the epidemic threshold, where dynamics are most complex and most informative for the emulator. Sub-threshold extinctions and large unconstrained outbreaks are down-weighted — not by filtering, but by the geometry of the posterior.

**Sampler settings:**
- 500 draws × 2 chains = **1,000 posterior samples**
- 2,000 tuning steps (warm-up)
- `target_accept = 0.95` (high acceptance rate for NUTS near a sharp potential)
- N = 100,000 nodes, m = 10, tmax = 80, 2 replicates per parameter set

This is **not iterative or adaptive** — NUTS runs once and returns a set of samples from the designed posterior. It is fundamentally different from importance resampling: there are no weights, no resampling steps, and no risk of sample impoverishment. The NUTS sampler navigates the posterior geometry directly using gradient information.

---

## Emulator Architecture

The MLP emulator takes `(τ, γ, ρ)` as input and outputs full `S(t)`, `I(t)`, `R(t)` time series. It is **physics-constrained** — `S + I + R = N` is enforced exactly by construction, not as a penalty.

**Defined in:** `src/step0_model.py` and `src/step0_model1.py`

```
[τ, γ, ρ]  →  StandardRFF  →  Fusion MLP  →  S-decoder (B-spline, monotone ↓)
                                           →  g(t)-decoder (B-spline + sigmoid)
                                           →  I = (N − S) · g(t)
                                           →  R = (N − S) · (1 − g(t))
```

| Component | Detail |
|-----------|--------|
| **StandardRFF** | Random Fourier Features — encodes all 3 params jointly into 128-dim space, captures τ/γ interactions (i.e. R₀) |
| **Fusion MLP** | 128 → latent_dim, learns nonlinear combinations |
| **S-decoder** | Monotone-decreasing B-spline via cumulative product of sigmoid retention rates — S(t) can only decrease |
| **g(t)-decoder** | Free B-spline + sigmoid → g ∈ (0,1), allows bell-curve I(t) for R₀ > 1 |
| **Conservation** | I and R derived from S, not predicted directly — exact conservation by design |

---

## Current Results

Results below are from the **MCMC Sampling** strategy (NUTS near R₀ = 1), 5 replicates, 300 test samples.

| Metric | Value | 95% CI |
|--------|-------|--------|
| R² (overall) | 0.8156 ± 0.1255 | [0.660, 0.971] |
| MAE — Infected I(t) | 3,157 ± 1,172 | [1,702, 4,613] |
| MAE — Susceptible S(t) | 11,608 ± 4,553 | — |
| MAE — Recovered R(t) | 9,210 ± 3,016 | — |
| Replicate CV (MAE_I) | 37.1%  | — |

**Interpretation:** R² = 0.82 indicates reasonable emulation quality overall, with the infected compartment I(t) — the key public health metric — best predicted. High CV (37.1%) across replicates indicates sensitivity to random initialisation; this is a known issue when training near the epidemic threshold where dynamics are most complex.

> Full per-replicate breakdown: `out/results/FINAL_DISSERTATION_RESULTS.txt`
> Validation results: `out/results/VALIDATION_REPORT.txt`

---

## Project Structure

```
abm-epidemic-emulator/
├── configs/                        # Hyperparameter configs — one per sampling strategy
│   ├── random_sampling.yaml
│   ├── lhs_sampling.yaml
│   └── mcmc_sampling.yaml
├── docs/                           # Methodology notes and pipeline diagrams
│   └── utils_flow.md
├── experiments/                    # One self-contained folder per sampling strategy
│   │
│   ├── random-sampling/
│   │   ├── scripts/                # ← Run all scripts from here, in order
│   │   │   ├── Step1_Random_sampling.py
│   │   │   ├── Step2_data_split_no_aug.py
│   │   │   ├── step3_train.py
│   │   │   ├── step4_validate.py
│   │   │   ├── step5_test.py
│   │   │   ├── step0_model.py      # MLP architecture
│   │   │   └── utils.py
│   │   ├── data/
│   │   │   ├── raw/                # ABM simulations output
│   │   │   └── split/              # Train / test split
│   │   └── out/
│   │       ├── trained-models/     # Saved .pt weights
│   │       ├── plots/
│   │       │   ├── validation_plots/
│   │       │   └── testing_plots/
│   │       └── results/
│   │           ├── validation/
│   │           └── testing/
│   │
│   ├── lhs-sampling/
│   │   ├── scripts/
│   │   │   ├── step1_LHS sampling.py   # ⚠ space in filename — quote when running
│   │   │   ├── Step2_data_split_no_aug.py
│   │   │   ├── step3_train.py
│   │   │   ├── step4_validate.py
│   │   │   ├── step5_test.py
│   │   │   ├── step0_model.py
│   │   │   └── utils.py
│   │   ├── data/
│   │   │   ├── raw/
│   │   │   └── split/
│   │   └── out/
│   │       ├── trained-models/
│   │       ├── plots/
│   │       └── results/
│   │
│   └── mcmc-sampling/              # Only experiment with data augmentation (Step 2A)
│       ├── scripts/
│       │   ├── step1_mcmc_sampling.py
│       │   ├── step2_split.py
│       │   ├── step2A_augmented.py     # ← unique to MCMC
│       │   ├── step3_train.py
│       │   ├── step4_validate.py
│       │   ├── step5_test.py
│       │   ├── step0_model.py
│       │   └── utils.py
│       ├── data/
│       │   ├── raw/
│       │   ├── split/
│       │   └── augmented/              # ← unique to MCMC
│       └── out/
│           ├── trained-models/
│           ├── plots/
│           │   ├── mcmc_sampling_plots/
│           │   ├── augmentation_plots/
│           │   ├── validation_plots/
│           │   └── testing_plots/
│           └── results/
│               ├── validation/
│               └── testing/
│
├── notebooks/                      # EDA notebooks — interactive exploration
│   ├── step1_random_sampling.ipynb
│   ├── step1_LHS_sampling.ipynb
│   ├── step2_split.ipynb
│   ├── step2_random_data_split_no_aug.ipynb
│   ├── Step2A_augmented_data.ipynb
│   ├── data_generation_mcmc.ipynb
│   └── python_basics.ipynb
├── src/                            # Source library (shared utilities and model definitions)
│   ├── experiment_paths.py         # Central path registry — all scripts import from here
│   ├── step0_model1.py             # MLP architecture (RFF + B-spline + physics conservation)
│   ├── utils_SIR.py                # Dataloaders, metrics, normalisation, EarlyStopping
│   ├── Average_ratio.py            # ⟨k²⟩/⟨k⟩ ratio computation for BA network
│   ├── Verification_test_data.py
│   └── Verification_train_data_before_spliting.py
├── .github/
│   └── workflows/
│       └── python-app.yml
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Clone and set up

```bash
git clone https://github.com/<your-username>/abm-epidemic-emulator.git
cd abm-epidemic-emulator
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **Note:** Large binary files (`.pkl`, `.pt`, `.npy`) are excluded from Git via `.gitignore`. Regenerate them by running the pipeline below, or request them from the author.

### 2. Run an experiment — navigate into the experiment folder and run scripts in order

Each experiment is fully self-contained. All scripts are inside `scripts/` and all outputs go to the experiment's own `data/` and `out/` subfolders. **Do not run scripts from `src/` directly.**

---

#### Strategy 1 — Random Sampling (Baseline)

```bash
cd experiments/random-sampling

python scripts/Step1_Random_sampling.py          # Generate ABM simulations → data/raw/
python scripts/Step2_data_split_no_aug.py        # Train/test split → data/split/
python scripts/step3_train.py                    # Train emulator → out/trained-models/
python scripts/step4_validate.py                 # Validation metrics → out/results/validation/
python scripts/step5_test.py                     # Final test → out/results/testing/
```

---

#### Strategy 2 — Latin Hypercube Sampling (LHS)

```bash
cd experiments/lhs-sampling

python "scripts/step1_LHS sampling.py"        
python scripts/Step2_data_split_no_aug.py
python scripts/step3_train.py
python scripts/step4_validate.py
python scripts/step5_test.py
```

---

#### Strategy 3 — MCMC Sampling (NUTS near R₀ = 1)

```bash
cd experiments/mcmc-sampling

python scripts/step1_mcmc_sampling.py            # NUTS warm-up (~5–10 min) then ABM runs
python scripts/step2_split.py                    # Train/test split → data/split/
python scripts/step2A_augmented.py               # Data augmentation → data/augmented/  ← MCMC only
python scripts/step3_train.py                    # Train emulator → out/trained-models/
python scripts/step4_validate.py                 # Validation → out/results/validation/
python scripts/step5_test.py                     # Final test → out/results/testing/
```

> **MCMC note:** The Step 1 script runs PyMC's NUTS sampler — 2,000 tuning steps and 500 draws per chain (×2 chains = 1,000 posterior samples) — before launching ABM simulations. Budget 10–15 minutes for Step 1.

> **Step 2A — data augmentation** is unique to the MCMC experiment. It generates additional simulations around the near-threshold region to increase training density there.

---

### 3. Explore interactively (recommended before any full pipeline run)

```bash
jupyter notebook notebooks/step1_random_sampling.ipynb    # EDA: random sampling coverage
jupyter notebook notebooks/step1_LHS_sampling.ipynb       # EDA: LHS parameter coverage
jupyter notebook notebooks/data_generation_mcmc.ipynb     # EDA: MCMC traces, R₀ posterior
jupyter notebook notebooks/Step2A_augmented_data.ipynb    # EDA: augmentation effects
```

---

## Data

| File | Location | Description |
|------|----------|-------------|
| `epidemic_data_age_adaptive_sobol.csv` | `data/raw/` | Simulated S(t), I(t), R(t) trajectories (MCMC NUTS strategy) |
| `epidemic_data_age_adaptive_sobol_parameters.csv` | `data/raw/` | Corresponding (τ, γ, ρ) input parameters |
| `epidemic_data_age_adaptive_sobol_split_augmented.csv` | `data/processed/` | Train/test split with data augmentation |
| `epidemic_data_age_adaptive_sobol_train_params.csv` | `data/processed/` | Training set parameter values |

> `.pkl` files are binary serialisations for fast loading — same content as the CSVs. Excluded from Git tracking.

---

## Dependencies

All Python dependencies are pinned in `requirements.txt`. Key packages:

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.10.0 | MLP training |
| `numpy` / `pandas` / `scipy` | 2.2.6 / 2.3.3 / 1.15.3 | Data stack |
| `scikit-learn` | 1.7.2 | Preprocessing, metrics |
| `EoN` | 1.2 | Epidemic network simulation (ABM) |
| `networkx` | 3.4.2 | Barabási-Albert graph construction |
| `pymc` + `arviz` | 5.25.1 / 0.23.4 | Bayesian/MCMC pipeline |
| `matplotlib` | 3.10.8 | Visualisations |
| `tqdm` | 4.67.3 | Progress tracking |

Python version: **3.10.10**
R: `deSolve` required for `src/demo-mcmc-sampler.R`

---


## Citation

```bibtex
@mastersthesis{munetsi2026abm,
  author  = {Munetsi, Gerald},
  title   = {Data-efficient emulation strategies for individual-based epidemic models},
  school  = {University of Cambridge},
  year    = {2026},
  program = {MPhil in Population Health Science (Health Data Science)}
}
```

---

## License

Academic research use only. Please contact the author before reuse or adaptation.

---

*MPhil Population Health Science · Health Data Science · University of Cambridge · 2026*
