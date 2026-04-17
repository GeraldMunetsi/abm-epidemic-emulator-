# Experiment: MCMC Sampling — NUTS near R0 = 1

**Sampling strategy:** PyMC NUTS with Gaussian potential concentrated at the epidemic threshold
**EDA notebook:** `../../notebooks/data_generation_mcmc.ipynb`
**Config:** `../../configs/mcmc_sampling.yaml`

---

## Strategy

This is the most principled of the three strategies. It uses **PyMC's NUTS sampler** (No-U-Turn
Sampler, a form of Hamiltonian Monte Carlo) to draw parameter sets `(tau, gamma, rho)` from a
designed distribution that concentrates near the epidemic threshold R0 = 1.

**Why target R0 = 1?** Near the threshold, SIR dynamics are at their most complex — the system
is poised between extinction and outbreak. Random and LHS sampling waste budget on trivially
simple dynamics far from the threshold. MCMC targets exactly where the emulator needs the most
training data.

**Priors — Uniform over the full parameter space:**

```
tau   ~ Uniform(0.0005, 0.024)
gamma ~ Uniform(0.01,   0.50)
rho   ~ Uniform(0.001,  0.010)
```

**Target potential — Gaussian centred at R0 = 1:**

```
log pi(theta) = -0.5 * ((R0 - 1) / sigma)^2     sigma = 0.30

where  R0 = (tau / gamma) * 34.0   (degree ratio for N=100,000, m=10 BA network)
```

This is **not iterative or adaptive**. NUTS runs once and returns posterior samples. No
importance weights, no resampling, no risk of sample impoverishment.

## Sampler settings

| Parameter             | Value  |
|-----------------------|--------|
| Draws per chain       | 500    |
| Tuning steps          | 2,000  |
| Chains                | 2      |
| Total posterior samples | 1,000 |
| target_accept         | 0.95   |
| sigma (R0 width)      | 0.30   |
| random_seed           | 42     |

---

## Pipeline — run scripts in this order

**This is the only experiment with Step 2A (data augmentation).**
All scripts are run from **within the `experiments/mcmc-sampling/` folder**.

```bash
# Navigate into this experiment folder first
cd experiments/mcmc-sampling

# Step 1 — MCMC NUTS sampling then ABM simulations
# NUTS warm-up runs first (~5-10 min), then 1,000 parameter sets are simulated
python scripts/step1_mcmc_sampling.py
# Writes to: data/raw/data_raw.pkl
#            data/raw/data_raw.csv
#            out/plots/mcmc_sampling_plots/

# Step 2 — Train/test split
python scripts/step2_split.py
# Writes to: data/split/data_split.pkl

# Step 2A — Data augmentation (MCMC only — generates extra near-threshold simulations)
python scripts/step2A_augmented.py
# Writes to: data/augmented/data_augmented.pkl
#            data/augmented/data_augmented.csv
#            out/plots/augmentation_plots/

# Step 3 — Train the MLP emulator (uses augmented data by default)
python scripts/step3_train.py
# Writes to: out/trained-models/
#            out/plots/

# Step 4 — Validate on held-out set
python scripts/step4_validate.py
# Writes to: out/results/validation/
#            out/plots/validation_plots/

# Step 5 — Final test evaluation
python scripts/step5_test.py
# Writes to: out/results/testing/
#            out/plots/testing_plots/
```

---

## Output folder map

```
experiments/mcmc-sampling/
├── scripts/                  <- you are here, run everything from the parent folder
├── data/
│   ├── raw/                  <- Step 1 output
│   ├── split/                <- Step 2 output
│   └── augmented/            <- Step 2A output (unique to MCMC)
└── out/
    ├── trained-models/           <- Step 3 saves .pt model weights here
    ├── plots/
    │   ├── mcmc_sampling_plots/  <- Step 1 trace plots, R0 distribution
    │   ├── augmentation_plots/   <- Step 2A figures
    │   ├── validation_plots/     <- Step 4 figures
    │   └── testing_plots/        <- Step 5 figures
    └── results/
        ├── validation/           <- Step 4 metrics (JSON, TXT)
        └── testing/              <- Step 5 metrics (JSON, TXT)
```

---

## Current results (5 replicates, 300 test samples)

| Metric              | Mean +/- SD       | 95% CI          |
|---------------------|-------------------|-----------------|
| R2 (overall)        | 0.8156 +/- 0.1255 | [0.660, 0.971]  |
| MAE - Infected I(t) | 3,157 +/- 1,172   | [1,702, 4,613]  |
| MAE - Susceptible   | 11,608 +/- 4,553  | -               |
| MAE - Recovered     | 9,210 +/- 3,016   | -               |
| Replicate CV (MAE_I)| 37.1%             | -               |

Full per-replicate details: `out/results/testing/`

---

## MCMC diagnostic checks (run EDA notebook before the full pipeline)

Open `../../notebooks/data_generation_mcmc.ipynb` and verify:

- `az.plot_trace()` — both chains should overlap and show stationarity for tau, gamma, rho
- `az.summary()` — R-hat should be ~1.00 and ESS should be adequate per parameter
- tau vs gamma scatter with R0=1 boundary — samples should cluster along (tau/gamma) = 1/34
- R0 distribution — should be bell-shaped and centred near 1.0
- Near-threshold proportion — fraction of samples with R0 in [0.8, 1.2] should far exceed
  what Random or LHS produces at the same budget

---

## Notes

- sigma = 0.30 controls how tightly samples concentrate near R0 = 1. Tighter (0.10) gives
  more threshold focus but harder NUTS convergence. Looser (0.50) gives more R0 diversity.
  A sensitivity analysis on sigma is a recommended dissertation extension.
- Network: N = 100,000 nodes, m = 10, degree ratio = 34.0.
- 2 replicates per posterior sample keeps ABM simulation cost manageable across 1,000 sets.
