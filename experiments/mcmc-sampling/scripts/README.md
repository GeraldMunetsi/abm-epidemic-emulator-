# Experiment: MCMC Sampling — NUTS near R0 = 1

**Sampling strategy:** PyMC NUTS with Gaussian potential concentrated at the epidemic threshold
**EDA notebook:** `../../notebooks/data_generation_mcmc.ipynb`
**Config:** `../../configs/mcmc_sampling.yaml`

---

## Strategy

This is the most principled of the three strategies. It uses **PyMC's NUTS sampler** (No-U-Turn
Sampler, a form of Hamiltonian Monte Carlo) to draw parameter sets `(tau, gamma, rho)` from a
designed distribution that concentrates near the epidemic threshold R₀ = 1.

**Why target R₀ = 1?** Near the threshold, SIR dynamics are at their most complex — the system
is poised between extinction and outbreak. Random and LHS sampling waste budget on trivially
simple dynamics far from the threshold. MCMC targets exactly where the emulator needs the most
training data.

**Priors — Uniform over the full parameter space:**

```
tau   ~ Uniform(0.0005, 0.024)
gamma ~ Uniform(0.01,   0.50)
rho   ~ Uniform(0.001,  0.010)
```

**Target potential — Gaussian centred at R₀ = 1:**

```
log π(θ) = -0.5 * ((R₀ - 1) / σ)²     σ = 0.30

where  R₀ = (tau / gamma) * 34.0   (degree ratio for N=100,000, m=10 BA network)
```

This is **not iterative or adaptive**. NUTS runs once and returns posterior samples. No
importance weights, no resampling, no risk of sample impoverishment.

## Sampler settings

| Parameter               | Value  |
|-------------------------|--------|
| Draws per chain         | 500    |
| Tuning steps            | 2,000  |
| Chains                  | 2      |
| Total posterior samples | 1,000  |
| target_accept           | 0.95   |
| sigma (R₀ width)        | 0.30   |
| random_seed             | 42     |

---

## Two pipeline variants

This experiment runs two parallel pipelines so you can directly compare the effect of
data augmentation on emulator performance.

| Variant | Description | Outputs |
|---------|-------------|---------|
| **With augmentation** (main) | Trains on split + augmented data | `out/trained-models/`, `out/results/testing/` |
| **Without augmentation** | Trains on split data only | `out/results/testing/mcmc_no_augmentation/` |

---

## Pipeline A — With data augmentation (main pipeline)

Run all scripts from **within the `experiments/mcmc-sampling/` folder**.

```bash
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

# Step 2A — Data augmentation (generates extra near-threshold simulations)
python scripts/step2A_augmented.py
# Writes to: data/augmented/data_augmented.pkl
#            data/augmented/data_augmented.csv
#            out/plots/augmentation_plots/

# Step 3 — Train the MLP emulator on augmented data
python scripts/step3_train.py
# Writes to: out/trained-models/  +  out/plots/

# Step 4 — Validate on held-out set
python scripts/step4_validate.py
# Writes to: out/results/validation/  +  out/plots/validation_plots/

# Step 5 — Final test evaluation
python scripts/step5_test2.py
# Writes to: out/results/testing/  +  out/plots/testing_plots/

# Step 6 — Cross-test on LHS data (relative MAE is the key metric)
python scripts/step6_test_lhs_data.py
# Reads test data: experiments/lhs-sampling/data/split/
# Writes to: out/results/testing/results_on_lhs_sampled_data/
#            out/plots/testing_plots/LHS_test_data/

# Step 6 — Cross-test on Random data
python scripts/step6_test_random_sampling_data.py
# Reads test data: experiments/random-sampling/data/split/
# Writes to: out/results/testing/results_on_random_sampled_data/
#            out/plots/testing_plots/random_test_data/
```

---

## Pipeline B — Without data augmentation (comparison run)

The no-augmentation scripts are self-contained inside their own subfolder.
Run these from the **project root** (`abm-epidemic-emulator/`):

```bash
# From project root:

# Step 3 — Train on split data only (no augmentation)
python experiments/mcmc-sampling/out/results/testing/mcmc_no_augmentation/Scripts/step3_training_no_aug.py
# Writes to: out/results/testing/mcmc_no_augmentation/trained models/

# Step 4 — Validate
python experiments/mcmc-sampling/out/results/testing/mcmc_no_augmentation/Scripts/step4_validate_no_aug.py
# Writes to: out/results/testing/mcmc_no_augmentation/validation/

# Step 5 — In-sample test
python experiments/mcmc-sampling/out/results/testing/mcmc_no_augmentation/Scripts/step5_test_no_aug.py
# Writes to: out/results/testing/mcmc_no_augmentation/testing/

# Step 6 — Cross-test on LHS data (no-aug model)
python experiments/mcmc-sampling/out/results/testing/mcmc_no_augmentation/Scripts/step6_lhs_test_no_aug.py
# Writes to: out/results/testing/mcmc_no_augmentation/lhs_testing_no_aug/

# Step 6 — Cross-test on Random data (no-aug model)
python experiments/mcmc-sampling/out/results/testing/mcmc_no_augmentation/Scripts/step6_random_test_no_aug.py
# Writes to: out/results/testing/mcmc_no_augmentation/random_testing_no_aug/
```

> Steps 1 and 2 are shared between both pipelines — both use the same raw and split data.
> Only the training onwards differs: Pipeline A uses augmented data, Pipeline B uses split data only.

---

## Output folder map

```
experiments/mcmc-sampling/
├── scripts/                      <- Pipeline A scripts live here
├── data/
│   ├── raw/                      <- Step 1 output (shared)
│   ├── split/                    <- Step 2 output (shared)
│   └── augmented/                <- Step 2A output (Pipeline A only)
└── out/
    ├── trained-models/               <- Pipeline A trained models (.pt)
    ├── plots/
    │   ├── mcmc_sampling_plots/      <- Step 1 trace plots, R₀ distribution
    │   ├── augmentation_plots/       <- Step 2A figures
    │   ├── validation_plots/         <- Step 4 figures
    │   └── testing_plots/
    │       ├── (in-sample plots)     <- Step 5 figures
    │       ├── LHS_test_data/        <- Step 6 cross-test on LHS
    │       └── random_test_data/     <- Step 6 cross-test on Random
    └── results/
        ├── validation/                             <- Pipeline A validation metrics
        ├── testing/
        │   ├── (in-sample test results)            <- Pipeline A in-sample test
        │   ├── results_on_lhs_sampled_data/        <- Pipeline A cross-test on LHS
        │   ├── results_on_random_sampled_data/     <- Pipeline A cross-test on Random
        │   └── mcmc_no_augmentation/               <- Pipeline B (all outputs here)
        │       ├── Scripts/
        │       ├── trained models/
        │       ├── validation/
        │       ├── testing/
        │       ├── lhs_testing_no_aug/
        │       └── random_testing_no_aug/
        └── (other results)
```

---

## Why relative MAE for cross-testing

LHS and Random datasets span the full parameter space, producing many large outbreaks with
peak I in the tens of thousands. The MCMC training data concentrates near R₀ = 1 where
outbreaks are smaller. Comparing absolute MAE counts across datasets with different epidemic
size distributions is statistically incorrect — it is the classic problem of dividing a mean
by a mean, which is distorted by Jensen's inequality (`E[X/Y] ≠ E[X]/E[Y]`).

**Relative MAE_I** corrects for this by normalising each sample's error against its own true
peak *before* averaging:

```
Relative MAE_I = mean over valid samples of ( MAE_I_i / peak_I_i ) × 100%

where  MAE_I_i  = mean absolute error on I(t) for sample i
       peak_I_i = ground-truth maximum of I(t) for sample i
       valid    = samples where peak_I_i >= 1  (excludes sub-critical extinction)
```

This gives a **percentage of peak infected count** — the emulator's typical error expressed
as a fraction of how big the outbreak actually was. A result of 15% means the model's I(t)
curve is, on average, off by 15% of the true epidemic peak, regardless of whether that peak
was 1,000 or 100,000 individuals. This makes cross-dataset comparisons scientifically valid.

---

## Current results (5 replicates, 300 test samples)

| Metric              | Mean ± SD         | 95% CI          |
|---------------------|-------------------|-----------------|
| R² (overall)        | 0.8156 ± 0.1255   | [0.660, 0.971]  |
| MAE — Infected I(t) | 3,157 ± 1,172     | [1,702, 4,613]  |
| MAE — Susceptible   | 11,608 ± 4,553    | —               |
| MAE — Recovered     | 9,210 ± 3,016     | —               |
| Replicate CV (MAE_I)| 37.1%             | —               |

Full per-replicate details: `out/results/testing/`

---

## MCMC diagnostic checks (run EDA notebook before the full pipeline)

Open `../../notebooks/data_generation_mcmc.ipynb` and verify:

- `az.plot_trace()` — both chains should overlap and show stationarity for tau, gamma, rho
- `az.summary()` — R-hat should be ~1.00 and ESS should be adequate per parameter
- tau vs gamma scatter with R₀=1 boundary — samples should cluster along (tau/gamma) = 1/34
- R₀ distribution — should be bell-shaped and centred near 1.0
- Near-threshold proportion — fraction of samples with R₀ in [0.8, 1.2] should far exceed
  what Random or LHS produces at the same budget

---

## Notes

- sigma = 0.30 controls how tightly samples concentrate near R₀ = 1. Tighter (0.10) gives
  more threshold focus but harder NUTS convergence. Looser (0.50) gives more R₀ diversity.
  A sensitivity analysis on sigma is a recommended dissertation extension.
- Network: N = 100,000 nodes, m = 10, degree ratio = 34.0.
- 2 replicates per posterior sample keeps ABM simulation cost manageable across 1,000 sets.
