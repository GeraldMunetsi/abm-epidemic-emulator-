# Experiment: Random Sampling (Baseline)

**Sampling strategy:** Uniform random draws from the full parameter box
**EDA notebook:** `../../notebooks/step1_random_sampling.ipynb`
**Config:** `../../configs/random_sampling.yaml`

---

## Strategy

Draws parameter sets `(tau, gamma, rho)` uniformly at random from the 3D parameter space.
No knowledge of the epidemic threshold is used. This is the **baseline** — all other
strategies are benchmarked against it.

**Why it matters:** Most random samples land in sub-threshold (R0 < 1, extinction) or
heavily super-threshold (R0 >> 1, trivial large outbreak) regions — wasting the simulation
budget where dynamics are uninformative. This is exactly the inefficiency that LHS and MCMC
are designed to fix.

---

## Two pipeline variants

This experiment runs two parallel pipelines so you can directly compare the effect of
data augmentation on emulator performance.

| Variant | Description | Outputs |
|---------|-------------|---------|
| **With augmentation** (main) | Trains on split + augmented data | `out/trained-models/`, `out/results/testing/` |
| **Without augmentation** | Trains on split data only | `out/results/uniform_random_no_augmentation/` |

---

## Pipeline A — With data augmentation (main pipeline)

Run all scripts from **within the `experiments/random-sampling/` folder**.

```bash
cd experiments/random-sampling

# Step 1 — Generate ABM simulations (uniform random draws)
python scripts/Step1_Random_sampling.py
# Writes to: data/raw/data_raw.pkl  +  data/raw/data_raw.csv

# Step 2 — Train/test split
python scripts/Step2_data_split.py
# Writes to: data/split/data_split.pkl

# Step 2A — Data augmentation (near-threshold simulations added)
python scripts/step2A_augmented.py
# Writes to: data/augmented/data_augmented.pkl + data_augmented.csv
#            out/plots/augmentation_plots/

# Step 3 — Train the MLP emulator on augmented data
python scripts/step3_train.py
# Writes to: out/trained-models/  +  out/plots/

# Step 4 — Validate
python scripts/step4_validate.py
# Writes to: out/results/validation/  +  out/plots/validation_plots/

# Step 5 — In-sample test
python scripts/step5_test.py
# Writes to: out/results/testing/random_test_data_results/
#            out/plots/testing_plots/

# Step 6 — Cross-test on MCMC data (relative MAE is the key metric)
python scripts/step6_test_on_mcmc_data.py
# Writes to: out/results/testing/mcmc_test_data_results/
#            out/plots/testing_plots/mcmc_test_data_plots/
```

---

## Pipeline B — Without data augmentation (comparison run)

The no-augmentation scripts are self-contained inside their own subfolder.
Run these from the **project root** (`abm-epidemic-emulator/`):

```bash
# From project root:

# Step 3 — Train on split data only (no augmentation)
python experiments/random-sampling/out/results/uniform_random_no_augmentation/scripts/step3_train_no_aug.py
# Writes to: out/results/uniform_random_no_augmentation/trained models/

# Step 4 — Validate
python experiments/random-sampling/out/results/uniform_random_no_augmentation/scripts/step4_validate_no_aug.py
# Writes to: out/results/uniform_random_no_augmentation/validation/

# Step 5 — In-sample test
python experiments/random-sampling/out/results/uniform_random_no_augmentation/scripts/step5_test_no_aug.py
# Writes to: out/results/uniform_random_no_augmentation/testing/

# Step 6 — Cross-test on MCMC data (no-aug model)
python experiments/random-sampling/out/results/uniform_random_no_augmentation/scripts/step6_test_mcmc_no_aug.py
# Writes to: out/results/uniform_random_no_augmentation/mcmc_testing_no_aug/
```

> Steps 1 and 2 are shared between both pipelines — both use the same raw and split data.
> Only the training onwards differs: Pipeline A uses augmented data, Pipeline B uses split data only.

---

## Output folder map

```
experiments/random-sampling/
├── scripts/                    <- Pipeline A scripts live here
├── data/
│   ├── raw/                    <- Step 1 output (shared)
│   ├── split/                  <- Step 2 output (shared)
│   └── augmented/              <- Step 2A output (Pipeline A only)
└── out/
    ├── trained-models/             <- Pipeline A trained models (.pt)
    ├── plots/
    │   ├── augmentation_plots/     <- Step 2A figures
    │   ├── validation_plots/       <- Step 4 figures
    │   └── testing_plots/
    │       ├── (in-sample plots)   <- Step 5 figures
    │       └── mcmc_test_data_plots/ <- Step 6 cross-test figures
    └── results/
        ├── validation/                         <- Pipeline A validation metrics
        ├── testing/
        │   ├── random_test_data_results/       <- Pipeline A in-sample test
        │   └── mcmc_test_data_results/         <- Pipeline A cross-test
        └── uniform_random_no_augmentation/     <- Pipeline B (all outputs here)
            ├── scripts/
            ├── trained models/
            ├── validation/
            ├── testing/
            └── mcmc_testing_no_aug/
```

---

## Why relative MAE for cross-testing

The MCMC dataset concentrates near R0 = 1 with smaller outbreaks. Comparing absolute MAE
counts between in-sample (Random data, large outbreaks) and cross-test (MCMC data, near-
threshold outbreaks) is not valid — the scales are completely different.

**Relative MAE_I** normalises each sample's error by its own true peak before averaging:

```
Relative MAE_I = mean( MAE_I_i / peak_I_i ) x 100%
```

This is scale-invariant and the correct metric for comparing performance across datasets
with different epidemic size distributions. Sub-critical extinction runs (peak_I < 1) are
excluded automatically.

---

## Expected performance

Random sampling is the **weakest strategy by design** — it produces the highest relative
MAE and lowest R2 compared to LHS and MCMC at the same sample budget, especially for near-
threshold dynamics (R0 ~ 1). Use these results as the lower bound when comparing strategies.

The augmentation vs no-augmentation comparison tests whether adding near-threshold
simulations post-hoc can partially compensate for the poor initial sampling distribution.
