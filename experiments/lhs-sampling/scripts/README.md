# Experiment: Latin Hypercube Sampling (LHS)

**Sampling strategy:** Stratified space-filling design using `scipy.stats.qmc.LatinHypercube`  
**EDA notebook:** `../../notebooks/step1_LHS_sampling.ipynb`  
**Config:** `../../configs/lhs_sampling.yaml`

---

## Strategy

Latin Hypercube Sampling (LHS) stratifies the 3D parameter space `(τ, γ, ρ)` into a grid and guarantees **exactly one sample per row and column** across each parameter dimension. This ensures uniform marginal coverage of the full parameter range — no dimension is over- or under-sampled — even with a small simulation budget.

LHS is more space-efficient than random sampling at the same budget, but it still does not specifically target the epidemic threshold. Samples are spread across the whole space, not concentrated where dynamics are most complex.

**Where it sits in the hierarchy:**
- Better than random at uniform coverage 
- Does not concentrate near R₀ = 1 where dynamics are hardest 
- Expected to outperform Random but underperform MCMC for near-threshold prediction

---

## Two pipeline variants

This experiment runs two parallel pipelines so you can directly compare the effect of
data augmentation on emulator performance.

| Variant | Description | Outputs |
|---------|-------------|---------|
| **With augmentation** (main) | Trains on split + augmented data | `out/trained-models/`, `out/results/testing/` |
| **Without augmentation** | Trains on split data only | `out/results/testing/lhs_no_augmentation/` |

---

## Pipeline A — With data augmentation (main pipeline)

Run all scripts from **within the `experiments/lhs-sampling/` folder**.

```bash
cd experiments/lhs-sampling

# Step 1 — Generate ABM simulations using LHS
python "scripts/step1_LHS sampling.py"
# Writes to: data/raw/data_raw.pkl  +  data/raw/data_raw.csv

# Step 2 — Train/test split
python scripts/Step2_data_split.py
# Writes to: data/split/data_split.pkl

# Step 2A — Data augmentation (near-threshold simulations added)
python scripts/step2_data_augmentation.py
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
python scripts/step5_test.py
# Writes to: out/results/testing/  +  out/plots/testing_plots/

# Step 6 — Cross-test on MCMC data (relative MAE is the key metric)
python scripts/step6_test_mcmc_data.py
# Reads test data: experiments/mcmc-sampling/data/augmented/
# Writes to: out/results/testing/mcmc_test_data_results/
#            out/plots/testing_plots/mcmc_test_data_plots/
```

> **Filename note:** `step1_LHS sampling.py` has a space in its name. Always wrap it in double quotes when calling from the terminal, or rename it to `step1_LHS_sampling.py` to avoid shell errors.

---

## Pipeline B — Without data augmentation (comparison run)

The no-augmentation scripts are self-contained inside their own subfolder.
Run these from the **project root** (`abm-epidemic-emulator/`):

```bash
# From project root:

# Step 3 — Train on split data only (no augmentation)
python "experiments/lhs-sampling/out/results/testing/lhs_no_augmentation/Scripts/step3_training_no_augmentation.py"
# Writes to: out/results/testing/lhs_no_augmentation/trained models/

# Step 4 — Validate
python "experiments/lhs-sampling/out/results/testing/lhs_no_augmentation/Scripts/step4_validate_no_aug.py"
# Writes to: out/results/testing/lhs_no_augmentation/validation/

# Step 5 — In-sample test
python "experiments/lhs-sampling/out/results/testing/lhs_no_augmentation/Scripts/step6_test_no_aug.py"
# Writes to: out/results/testing/lhs_no_augmentation/testing/

# Step 6 — Cross-test on MCMC data (no-aug model)
python "experiments/lhs-sampling/out/results/testing/lhs_no_augmentation/Scripts/step6_test_mcmc_no_aug.py"
# Writes to: out/results/testing/lhs_no_augmentation/mcmc_testing/
```

> Steps 1 and 2 are shared between both pipelines — both use the same raw and split data.
> Only the training onwards differs: Pipeline A uses augmented data, Pipeline B uses split data only.

---

## Output folder map

```
experiments/lhs-sampling/
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
        │   ├── (in-sample test results)        <- Pipeline A in-sample test
        │   └── mcmc_test_data_results/         <- Pipeline A cross-test on MCMC
        └── lhs_no_augmentation/                <- Pipeline B (all outputs here)
            ├── Scripts/
            ├── trained models/
            ├── validation/
            ├── testing/
            └── mcmc_testing/
```

---

## Key parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| N (network nodes) | 100,000 | Barabási-Albert graph |
| m (BA attachment) | 10 | ⟨k²⟩/⟨k⟩ ≈ 34 |
| n_samples | 1,000 | LHS draws (one per stratum per dimension) |
| Discrepancy | Monitored | Lower = more uniform coverage |
| tmax | 80 | Simulation time horizon |
| n_timepoints | 80 | Fixed interpolation grid |

R₀ range covered: approximately [0.12, 4.98] — uniform across the full parameter box.

---

## Why relative MAE for cross-testing

The MCMC dataset concentrates near R₀ = 1 with smaller outbreaks. Comparing absolute MAE
counts between in-sample (LHS data, large outbreaks) and cross-test (MCMC data, near-
threshold outbreaks) is not valid — the scales are completely different.

**Relative MAE_I** normalises each sample's error by its own true peak before averaging:

```
Relative MAE_I = mean( MAE_I_i / peak_I_i ) × 100%
```

This is scale-invariant and the correct metric for comparing performance across datasets
with different epidemic size distributions. Sub-critical extinction runs (peak_I < 1) are
excluded automatically.

---

## EDA notebook

Before running the full pipeline, use `../../notebooks/step1_LHS_sampling.ipynb` to:
- Verify LHS discrepancy (should be near zero — confirms good coverage)
- Plot τ vs γ scatter coloured by R₀ value
- Compare R₀ distribution against Random sampling
- Check what fraction of samples fall near the threshold vs. Random
- Inspect sample SIR trajectories across the R₀ spectrum
