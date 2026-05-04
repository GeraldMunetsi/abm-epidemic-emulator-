# Experiment: Latin Hypercube Sampling (LHS)

**Sampling strategy:** Stratified space-filling design using `scipy.stats.qmc.LatinHypercube`  
**EDA notebook:** `../../notebooks/step1_LHS_sampling.ipynb`  
**Config:** `../../configs/lhs_sampling.yaml`

---

## Strategy

Latin Hypercube Sampling (LHS) stratifies the 3D parameter space `(τ, γ, ρ)` into a grid and guarantees **exactly one sample per row and column** across each parameter dimension. This ensures uniform marginal coverage of the full parameter range — no dimension is over- or under-sampled — even with a small simulation budget.

LHS is more space-efficient than random sampling at the same budget, but it still does not specifically target the epidemic threshold. Samples are spread across the whole space, not concentrated where dynamics are most complex.

**Where it sits in the hierarchy:**
- Better than random at uniform coverage ✓
- Does not concentrate near R₀ = 1 where dynamics are hardest ✗
- Expected to outperform Random but underperform MCMC for near-threshold prediction

---

## Pipeline — run scripts in this order

All scripts are run from **within the `experiments/lhs-sampling/` folder**. Outputs go automatically to the correct subfolders.

```bash
# Navigate to this experiment folder first
cd experiments/lhs-sampling

# Step 1 — Generate ABM simulations using LHS sampling method 

python "scripts/step1_LHS sampling.py"
# Output → data/raw/data_raw.pkl  +  data/raw/data_raw.csv

# Step 2 — Train/test split (no augmentation for this strategy)
python scripts/Step2_data_split_no_aug.py
# Output → data/split/data_split.pkl

# Step 3 — Train the MLP emulator
python scripts/step3_train.py
# Output → out/trained-models/ + out/plots/

# Step 4 — Validate on held-out set
python scripts/step4_validate.py
# Output → out/results/validation/  +  out/plots/validation_plots/

# Step 5 — Final test evaluation
python scripts/step5_test.py
# Output → out/results/testing/  +  out/plots/testing_plots/
```

> **Filename note:** `step1_LHS sampling.py` has a space in its name (not an underscore). Always wrap it in double quotes when calling from the terminal, or rename it to `step1_LHS_sampling.py` to avoid this.

---

## Output folder map

```
experiments/lhs-sampling/
├── scripts/          ← you are here
├── data/
│   ├── raw/          ← Step 1 writes here
│   └── split/        ← Step 2 writes here
└── out/
    ├── trained-models/   ← Step 3 writes .pt weights here
    ├── plots/
    │   ├── validation_plots/   ← Step 4 figures
    │   └── testing_plots/      ← Step 5 figures
    └── results/
        ├── validation/         ← Step 4 metrics (JSON, TXT)
        └── testing/            ← Step 5 metrics (JSON, TXT)
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

## Step 6 — Cross-testing on MCMC data

After completing the in-sample pipeline (Steps 1–5), run the cross-test to see how the LHS-trained model handles near-threshold dynamics it was not specifically trained on.

```bash
# Still inside experiments/lhs-sampling/
python scripts/step6_test_mcmc_data.py
# Reads models  : out/trained-models/
# Reads test data: experiments/mcmc-sampling/data/augmented/
# Writes results : out/results/testing/mcmc_test_data_results/
# Writes plots   : out/plots/testing_plots/mcmc_test_data_plots/
```

### Why relative MAE is the right metric here

The MCMC dataset is concentrated near R₀ = 1 — average outbreak size is much smaller than in the LHS dataset (which spans the full parameter space). Comparing absolute MAE counts across datasets with different epidemic size distributions is misleading. A model that produces absolute MAE of 500 on MCMC data and 5,000 on LHS data is not necessarily 10× worse on LHS — the outbreaks are just 10× larger.

**Relative MAE_I** normalises per-sample before averaging:

```
Relative MAE_I = mean( MAE_I_i / peak_I_i ) × 100%
```

This gives a percentage error relative to each epidemic's own peak, making results directly comparable across datasets and across strategies.

---

## EDA notebook

Before running the full pipeline, use `../../notebooks/step1_LHS_sampling.ipynb` to:
- Verify LHS discrepancy (should be near zero — confirms good coverage)
- Plot τ vs γ scatter coloured by R₀ value
- Compare R₀ distribution against Random sampling
- Check what fraction of samples fall near the threshold vs. Random
- Inspect sample SIR trajectories across the R₀ spectrum
