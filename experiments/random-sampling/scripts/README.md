# Experiment: Random Sampling (Baseline)

**Sampling strategy:** Uniform random draws from the full parameter box  
**EDA notebook:** `../../notebooks/step1_random_sampling.ipynb`  
**Config:** `../../configs/random_sampling.yaml`

---

## Strategy

Draws parameter sets `(τ, γ, ρ)` uniformly at random from the parameter space. No knowledge of the epidemic threshold is used — samples are equally likely to land anywhere in the 3D parameter box. This is the **baseline** against which LHS and MCMC strategies are benchmarked.

**Why it matters:** Most random samples will land in sub-threshold (R₀ < 1, extinction) or heavily super-threshold (R₀ >> 1, trivial outbreak) regions where epidemic dynamics are uninformative. The emulator wastes simulation budget learning these uninformative regions. This is the exact inefficiency that LHS and MCMC are designed to fix.

---

## Pipeline — run scripts in this order

All scripts are run from **within the `experiments/random-sampling/` folder**. Outputs go automatically to the correct subfolders.

```bash
# Navigate to this experiment folder first
cd experiments/random-sampling

# Step 1 — Generate ABM simulations
python scripts/Step1_Random_sampling.py
# Output → data/raw/data_raw.pkl  +  data/raw/data_raw.csv

# Step 2 — Train/test split (no augmentation for this strategy)
python scripts/Step2_data_split_no_aug.py
# Output → data/split/data_split.pkl

# Step 3 — Train the MLP emulator
python scripts/step3_train.py
# Output → out/trained-models/  +  out/plots/

# Step 4 — Validate on held-out set
python scripts/step4_validate.py
# Output → out/results/validation/  +  out/plots/validation_plots/

# Step 5 — Final test evaluation
python scripts/step5_test.py
# Output → out/results/testing/  +  out/plots/testing_plots/
```

---

## Output folder map

```
experiments/random-sampling/
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
| n_samples | 1,000 | Uniform random draws |
| n_replicates | 2 | Stochastic replicates per parameter set |
| tmax | 80 | Simulation time horizon |
| n_timepoints | 80 | Fixed interpolation grid |

---

## Expected performance

Random sampling is the **weakest strategy** by design. It will produce the highest MAE and lowest R² compared to LHS and MCMC at the same sample budget — especially for epidemic trajectories near the threshold (R₀ ≈ 1), where dynamics are most complex and most important for public health decisions. Use this as the lower bound when comparing strategies.

---

## EDA notebook

Before running the full pipeline, use `../../notebooks/step1_random_sampling.ipynb` to:
- Plot the R₀ distribution of sampled parameters
- Visualise τ vs γ scatter (with R₀ = 1 boundary line)
- Inspect example SIR trajectories by R₀ class (extinction / outbreak)
- Check what fraction of samples fall near the threshold (R₀ ∈ [0.8, 1.2])
