# Regression Analysis

Statistical validation of the emulator's generalization across sampling strategies
(Random / LHS / MCMC) used to generate training data, with and without data
augmentation. The unit of analysis is a **train→test condition**, replicated
across independent model fits, with `relative_MAE_I` (relative MAE on the
Infected trajectory) as the primary outcome.

## Contents

- **`Results_Combined.ipynb`** — the full analysis notebook (see breakdown below).
- **`data/`** — per-condition replicate results.
  - `replicate_results_{TRAIN}_to_{TEST}_aug{0,1}.csv` — one file per
    train-strategy × test-strategy × augmentation combination (train ∈ {LHS,
    MCMC, UNIFORM_RANDOM}, test ∈ {LHS, MCMC, UNIFORM_RANDOM}); condition
    count = —, replicates per condition = —.
  - `per_sample_peak_*.csv` — per-sample epidemic peak (`I_max`) predictions
    for a subset of conditions.
  - `master_results.csv` — all `replicate_results_*.csv` files concatenated
    (built by the notebook's first cell; row count = —). Columns include
    `train_strategy`, `test_strategy`,
    `augmentation`, `in_domain`, `relative_MAE_I`, `R2_I`/`R2_S`/`R2_R`/
    `R2_overall`, `MAE_I`/`MAE_S`/`MAE_R`, `RMSE`, `MSE`, `mean_peak_I`,
    `peak_I`, `n_train_simulations`, `n_test_samples`, `condition_id`
    (`train_test_augmentation`), `model_path`, `training_epoch`.
- **`Analysis_plots/`** — every figure and text summary the notebook produces
  (regenerated on re-run; safe to delete and rebuild).

## What the notebook does

1. **Data assembly** — loads and concatenates all `replicate_results_*.csv`
   into `master_results.csv`; derives `condition_id` and
   `log_relative_MAE_I`.
2. **Regression models** — OLS on `log_relative_MAE_I ~ train_strategy +
   test_strategy + augmentation + train_strategy:augmentation`, both
   ordinary and with cluster-robust SEs (clustered on `condition_id`), plus
   VIF checks for multicollinearity.
3. **Distributional checks** — histograms of `R2_I` and `log(R2_I)`,
   regression diagnostics (linearity/normality), log-transform
   justification for the skewed MAE outcome.
4. **Effect sizes & hypothesis testing** — joint F-tests, Cohen's d,
   interaction confidence intervals, ranking of strategies with 95% CIs.
5. **Cluster/generalization diagnostics** — cluster adequacy & independence
   checks, in-domain vs. out-of-domain generalization gap, mean relative MAE
   and coefficient-of-variation by strategy.
6. **Domain plots** — stochastic SIR behaviour near the epidemic threshold,
   sampling-strategy coverage, network- vs. homogeneous-mixing comparison,
   distribution of epidemic peak `I_max` by test strategy, clamped B-spline
   basis functions, 2-component Gaussian mixture on pooled epidemic peak.
7. **Compute cost** — training/simulation time comparison across strategies
   (parallel and heatmap views).

## Regenerating

Run the notebook top to bottom from the repo root (it resolves
`experiments/Regression/data` automatically) — every cell recreates its
output under `Analysis_plots/`.
