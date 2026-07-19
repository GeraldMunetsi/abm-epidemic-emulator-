# Experiment: MCMC Sampling — NUTS near R0 = 1

## Sampling method

PyMC NUTS (Hamiltonian Monte Carlo) with uniform priors on `(tau, gamma, rho)`, plus a
Gaussian potential `-0.5 * ((R0 - 1) / sigma)^2` added via `pm.Potential` that concentrates
posterior draws near the epidemic threshold R₀=1 (`sigma` controls how tight the
concentration is), rather than covering the full parameter box. Referred to as
Near-Threshold Sampling (NTS) in the dissertation — it deliberately over-samples the
regime where the emulator's predictions matter most (near the outbreak/no-outbreak
boundary) instead of treating all of parameter space as equally important.

## How to run

All scripts read/write relative to the repo root and take no required CLI arguments
(defaults are hardcoded, e.g. `initial_samples=10000`, `N=100000`) — run them in order
with plain `python`, from the repository root:

```bash
python "experiments/mcmc-sampling/scripts/step1_mcmc_sampling.py"           # generate ABM data (NUTS-sampled params near R0=1)
python "experiments/mcmc-sampling/scripts/step2_split.py"                   # train/val/test split by parameter set
python "experiments/mcmc-sampling/scripts/step2A_augmented.py"              # data augmentation
python "experiments/mcmc-sampling/scripts/step3_train.py"                   # train MLP replicates
python "experiments/mcmc-sampling/scripts/step4_validate.py"                # validate on held-out val set
python "experiments/mcmc-sampling/scripts/step5_test2.py"                   # in-sample test
python "experiments/mcmc-sampling/scripts/step6_test_lhs_data.py"           # cross-test: MCMC-trained model on LHS data
python "experiments/mcmc-sampling/scripts/step6_test_random_sampling_data.py" # cross-test: MCMC-trained model on Random data
```

`step3`–`step6` accept optional flags (e.g. `--epochs`, `--models_dir`, `--data`) that
override the hardcoded defaults — see each script's `argparse` block if you need to point
at different inputs/outputs. Supporting/diagnostic scripts (not part of the main pipeline):
`Ratio generation.py` (estimates the BA network's `<k²>/<k>` ratio used to compute R₀),
`conservation_check.py` and `other_plot.py` (post-hoc diagnostic plots), and
`plot_training_curves_all.py`.

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
