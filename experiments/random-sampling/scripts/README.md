# Experiment: Random Sampling (Baseline)

## Sampling method

Plain uniform random draws of `(tau, gamma, rho)` over the parameter box, via
`numpy.random.default_rng().uniform(...)` — no stratification or space-filling structure
imposed. This is the baseline against which LHS and MCMC sampling are compared.

## How to run

All scripts read/write relative to the repo root and take no required CLI arguments
(defaults are hardcoded, e.g. `n_samples=4000`, `N=100000`) — run them in order with plain
`python`, from the repository root:

```bash
python "experiments/random-sampling/scripts/Step1_Random_sampling.py"   # generate ABM data (uniform random params)
python "experiments/random-sampling/scripts/Step2_data_split.py"        # train/val/test split by parameter set
python "experiments/random-sampling/scripts/step2A_augmented.py"        # data augmentation
python "experiments/random-sampling/scripts/step3_train.py"             # train MLP replicates
python "experiments/random-sampling/scripts/step4_validate.py"          # validate on held-out val set
python "experiments/random-sampling/scripts/step5_test.py"              # in-sample test
python "experiments/random-sampling/scripts/step6_test_on_lhs_data.py"  # cross-test: Random-trained model on LHS data
python "experiments/random-sampling/scripts/step6_test_on_mcmc_data.py" # cross-test: Random-trained model on MCMC data
```

`step3`–`step6` accept optional flags (e.g. `--epochs`, `--models_dir`, `--data`) that
override the hardcoded defaults — see each script's `argparse` block if you need to point
at different inputs/outputs.

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
        │   ├── mcmc_test_data_results/         <- Pipeline A cross-test on MCMC
        │   └── lhs_test_data_results/          <- Pipeline A cross-test on LHS
        └── uniform_random_no_augmentation/     <- Pipeline B (all outputs here)
            ├── scripts/
            ├── trained models/
            ├── validation/
            ├── testing/
            └── mcmc_testing_no_aug/
```
