# Experiment: Latin Hypercube Sampling (LHS)

## Sampling method

Scrambled Latin Hypercube sampling (`scipy.stats.qmc.LatinHypercube`, `optimization="random-cd"`)
over `(tau, gamma, rho)` — stratifies each parameter into equal-probability bins so the
2000-point design fills the space more evenly than uniform random sampling, reducing gaps
and clustering.

## How to run

All scripts read/write relative to the repo root and take no required CLI arguments
(defaults are hardcoded, e.g. `n_samples=4000`, `N=100000`) — run them in order with plain
`python`, from the repository root:

```bash
python "experiments/lhs-sampling/scripts/step1_LHS sampling.py"     # generate ABM data (LHS-sampled params)
python "experiments/lhs-sampling/scripts/Step2_data_split.py"       # train/val/test split by parameter set
python "experiments/lhs-sampling/scripts/step2_data_augmentation.py" # data augmentation
python "experiments/lhs-sampling/scripts/step3_train.py"            # train MLP replicates
python "experiments/lhs-sampling/scripts/step4_validate.py"         # validate on held-out val set
python "experiments/lhs-sampling/scripts/step5_test.py"             # in-sample test
python "experiments/lhs-sampling/scripts/step6_test_mcmc_data.py"   # cross-test: LHS-trained model on MCMC data
python "experiments/lhs-sampling/scripts/step7_test_random.py"      # cross-test: LHS-trained model on Random data
```



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
        │   ├── lhs_test_data_results/          <- Pipeline A in-sample test
        │   ├── mcmc_test_data_results/         <- Pipeline A cross-test on MCMC
        │   └── random_test_data_results/       <- Pipeline A cross-test on Random
        └── lhs_no_augmentation/                <- Pipeline B (all outputs here)
            ├── Scripts/
            ├── trained models/
            ├── validation/
            ├── testing/
            └── mcmc_testing/
```
