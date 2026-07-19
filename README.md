# ABM Epidemic Emulator — MLP Surrogate for SIR on [Barabási-Albert Networks](https://en.wikipedia.org/wiki/Barab%C3%A1si%E2%80%93Albert_model)

This project trains a physics-constrained MLP to emulate the output of a stochastic SIR
agent-based model (ABM) run on Barabási–Albert networks, replacing expensive network
simulations with a fast surrogate. The emulator maps epidemic parameters `(tau, gamma, rho)`
(infection rate, recovery rate, initial fraction infected) to full S/I/R trajectories over
time. Three parameter-sampling strategies are compared as separate, self-contained
experiments, since how the training parameter space is sampled turns out to strongly affect
how well the emulator generalises:

| Strategy | Folder | Idea |
|---|---|---|
| **Random** | [`experiments/random-sampling/`](experiments/random-sampling/) | Uniform random draws over the parameter box — baseline, no structure imposed on coverage. |
| **LHS** | [`experiments/lhs-sampling/`](experiments/lhs-sampling/) | Latin Hypercube (scrambled) — stratifies each parameter for space-filling coverage. |
| **MCMC (NUTS)** | [`experiments/mcmc-sampling/`](experiments/mcmc-sampling/) | PyMC HMC sampling biased toward the near-critical epidemic threshold (R₀≈1), called Near-Threshold Sampling (NTS) in the dissertation. |

Each experiment folder has its own README with a fuller description of its sampling method
and the exact commands to run its pipeline end-to-end. `experiments/Ablation studies/`
tests removing individual architecture components (RFF, B-spline, conservation loss), and
`experiments/Regression/` runs cross-strategy statistical comparisons.

## Project Structure

Each sampling strategy lives in its own **fully self-contained** experiment folder —
scripts, data, and model definition are all local to that folder (there is no shared
`src/` library; `step0_model.py` and `utils.py` are duplicated per experiment so each
one can evolve independently). There is also no top-level `data/`, `docs/`, or
`notebooks/` folder — all data and EDA live inside the relevant experiment.

```
abm-epidemic-emulator/
├── configs/                        # Hyperparameter configs — one per sampling strategy
│   ├── random_sampling.yaml
│   ├── lhs_sampling.yaml
│   ├── mcmc_sampling.yaml
│   └── mcmc_adaptive_IS.yaml
├── experiments/                    # One self-contained folder per sampling strategy
│   │
│   ├── random-sampling/
│   │   ├── scripts/                # ← Pipeline A: run from experiments/random-sampling/
│   │   │   ├── Step1_Random_sampling.py
│   │   │   ├── Step2_data_split.py
│   │   │   ├── step2A_augmented.py          # Data augmentation (Pipeline A)
│   │   │   ├── step3_train.py
│   │   │   ├── step4_validate.py
│   │   │   ├── step5_test.py                # in-sample test
│   │   │   ├── step6_test_on_mcmc_data.py   # cross-test: Random model on MCMC data
│   │   │   ├── step6_test_on_lhs_data.py    # cross-test: Random model on LHS data
│   │   │   ├── step0_model.py               # MLP architecture
│   │   │   └── utils.py
│   │   ├── data/
│   │   │   ├── raw/                # Step 1 output (shared by both pipelines)
│   │   │   ├── split/              # Step 2 output (shared by both pipelines)
│   │   │   └── augmented/          # Step 2A output (Pipeline A only)
│   │   └── out/
│   │       ├── trained models/     # Pipeline A saved .pt weights
│   │       ├── plots/
│   │       │   ├── augmentation_plots/
│   │       │   ├── validation_plots/
│   │       │   └── testing_plots/
│   │       └── results/
│   │           ├── validation/
│   │           ├── testing/
│   │           └── uniform_random_no_augmentation/  # Pipeline B (all outputs)
│   │               └── scripts/
│   │
│   ├── lhs-sampling/
│   │   ├── scripts/                # ← Pipeline A: run from experiments/lhs-sampling/
│   │   │   ├── step1_LHS sampling.py
│   │   │   ├── Step2_data_split.py
│   │   │   ├── step2_data_augmentation.py   # Data augmentation
│   │   │   ├── step3_train.py
│   │   │   ├── step4_validate.py
│   │   │   ├── step5_test.py                # in-sample test
│   │   │   ├── step6_test_mcmc_data.py      # cross-test: LHS model on MCMC data
│   │   │   ├── step7_test_random.py         # cross-test: LHS model on Random data
│   │   │   ├── step0_model.py
│   │   │   └── utils.py
│   │   ├── data/
│   │   │   ├── raw/
│   │   │   ├── split/
│   │   │   └── augmented/          # Step 2A output
│   │   └── out/
│   │       ├── trained-models/
│   │       ├── plots/
│   │       │   ├── augmentation_plots/
│   │       │   ├── validation_plots/
│   │       │   └── testing_plots/
│   │       └── results/
│   │           ├── validation/
│   │           ├── testing/
│   │           └── lhs_no_augmentation/     # Pipeline B (all outputs)
│   │               └── Scripts/
│   │
│   ├── mcmc-sampling/
│   │   ├── scripts/                # ← Pipeline A: run from experiments/mcmc-sampling/
│   │   │   ├── step1_mcmc_sampling.py
│   │   │   ├── step2_split.py
│   │   │   ├── step2A_augmented.py          # Data augmentation
│   │   │   ├── step3_train.py
│   │   │   ├── step4_validate.py
│   │   │   ├── step5_test2.py               # in-sample test
│   │   │   ├── step6_test_lhs_data.py       # cross-test: MCMC model on LHS data
│   │   │   ├── step6_test_random_sampling_data.py  # cross-test: MCMC model on Random data
│   │   │   ├── step0_model.py
│   │   │   └── utils.py
│   │   ├── data/
│   │   │   ├── raw/
│   │   │   ├── split/
│   │   │   └── augmented/          # Step 2A
│   │   └── out/
│   │       ├── trained-models/
│   │       ├── plots/
│   │       │   ├── mcmc_sampling_plots/
│   │       │   ├── augmentation_plots/
│   │       │   ├── validation_plots/
│   │       │   └── testing_plots/
│   │       └── results/
│   │           ├── validation/
│   │           ├── testing/
│   │           │   ├── results_on_lhs_sampled_data/
│   │           │   └── results_on_random_sampled_data/
│   │           └── mcmc_no_augmentation/    # Pipeline B (all outputs)
│   │               └── Scripts/
│   │
│   ├── Ablation studies/           # Architecture ablations (RFF / B-spline / conservation)
│   │   ├── step0_model.py
│   │   ├── step1_train.py
│   │   ├── step2_validate.py
│   │   ├── step3_plot_training_curves.py
│   │   ├── step3_ablation_test_table.py
│   │   ├── utils.py
│   │   └── out/
│   │
│   └── Regression/                 # Cross-strategy statistical validation
│       ├── Results_Combined.ipynb
│       ├── data/
│       └── Analysis_plots/
│
├── .github/
│   └── workflows/
│       └── python-app.yml
├── .gitignore
├── requirements.txt
└── README.md
```
