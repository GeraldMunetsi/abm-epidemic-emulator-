"""
experiment_paths.py
===================
Single source of truth for all file paths across the project.

Every script (step1, step3, step4, step5) imports from here.
This is what makes the scripts communicate — they all resolve
paths from the same definitions.

Usage
-----
    from experiment_paths import get_paths, get_training_data, makedirs

    paths = get_paths('mcmc')
    makedirs('mcmc')

    data_file = get_training_data('mcmc')   # augmented > split > raw
    model_dir = paths['models']
    plots_dir = paths['plots']

Folder layout per experiment
-----------------------------
    experiments/<name>/
    ├── data/
    │   ├── raw/          ← step1 saves here
    │   ├── split/        ← step2 (split notebook) saves here
    │   └── augmented/    ← step2A (MCMC only) saves here
    ├── out/
    │   ├── trained-models/   ← step3 saves model weights (.pt)
    │   ├── plots/            ← step3, step4, step5 save figures
    │   └── results/
    │       ├── validation/   ← step4 saves reports + CSVs here
    │       └── testing/      ← step5 saves reports + CSVs here
    └── README.md
"""

from pathlib import Path

# ── Experiment registry ────────────────────────────────────────────────────────
EXPERIMENTS = {
    'random' : Path('experiments/random-sampling'),
    'lhs'    : Path('experiments/lhs-sampling'),
    'mcmc'   : Path('experiments/mcmc-sampling'),
}

# Standard filenames written by step1 scripts
RAW_PKL = 'data_raw.pkl'
RAW_CSV = 'data_raw.csv'

# Standard filenames written by step2 (split notebook)
SPLIT_PKL = 'data_split.pkl'

# Standard filenames written by step2A (augmentation, MCMC only)
AUGMENTED_PKL = 'data_augmented.pkl'
AUGMENTED_CSV = 'data_augmented.csv'


# ── Path resolver ──────────────────────────────────────────────────────────────

def get_paths(experiment: str) -> dict:
    """
    Return all standardised Path objects for a given experiment.

    Parameters
    ----------
    experiment : str
        One of 'random', 'lhs', 'mcmc'

    Returns
    -------
    dict with keys:
        base, data_raw, data_split, data_augmented,
        models, plots, results_val, results_test
    """
    if experiment not in EXPERIMENTS:
        raise ValueError(
            f"Unknown experiment '{experiment}'. "
            f"Choose from: {list(EXPERIMENTS.keys())}"
        )

    base = EXPERIMENTS[experiment]

    return {
        'base'          : base,
        'data_raw'      : base / 'data' / 'raw',
        'data_split'    : base / 'data' / 'split',
        'data_augmented': base / 'data' / 'augmented',   # MCMC only
        'models'        : base / 'out' / 'trained-models',
        'plots'         : base / 'out' / 'plots',
        'results_val'   : base / 'out' / 'results' / 'validation',
        'results_test'  : base / 'out' / 'results' / 'testing',
    }


def makedirs(experiment: str) -> dict:
    """
    Create all directories for an experiment (safe, idempotent).

    Returns the paths dict so you can chain:
        paths = makedirs('mcmc')
    """
    paths = get_paths(experiment)
    for key, path in paths.items():
        if key != 'base':
            path.mkdir(parents=True, exist_ok=True)
    return paths


def get_training_data(experiment: str) -> Path:
    """
    Return the most processed data file available for training.

    Priority: augmented (.pkl) > split (.pkl) > raw (.pkl)

    This is what step3_train.py calls so it always picks up
    the best available data automatically.

    Raises
    ------
    FileNotFoundError if no data exists yet for this experiment.
    """
    paths = get_paths(experiment)

    # 1. Augmented (MCMC only — step2A output)
    aug_file = paths['data_augmented'] / AUGMENTED_PKL
    if aug_file.exists():
        print(f"  [data] Using augmented data: {aug_file}")
        return aug_file

    # 2. Split (step2 notebook output)
    split_file = paths['data_split'] / SPLIT_PKL
    if split_file.exists():
        print(f"  [data] Using split data: {split_file}")
        return split_file

    # 3. Raw (step1 output) — fallback
    raw_file = paths['data_raw'] / RAW_PKL
    if raw_file.exists():
        print(f"  [data] Using raw data: {raw_file}")
        return raw_file

    raise FileNotFoundError(
        f"No data found for experiment '{experiment}'.\n"
        f"Run the corresponding step1 script first:\n"
        f"  random → python src/Step1_Random_sampling.py --experiment random\n"
        f"  lhs    → python src/step1_LHS_sampling.py   --experiment lhs\n"
        f"  mcmc   → python src/step1_MCMC_sampling.py  --experiment mcmc"
    )


def describe(experiment: str):
    """Print a summary of what exists for an experiment."""
    paths = get_paths(experiment)
    print(f"\n{'='*55}")
    print(f"  Experiment : {experiment}  ({paths['base']})")
    print(f"{'='*55}")

    checks = [
        ('Raw data',       paths['data_raw']       / RAW_PKL),
        ('Split data',     paths['data_split']      / SPLIT_PKL),
        ('Augmented data', paths['data_augmented']  / AUGMENTED_PKL),
        ('Trained models', paths['models']),
        ('Plots',          paths['plots']),
        ('Val results',    paths['results_val']),
        ('Test results',   paths['results_test']),
    ]

    for label, path in checks:
        exists = '✓' if path.exists() else '✗'
        print(f"  {exists}  {label:<18} {path}")
    print()
