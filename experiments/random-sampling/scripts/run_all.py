"""
run_all.py — Master pipeline runner for the Random Sampling experiment.

Runs Pipeline A (with data augmentation) followed by Pipeline B (no augmentation)
in the correct order, from the correct working directories.

HOW TO RUN:
    cd experiments/random-sampling
    python scripts/run_all.py

To run only one pipeline:
    python scripts/run_all.py --pipeline A
    python scripts/run_all.py --pipeline B
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Directory layout 
SCRIPTS_DIR    = Path(__file__).parent                              # …/random-sampling/scripts/
EXPERIMENT_DIR = SCRIPTS_DIR.parent                                 # …/random-sampling/
PROJECT_ROOT   = EXPERIMENT_DIR.parent.parent                       # …/abm-epidemic-emulator/
NO_AUG_SCRIPTS = (EXPERIMENT_DIR
                  / "out/results/uniform_random_no_augmentation/scripts")

# Helper ──────────────────────────────────────────────────────────────────
def run_step(label: str, script: Path, cwd: Path) -> None:
    """Run a single pipeline step.  Aborts the whole run on failure."""
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  {label}")
    print(f"  script : {script.relative_to(PROJECT_ROOT)}")
    print(f"  cwd    : {cwd.relative_to(PROJECT_ROOT)}")
    print(sep)

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(cwd),
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n FAILED after {elapsed:.1f}s — stopping run.")
        sys.exit(result.returncode)

    print(f"\n Done in {elapsed:.1f}s")


# Pipeline A — With data augmentation 
def run_pipeline_A():
    print("\n" + "═" * 60)
    print("  PIPELINE A — With data augmentation")
    print("  All scripts run from: experiments/random-sampling/")
    print("═" * 60)

    steps = [
        ("Step 1  — Generate ABM simulations (uniform random)",
         SCRIPTS_DIR / "Step1_Random_sampling.py",
         EXPERIMENT_DIR),

        ("Step 2  — Train/test split",
         SCRIPTS_DIR / "Step2_data_split.py",
         EXPERIMENT_DIR),

        ("Step 2A — Data augmentation (near-threshold simulations)",
         SCRIPTS_DIR / "step2A_augmented.py",
         EXPERIMENT_DIR),

        ("Step 3  — Train MLP emulator on augmented data",
         SCRIPTS_DIR / "step3_train.py",
         EXPERIMENT_DIR),

        ("Step 4  — Validate on held-out set",
         SCRIPTS_DIR / "step4_validate.py",
         EXPERIMENT_DIR),

        ("Step 5  — In-sample test (Random test data)",
         SCRIPTS_DIR / "step5_test.py",
         EXPERIMENT_DIR),

        ("Step 6  — Cross-test on MCMC data (relative MAE)",
         SCRIPTS_DIR / "step6_test_on_mcmc_data.py",
         EXPERIMENT_DIR),
    ]

    for label, script, cwd in steps:
        run_step(label, script, cwd)

    print("\n Pipeline A complete.\n")


# Pipeline B Without data augmentation 
def run_pipeline_B():
    print("\n" + "═" * 60)
    print("  PIPELINE B — Without data augmentation (comparison)")
    print("  Scripts run from: project root")
    print("  Uses same raw + split data as Pipeline A (Steps 1–2 shared)")
    print("═" * 60)

    steps = [
        ("Step 3  — Train MLP emulator on split data only (no augmentation)",
         NO_AUG_SCRIPTS / "step3_train_no_aug.py",
         PROJECT_ROOT),

        ("Step 4  — Validate (no-aug model)",
         NO_AUG_SCRIPTS / "step4_validate_no_aug.py",
         PROJECT_ROOT),

        ("Step 5  — In-sample test (no-aug model)",
         NO_AUG_SCRIPTS / "step5_test_no_aug.py",
         PROJECT_ROOT),

        ("Step 6  — Cross-test on MCMC data (no-aug model, relative MAE)",
         NO_AUG_SCRIPTS / "step6_test_mcmc_no_aug.py",
         PROJECT_ROOT),
    ]

    for label, script, cwd in steps:
        run_step(label, script, cwd)

    print("\n Pipeline B complete.\n")


#  Entry point 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Random Sampling emulator pipeline (A=augmented, B=no-augmentation)."
    )
    parser.add_argument(
        "--pipeline",
        choices=["A", "B", "both"],
        default="both",
        help="Which pipeline to run (default: both)",
    )
    args = parser.parse_args()

    total_start = time.time()

   
    print("ABM Epidemic Emulator — Random Sampling Experiment")
  

    if args.pipeline in ("A", "both"):
        run_pipeline_A()

    if args.pipeline in ("B", "both"):
        run_pipeline_B()

    total = time.time() - total_start
    minutes, seconds = divmod(int(total), 60)
    print(f"All pipelines finished in {minutes}m {seconds}s")
