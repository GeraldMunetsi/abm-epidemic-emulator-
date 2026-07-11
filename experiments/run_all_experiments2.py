"""
run_all_experiments.py
======================

Orchestrator for the ABM emulator experiment matrix.

Goal
----
Run ALL 18 experimental conditions in one go, without modifying any code
inside the existing pipeline (lhs-sampling/, mcmc-sampling/, random-sampling/).

The 18 conditions
    train_design (3): LHS | MCMC | Random
    augmented   (2): True | False      3 x 2 = 6 unique trained models
    eval_design (3): LHS | MCMC | Random
    Total = 3 * 2 * 3 = 18 conditions

For every condition the orchestrator records:
    * validation/test scores (whatever the existing eval script writes)
    * the path of the trained-model artifact used
    * timestamp

Output

A single tidy CSV at:
    experiments/runs/results/all_experiments_results.csv

Run from the project root
    cd "<repo root>"
    python experiments/run_all_experiments.py --dry-run        # list 18 conditions
    python experiments/run_all_experiments.py                  # run everything
    python experiments/run_all_experiments.py --only LHS_aug_eval_MCMC   # one condition

Design decisions (read these before changing the file)



2. Each (train_design, augmented) pair is trained ONCE (6 jobs total). The
   resulting model artifact is then re-used for the 3 eval_design evaluations
   -- saving 12 redundant retrainings.

3. The orchestrator is idempotent. Re-running skips conditions already
   recorded in results.csv and skips trainings whose model directory already
   contains a checkpoint. Safe to relaunch after a timeout or crash.

4. Cross-design evaluation re-uses the pipeline's existing cross-strategy
   test scripts (step5_test, step6_*, step7_*). These already accept
   --train_strategy / --test_strategy / --augmentation flags, so we just
   wire them up correctly.


"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd



# 1. PATHS & CONSTANTS

# project_root is the parent of the parent.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Pipeline scripts are invoked with cwd=PROJECT_ROOT because the upstream code
# uses paths like Path("experiments/lhs-sampling/data/augmented") relative to
# the project root.
PIPELINE_PYTHON = sys.executable

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
DESIGN_DIR = {
    "LHS":    EXPERIMENTS_DIR / "lhs-sampling",
    "MCMC":   EXPERIMENTS_DIR / "mcmc-sampling",
    "Random": EXPERIMENTS_DIR / "random-sampling",
}

DESIGNS = ["LHS", "MCMC", "Random"]
AUGMENTATION_FLAGS = [True, False]   # only used at TRAINING time

# Training data file conventions (verified by inspecting data/augmented and
# data/split folders).
AUGMENTED_PKL = "abm-data_split_augmented.pkl"
SPLIT_PKL = "abm-data_split.pkl"

# Where the orchestrator stores its own (non-pipeline) artifacts.
ORCH_ROOT = EXPERIMENTS_DIR / "runs"
MODELS_ROOT= ORCH_ROOT / "models"        # 6 sub-folders, one per (train_design, augmented)
VAL_RESULTS_ROOT = ORCH_ROOT / "validation"    # 6 sub-folders, one per trained model
RESULTS_ROOT= ORCH_ROOT / "results"       # 18 sub-folders, one per test condition
PLOTS_ROOT= ORCH_ROOT / "plots"         # plots from validation + test
LOGS_ROOT= ORCH_ROOT / "logs"

# Two separate top-level CSVs (per user spec):
#   validation_results.csv : 6 rows  -- one per (train_design, augmented)
#   test_results.csv       : 18 rows -- one per (train_design, augmented, eval_design)
VAL_RESULTS_CSV  = VAL_RESULTS_ROOT / "validation_results.csv"
TEST_RESULTS_CSV = RESULTS_ROOT / "test_results.csv"

for d in (MODELS_ROOT, VAL_RESULTS_ROOT, RESULTS_ROOT, PLOTS_ROOT, LOGS_ROOT):
    d.mkdir(parents=True, exist_ok=True)


# 2. SCRIPT LOOKUP TABLES

# Map (train_design, eval_design) -> which existing script we call to evaluate
# a model trained on train_design against eval_design's data. These are the
# step5 / step6 / step7 scripts already in the pipeline.
# we just call them.

CROSS_EVAL_SCRIPT = {
    ("LHS",    "LHS"):    "step5_test.py",
    ("LHS",    "MCMC"):   "step6_test_mcmc_data.py",
    ("LHS",    "Random"): "step7_test_random.py",

    ("MCMC",   "MCMC"):   "step5_test2.py",
    ("MCMC",   "LHS"):    "step6_test_lhs_data.py",
    ("MCMC",   "Random"): "step6_test_random_sampling_data.py",

    ("Random", "Random"): "step5_test.py",
    ("Random", "LHS"):    "step6_test_on_lhs_data.py",
    ("Random", "MCMC"):   "step6_test_on_mcmc_data.py",
}

# Step1 sampling scripts (no argparse; paths are hard-coded inside).
SAMPLING_SCRIPT = {
    "LHS":    "step1_LHS sampling.py",     # note the space in the filename
    "MCMC":   "step1_mcmc_sampling.py",
    "Random": "Step1_Random_sampling.py",
}

# Step2 split scripts (have argparse; defaults read raw/, write split/).
SPLIT_SCRIPT = {
    "LHS":    "Step2_data_split.py",
    "MCMC":   "step2_split.py",
    "Random": "Step2_data_split.py",
}

# Step2 augmentation scripts (no argparse; hard-coded paths).
AUGMENT_SCRIPT = {
    "LHS":    "step2_data_augmentation.py",
    "MCMC":   "step2A_augmented.py",
    "Random": "step2A_augmented.py",
}

# Output filenames produced by the pipeline (for idempotency checks).
RAW_PKL_NAME = "abm-data_raw.pkl"

# The "test_strategy" string the cross-test scripts expect on the CLI.
# (Random folder's step5_test.py was authored with TEST_STRATEGY='UNIFORM_RANDOM'
# but the argparse choices in step6/7 only allow 'MCMC','LHS','Random' -- so we
# pass 'Random'. If any script complains, you'll see it in the per-job log.)
def cli_strategy(name: str) -> str:
    return name  # LHS / MCMC / Random -- matches argparse choices


# 3. EXPERIMENT MATRIX 
@dataclass(frozen=True)
class TrainConfig:
    train_design: str
    augmented: bool

    @property
    def tag(self) -> str:
        return f"{self.train_design}_{'aug' if self.augmented else 'noaug'}"


@dataclass(frozen=True)
class Condition:
    train_design: str
    augmented: bool
    eval_design: str

    @property
    def tag(self) -> str:
        return f"{self.train_design}_{'aug' if self.augmented else 'noaug'}_eval_{self.eval_design}"


def enumerate_train_configs() -> list[TrainConfig]:
    return [TrainConfig(d, a) for d, a in itertools.product(DESIGNS, AUGMENTATION_FLAGS)]


def enumerate_conditions() -> list[Condition]:
    return [
        Condition(d, a, e)
        for d, a, e in itertools.product(DESIGNS, AUGMENTATION_FLAGS, DESIGNS)
    ]



# 4. DATA-PATH RESOLUTION

def training_data_path(design: str, augmented: bool) -> Path:
    """Pickle file used for TRAINING. Pipeline-conformant path."""
    if augmented:
        return DESIGN_DIR[design] / "data" / "augmented" / AUGMENTED_PKL
    return DESIGN_DIR[design] / "data" / "split" / SPLIT_PKL


def eval_data_path(design: str) -> Path:
    """Pickle file used by the cross-design TEST scripts. Pipeline-conformant."""
    return DESIGN_DIR[design] / "data" / "split" / SPLIT_PKL

# 5. PER-CONDITION OUTPUT DIRS


def model_dir(tc: TrainConfig) -> Path:
    """Where step3_train.py will drop its checkpoint(s)."""
    return MODELS_ROOT / tc.tag


def condition_results_dir(c: Condition) -> Path:
    return RESULTS_ROOT / c.tag


def condition_plots_dir(c: Condition) -> Path:
    return condition_results_dir(c) / "plots"


# 6. SUBPROCESS RUNNERS  (no pipeline modification)


def _run(cmd: list[str], log_file: Path) -> int:
    """Run subprocess from PROJECT_ROOT, tee output to log_file, return rc.

    Forces PYTHONIOENCODING=utf-8 in the child so pipeline scripts that print
    unicode characters (e.g. the arrow '→') don't crash on Windows cp1252.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.info("RUN: %s", " ".join(map(str, cmd)))
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"   # belt-and-braces: enables Python 3.7+ UTF-8 mode
    with open(log_file, "w", encoding="utf-8") as f:
        proc = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            env=env,
            check=False,
        )
    return proc.returncode


def raw_data_path(design: str) -> Path:
    return DESIGN_DIR[design] / "data" / "raw" / RAW_PKL_NAME

def split_data_path(design: str) -> Path:
    return DESIGN_DIR[design] / "data" / "split" / SPLIT_PKL

def augmented_data_path(design: str) -> Path:
    return DESIGN_DIR[design] / "data" / "augmented" / AUGMENTED_PKL


def sample_design(design: str, force: bool = False) -> None:
    """Run step1 sampling. force=True overrides + regenerates the raw pkl."""
    out = raw_data_path(design)
    if out.exists() and not force:
        logging.info("Skipping sampling (already present, use --force-data-prep to override): %s", out.name)
        return
    if force and out.exists():
        logging.info("OVERRIDE sampling: removing %s", out.name)
        out.unlink()
    script = DESIGN_DIR[design] / "scripts" / SAMPLING_SCRIPT[design]
    rc = _run([PIPELINE_PYTHON, str(script)],
              LOGS_ROOT / f"sample_{design}.log")
    if rc != 0:
        raise RuntimeError(
            f"Sampling failed for {design} (rc={rc}). "
            f"See {LOGS_ROOT / f'sample_{design}.log'}"
        )


def split_design(design: str, force: bool = False) -> None:
    """Run step2 split. force=True overrides + regenerates the split pkl."""
    out = split_data_path(design)
    if out.exists() and not force:
        logging.info("Skipping split (already present, use --force-data-prep to override): %s", out.name)
        return
    if force and out.exists():
        logging.info("OVERRIDE split: removing %s", out.name)
        out.unlink()
    script = DESIGN_DIR[design] / "scripts" / SPLIT_SCRIPT[design]
    rc = _run([PIPELINE_PYTHON, str(script)],
              LOGS_ROOT / f"split_{design}.log")
    if rc != 0:
        raise RuntimeError(
            f"Split failed for {design} (rc={rc}). "
            f"See {LOGS_ROOT / f'split_{design}.log'}"
        )


def augment_design(design: str, force: bool = False) -> None:
    """Run step2 augmentation. force=True overrides + regenerates the augmented pkl."""
    out = augmented_data_path(design)
    if out.exists() and not force:
        logging.info("Skipping augmentation (already present, use --force-data-prep to override): %s", out.name)
        return
    if force and out.exists():
        logging.info("OVERRIDE augmentation: removing %s", out.name)
        out.unlink()
    script = DESIGN_DIR[design] / "scripts" / AUGMENT_SCRIPT[design]
    rc = _run([PIPELINE_PYTHON, str(script)],
              LOGS_ROOT / f"augment_{design}.log")
    if rc != 0:
        raise RuntimeError(
            f"Augmentation failed for {design} (rc={rc}). "
            f"See {LOGS_ROOT / f'augment_{design}.log'}"
        )


def train_model(tc: TrainConfig) -> Path:
    """Run step3_train.py for one (train_design, augmented). Returns model dir."""
    out_dir = model_dir(tc)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Re-use existing artifacts (idempotent restart).
    if any(out_dir.glob("*.pt")):
        logging.info("Skipping training: %s already has checkpoints", out_dir.name)
        return out_dir

    train_script = DESIGN_DIR[tc.train_design] / "scripts" / "step3_train.py"
    cmd = [
        PIPELINE_PYTHON, str(train_script),
        "--input",      str(training_data_path(tc.train_design, tc.augmented)),
        "--output_dir", str(out_dir),
    ]
    rc = _run(cmd, LOGS_ROOT / f"train_{tc.tag}.log")
    if rc != 0:
        raise RuntimeError(
            f"Training failed for {tc.tag} (rc={rc}). "
            f"See {LOGS_ROOT / f'train_{tc.tag}.log'}"
        )
    return out_dir


def validate_model(tc: TrainConfig, models_dir: Path) -> Path:
    """Run step4_validate.py for one trained model on its OWN-design val split.

    Returns the validation results directory.
    """
    val_dir   = VAL_RESULTS_ROOT / tc.tag
    plots_dir = val_dir / "plots"        # keep plots inside the validation subfolder
    val_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Idempotent: skip running step4_validate.py only if the PLOTS already
    # exist. This is the strongest signal of "I already produced the
    # deliverable for this design" -- a log file alone would let an
    # incomplete run be misread as done, which previously left plot
    # folders empty after re-running.
    if any(plots_dir.glob("*.png")):
        logging.info("Skipping validation run (plots already present): %s", tc.tag)
        return val_dir

    validate_script = DESIGN_DIR[tc.train_design] / "scripts" / "step4_validate.py"
    cmd = [
        PIPELINE_PYTHON, str(validate_script),
        "--models_dir", str(models_dir),
        "--data",       str(eval_data_path(tc.train_design)),  # own-design split.pkl
        "--output_dir", str(val_dir),
        "--plots_dir",  str(plots_dir),
    ]
    rc = _run(cmd, LOGS_ROOT / f"validate_{tc.tag}.log")
    if rc != 0:
        # Soft-fail on validation: many pipeline crashes happen in the plotting
        # tail AFTER the metrics are computed. We log loudly but continue so
        # the rest of the 18-condition sweep isn't lost.
        logging.error(
            "Validation script exited with rc=%d for %s. "
            "Continuing; harvest_metrics will pick up whatever was written. "
            "See %s",
            rc, tc.tag, LOGS_ROOT / f"validate_{tc.tag}.log"
        )
    return val_dir


def evaluate_condition(c: Condition, models_dir: Path) -> Path:
    """Run the appropriate cross-design test script for this condition.

    Returns the per-condition results directory (where CSVs/JSONs landed).
    """
    script_name = CROSS_EVAL_SCRIPT[(c.train_design, c.eval_design)]
    script_path = DESIGN_DIR[c.train_design] / "scripts" / script_name

    res_dir   = condition_results_dir(c);  res_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = condition_plots_dir(c);    plots_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        PIPELINE_PYTHON, str(script_path),
        "--models_dir",     str(models_dir),
        "--data",           str(eval_data_path(c.eval_design)),
        "--output_dir",     str(res_dir),
        "--plots_dir",      str(plots_dir),
        "--train_strategy", cli_strategy(c.train_design),
        "--test_strategy",  cli_strategy(c.eval_design),
        "--augmentation",   "1" if c.augmented else "0",
    ]
    rc = _run(cmd, LOGS_ROOT / f"eval_{c.tag}.log")
    if rc != 0:
        # Soft-fail on test: same logic as validation. One condition's plot
        # bug shouldn't kill the other 17.
        logging.error(
            "Test script exited with rc=%d for %s. "
            "Continuing; harvest_metrics will pick up whatever was written. "
            "See %s",
            rc, c.tag, LOGS_ROOT / f"eval_{c.tag}.log"
        )
    return res_dir


# 7. METRIC HARVESTING

#
# The pipeline writes per-replicate result CSVs into the --output_dir we pass.
# We don't know the exact filename pattern up-front, so we sweep the directory
# for CSVs and pull summary statistics from whatever is there.

_LOG_METRIC_PATTERNS = [
    # "Mean R²: 0.9629 ± 0.0038"   (the ² and ± may render as garbage on cp1252
    # but the numbers themselves are still ASCII)
    (re.compile(r"^Mean\s+R\S{0,3}\s*:\s*([-\d.eE+]+)\s*[^\d-]+\s*([-\d.eE+]+)"),
        ("mean_R2", "std_R2")),
    (re.compile(r"^Mean\s+MAE_?I?\s*:\s*([-\d.eE+]+)\s*[^\d-]+\s*([-\d.eE+]+)"),
        ("mean_MAE_I", "std_MAE_I")),
    (re.compile(r"^CV\s*\(MAE_?I?\)\s*:\s*([-\d.eE+]+)\s*%"),
        ("cv_MAE_I_pct",)),
    (re.compile(r"^R\S{0,3}\s*=\s*([-\d.eE+]+)\s*\|\s*MAE_?I?\s*=\s*([-\d.eE+]+)"),
        ("rep_R2", "rep_MAE_I")),
]


def _harvest_from_log(log_file: Path) -> dict[str, Any]:
    """Extract summary metrics by regex-parsing a subprocess log.

    The pipeline scripts print 'Mean R²: 0.96 ± 0.04', 'Mean MAE_I: 1016 ± 19',
    'CV (MAE_I): 1.89%' and per-replicate 'R² = ... | MAE_I = ...'. We grab
    summary stats and aggregate the replicate lines.
    """
    metrics: dict[str, Any] = {}
    if not log_file.exists():
        return metrics
    rep_r2: list[float] = []
    rep_mae: list[float] = []
    try:
        text = log_file.read_text(encoding="utf-8", errors="replace")
    except Exception as e:        # noqa: BLE001
        logging.warning("Could not read log %s: %s", log_file, e)
        return metrics
    for line in text.splitlines():
        line = line.strip()
        for pat, keys in _LOG_METRIC_PATTERNS:
            m = pat.search(line)
            if not m:
                continue
            vals = [float(x) for x in m.groups()]
            if keys == ("rep_R2", "rep_MAE_I"):
                rep_r2.append(vals[0]); rep_mae.append(vals[1])
            else:
                for k, v in zip(keys, vals):
                    metrics[k] = v
    if rep_r2:
        metrics["n_replicates"] = len(rep_r2)
        metrics.setdefault("mean_R2", sum(rep_r2) / len(rep_r2))
    if rep_mae:
        metrics.setdefault("mean_MAE_I", sum(rep_mae) / len(rep_mae))
    if metrics:
        metrics["_metrics_source_file"] = log_file.name
    return metrics


def harvest_metrics(results_dir: Path, fallback_log: Path | None = None) -> dict[str, Any]:
    """Best-effort extraction of summary metrics.

    Priority:
        1. Any 'master_*' CSV in results_dir -> mean of numeric columns.
        2. Any other CSV in results_dir       -> mean of numeric columns.
        3. fallback_log (subprocess stdout)   -> regex extraction of summary lines.
    """
    metrics: dict[str, Any] = {}
    csvs = sorted(results_dir.glob("*.csv"))
    if csvs:
        master_candidates = [p for p in csvs if "master" in p.name.lower()]
        target = master_candidates[0] if master_candidates else csvs[0]
        try:
            df = pd.read_csv(target)
            numeric = df.select_dtypes(include="number")
            for col in numeric.columns:
                metrics[col] = float(numeric[col].mean())
            metrics["_metrics_source_file"] = target.name
            metrics["_n_rows"] = int(len(df))
            return metrics
        except Exception as e:        # noqa: BLE001
            logging.warning("Could not parse %s: %s", target, e)
    # Fall back to log scraping.
    if fallback_log is not None:
        metrics = _harvest_from_log(fallback_log)
        if metrics:
            return metrics
    logging.warning("No metrics produced in %s", results_dir)
    return metrics



# 8. CSV WRITER


def _load_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _append_row(path: Path, row: dict[str, Any]) -> None:
    df_row = pd.DataFrame([row])
    if path.exists():
        df_row.to_csv(path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(path, mode="w", header=True, index=False)


# ---- TEST CSV ----------------------------------------------------------------

def load_existing_test_results() -> pd.DataFrame | None:
    return _load_csv(TEST_RESULTS_CSV)


def test_already_recorded(c: Condition, df: pd.DataFrame | None) -> bool:
    if df is None or df.empty:
        return False
    mask = (
        (df["train_design"] == c.train_design)
        & (df["augmented"]  == c.augmented)
        & (df["eval_design"] == c.eval_design)
    )
    return bool(mask.any())


def append_test_row(row: dict[str, Any]) -> None:
    _append_row(TEST_RESULTS_CSV, row)


# ---- VALIDATION CSV ----------------------------------------------------------

def load_existing_val_results() -> pd.DataFrame | None:
    return _load_csv(VAL_RESULTS_CSV)


def val_already_recorded(tc: TrainConfig, df: pd.DataFrame | None) -> bool:
    if df is None or df.empty:
        return False
    mask = (
        (df["train_design"] == tc.train_design)
        & (df["augmented"]  == tc.augmented)
    )
    return bool(mask.any())


def append_val_row(row: dict[str, Any]) -> None:
    _append_row(VAL_RESULTS_CSV, row)



# 9. ORCHESTRATION


def filter_conditions(conds: list[Condition], only: str | None) -> list[Condition]:
    if not only:
        return conds
    pat = re.compile(only)
    return [c for c in conds if pat.search(c.tag)]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="List the 18 conditions and exit.")
    parser.add_argument("--only", type=str, default=None,
                        help="Regex on condition tag, e.g. 'LHS_aug_eval_MCMC'.")
    parser.add_argument("--skip-training", action="store_true",
                        help="Assume models are already trained; only validate + test.")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip Phase 1.5 (step4_validate.py). Go straight to test.")
    parser.add_argument("--skip-test", action="store_true",
                        help="Skip Phase 2 (cross-design test). Only train + validate.")
    parser.add_argument("--skip-data-prep", action="store_true",
                        help="Skip Phase 0 (sampling + split + augmentation).")
    parser.add_argument("--force-data-prep", action="store_true",
                        help="Force Phase 0 to regenerate raw/split/augmented data "
                             "even when files already exist (overrides idempotency).")
    args = parser.parse_args(argv)

    wall_start  = time.time()
    prep_times:  dict[str, float] = {}
    train_times: dict[str, float] = {}
    val_times:   dict[str, float] = {}
    test_times:  dict[str, float] = {}

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOGS_ROOT / "orchestrator.log"),
        ],
    )

    conditions = filter_conditions(enumerate_conditions(), args.only)

    # Only train the recipes that are ACTUALLY referenced by the surviving
    # conditions. Without this filter, --only would still trigger all 6
    # trainings even when you only asked for one evaluation. The set comprehension
    # below produces the minimal cover: e.g. --only "LHS_aug_eval_MCMC" reduces
    # the training plan from 6 jobs to 1.
    needed_train_tags = {
        TrainConfig(c.train_design, c.augmented).tag for c in conditions
    }
    train_configs = [
        tc for tc in enumerate_train_configs() if tc.tag in needed_train_tags
    ]

    logging.info("Project root : %s", PROJECT_ROOT)
    logging.info("Plan         : %d trainings, %d evaluation conditions",
                 len(train_configs), len(conditions))
    if args.only:
        logging.info("Filter --only %r selected trainings: %s",
                     args.only, sorted(needed_train_tags))

    if args.dry_run:
        print(f"{'tag':45s}  train_design  augmented  eval_design")
        for c in conditions:
            print(f"{c.tag:45s}  {c.train_design:12s}  {str(c.augmented):9s}  {c.eval_design}")
        return

    # ---- Phase 0: data preparation (sampling + split + augmentation) -----
    # Runs ONCE per design needed by surviving conditions. Augmentation is
    # only executed when at least one surviving condition has augmented=True
    # for that design. Each step is idempotent (skip if output exists).
    if not args.skip_data_prep:
        designs_needed = sorted({tc.train_design for tc in train_configs})
        aug_needed = {tc.train_design for tc in train_configs if tc.augmented}
        force = args.force_data_prep
        if force:
            logging.warning("--force-data-prep set: existing raw/split/augmented files will be regenerated.")
        for design in designs_needed:
            t0 = time.time()
            sample_design(design, force=force)
            split_design(design,  force=force)
            if design in aug_needed:
                augment_design(design, force=force)
            elapsed = time.time() - t0
            prep_times[design] = elapsed
            logging.info("Data prep for %s in %.1fs", design, elapsed)
    else:
        logging.info("Skipping Phase 0 (data prep) per --skip-data-prep")

    # ---- Phase 1: training -----------------------------------------------
    trained: dict[str, Path] = {}
    if not args.skip_training:
        for tc in train_configs:
            t0 = time.time()
            trained[tc.tag] = train_model(tc)
            elapsed = time.time() - t0
            train_times[tc.tag] = elapsed
            logging.info("Trained %s in %.1fs", tc.tag, elapsed)
    else:
        for tc in train_configs:
            trained[tc.tag] = model_dir(tc)

    # Phase 1.5: validation (own-design) 
    # Runs step4_validate.py on each trained model against its OWN design's
    # split.pkl val portion. Produces up to 6 rows in validation_results.csv
    # (fewer when --only narrows the training set).
    if not args.skip_validation:
        existing_val_df = load_existing_val_results()
        for tc in train_configs:
            if val_already_recorded(tc, existing_val_df):
                logging.info("SKIP validation (already in CSV): %s", tc.tag)
                continue
            t0 = time.time()
            val_dir = validate_model(tc, trained[tc.tag])
            val_metrics = harvest_metrics(
                val_dir,
                fallback_log=LOGS_ROOT / f"validate_{tc.tag}.log",
            )
            row = {
                "train_design": tc.train_design,
                "augmented":    tc.augmented,
                "val_design":   tc.train_design,   # own-design validation
                "models_dir":   str(trained[tc.tag]),
                "val_dir":      str(val_dir),
                "timestamp":    time.strftime("%Y-%m-%d %H:%M:%S"),
                "val_seconds":  round(time.time() - t0, 2),
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            elapsed = time.time() - t0
            val_times[tc.tag] = elapsed
            append_val_row(row)
            logging.info("Validated %s in %.1fs -> %s",
                         tc.tag, elapsed, VAL_RESULTS_CSV.name)
    else:
        logging.info("Skipping Phase 1.5 (validation) per --skip-validation")

    # Phase 2: cross-design test 
    if args.skip_test:
        logging.info("Skipping Phase 2 (test) per --skip-test")
        logging.info("DONE. Validation CSV: %s", VAL_RESULTS_CSV)
        return

    existing_test_df = load_existing_test_results()

    for c in conditions:
        if test_already_recorded(c, existing_test_df):
            logging.info("SKIP test (already in CSV): %s", c.tag)
            continue

        tc_tag = TrainConfig(c.train_design, c.augmented).tag
        models_path = trained[tc_tag]

        t0 = time.time()
        res_dir = evaluate_condition(c, models_path)
        metrics = harvest_metrics(
            res_dir,
            fallback_log=LOGS_ROOT / f"eval_{c.tag}.log",
        )
        elapsed = time.time() - t0
        test_times[c.tag] = elapsed

        row = {
            "train_design":  c.train_design,
            "augmented":     c.augmented,
            "eval_design":   c.eval_design,
            "models_dir":    str(models_path),
            "results_dir":   str(res_dir),
            "timestamp":     time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_seconds":  round(elapsed, 2),
            **{f"test_{k}": v for k, v in metrics.items()},
        }
        append_test_row(row)
        logging.info("Recorded %s in %.1fs -> %s", c.tag, elapsed, TEST_RESULTS_CSV.name)

    # TIMING SUMMARY 
    total_prep  = sum(prep_times.values())
    total_train = sum(train_times.values())
    total_val   = sum(val_times.values())
    total_test  = sum(test_times.values())
    wall        = time.time() - wall_start
    sep         = "=" * 72

    logging.info(sep)
    logging.info("TIMING SUMMARY")
    logging.info(sep)
    logging.info("Phase            Conditions   Total (s)    Total (min)   Mean (s)")
    logging.info("-" * 72)
    n_p, n_t, n_v, n_e = (len(prep_times), len(train_times),
                          len(val_times),  len(test_times))
    logging.info("Data prep        %-12d %-11.1f  %-12.2f  %.1f",
                 n_p, total_prep,  total_prep/60,  (total_prep/n_p if n_p else 0.0))
    logging.info("Training         %-12d %-11.1f  %-12.2f  %.1f",
                 n_t, total_train, total_train/60, (total_train/n_t if n_t else 0.0))
    logging.info("Validation       %-12d %-11.1f  %-12.2f  %.1f",
                 n_v, total_val,   total_val/60,   (total_val/n_v if n_v else 0.0))
    logging.info("Test             %-12d %-11.1f  %-12.2f  %.1f",
                 n_e, total_test,  total_test/60,  (total_test/n_e if n_e else 0.0))
    logging.info("-" * 72)
    logging.info("Wall-clock total                %-11.1f  %-12.2f  hours: %.2f",
                 wall, wall/60, wall/3600)
    logging.info(sep)

    # Per-condition breakdown
    if prep_times:
        logging.info("Per-design data-prep breakdown (s):")
        for k, v in prep_times.items():     logging.info("  %-25s %.1f", k, v)
    if train_times:
        logging.info("Per-training breakdown (s):")
        for k, v in train_times.items():    logging.info("  %-25s %.1f", k, v)
    if val_times:
        logging.info("Per-validation breakdown (s):")
        for k, v in val_times.items():      logging.info("  %-25s %.1f", k, v)
    if test_times:
        logging.info("Per-test breakdown (s):")
        for k, v in test_times.items():     logging.info("  %-25s %.1f", k, v)

    # Persist to JSON next to the CSVs so it's machine-readable too.
    timings_path = ORCH_ROOT / "timings.json"
    timings_path.write_text(json.dumps({
        "wall_clock_seconds": round(wall, 2),
        "wall_clock_minutes": round(wall/60, 2),
        "wall_clock_hours":   round(wall/3600, 3),
        "data_prep_total_s":  round(total_prep, 2),
        "training_total_s":   round(total_train, 2),
        "validation_total_s": round(total_val, 2),
        "test_total_s":       round(total_test, 2),
        "per_data_prep":  {k: round(v, 2) for k, v in prep_times.items()},
        "per_training":   {k: round(v, 2) for k, v in train_times.items()},
        "per_validation": {k: round(v, 2) for k, v in val_times.items()},
        "per_test":       {k: round(v, 2) for k, v in test_times.items()},
        "completed_at":   time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2))

    logging.info("DONE.")
    logging.info("  Validation CSV : %s", VAL_RESULTS_CSV)
    logging.info("  Test CSV       : %s", TEST_RESULTS_CSV)
    logging.info("  Timings JSON   : %s", timings_path)


#Entry

if __name__ == "__main__":
    main()
