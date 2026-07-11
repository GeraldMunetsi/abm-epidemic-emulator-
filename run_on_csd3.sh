#!/bin/bash
# ============================================================================
# run_on_csd3.sh
# ----------------------------------------------------------------------------
# SLURM submission script for Cambridge CSD3.
# Submits the full ABM emulator sweep as ONE job: 6 trainings + 6 validations
# + 18 cross-design tests, all sequential inside a single GPU allocation.
#
# Submit with:    sbatch run_on_csd3.sh
# Check status:   squeue -u $USER
# Cancel:         scancel <jobid>
# Live tail:      tail -f experiments/runs/logs/orchestrator.log
#
# To run just one condition (smoke test) replace the python line with:
#   python experiments/run_all_experiments.py --only "LHS_aug_eval_MCMC"
# ============================================================================

# ----------------------------------------------------------------------------
# 1. SLURM RESOURCE REQUESTS
# ----------------------------------------------------------------------------

#SBATCH --job-name=abm-emu                       # shown in squeue
#SBATCH --account=YOUR-PROJECT-CODE-GPU          # <- EDIT: your CSD3 project
#SBATCH --partition=ampere                       # A100 GPUs; alt: pascal, wilkes3
#SBATCH --nodes=1
#SBATCH --gres=gpu:1                             # one GPU is enough (sequential)
#SBATCH --cpus-per-task=8                        # for data loading + matplotlib
#SBATCH --mem=32G                                # adjust if MCMC_aug OOMs
#SBATCH --time=10:00:00                          # 10 hours wall time, see notes below
#SBATCH --output=experiments/runs/logs/slurm-%j.out
#SBATCH --error=experiments/runs/logs/slurm-%j.err
#SBATCH --mail-type=END,FAIL                     # email on completion/failure
#SBATCH --mail-user=YOUR-CRSID@cam.ac.uk         # <- EDIT

# ----------------------------------------------------------------------------
# 2. CSD3 ENVIRONMENT SETUP
# ----------------------------------------------------------------------------

# CSD3 uses Lmod modules. Load Python + CUDA matched to your venv.
. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp                    # default for ampere partition
module load python/3.10
module load cuda/11.8                            # match the version your venv uses

# Force UTF-8 for everything (avoids the Windows-style cp1252 crash on Linux too,
# though Linux defaults are already UTF-8). Belt-and-braces.
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

# Disable buffering so log files flush in real time and you can `tail -f`.
export PYTHONUNBUFFERED=1

# ----------------------------------------------------------------------------
# 3. PROJECT LOCATION
# ----------------------------------------------------------------------------

# Recommended: clone the repo into RDS hpc-work, not $HOME (too small).
PROJECT_ROOT=/rds/user/$USER/hpc-work/abm-epidemic-emulator

cd "$PROJECT_ROOT" || { echo "Project not found at $PROJECT_ROOT"; exit 1; }

# Activate the venv that lives inside the project.
source venv/bin/activate

# Quick sanity prints (land in slurm-<jobid>.out).
echo "==========================================="
echo "Job ID         : $SLURM_JOB_ID"
echo "Submit host    : $SLURM_SUBMIT_HOST"
echo "Compute node   : $(hostname)"
echo "Date           : $(date)"
echo "Project root   : $PROJECT_ROOT"
echo "Python         : $(which python)"
echo "Python version : $(python --version)"
echo "CUDA visible   : $CUDA_VISIBLE_DEVICES"
nvidia-smi | head -20
echo "==========================================="

# ----------------------------------------------------------------------------
# 4. RUN THE ORCHESTRATOR
# ----------------------------------------------------------------------------

# Exits 0 on success. The orchestrator is idempotent, so if SLURM kills the
# job at the wall-time limit, you can re-submit and it picks up where it left
# off (skips done trainings/validations/tests).
python experiments/run_all_experiments.py

# ----------------------------------------------------------------------------
# 5. POST-JOB SUMMARY
# ----------------------------------------------------------------------------

echo ""
echo "=========== POST-RUN SUMMARY ==========="
echo "Models trained:"
ls experiments/runs/models/ 2>/dev/null
echo ""
echo "Validation rows:"
[ -f experiments/runs/validation/validation_results.csv ] && \
    wc -l experiments/runs/validation/validation_results.csv
echo "Test rows:"
[ -f experiments/runs/results/test_results.csv ] && \
    wc -l experiments/runs/results/test_results.csv
echo "Wall time used:"
sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,MaxRSS,State 2>/dev/null
echo "========================================"
