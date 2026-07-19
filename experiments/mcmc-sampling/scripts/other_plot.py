"""
ABC Posterior Predictive Check and Conservation Error plots for MCMC→MCMC condition.

Run from repo root:
    python experiments/mcmc-sampling/scripts/other_plot.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path("experiments/mcmc-sampling/scripts")))
import torch
from step0_model import create_hybrid_mlp_model
from utils import create_dataloaders, get_device

MODELS_DIR = Path("experiments/mcmc-sampling/out/trained-models")
DATA_PATH  = Path("experiments/mcmc-sampling/data/split/abm-data_split.pkl")
OUT_DIR    = Path("experiments/Regression")
N          = 100000


# ── DATA LOADER ───────────────────────────────────────────────────────────────

def _run_inference(model, loader, device, n_ts):
    """Run a model over a DataLoader; return (preds_np, targets_np)."""
    preds_list, targs_list = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds_list.append(model(batch, n_timesteps=n_ts).cpu())
            targs_list.append(batch.y.cpu())
    preds_np  = torch.cat(preds_list, dim=0).numpy()
    targets_np = torch.cat(targs_list, dim=0).numpy()
    return preds_np, targets_np


def load_predictions(batch_size: int = 35):
    """
    Load all MCMC replicate checkpoints and run inference on the MCMC test and val sets.
    Returns:
        all_preds_test : np.ndarray  (n_reps, n_test, T, 3)
        targets_test   : np.ndarray  (n_test, T, 3)
        all_preds_cal  : np.ndarray  (n_reps, n_val,  T, 3)  — val split used as calibration
        targets_cal    : np.ndarray  (n_val,  T, 3)
    """
    device = get_device()

    dataloaders = create_dataloaders(str(DATA_PATH), batch_size=batch_size)
    test_loader = dataloaders["test"]
    val_loader  = dataloaders["val"]
    n_ts        = dataloaders["metadata"]["n_timepoints"]

    model_paths = sorted(
        MODELS_DIR.glob("best_balanced_mlp_model_*.pt"),
        key=lambda p: int(p.stem.split("_")[-1])
    )
    print(f"Found {len(model_paths)} replicate(s) in {MODELS_DIR}")

    test_preds_list, cal_preds_list = [], []
    targets_test = None
    targets_cal  = None

    for path in model_paths:
        print(f"  Loading: {path.name}")
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config     = checkpoint.get("config", {
            "n_params": 3, "n_fourier": 64, "sigma": 1.0,
            "fusion_hidden": 128, "latent_dim": 64, "decoder_hidden": 64,
            "dropout": 0.3, "n_knots": 8, "n_timepoints": n_ts,
            "total_population": N,
        })
        state_dict = checkpoint["model_state_dict"]
        state_dict.pop("temporal_decoder.t_grid", None)
        model = create_hybrid_mlp_model(config)
        model.load_state_dict(state_dict, strict=True)
        model.to(device).eval()

        preds_test, tgt_test = _run_inference(model, test_loader, device, n_ts)
        preds_cal,  tgt_cal  = _run_inference(model, val_loader,  device, n_ts)

        test_preds_list.append(preds_test)
        cal_preds_list.append(preds_cal)
        if targets_test is None:
            targets_test = tgt_test
            targets_cal  = tgt_cal

    all_preds_test = np.stack(test_preds_list, axis=0)
    all_preds_cal  = np.stack(cal_preds_list,  axis=0)
    print(f"Test predictions : {all_preds_test.shape}  |  Targets: {targets_test.shape}")
    print(f"Cal  predictions : {all_preds_cal.shape}   |  Targets: {targets_cal.shape}")
    return all_preds_test, targets_test, all_preds_cal, targets_cal


# ── ABC POSTERIOR PREDICTIVE CHECK 

def plot_abc_ppc(all_preds: np.ndarray, targets_np: np.ndarray,
                 out_dir: Path, n_samples: int =1000) -> None:
    """
    Overlays n_samples randomly drawn I(t) ground-truth trajectories with
    all replicate predictions, plus bold mean lines — standard ABC PPC style.
    """
    n_reps, n_test, T, _ = all_preds.shape
    indices = np.sort(np.random.choice(n_test, size=min(n_samples, n_test), replace=False))
    t = np.arange(T)

    fig, ax = plt.subplots(figsize=(14, 5))

    gt_labelled  = False
    rep_labelled = False
    for idx in indices:
        ax.plot(t, targets_np[idx, :, 1],
                color="steelblue", alpha=0.12, linewidth=0.7,
                label="Ground truth" if not gt_labelled else "")
        gt_labelled = True
        for rep in range(n_reps):
            ax.plot(t, all_preds[rep, idx, :, 1],
                    color="firebrick", alpha=0.07, linewidth=0.7,
                    label="Replicate prediction" if not rep_labelled else "")
            rep_labelled = True

    mean_pred_I = all_preds[:, indices, :, 1].mean(axis=(0, 1))
    mean_gt_I   = targets_np[indices, :, 1].mean(axis=0)
    ax.plot(t, mean_pred_I, color="darkred",  linewidth=2.2, label="Mean prediction",   zorder=5)
    ax.plot(t, mean_gt_I,   color="darkblue", linewidth=2.2, label="Mean ground truth", zorder=5)

    ax.set_xlabel("Time step", fontsize=11)
    ax.set_ylabel("Infected count", fontsize=11)
    ax.set_title(f"MCMC → MCMC — ABC Posterior Predictive Check: I(t)  "
                 f"(n={len(indices)} samples, {n_reps} replicates)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3, linestyle="--")

    fig.tight_layout()
    out = out_dir / "fig_abc_ppc_MCMC_to_MCMC.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── CONSERVATION ERROR OVER TIME ──────────────────────────────────────────────

def plot_conservation_over_time(all_preds: np.ndarray, out_dir: Path) -> None:
    """
    Plots mean |S(t)+I(t)+R(t) − N| / N × 100 vs time.
    Individual replicate mean lines show fluctuation; shaded band = ±1σ.
    """
    total   = all_preds.sum(axis=-1)           # (n_reps, n_test, T)
    err_pct = np.abs(total - N) / N * 100      # % of N

    per_rep  = err_pct.mean(axis=1)            # (n_reps, T) — mean over test samples
    mean_err = per_rep.mean(axis=0)            # (T,)
    std_err  = per_rep.std(axis=0)             # (T,)

    T = mean_err.shape[0]
    t = np.arange(T)

    fig, ax = plt.subplots(figsize=(12, 4))

    for rep in range(per_rep.shape[0]):
        ax.plot(t, per_rep[rep], color="salmon", linewidth=0.9, alpha=0.55)

    ax.fill_between(t, mean_err - std_err, mean_err + std_err,
                    color="firebrick", alpha=0.18, label="Mean ± 1σ")
    ax.plot(t, mean_err, color="firebrick", linewidth=2.0,
            label="Mean |S+I+R−N| / N (%)", zorder=4)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.set_xlabel("Time step", fontsize=11)
    ax.set_ylabel("|S+I+R−N| / N  (%)", fontsize=11)
    ax.set_title("MCMC → MCMC — Conservation Error: |S+I+R−N| vs Time",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")

    fig.tight_layout()
    out = out_dir / "fig_conservation_error_MCMC_to_MCMC.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── CONFORMAL PREDICTION ──────────────────────────────────────────────────────

def _conformal_quantile(all_preds_cal: np.ndarray,
                        targets_cal:   np.ndarray,
                        alpha: float = 0.05) -> float:
    """
    Split conformal prediction using the supremum-norm nonconformity score on I(t).

    Score for calibration sample i:  s_i = max_t |ŷ_I(t) - y_I(t)|
    Quantile level uses the finite-sample correction:
        q_level = ceil((n_cal + 1)(1 − α)) / n_cal
    which guarantees marginal coverage ≥ (1 − α) on exchangeable data.
    """
    mean_pred_cal = all_preds_cal.mean(axis=0)          # (n_cal, T, 3)
    residuals_I   = np.abs(mean_pred_cal[:, :, 1]
                           - targets_cal[:, :, 1])       # (n_cal, T)
    scores  = residuals_I.max(axis=1)                    # (n_cal,) — one score per sample
    n_cal   = len(scores)
    q_level = min(np.ceil((1 - alpha) * (n_cal + 1)) / n_cal, 1.0)
    q_hat   = float(np.quantile(scores, q_level))
    return q_hat


def plot_conformal_uncertainty(all_preds_test: np.ndarray,
                               targets_test:   np.ndarray,
                               all_preds_cal:  np.ndarray,
                               targets_cal:    np.ndarray,
                               out_dir: Path,
                               alpha: float = 0.05,
                               n_display: int = 15) -> None:
    """
    Two-panel conformal prediction plot for I(t).
    Nonconformity score: sup-norm  max_t |Î(t) - I(t)|
    Guarantee: ≥ (1-α) fraction of complete trajectories lie
               entirely within the band at every timestep.
    """
    q_hat = _conformal_quantile(all_preds_cal, targets_cal, alpha)
    print(f"Conformal q̂ = {q_hat:.2f}  (α={alpha}, "
          f"cal n={targets_cal.shape[0]})")

    n_reps, n_test, T, _ = all_preds_test.shape
    mean_pred = all_preds_test.mean(axis=0)          # (n_test, T, 3)

    lower = mean_pred[:, :, 1] - q_hat              # (n_test, T)
    upper = mean_pred[:, :, 1] + q_hat              # (n_test, T)

    # ── Coverage ─────────────────────────────────────────────────────────
    inside = ((targets_test[:, :, 1] >= lower) &
              (targets_test[:, :, 1] <= upper))      # (n_test, T) bool

    # CORRECT: per-sample joint coverage
    # A trajectory is covered iff ALL timesteps are inside the band
    covered_per_sample = inside.all(axis=1)          # (n_test,)
    emp_cov = covered_per_sample.mean() * 100        # scalar — the guaranteed quantity

    # Per-timestep marginal coverage (informational only — will exceed 95%)
    cov_t = inside.mean(axis=0) * 100               # (T,)

    # ── Plot ─────────────────────────────────────────────────────────────
    indices = np.sort(
        np.random.choice(n_test, size=min(n_display, n_test), replace=False)
    )
    t = np.arange(T)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Left panel: trajectories
    ax = axes[0]
    for i, idx in enumerate(indices):
        color = 'steelblue' if covered_per_sample[idx] else 'darkorange'
        ax.plot(t, targets_test[idx, :, 1],
                color=color, alpha=0.55, linewidth=0.9,
                label='Covered trajectory' if (i == 0 and covered_per_sample[idx])
                      else 'Uncovered trajectory' if (i == 0 and not covered_per_sample[idx])
                      else '')
        ax.fill_between(t, lower[idx], upper[idx],
                        color='firebrick', alpha=0.08)
        ax.plot(t, mean_pred[idx, :, 1],
                color='firebrick', alpha=0.65, linewidth=0.9,
                label='Mean prediction ± q̂' if i == 0 else '')

    ax.fill_between([], [], color='firebrick', alpha=0.30,
                    label=rf'{int((1-alpha)*100)}% conformal band '
                          rf'($\hat{{q}}$={q_hat:.0f})')
    ax.set_xlabel('Time step', fontsize=11)
    ax.set_ylabel('Infected count', fontsize=11)
    ax.set_title(f'Conformal Prediction Bands on I(t)\n'
                 f'(n={len(indices)} samples shown, '
                 f'blue=covered, orange=uncovered)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Right panel: two coverage curves
    ax2 = axes[1]
    ax2.plot(t, cov_t, color='steelblue', linewidth=2.0,
             label='Per-timestep marginal coverage')
    ax2.axhline((1 - alpha) * 100, color='firebrick',
                linestyle='--', linewidth=1.5,
                label=f'Target {int((1-alpha)*100)}% coverage')
    ax2.axhline(emp_cov, color='darkgreen',
                linestyle=':', linewidth=1.5,
                label=f'Joint trajectory coverage: {emp_cov:.1f}%')
    ax2.set_xlabel('Time step', fontsize=11)
    ax2.set_ylabel('Coverage (%)', fontsize=11)
    ax2.set_title('Marginal vs Joint Coverage\n'
                  '(marginal >> 95% expected with sup-norm score)',
                  fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 105])
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--')

    fig.suptitle(
        rf'MCMC $\rightarrow$ MCMC — Split Conformal Prediction  '
        rf'($\alpha$={alpha},  $\hat{{q}}$={q_hat:.0f})'
        f'\nJoint trajectory coverage: {emp_cov:.1f}%  '
        f'(target ≥ {int((1-alpha)*100)}%)  |  '
        f'Cal n={targets_cal.shape[0]}  |  Test n={n_test}',
        fontsize=13, fontweight='bold'
    )
    fig.tight_layout()
    out = out_dir / 'fig_conformal_uncertainty_MCMC_to_MCMC.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")
    print(f"Joint trajectory coverage: {emp_cov:.2f}%  "
          f"(guarantee: ≥{(1-alpha)*100:.0f}%)")
    



# TIME-ADAPTIVE CONFORMAL PREDICTION

def _time_adaptive_quantiles(all_preds_cal, targets_cal, alpha=0.10):
    """
    Per-timestep split-conformal quantile q_hat(t) (rather than a single sup-norm
    q_hat), so band width can vary over time instead of being constant.
    """
    mean_pred_cal = all_preds_cal.mean(axis=0)           # (n_cal, T, 3)
    residuals_I   = np.abs(mean_pred_cal[:, :, 1]
                           - targets_cal[:, :, 1])       # (n_cal, T)
    n_cal   = residuals_I.shape[0]
    q_level = min(np.ceil((1 - alpha) * (n_cal + 1)) / n_cal, 1.0)
    return np.quantile(residuals_I, q_level, axis=0)    # (T,)


def plot_adaptive_conformal_uncertainty(all_preds_test, targets_test,
                                        all_preds_cal, targets_cal,
                                        out_dir, alpha=0.10, n_display=15):
    """
    Three-panel time-adaptive conformal prediction plot: prediction bands using
    q_hat(t) instead of a single global q_hat, per-timestep coverage, and a
    comparison of adaptive vs constant (sup-norm) band width over time.
    """
    q_hat_t   = _time_adaptive_quantiles(all_preds_cal, targets_cal, alpha)
    q_hat_sup = _conformal_quantile(all_preds_cal, targets_cal, alpha)

    _, n_test, T, _ = all_preds_test.shape
    mean_pred = all_preds_test.mean(axis=0)

    lower = mean_pred[:, :, 1] - q_hat_t
    upper = mean_pred[:, :, 1] + q_hat_t

    inside    = ((targets_test[:, :, 1] >= lower) &
                 (targets_test[:, :, 1] <= upper))
    cov_t     = inside.mean(axis=0) * 100
    joint_cov = inside.all(axis=1).mean() * 100

    indices = np.sort(
        np.random.choice(n_test, size=min(n_display, n_test), replace=False)
    )
    t = np.arange(T)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Panel 1 - Adaptive bands
    ax = axes[0]
    for i, idx in enumerate(indices):
        ax.fill_between(t, lower[idx], upper[idx], color='firebrick', alpha=0.08)
        ax.plot(t, targets_test[idx, :, 1],
                color='steelblue', alpha=0.55, linewidth=0.9,
                label='Ground truth' if i == 0 else '')
        ax.plot(t, mean_pred[idx, :, 1],
                color='firebrick', alpha=0.55, linewidth=0.9,
                label='Mean prediction' if i == 0 else '')
    ax.fill_between([], [], color='firebrick', alpha=0.30,
                    label=f'{int((1-alpha)*100)}% adaptive band')
    ax.set_xlabel('Time step', fontsize=11)
    ax.set_ylabel('Infected count', fontsize=11)
    ax.set_title(f'Time-Adaptive Conformal Bands on I(t)\n(n={len(indices)} samples shown)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Panel 2 - Per-timestep coverage
    ax2 = axes[1]
    ax2.plot(t, cov_t, color='steelblue', linewidth=2.0,
             label='Per-timestep coverage (adaptive)')
    ax2.axhline((1 - alpha) * 100, color='firebrick', linestyle='--', linewidth=1.5,
                label=f'Target {int((1-alpha)*100)}% coverage')
    ax2.axhline(joint_cov, color='darkgreen', linestyle=':', linewidth=1.5,
                label=f'Joint trajectory coverage: {joint_cov:.1f}%')
    ax2.set_ylim([0, 105])
    ax2.set_xlabel('Time step', fontsize=11)
    ax2.set_ylabel('Coverage (%)', fontsize=11)
    ax2.set_title('Per-Timestep Marginal Coverage\n(adaptive targets ~90% at each t)',
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--')

    # Panel 3 - Band width comparison
    ax3 = axes[2]
    ax3.plot(t, 2 * q_hat_t, color='steelblue', linewidth=2.0,
             label='Adaptive  2 x q_hat(t)')
    ax3.axhline(2 * q_hat_sup, color='firebrick', linestyle='--', linewidth=1.5,
                label=f'Sup-norm  2 x q_hat = {2*q_hat_sup:.0f}  (constant)')
    ax3.fill_between(t, 0, 2 * q_hat_t, color='steelblue', alpha=0.15)
    ax3.set_xlabel('Time step', fontsize=11)
    ax3.set_ylabel('Band width  (2 x q_hat)', fontsize=11)
    ax3.set_title('Band Width: Adaptive vs Global Sup-norm\n(adaptive shrinks once epidemic ends)',
                  fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, linestyle='--')

    fig.suptitle(
        f'MCMC -> MCMC - Time-Adaptive Conformal Prediction  (alpha={alpha})\n'
        f'Mean marginal coverage: {cov_t.mean():.1f}%  |  '
        f'Joint: {joint_cov:.1f}%  |  '
        f'Cal n={targets_cal.shape[0]}  |  Test n={n_test}',
        fontsize=13, fontweight='bold'
    )
    fig.tight_layout()
    out = out_dir / 'fig_adaptive_conformal_MCMC_to_MCMC.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")
    print(f"Adaptive - mean marginal coverage: {cov_t.mean():.2f}%  "
          f"| joint: {joint_cov:.2f}%")


# MONDRIAN CONFORMAL PREDICTION

GROUP_LABELS  = ['Small epidemic', 'Medium epidemic', 'Large epidemic']
GROUP_COLOURS = ['steelblue', 'darkorange', 'firebrick']


def _mondrian_calibrate(all_preds_cal, targets_cal, alpha=0.10, n_groups=3):
    """
    Partition calibration samples into n_groups by predicted peak I(t),
    then compute a sup-norm q̂ within each group.

    Grouping uses the *predicted* peak so the same rule applies at test time
    without needing the ground truth.

    Returns:
        boundaries  : np.ndarray (n_groups-1,) — predicted-peak thresholds
        q_hat_grp   : np.ndarray (n_groups,)   — q̂ per group
        group_sizes : list[int]
    """
    mean_pred_cal = all_preds_cal.mean(axis=0)           # (n_cal, T, 3)
    pred_peak_cal = mean_pred_cal[:, :, 1].max(axis=1)   # (n_cal,)

    # Equal-frequency boundaries from calibration predicted peaks
    pcts       = np.linspace(0, 100, n_groups + 1)[1:-1]
    boundaries = np.percentile(pred_peak_cal, pcts)       # (n_groups-1,)
    groups_cal = np.digitize(pred_peak_cal, boundaries)   # 0 … n_groups-1

    residuals_I = np.abs(mean_pred_cal[:, :, 1]
                         - targets_cal[:, :, 1])          # (n_cal, T)
    scores      = residuals_I.max(axis=1)                 # (n_cal,) sup-norm

    q_hat_grp   = np.zeros(n_groups)
    group_sizes = []
    for g in range(n_groups):
        mask    = groups_cal == g
        s_g     = scores[mask]
        n_g     = len(s_g)
        group_sizes.append(n_g)
        q_level = min(np.ceil((1 - alpha) * (n_g + 1)) / n_g, 1.0) if n_g > 0 else 1.0
        q_hat_grp[g] = float(np.quantile(s_g, q_level)) if n_g > 0 else np.inf

    return boundaries, q_hat_grp, group_sizes


def plot_mondrian_conformal_uncertainty(all_preds_test, targets_test,
                                        all_preds_cal, targets_cal,
                                        out_dir, alpha=0.10,
                                        n_groups=3, n_display=15):
    """
    Three-panel Mondrian conformal prediction plot.

    Panel 1 — Bands coloured by epidemic-size group.
    Panel 2 — Joint coverage per group (each should meet the 1-alpha target).
    Panel 3 — q̂ per group (shows how band width scales with epidemic size).
    """
    boundaries, q_hat_grp, group_sizes = _mondrian_calibrate(
        all_preds_cal, targets_cal, alpha, n_groups
    )

    _, n_test, T, _ = all_preds_test.shape
    mean_pred = all_preds_test.mean(axis=0)               # (n_test, T, 3)

    # Assign test samples to groups by predicted peak
    pred_peak_test = mean_pred[:, :, 1].max(axis=1)       # (n_test,)
    groups_test    = np.digitize(pred_peak_test, boundaries)

    # Per-sample q̂ from group membership
    q_per_sample = q_hat_grp[groups_test]                 # (n_test,)
    lower = mean_pred[:, :, 1] - q_per_sample[:, None]
    upper = mean_pred[:, :, 1] + q_per_sample[:, None]

    inside    = ((targets_test[:, :, 1] >= lower) &
                 (targets_test[:, :, 1] <= upper))        # (n_test, T)

    # Per-group joint coverage
    joint_cov_grp = []
    for g in range(n_groups):
        mask = groups_test == g
        if mask.sum() == 0:
            joint_cov_grp.append(np.nan)
        else:
            joint_cov_grp.append(inside[mask].all(axis=1).mean() * 100)

    overall_joint = inside.all(axis=1).mean() * 100

    indices = np.sort(
        np.random.choice(n_test, size=min(n_display, n_test), replace=False)
    )
    t = np.arange(T)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Panel 1 — Bands coloured by group
    ax = axes[0]
    labelled = [False] * n_groups
    for idx in indices:
        g     = groups_test[idx]
        col   = GROUP_COLOURS[g]
        label = GROUP_LABELS[g] if not labelled[g] else ''
        ax.fill_between(t, lower[idx], upper[idx], color=col, alpha=0.10)
        ax.plot(t, targets_test[idx, :, 1],
                color=col, alpha=0.60, linewidth=0.9, label=label)
        ax.plot(t, mean_pred[idx, :, 1],
                color=col, alpha=0.40, linewidth=0.7, linestyle='--')
        labelled[g] = True
    ax.set_xlabel('Time step', fontsize=11)
    ax.set_ylabel('Infected count', fontsize=11)
    ax.set_title(f'Mondrian Conformal Bands on I(t)\n(n={len(indices)} shown, coloured by group)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Panel 2 — Joint coverage per group
    ax2 = axes[1]
    x   = np.arange(n_groups)
    bars = ax2.bar(x, joint_cov_grp, color=GROUP_COLOURS, alpha=0.80, width=0.5)
    ax2.axhline((1 - alpha) * 100, color='black', linestyle='--', linewidth=1.5,
                label=f'Target {int((1-alpha)*100)}% coverage')
    for bar, cov, n_g in zip(bars, joint_cov_grp, group_sizes):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.8,
                 f'{cov:.1f}%\n(n={n_g})',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(GROUP_LABELS[:n_groups], fontsize=10)
    ax2.set_ylabel('Joint trajectory coverage (%)', fontsize=11)
    ax2.set_ylim(0, 110)
    ax2.set_title('Joint Coverage per Mondrian Group\n(each group independently guaranteed)',
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    # Panel 3 — q̂ per group
    ax3 = axes[2]
    ax3.bar(x, q_hat_grp, color=GROUP_COLOURS, alpha=0.80, width=0.5)
    for i, (q, n_g) in enumerate(zip(q_hat_grp, group_sizes)):
        ax3.text(i, q + 10, f'{q:.0f}\n(n={n_g})',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(GROUP_LABELS[:n_groups], fontsize=10)
    ax3.set_ylabel('q̂  (sup-norm quantile)', fontsize=11)
    ax3.set_title('Calibrated q̂ per Group\n(larger epidemics need wider bands)',
                  fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)

    fig.suptitle(
        f'MCMC -> MCMC - Mondrian Conformal Prediction  (alpha={alpha}, {n_groups} groups)\n'
        f'Overall joint coverage: {overall_joint:.1f}%  |  '
        f'Cal n={targets_cal.shape[0]}  |  Test n={n_test}',
        fontsize=13, fontweight='bold'
    )
    fig.tight_layout()
    out = out_dir / 'fig_mondrian_conformal_MCMC_to_MCMC.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")
    for g in range(n_groups):
        print(f"  {GROUP_LABELS[g]:20s}  q̂={q_hat_grp[g]:.0f}  "
              f"n_cal={group_sizes[g]}  joint_cov={joint_cov_grp[g]:.1f}%")


# MONDRIAN + TIME-ADAPTIVE CONFORMAL PREDICTION

def _mondrian_adaptive_calibrate(all_preds_cal, targets_cal, alpha=0.10, n_groups=3):
    """
    Per-group, per-timestep calibration.

    For each Mondrian group g and each time step t:
        q̂_g(t) = quantile_{ceil((n_g+1)(1-α))/n_g} of {|ŷ_i(t) - y_i(t)| : i in group g}

    Returns:
        boundaries   : np.ndarray (n_groups-1,)
        q_hat_grp_t  : np.ndarray (n_groups, T)
        group_sizes  : list[int]
    """
    mean_pred_cal = all_preds_cal.mean(axis=0)           # (n_cal, T, 3)
    pred_peak_cal = mean_pred_cal[:, :, 1].max(axis=1)   # (n_cal,)

    pcts       = np.linspace(0, 100, n_groups + 1)[1:-1]
    boundaries = np.percentile(pred_peak_cal, pcts)
    groups_cal = np.digitize(pred_peak_cal, boundaries)

    residuals_I = np.abs(mean_pred_cal[:, :, 1]
                         - targets_cal[:, :, 1])          # (n_cal, T)
    T = residuals_I.shape[1]

    q_hat_grp_t = np.zeros((n_groups, T))
    group_sizes = []
    for g in range(n_groups):
        mask = groups_cal == g
        r_g  = residuals_I[mask]                          # (n_g, T)
        n_g  = len(r_g)
        group_sizes.append(n_g)
        q_level = min(np.ceil((1 - alpha) * (n_g + 1)) / n_g, 1.0) if n_g > 0 else 1.0
        q_hat_grp_t[g] = np.quantile(r_g, q_level, axis=0) if n_g > 0 else np.inf

    return boundaries, q_hat_grp_t, group_sizes


def plot_mondrian_adaptive_conformal(all_preds_test, targets_test,
                                     all_preds_cal, targets_cal,
                                     out_dir, alpha=0.10,
                                     n_groups=3, n_display=15):
    """
    Three-panel Mondrian + time-adaptive conformal prediction plot.

    Panel 1 — Adaptive bands coloured by group (tight tails, no negatives).
    Panel 2 — Per-timestep marginal coverage per group (each tracks ~90%).
    Panel 3 — Band width over time per group.
    """
    boundaries, q_hat_grp_t, group_sizes = _mondrian_adaptive_calibrate(
        all_preds_cal, targets_cal, alpha, n_groups
    )

    _, n_test, T, _ = all_preds_test.shape
    mean_pred = all_preds_test.mean(axis=0)               # (n_test, T, 3)

    pred_peak_test = mean_pred[:, :, 1].max(axis=1)
    groups_test    = np.digitize(pred_peak_test, boundaries)

    # Per-sample q̂(t) from group membership  (n_test, T)
    q_per_sample_t = q_hat_grp_t[groups_test]
    lower = mean_pred[:, :, 1] - q_per_sample_t
    upper = mean_pred[:, :, 1] + q_per_sample_t

    inside = ((targets_test[:, :, 1] >= lower) &
              (targets_test[:, :, 1] <= upper))           # (n_test, T)

    # Per-group per-timestep marginal coverage
    cov_t_grp = []
    for g in range(n_groups):
        mask = groups_test == g
        if mask.sum() == 0:
            cov_t_grp.append(np.full(T, np.nan))
        else:
            cov_t_grp.append(inside[mask].mean(axis=0) * 100)

    overall_joint = inside.all(axis=1).mean() * 100
    mean_marginal = inside.mean() * 100

    indices = np.sort(
        np.random.choice(n_test, size=min(n_display, n_test), replace=False)
    )
    t = np.arange(T)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Panel 1 — Adaptive bands coloured by group
    ax = axes[0]
    labelled = [False] * n_groups
    for idx in indices:
        g     = groups_test[idx]
        col   = GROUP_COLOURS[g]
        label = GROUP_LABELS[g] if not labelled[g] else ''
        ax.fill_between(t, lower[idx], upper[idx], color=col, alpha=0.10)
        ax.plot(t, targets_test[idx, :, 1],
                color=col, alpha=0.60, linewidth=0.9, label=label)
        ax.plot(t, mean_pred[idx, :, 1],
                color=col, alpha=0.40, linewidth=0.7, linestyle='--')
        labelled[g] = True
    ax.set_xlabel('Time step', fontsize=11)
    ax.set_ylabel('Infected count', fontsize=11)
    ax.set_title(f'Mondrian + Adaptive Bands on I(t)\n(n={len(indices)} shown, coloured by group)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Panel 2 — Per-timestep coverage per group
    ax2 = axes[1]
    for g in range(n_groups):
        ax2.plot(t, cov_t_grp[g], color=GROUP_COLOURS[g], linewidth=1.8,
                 label=f'{GROUP_LABELS[g]} (n={group_sizes[g]})')
    ax2.axhline((1 - alpha) * 100, color='black', linestyle='--', linewidth=1.5,
                label=f'Target {int((1-alpha)*100)}%')
    ax2.set_ylim([0, 105])
    ax2.set_xlabel('Time step', fontsize=11)
    ax2.set_ylabel('Coverage (%)', fontsize=11)
    ax2.set_title('Per-Timestep Marginal Coverage per Group\n(each group independently calibrated)',
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--')

    # Panel 3 — Band width over time per group
    ax3 = axes[2]
    for g in range(n_groups):
        ax3.plot(t, 2 * q_hat_grp_t[g], color=GROUP_COLOURS[g], linewidth=1.8,
                 label=GROUP_LABELS[g])
        ax3.fill_between(t, 0, 2 * q_hat_grp_t[g],
                         color=GROUP_COLOURS[g], alpha=0.08)
    ax3.set_xlabel('Time step', fontsize=11)
    ax3.set_ylabel('Band width  (2 x q̂(t))', fontsize=11)
    ax3.set_title('Adaptive Band Width per Group\n(wider for larger epidemics, shrinks at tail)',
                  fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, linestyle='--')

    fig.suptitle(
        f'MCMC -> MCMC - Mondrian + Time-Adaptive Conformal  (alpha={alpha}, {n_groups} groups)\n'
        f'Mean marginal coverage: {mean_marginal:.1f}%  |  '
        f'Overall joint: {overall_joint:.1f}%  |  '
        f'Cal n={targets_cal.shape[0]}  |  Test n={n_test}',
        fontsize=13, fontweight='bold'
    )
    fig.tight_layout()
    out = out_dir / 'fig_mondrian_adaptive_conformal_MCMC_to_MCMC.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")
    print(f"Mondrian+Adaptive — mean marginal: {mean_marginal:.2f}%  "
          f"| overall joint: {overall_joint:.2f}%")


# MAIN

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_preds_test, targets_test, all_preds_cal, targets_cal = load_predictions()
    plot_abc_ppc(all_preds_test, targets_test, OUT_DIR, n_samples=100)
    plot_conservation_over_time(all_preds_test, OUT_DIR)
    plot_conformal_uncertainty(
        all_preds_test, targets_test,
        all_preds_cal,  targets_cal,
        OUT_DIR, alpha=0.10, n_display=15,
    )
    plot_adaptive_conformal_uncertainty(
        all_preds_test, targets_test,
        all_preds_cal,  targets_cal,
        OUT_DIR, alpha=0.10, n_display=15,
    )
    plot_mondrian_conformal_uncertainty(
        all_preds_test, targets_test,
        all_preds_cal,  targets_cal,
        OUT_DIR, alpha=0.10, n_groups=3, n_display=15,
    )
    plot_mondrian_adaptive_conformal(
        all_preds_test, targets_test,
        all_preds_cal,  targets_cal,
        OUT_DIR, alpha=0.10, n_groups=3, n_display=15,
    )
    print("\nDone.")
