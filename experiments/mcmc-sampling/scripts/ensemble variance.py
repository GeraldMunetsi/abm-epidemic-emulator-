"""
Ensemble variance vs R₀ for the MCMC-trained SIR emulator.

For every test sample the script computes:
  R₀  = tau / gamma          (basic reproduction number proxy)
  σ²  = Var over replicates of peak I(t)   (ensemble disagreement)
  σ   = √σ² normalised by N × 100          (% population, y-axis)

The scatter reveals how much the trained ensemble disagrees
as epidemic severity (R₀) increases.

Run from repo root:
    python "experiments/mcmc-sampling/scripts/ensemble variance.py"
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from step0_model import create_hybrid_mlp_model
from utils import create_dataloaders, get_device, PARAM_MINS, PARAM_MAXS

# ── CONSTANTS ──────────────────────────────────────────────────────────────────
N            = 100_000
n_knots      = 8
n_timepoints = 250

MODELS_DIR = Path("experiments/mcmc-sampling/out/trained-models")
DATA_PATH  = Path("experiments/mcmc-sampling/data/split/abm-data_split.pkl")
OUT_DIR    = Path("experiments/mcmc-sampling/out/plots")


# ── DATA LOADING ───────────────────────────────────────────────────────────────
def load_replicate_model(model_path: Path, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config     = checkpoint.get('config', {
        'n_params'        : 3,
        'n_fourier'       : 64,
        'sigma'           : 1.0,
        'fusion_hidden'   : 128,
        'latent_dim'      : 64,
        'decoder_hidden'  : 64,
        'dropout'         : 0.3,
        'n_knots'         : n_knots,
        'n_timepoints'    : n_timepoints,
        'total_population': N,
    })
    state_dict = checkpoint['model_state_dict']
    state_dict.pop('temporal_decoder.t_grid', None)
    model = create_hybrid_mlp_model(config)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model


def evaluate_model(model, loader, device):
    """
    Returns:
        preds  : (n_test, T, 3)  numpy
        params : (n_test, 3)     numpy  — raw (tau, gamma, rho)
    """
    all_preds, all_params = [], []
    param_mins = torch.tensor(PARAM_MINS)
    param_maxs = torch.tensor(PARAM_MAXS)

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            all_preds.append(model(batch, n_timesteps=n_timepoints).cpu())
            raw = batch.params_norm.cpu() * (param_maxs - param_mins) + param_mins
            all_params.append(raw)

    return (
        torch.cat(all_preds,  dim=0).numpy(),   # (n_test, T, 3)
        torch.cat(all_params, dim=0).numpy(),   # (n_test, 3)
    )


def collect_predictions(models_dir: Path, loader, device):
    """
    Runs every replicate checkpoint on the loader.

    Returns:
        all_preds : (n_reps, n_test, T, 3)
        params    : (n_test, 3)  raw (tau, gamma, rho)
    """
    model_paths = sorted(
        models_dir.glob("best_balanced_mlp_model_*.pt"),
        key=lambda p: int(p.stem.split('_')[-1])
    )
    if not model_paths:
        raise FileNotFoundError(f"No checkpoints found in {models_dir.resolve()}")

    print(f"Found {len(model_paths)} replicate(s)")
    stacked, params = [], None

    for i, path in enumerate(model_paths, 1):
        print(f"  [{i}/{len(model_paths)}] {path.name}")
        model = load_replicate_model(path, device)
        preds, p = evaluate_model(model, loader, device)
        stacked.append(preds)
        if params is None:
            params = p

    return np.stack(stacked, axis=0), params   # (n_reps, n_test, T, 3), (n_test, 3)


# ── PLOT ───────────────────────────────────────────────────────────────────────
def plot_ensemble_variance_vs_R0(
    all_preds : np.ndarray,   # (n_reps, n_test, T, 3)
    params    : np.ndarray,   # (n_test, 3) — [tau, gamma, rho]
    out_dir   : Path,
) -> None:
    """
    Scatter + binned-mean trend of ensemble σ(peak I) / N vs R₀ = tau/gamma.
    Points coloured by ρ (initial infection fraction).
    """
    tau   = params[:, 0]   # (n_test,)
    gamma = params[:, 1]
    rho   = params[:, 2]

    R0 = tau / gamma       # (n_test,)  — epidemic severity proxy

    # Peak I for each replicate, each sample
    I_preds   = all_preds[:, :, :, 1]            # (n_reps, n_test, T)
    peak_I    = I_preds.max(axis=2)              # (n_reps, n_test)

    # Ensemble statistics across replicates
    ens_std  = peak_I.std(axis=0)               # (n_test,)
    ens_mean = peak_I.mean(axis=0)              # (n_test,)
    ens_cv   = np.where(ens_mean > 0, ens_std / ens_mean * 100, 0.0)  # CV in %
    ens_std_pct = ens_std / N * 100             # σ as % of N

    # ── Binned trend ───────────────────────────────────────────────────────────
    n_bins   = 30
    bin_edges = np.percentile(R0, np.linspace(0, 100, n_bins + 1))
    bin_edges = np.unique(bin_edges)
    bin_idx   = np.digitize(R0, bin_edges[1:-1])   # 0-indexed bins

    bin_R0_med  = np.array([np.median(R0[bin_idx == b])      for b in range(n_bins)
                             if (bin_idx == b).any()])
    bin_std_med = np.array([np.median(ens_std_pct[bin_idx == b]) for b in range(n_bins)
                             if (bin_idx == b).any()])
    bin_std_lo  = np.array([np.percentile(ens_std_pct[bin_idx == b], 25) for b in range(n_bins)
                             if (bin_idx == b).any()])
    bin_std_hi  = np.array([np.percentile(ens_std_pct[bin_idx == b], 75) for b in range(n_bins)
                             if (bin_idx == b).any()])

    # ── Figure: 2-panel ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Ensemble Variance vs R₀  (R₀ = τ/γ)",
                 fontsize=14, fontweight="bold")

    # Panel A — σ(peak I) / N vs R₀
    ax = axes[0]
    sc = ax.scatter(R0, ens_std_pct, c=rho, cmap="plasma",
                    s=12, alpha=0.45, linewidths=0, rasterized=True)
    ax.fill_between(bin_R0_med, bin_std_lo, bin_std_hi,
                    color="steelblue", alpha=0.25, label="IQR (binned)")
    ax.plot(bin_R0_med, bin_std_med, color="steelblue", linewidth=2.0,
            label="Median (binned)", zorder=4)

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("ρ  (initial infected fraction)", fontsize=9)

    ax.set_xlabel("R₀ = τ / γ", fontsize=11)
    ax.set_ylabel("Ensemble σ(peak I) / N  (%)", fontsize=11)
    ax.set_title("Replicate spread on peak I", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.text(0.02, 0.97, "a)", transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top")

    # Panel B — CV(peak I) vs R₀
    ax2 = axes[1]
    sc2 = ax2.scatter(R0, ens_cv, c=rho, cmap="plasma",
                      s=12, alpha=0.45, linewidths=0, rasterized=True)
    bin_cv_med = np.array([np.median(ens_cv[bin_idx == b]) for b in range(n_bins)
                            if (bin_idx == b).any()])
    bin_cv_lo  = np.array([np.percentile(ens_cv[bin_idx == b], 25) for b in range(n_bins)
                            if (bin_idx == b).any()])
    bin_cv_hi  = np.array([np.percentile(ens_cv[bin_idx == b], 75) for b in range(n_bins)
                            if (bin_idx == b).any()])

    ax2.fill_between(bin_R0_med, bin_cv_lo, bin_cv_hi,
                     color="darkorange", alpha=0.25, label="IQR (binned)")
    ax2.plot(bin_R0_med, bin_cv_med, color="darkorange", linewidth=2.0,
             label="Median (binned)", zorder=4)

    fig.colorbar(sc2, ax=ax2).set_label("ρ  (initial infected fraction)", fontsize=9)
    ax2.set_xlabel("R₀ = τ / γ", fontsize=11)
    ax2.set_ylabel("CV(peak I)  =  σ / mean  (%)", fontsize=11)
    ax2.set_title("Coefficient of variation of peak I", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.text(0.02, 0.97, "b)", transform=ax2.transAxes,
             fontsize=13, fontweight="bold", va="top")

    fig.tight_layout()
    out = out_dir / "fig_ensemble_variance_vs_R0.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # Print summary
    print(f"\nR₀ range : [{R0.min():.4f}, {R0.max():.4f}]")
    print(f"σ(peak I)/N range : [{ens_std_pct.min():.4f}, {ens_std_pct.max():.4f}] %")
    print(f"CV(peak I) range  : [{ens_cv.min():.4f},  {ens_cv.max():.4f}] %")
    print(f"Spearman ρ (R₀ vs σ/N) : ", end="")
    from scipy.stats import spearmanr
    r, p = spearmanr(R0, ens_std_pct)
    print(f"{r:.4f}  (p={p:.2e})")


def plot_epistemic_uncertainty_vs_R0(
    all_preds : np.ndarray,   # (n_reps, n_test, T, 3)
    out_dir   : Path,
) -> None:
    """
    Two-panel epistemic uncertainty figure.

    Left  — binned mean inter-model σ(I(t)) vs R₀, with ±1σ band across
            samples in each bin and a dashed R₀ = 1 threshold line.
    Right — temporal profile of inter-model σ(I(t)) for four target R₀
            values, showing when in the epidemic the ensemble disagrees most.
    """
    I_preds   = all_preds[:, :, :, 1]           # (n_reps, n_test, T)

    # Epidemic severity = peak I / N (%).
    # This avoids R₀ estimation entirely: the final-size formula always returns
    # ≥ 1 when ρ > 0, and tau/gamma is sub-unity because the ABM effective R₀
    # = tau·<k>/gamma involves the unknown mean network degree <k>.
    # Peak I/N directly quantifies how large the epidemic was, spanning
    # [0, 100%] with 0 ≈ no outbreak and larger values ≈ more severe epidemic.
    mean_I_preds = I_preds.mean(axis=0)                  # (n_test, T)
    peak_I_frac  = mean_I_preds.max(axis=1) / N * 100   # (n_test,)  % of N
    severity     = peak_I_frac
    ens_std_t    = I_preds.std(axis=0)                  # (n_test, T)  std across replicates

    # Mean std over time per sample (left-panel y-value)
    mean_t_std = ens_std_t.mean(axis=1)          # (n_test,)

    # ── Binning by epidemic severity ───────────────────────────────────────────
    n_bins    = 20
    bin_edges = np.percentile(severity, np.linspace(0, 100, n_bins + 1))
    bin_edges = np.unique(bin_edges)
    bin_idx   = np.digitize(severity, bin_edges[1:-1])

    valid_bins  = [b for b in range(len(bin_edges) - 1) if (bin_idx == b).any()]
    bin_sev_med = np.array([np.median(severity[bin_idx == b])    for b in valid_bins])
    bin_mean    = np.array([mean_t_std[bin_idx == b].mean()      for b in valid_bins])
    bin_std_err = np.array([mean_t_std[bin_idx == b].std()       for b in valid_bins])

    # ── Data-driven target severity values (5th / 25th / 50th / 90th pct) ────
    T    = ens_std_t.shape[1]
    time = np.arange(T)

    target_percentiles = [5, 25, 50, 90]
    target_sevs        = [float(np.percentile(severity, p)) for p in target_percentiles]
    target_colors      = ['#aec6e8', '#4393c3', '#08306b', 'darkorange']
    sev_max            = severity.max()

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Epistemic Uncertainty vs Epidemic Severity  (peak I / N %)",
                 fontsize=15, fontweight='bold')

    # Left panel — binned mean uncertainty vs epidemic severity
    ax_l.plot(bin_sev_med, bin_mean, color='steelblue', linewidth=2.0,
              marker='o', markersize=5, zorder=4)
    ax_l.fill_between(bin_sev_med,
                      np.maximum(bin_mean - bin_std_err, 0),
                      bin_mean + bin_std_err,
                      color='steelblue', alpha=0.25)
    for target, color, pct in zip(target_sevs, target_colors, target_percentiles):
        ax_l.axvline(target, color=color, linewidth=1.2,
                     linestyle='--', alpha=0.8,
                     label=f'p{pct}: {target:.2f}%')
    # Epidemic threshold: below this peak I/N the outbreak barely took off (R₀ ≈ 1).
    # Derived empirically as the 10th percentile — most subcritical runs cluster here.
    threshold_pct = float(np.percentile(severity, 10))
    ax_l.axvline(threshold_pct, color='crimson', linewidth=1.8,
                 linestyle='-', alpha=0.9,
                 label=f'R₀ ≈ 1  (p10 = {threshold_pct:.2f}%)')
    ax_l.set_xlabel('Peak I / N  (%)', fontweight='bold')
    ax_l.set_ylabel('Mean inter-model std of I(t)', fontweight='bold')
    ax_l.set_title('Ensemble variance by epidemic severity bin')
    ax_l.set_xlim(left=0, right=sev_max * 1.05)
    ax_l.legend(fontsize=9, framealpha=0.85)
    ax_l.grid(True, alpha=0.3)

    # Right panel — temporal uncertainty profile for selected severity levels
    for target, color, pct in zip(target_sevs, target_colors, target_percentiles):
        tol  = max(sev_max * 0.04, target * 0.15)
        mask = np.abs(severity - target) < tol
        if mask.sum() == 0:
            print(f"  [skip] no samples near peak I/N={target:.3f}%")
            continue
        curve = ens_std_t[mask].mean(axis=0)
        ax_r.plot(time, curve, color=color, linewidth=2.2,
                  label=f'peak I/N~{target:.2f}%  (p{pct})')

    ax_r.set_xlabel('Time', fontweight='bold')
    ax_r.set_ylabel('Inter-model std of I(t)', fontweight='bold')
    ax_r.set_title('Ensemble std I(t) for selected severity levels')
    ax_r.legend(fontsize=9, framealpha=0.85)
    ax_r.grid(True, alpha=0.3)

    fig.tight_layout()
    out = out_dir / 'fig_epistemic_uncertainty_vs_severity.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


# ── ENTRY POINT ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = get_device()

    print(f"Loading data: {DATA_PATH}")
    dataloaders = create_dataloaders(str(DATA_PATH), batch_size=64)
    test_loader = dataloaders.get('test', dataloaders['val'])
    print(f"Test samples: {len(test_loader.dataset)}")

    all_preds, params = collect_predictions(MODELS_DIR, test_loader, device)
    print(f"Predictions shape: {all_preds.shape}  (n_reps, n_test, T, 3)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_ensemble_variance_vs_R0(all_preds, params, OUT_DIR)
    plot_epistemic_uncertainty_vs_R0(all_preds, OUT_DIR)
