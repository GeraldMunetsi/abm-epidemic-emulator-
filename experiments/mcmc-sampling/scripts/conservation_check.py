"""
Conservation check: |S(t)+I(t)+R(t) − N| vs time for the MCMC-trained emulator.

Loads every best_balanced_mlp_model_*.pt from MODELS_DIR, runs inference on
the MCMC test set, stacks predictions across replicates, then plots the mean
absolute conservation error (% of N) as a function of time step.

Run from repo root:
    python "experiments/mcmc-sampling/scripts/conservation_check.py"
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from step0_model import create_hybrid_mlp_model
from utils import create_dataloaders, get_device

# ── CONSTANTS ──────────────────────────────────────────────────────────────────
N = 100_000
n_knots = 8
n_timepoints = 250

MODELS_DIR = Path("experiments/mcmc-sampling/out/trained-models")
DATA_PATH  = Path("experiments/mcmc-sampling/data/split/abm-data_split.pkl")
OUT_DIR    = Path("experiments/mcmc-sampling/out/plots/conservation_plot")


# ── MODEL LOADING ──────────────────────────────────────────────────────────────
def load_replicate_model(model_path: Path, device: torch.device):
    """Load a single replicate checkpoint (falling back to a default config if none was saved) and return it in eval mode."""
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
    print(f"  {model_path.name}  epoch={checkpoint.get('epoch')}  "
          f"val R²={checkpoint.get('val_metrics', {}).get('R2', float('nan')):.4f}")
    return model


def evaluate_model(model: torch.nn.Module, loader, device: torch.device) -> np.ndarray:
    """Returns predictions (n_test, T, 3) as numpy array."""
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds = model(batch, n_timesteps=n_timepoints)
            all_preds.append(preds.cpu())
    return torch.cat(all_preds, dim=0).numpy()   # (n_test, T, 3)


def collect_all_predictions(models_dir: Path, loader, device: torch.device) -> np.ndarray:
    """
    Run every replicate on the test loader.

    Returns:
        all_preds : (n_reps, n_test, T, 3)
    """
    model_paths = sorted(
        models_dir.glob("best_balanced_mlp_model_*.pt"),
        key=lambda p: int(p.stem.split('_')[-1])
    )
    if not model_paths:
        raise FileNotFoundError(f"No model checkpoints found in {models_dir.resolve()}")

    print(f"\nFound {len(model_paths)} replicate(s) in {models_dir}")
    stacked = []
    for path in model_paths:
        model = load_replicate_model(path, device)
        preds = evaluate_model(model, loader, device)
        stacked.append(preds)

    return np.stack(stacked, axis=0)   # (n_reps, n_test, T, 3)


# ── CONSERVATION ERROR OVER TIME ───────────────────────────────────────────────
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
    ax.set_title("MCMC trained and  MCMC tested — Conservation Error",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")

    fig.tight_layout()
    out = out_dir / "fig_conservation_error_MCMC_to_MCMC.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── CONSERVATION ERROR: AUG vs NO-AUG ACROSS SAMPLING STRATEGIES ──────────────
RUNS_MODELS_DIR = Path("experiments/runs/models")

STRATEGIES = {
    "LHS"    : ("LHS_aug","LHS_noaug"),
    "MCMC"   : ("MCMC_aug", "MCMC_noaug"),
    "Random" : ("Random_aug", "Random_noaug"),
}

AUG_STYLE = {
    "aug"  : dict(color="steelblue", linestyle="-",  label="Augmented",       lw=2.0),
    "noaug": dict(color="tomato",    linestyle="--", label="No augmentation",  lw=2.0),
}
FILL_ALPHA = 0.15


def _conservation_curve(all_preds: np.ndarray):
    """
    Compute per-replicate mean conservation error curve.

    Args:
        all_preds : (n_reps, n_test, T, 3)

    Returns:
        mean_t : (T,)   mean over replicates and test samples
        std_t  : (T,)   std over replicates (each rep averaged over samples first)
    """
    total   = all_preds.sum(axis=-1)        # (n_reps, n_test, T)
    err_pct = np.abs(total - N) / N * 100   # % of N

    per_rep  = err_pct.mean(axis=1)         # (n_reps, T)
    return per_rep.mean(axis=0), per_rep.std(axis=0)


def plot_conservation_aug_vs_noaug(loader, device: torch.device, out_dir: Path) -> None:
    """
    3-panel figure: one column per sampling strategy (LHS / MCMC / Random).
    Each panel overlays augmented (solid blue) vs no-augmentation (dashed red)
    conservation error curves with ±1σ shaded bands.

    Models are loaded from experiments/runs/models/{STRATEGY}_{aug|noaug}/.
    All conditions are evaluated on the same test loader.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    fig.suptitle(
        "Conservation Error: Augmented vs No Augmentation",
        fontsize=14, fontweight="bold",
    )

    t = np.arange(n_timepoints)

    for col, (strategy, (aug_dir, noaug_dir)) in enumerate(STRATEGIES.items()):
        ax = axes[col]

        for aug_key, folder in [("aug", aug_dir), ("noaug", noaug_dir)]:
            models_dir = RUNS_MODELS_DIR / folder
            style      = AUG_STYLE[aug_key]

            if not models_dir.exists():
                print(f"  [skip] {models_dir} — not found")
                ax.text(0.5, 0.5, f"{folder}\nnot found",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=9, color="grey")
                continue

            print(f"\n[{strategy} / {aug_key}]  Loading from {models_dir}")
            try:
                all_preds = collect_all_predictions(models_dir, loader, device)
            except FileNotFoundError as exc:
                print(f"  [skip] {exc}")
                continue

            mean_t, std_t = _conservation_curve(all_preds)

            ax.fill_between(t, mean_t - std_t, mean_t + std_t,
                            color=style["color"], alpha=FILL_ALPHA)
            ax.plot(t, mean_t,
                    color=style["color"], linewidth=style["lw"],
                    linestyle=style["linestyle"], label=style["label"])

        ax.axhline(0, color="black", linewidth=0.7, linestyle=":", alpha=0.5)
        ax.set_title(f"{strategy}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Time step", fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")

        panel_label = chr(ord("a") + col) + ")"
        ax.text(0.02, 0.97, panel_label, transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top")

        if col == 0:
            ax.set_ylabel("|S+I+R−N| / N  (%)", fontsize=10)

    # Single legend on the last panel
    axes[-1].legend(fontsize=9, loc="upper right")

    fig.tight_layout()
    out = out_dir / "fig_conservation_aug_vs_noaug.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


# ── ENTRY POINT ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = get_device()

    print(f"Loading test data: {DATA_PATH}")
    dataloaders = create_dataloaders(str(DATA_PATH), batch_size=64)
    test_loader = dataloaders.get('test', dataloaders['val'])
    n_samples   = len(test_loader.dataset)
    print(f"Test samples: {n_samples}")

    all_preds = collect_all_predictions(MODELS_DIR, test_loader, device)
    print(f"\nPredictions shape: {all_preds.shape}  (n_reps, n_test, T, 3)")

    # Global max deviation as a sanity header
    total   = all_preds.sum(axis=-1)
    max_dev = np.abs(total - N).max()
    mean_dev_pct = np.abs(total - N).mean() / N * 100
    print(f"Max |S+I+R−N|        : {max_dev:.4f}")
    print(f"Mean |S+I+R−N| / N   : {mean_dev_pct:.6f} %")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_conservation_over_time(all_preds, OUT_DIR)

    # ── Aug vs no-aug comparison across all sampling strategies ────────────────
    print("\n" + "=" * 60)
    print("AUG vs NO-AUG CONSERVATION CHECK")
    print("=" * 60)
    plot_conservation_aug_vs_noaug(test_loader, device, OUT_DIR)
