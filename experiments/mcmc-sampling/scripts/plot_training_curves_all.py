"""
Training curves for all 6 sampling-strategy conditions.

Produces per-condition 4-panel figures (matching the ablation-study style):
    a) Train (--) vs Validation (—) Loss  [log scale]
    b) R²_I of I Compartment Evolution
    c) Relative MAE_I (% of N)
    d) Convergence — last 20 epochs validation loss  [dot markers]

Also produces a combined 6-row × 4-col overview figure.

Run from repo root:
    python "experiments/mcmc-sampling/scripts/plot_training_curves_all.py"
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── CONSTANTS ──────────────────────────────────────────────────────────────────
N             = 100_000
LAST_N_EPOCHS = 20
MODELS_ROOT   = Path("experiments/runs/models")
OUT_DIR       = Path("experiments/runs/out/training_curves")

CONDITIONS = {
    "LHS_aug"     : "LHS — Augmented",
    "LHS_noaug"   : "LHS — No Augmentation",
    "MCMC_aug"    : "MCMC — Augmented",
    "MCMC_noaug"  : "MCMC — No Augmentation",
    "Random_aug"  : "Random — Augmented",
    "Random_noaug": "Random — No Augmentation",
}


# ── DATA HELPERS ───────────────────────────────────────────────────────────────
def load_histories(condition_dir: Path) -> list:
    files = sorted(
        condition_dir.glob("training_history_*.npy"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    return [np.load(f, allow_pickle=True).item() for f in files]


def stack(histories: list, key: str) -> np.ndarray:
    """Stack a metric across replicates, NaN-padding to equal length."""
    arrays  = [np.array(h[key]) for h in histories]
    max_len = max(len(a) for a in arrays)
    out     = np.full((len(arrays), max_len), np.nan)
    for i, a in enumerate(arrays):
        out[i, :len(a)] = a
    return out   # (n_reps, epochs)


def _legend(ax, n_reps: int, fontsize: int = 7) -> None:
    """Two-column legend matching the reference style."""
    ax.legend(fontsize=fontsize, ncol=2, loc="best",
              framealpha=0.85, edgecolor="lightgrey")


# ── SINGLE-CONDITION 4-PANEL FIGURE ───────────────────────────────────────────
def plot_condition(condition_key: str, histories: list, out_path: Path) -> None:
    label    = CONDITIONS[condition_key]
    n_reps   = len(histories)
    rep_cols = plt.cm.tab10(np.linspace(0, 0.9, n_reps))

    train_loss = stack(histories, "train_loss")
    val_loss   = stack(histories, "val_loss")
    val_r2_I   = stack(histories, "val_r2_I")
    val_mae_i  = stack(histories, "val_mae_i")

    epochs = val_loss.shape[1]
    ep     = np.arange(1, epochs + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training Summary", fontsize=16, fontweight="bold")

    # subtitle with condition name
    fig.text(0.5, 0.935, label, ha="center", va="top",
             fontsize=11, color="#444444")

    ax_loss = axes[0, 0]
    ax_r2   = axes[0, 1]
    ax_mae  = axes[1, 0]
    ax_conv = axes[1, 1]

    for i in range(n_reps):
        c   = rep_cols[i]
        lbl = f"Model {i + 1}"

        # a) Loss — train dashed, val solid
        ax_loss.plot(ep, train_loss[i], color=c, linewidth=1.0,
                     linestyle="--", alpha=0.55)
        ax_loss.plot(ep, val_loss[i], color=c, linewidth=1.2,
                     alpha=0.8, label=lbl)

        # b) R²_I
        ax_r2.plot(ep, val_r2_I[i], color=c, linewidth=1.2,
                   alpha=0.8, label=lbl)

        # c) Relative MAE_I
        ax_mae.plot(ep, val_mae_i[i] / N * 100, color=c, linewidth=1.2,
                    alpha=0.8, label=lbl)

        # d) Convergence — last LAST_N_EPOCHS epochs, with dot markers
        tail = max(0, epochs - LAST_N_EPOCHS)
        ax_conv.plot(ep[tail:], val_loss[i, tail:], color=c,
                     linewidth=1.4, alpha=0.85, marker="o",
                     markersize=3, label=lbl)

    # ── a) format ──────────────────────────────────────────────────────────────
    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("Epoch", fontweight="bold")
    ax_loss.set_ylabel("Loss (log scale)", fontweight="bold")
    ax_loss.set_title("Training (- -) vs Validation (—) Loss")
    _legend(ax_loss, n_reps)
    ax_loss.grid(True, alpha=0.3)

    # ── b) format ──────────────────────────────────────────────────────────────
    ax_r2.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.45)
    ax_r2.set_xlabel("Epoch", fontweight="bold")
    ax_r2.set_ylabel("R²", fontweight="bold")
    ax_r2.set_title("R² of I Compartment Evolution")
    _legend(ax_r2, n_reps)
    ax_r2.grid(True, alpha=0.3)

    # ── c) format ──────────────────────────────────────────────────────────────
    ax_mae.set_xlabel("Epoch", fontweight="bold")
    ax_mae.set_ylabel("Relative MAE_I (% of N)", fontweight="bold")
    ax_mae.set_title("Relative MAE_I Evolution")
    _legend(ax_mae, n_reps)
    ax_mae.grid(True, alpha=0.3)

    # ── d) format ──────────────────────────────────────────────────────────────
    ax_conv.set_xlabel("Epoch", fontweight="bold")
    ax_conv.set_ylabel("Validation Loss", fontweight="bold")
    ax_conv.set_title(f"Convergence (Last {LAST_N_EPOCHS} Epochs)")
    _legend(ax_conv, n_reps)
    ax_conv.grid(True, alpha=0.3)

    # ── panel labels a)–d) ─────────────────────────────────────────────────────
    for ax, lbl in zip([ax_loss, ax_r2, ax_mae, ax_conv],
                       ["a)", "b)", "c)", "d)"]):
        ax.text(0.02, 0.98, lbl, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top", ha="left")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── COMBINED OVERVIEW: 6-ROW × 4-COL ──────────────────────────────────────────
def plot_combined(all_histories: dict, out_path: Path) -> None:
    """
    One row per condition, four columns matching the single-condition panels.
    Row labels on the left; column titles at the top of the first row.
    """
    n_cond    = len(all_histories)
    fig, axes = plt.subplots(n_cond, 4, figsize=(22, 4.5 * n_cond))
    fig.suptitle("Training Summary — All Conditions",
                 fontsize=15, fontweight="bold")

    col_titles = [
        "Train (--) vs Val (—) Loss",
        "R²_I Evolution",
        "Relative MAE_I (% of N)",
        f"Convergence (Last {LAST_N_EPOCHS} Ep.)",
    ]

    for row, (cond_key, histories) in enumerate(all_histories.items()):
        label    = CONDITIONS[cond_key]
        n_reps   = len(histories)
        rep_cols = plt.cm.tab10(np.linspace(0, 0.9, n_reps))

        train_loss = stack(histories, "train_loss")
        val_loss   = stack(histories, "val_loss")
        val_r2_I   = stack(histories, "val_r2_I")
        val_mae_i  = stack(histories, "val_mae_i")

        epochs = val_loss.shape[1]
        ep     = np.arange(1, epochs + 1)

        ax_loss = axes[row, 0]
        ax_r2   = axes[row, 1]
        ax_mae  = axes[row, 2]
        ax_conv = axes[row, 3]

        for i in range(n_reps):
            c   = rep_cols[i]
            lbl = f"Model {i + 1}"

            ax_loss.plot(ep, train_loss[i], color=c, lw=0.8,
                         linestyle="--", alpha=0.5)
            ax_loss.plot(ep, val_loss[i], color=c, lw=1.1,
                         alpha=0.8, label=lbl)

            ax_r2.plot(ep, val_r2_I[i], color=c, lw=1.1,
                       alpha=0.8, label=lbl)

            ax_mae.plot(ep, val_mae_i[i] / N * 100, color=c, lw=1.1,
                        alpha=0.8, label=lbl)

            tail = max(0, epochs - LAST_N_EPOCHS)
            ax_conv.plot(ep[tail:], val_loss[i, tail:], color=c,
                         lw=1.3, alpha=0.85, marker="o",
                         markersize=2.5, label=lbl)

        # Format
        ax_loss.set_yscale("log")
        ax_r2.axhline(0, color="black", lw=0.7, linestyle="--", alpha=0.4)

        for ax in [ax_loss, ax_r2, ax_mae, ax_conv]:
            ax.grid(True, alpha=0.3)
            if row == n_cond - 1:
                ax.set_xlabel("Epoch", fontsize=9, fontweight="bold")

        ax_loss.set_ylabel(label + "\n\nLoss", fontsize=8)
        ax_r2.set_ylabel("R²_I", fontsize=8)
        ax_mae.set_ylabel("Rel. MAE_I (%N)", fontsize=8)
        ax_conv.set_ylabel("Val Loss", fontsize=8)

        # Column titles on first row only
        if row == 0:
            for ci, title in enumerate(col_titles):
                axes[0, ci].set_title(title, fontsize=10, fontweight="bold")

        # Panel letter top-left of loss column
        ax_loss.text(0.02, 0.97, f"{chr(ord('a') + row)})",
                     transform=ax_loss.transAxes,
                     fontsize=10, fontweight="bold", va="top")

        # Legend on convergence column only (keeps figure uncluttered)
        ax_conv.legend(fontsize=6, ncol=2, loc="upper right",
                       framealpha=0.8, edgecolor="lightgrey")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── ENTRY POINT ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_histories = {}

    for cond_key, label in CONDITIONS.items():
        cond_dir = MODELS_ROOT / cond_key
        if not cond_dir.exists():
            print(f"[skip] {cond_dir} not found")
            continue

        histories = load_histories(cond_dir)
        if not histories:
            print(f"[skip] {cond_key} — no training_history_*.npy files")
            continue

        print(f"Loaded {len(histories):2d} replicates for '{cond_key}'")
        all_histories[cond_key] = histories

        plot_condition(
            cond_key,
            histories,
            OUT_DIR / cond_key / "training_curves.png",
        )

    if len(all_histories) > 1:
        plot_combined(
            all_histories,
            OUT_DIR / "fig_training_curves_all_conditions.png",
        )

    print("\nDone.")
