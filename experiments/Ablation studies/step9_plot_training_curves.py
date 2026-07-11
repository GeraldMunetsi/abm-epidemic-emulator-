

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path("experiments/Ablation studies/out/trained-models")
OUT_DIR  = Path("experiments/Ablation studies/out")

CONDITIONS = {
    'full' : 'Full model',
    'no_rff': 'No RFF',
    'no_spline': 'No B-spline',
}
COLOURS = {
    'full': 'blue',
    'no_rff' : 'darkorange',
    'no_spline': 'firebrick',
}


def load_histories(condition_dir: Path) -> list:
    files = sorted(
        condition_dir.glob("training_history_*.npy"),
        key=lambda p: int(p.stem.split("_")[-1])
    )
    return [np.load(f, allow_pickle=True).item() for f in files]


def stack_metric(histories: list, key: str) -> np.ndarray:
    arrays  = [np.array(h[key]) for h in histories]
    max_len = max(len(a) for a in arrays)
    padded  = np.full((len(arrays), max_len), np.nan)
    for i, a in enumerate(arrays):
        padded[i, :len(a)] = a
    return padded  # (n_reps, epochs)


def mean_std(arr: np.ndarray):
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


def _draw_panels(axes, histories, colour, n_reps_legend=False):
    """Fill three axes with the three panels. Reused by both plot functions."""
    loss_train = stack_metric(histories, 'train_loss')
    loss_val   = stack_metric(histories, 'val_loss')
    r2_train   = stack_metric(histories, 'train_r2_I')
    r2_val     = stack_metric(histories, 'val_r2_I')

    epochs = r2_val.shape[1]
    t      = np.arange(1, epochs + 1)

    loss_tr_m, loss_tr_s = mean_std(loss_train)
    loss_v_m,  loss_v_s  = mean_std(loss_val)
    r2_tr_m,   r2_tr_s   = mean_std(r2_train)
    r2_v_m,    r2_v_s    = mean_std(r2_val)

    # Panel 1 — Loss vs Epoch
    ax = axes[0]
    ax.plot(t, loss_tr_m, color=colour, linewidth=1.5, linestyle='--', label='Train')
    ax.fill_between(t, loss_tr_m - loss_tr_s, loss_tr_m + loss_tr_s,
                    color=colour, alpha=0.12)
    ax.plot(t, loss_v_m, color=colour, linewidth=2.0, label='Validation')
    ax.fill_between(t, loss_v_m - loss_v_s, loss_v_m + loss_v_s,
                    color=colour, alpha=0.20)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Loss vs Epoch', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Panel 2 — Train vs Validation R²_I
    ax2 = axes[1]
    ax2.plot(t, r2_tr_m, color=colour, linewidth=1.5, linestyle='--',
             label='Train $R^2_I$')
    ax2.fill_between(t, r2_tr_m - r2_tr_s, r2_tr_m + r2_tr_s,
                     color=colour, alpha=0.12)
    ax2.plot(t, r2_v_m, color=colour, linewidth=2.0, label='Validation $R^2_I$')
    ax2.fill_between(t, r2_v_m - r2_v_s, r2_v_m + r2_v_s,
                     color=colour, alpha=0.20)
    ax2.axhline(0, color='black', linewidth=0.7, linestyle=':')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('$R^2_I$', fontsize=11)
    ax2.set_title('Train vs Validation $R^2_I$', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--')

    # Panel 3 — Validation R²_I per replicate
    ax3 = axes[2]
    colors_rep = plt.cm.tab10(np.linspace(0, 0.9, len(histories)))
    for rep_idx, rep_r2 in enumerate(r2_val):
        lbl = f'Rep {rep_idx+1}' if n_reps_legend else ''
        ax3.plot(t, rep_r2, color=colors_rep[rep_idx],
                 linewidth=1.0, alpha=0.65, label=lbl)
    ax3.plot(t, r2_v_m, color='black', linewidth=2.0,
             linestyle='--', label='Mean', zorder=5)
    ax3.axhline(0, color='black', linewidth=0.7, linestyle=':')
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Validation $R^2_I$', fontsize=11)
    ax3.set_title('Validation $R^2_I$ per Replicate', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=7, ncol=2)
    ax3.grid(True, alpha=0.3, linestyle='--')


def plot_condition(label, histories, out_path, colour):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'Training Curves — {label}', fontsize=13, fontweight='bold')
    _draw_panels(axes, histories, colour, n_reps_legend=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_combined(all_histories: dict, out_path: Path) -> None:
    """3-row x 3-col: rows = panel, cols = condition."""
    conditions = list(CONDITIONS.keys())
    row_titles = [
        'Loss vs Epoch',
        'Train vs Validation $R^2_I$',
        'Validation $R^2_I$ per Replicate',
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    fig.suptitle('Ablation Study — Training Curves Comparison',
                 fontsize=14, fontweight='bold')

    for col, cond in enumerate(conditions):
        label   = CONDITIONS[cond]
        colour  = COLOURS[cond]
        hists   = all_histories[cond]

        loss_train = stack_metric(hists, 'train_loss')
        loss_val = stack_metric(hists, 'val_loss')
        r2_train = stack_metric(hists, 'train_r2_I')
        r2_val = stack_metric(hists, 'val_r2_I')

        epochs = r2_val.shape[1]
        t = np.arange(1, epochs + 1)

        loss_tr_m, loss_tr_s = mean_std(loss_train)
        loss_v_m,  loss_v_s = mean_std(loss_val)
        r2_tr_m,   r2_tr_s = mean_std(r2_train)
        r2_v_m,    r2_v_s = mean_std(r2_val)

        # Column header
        axes[0, col].set_title(label, fontsize=12, fontweight='bold')

        # Row 0 — Loss
        ax = axes[0, col]
        ax.plot(t, loss_tr_m, colour, linewidth=1.4, linestyle='--', label='Train')
        ax.fill_between(t, loss_tr_m - loss_tr_s, loss_tr_m + loss_tr_s,
                        color=colour, alpha=0.12)
        ax.plot(t, loss_v_m, colour, linewidth=2.0, label='Validation')
        ax.fill_between(t, loss_v_m - loss_v_s, loss_v_m + loss_v_s,
                        color=colour, alpha=0.20)
        ax.set_ylabel('Loss' if col == 0 else '', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Row 1 — Train vs Val R²_I
        ax = axes[1, col]
        ax.plot(t, r2_tr_m, colour, linewidth=1.4, linestyle='--',
                label='Train $R^2_I$')
        ax.fill_between(t, r2_tr_m - r2_tr_s, r2_tr_m + r2_tr_s,
                        color=colour, alpha=0.12)
        ax.plot(t, r2_v_m, colour, linewidth=2.0, label='Val $R^2_I$')
        ax.fill_between(t, r2_v_m - r2_v_s, r2_v_m + r2_v_s,
                        color=colour, alpha=0.20)
        ax.axhline(0, color='black', linewidth=0.6, linestyle=':')
        ax.set_ylabel('$R^2_I$' if col == 0 else '', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Row 2 — Val R²_I per replicate
        ax = axes[2, col]
        colors_rep = plt.cm.tab10(np.linspace(0, 0.9, len(hists)))
        for rep_idx, rep_r2 in enumerate(r2_val):
            ax.plot(t, rep_r2, color=colors_rep[rep_idx],
                    linewidth=1.0, alpha=0.65)
        ax.plot(t, r2_v_m, color='black', linewidth=2.0,
                linestyle='--', label='Mean', zorder=5)
        ax.axhline(0, color='black', linewidth=0.6, linestyle=':')
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Val $R^2_I$' if col == 0 else '', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')

    # Row labels
    for row, rl in enumerate(row_titles):
        axes[row, 0].set_ylabel(f'{rl}\n', fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


N_POPULATION  = 100000
LAST_N_EPOCHS = 20
SMOOTH_WINDOW = 7          # rolling-mean window for mean lines
SKIP_EPOCHS   = 3          # skip initial transient when setting auto y-limits


def _smooth(arr: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    """Centered rolling mean — edge values are padded to avoid boundary dips."""
    if len(arr) < window or window < 2:
        return arr
    pad = window // 2
    padded = np.pad(arr, (pad, pad), mode='edge')
    return np.convolve(padded, np.ones(window) / window, mode='valid')[:len(arr)]


def plot_training_summary(all_histories: dict, out_path: Path) -> None:
    """
    4-panel Training Summary figure comparing all three ablation conditions.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training Summary", fontsize=16, fontweight="bold")

    cond_labels = {
        'full' : 'Full model',
        'no_rff': 'No RFF',
        'no_spline': 'No B-spline',
    }

    ax_loss = axes[0, 0]
    ax_r2 = axes[0, 1]
    ax_mae = axes[1, 0]
    ax_conv = axes[1, 1]

    # skip the initial transient for R²_I and MAE panels so y-axis
    # auto-scales to the stable convergence region only
    STABLE_FROM = 10

    for cond, histories in all_histories.items():
        colour = COLOURS[cond]
        label  = cond_labels.get(cond, cond)

        train_loss = stack_metric(histories, 'train_loss')
        val_loss   = stack_metric(histories, 'val_loss')
        val_r2_I   = stack_metric(histories, 'val_r2_I')
        # use proper rel-MAE if available (requires retrained models);
        # fall back to MAE_I/N for older history files
        has_rel = all('val_rel_mae_i' in h for h in histories)
        if has_rel:
            val_rel_mae_i = stack_metric(histories, 'val_rel_mae_i')
        else:
            val_mae_i     = stack_metric(histories, 'val_mae_i')
            val_rel_mae_i = val_mae_i / N_POPULATION * 100

        epochs = val_loss.shape[1]
        ep = np.arange(1, epochs + 1)
        S  = min(STABLE_FROM, epochs - 1)   # safe slice index

        tl_m, tl_s= mean_std(train_loss)
        vl_m, vl_s = mean_std(val_loss)
        r2_m, r2_s = mean_std(val_r2_I)
        rel_mae_m, rel_mae_s = mean_std(val_rel_mae_i)

        # smooth mean lines; keep raw std bands (they already reflect spread)
        tl_s_smooth = _smooth(tl_m)
        vl_s_smooth = _smooth(vl_m)
        r2_smooth   = _smooth(r2_m)
        mae_smooth  = _smooth(rel_mae_m)

        # a) Loss — full range, log scale handles the transient fine
        ax_loss.plot(ep, tl_s_smooth, color=colour, linewidth=1.4,
                     linestyle='--', alpha=0.7)
        ax_loss.plot(ep, vl_s_smooth, color=colour, linewidth=2.0,
                     label=label)
        ax_loss.fill_between(ep, vl_m - vl_s, vl_m + vl_s,
                             color=colour, alpha=0.12)

        # b) R²_I — slice from STABLE_FROM so y auto-scales to convergence region
        ax_r2.plot(ep[S:], r2_smooth[S:], color=colour, linewidth=2.0, label=label)
        ax_r2.fill_between(ep[S:], (r2_m - r2_s)[S:], (r2_m + r2_s)[S:],
                           color=colour, alpha=0.12)

        # c) Relative MAE_I — same slice, drops the No B-spline spike entirely
        ax_mae.plot(ep[S:], mae_smooth[S:], color=colour, linewidth=2.0, label=label)
        ax_mae.fill_between(ep[S:],
                            (rel_mae_m - rel_mae_s)[S:],
                            (rel_mae_m + rel_mae_s)[S:],
                            color=colour, alpha=0.12)

        # d) Convergence — last epochs with dot markers
        tail = max(0, epochs - LAST_N_EPOCHS)
        ax_conv.plot(ep[tail:], _smooth(vl_m[tail:], window=3),
                     color=colour, linewidth=1.8,
                     marker='o', markersize=4, label=label)
        ax_conv.fill_between(ep[tail:],
                             vl_m[tail:] - vl_s[tail:],
                             vl_m[tail:] + vl_s[tail:],
                             color=colour, alpha=0.12)

    # ── Format a) 
    ax_loss.set_yscale('log')
    ax_loss.set_xlabel('Epoch', fontweight='bold')
    ax_loss.set_ylabel('Loss (log scale)', fontweight='bold')
    ax_loss.set_title('Training (- -) vs Validation (—) Loss')
    ax_loss.legend(fontsize=10, framealpha=0.85, edgecolor='lightgrey')
    ax_loss.grid(True, alpha=0.3)

    # Format b) 
    ax_r2.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.45)
    ax_r2.set_xlabel('Epoch', fontweight='bold')
    ax_r2.set_ylabel('R²_I', fontweight='bold')
    ax_r2.set_title('R² of I Compartment Evolution')
    ax_r2.legend(fontsize=10, framealpha=0.85, edgecolor='lightgrey')
    ax_r2.grid(True, alpha=0.3)
    ax_r2.set_ylim(-0.1, 1.05)

    # Format c) 
    ax_mae.set_xlabel('Epoch', fontweight='bold')
    ax_mae.set_ylabel('Rel-MAE_I  (% of mean I)', fontweight='bold')
    ax_mae.set_title('Relative MAE_I Evolution')
    ax_mae.legend(fontsize=10, framealpha=0.85, edgecolor='lightgrey')
    ax_mae.grid(True, alpha=0.3)

    #Format d) 
    ax_conv.set_xlabel('Epoch', fontweight='bold')
    ax_conv.set_ylabel('Validation Loss', fontweight='bold')
    ax_conv.set_title(f'Convergence (Last {LAST_N_EPOCHS} Epochs)')
    ax_conv.legend(fontsize=10, framealpha=0.85, edgecolor='lightgrey')
    ax_conv.grid(True, alpha=0.3)

    # Panel labels 
    for ax, lbl in zip([ax_loss, ax_r2, ax_mae, ax_conv],
                       ['a)', 'b)', 'c)', 'd)']):
        ax.text(0.02, 0.98, lbl, transform=ax.transAxes,
                fontsize=13, fontweight='bold', va='top', ha='left')

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


DATA_PATH= Path("experiments/mcmc-sampling/data/split/abm-data_split.pkl")
N_POP = 100000
N_TIMEPOINTS = 250


def plot_conservation_error(out_path: Path) -> None:
    """
    Loads replicate checkpoints for each ablation condition, runs inference on
    the test split, and plots mean |S(t)+I(t)+R(t)−N| / N (%) vs time step
    with ±1σ shaded band.  One line per condition.
    """
    try:
        import torch
        from step0_model import create_ablation_model
        from utils import create_dataloaders, get_device
    except ImportError as exc:
        return

    device = get_device()
    dataloaders = create_dataloaders(str(DATA_PATH), batch_size=64)
    test_loader = dataloaders.get('test', dataloaders['val'])
    t = np.arange(N_TIMEPOINTS)

    fig, (ax_all, ax_zoom) = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Conservation Error by Ablation Condition",
                 fontsize=14, fontweight='bold')

    curves = {}   # cond -> (mean_t, std_t)

    for cond, label in CONDITIONS.items():
        colour   = COLOURS[cond]
        cond_dir = BASE_DIR / cond

        model_paths = sorted(
            cond_dir.glob("best_balanced_mlp_model_*.pt"),
            key=lambda p: int(p.stem.split('_')[-1]),
        ) if cond_dir.exists() else []

        if not model_paths:
            print(f"[skip] {cond} — no checkpoints in {cond_dir}")
            continue

        print(f"[conservation] {cond}: {len(model_paths)} replicate(s)")
        per_rep_curves = []

        for mp in model_paths:
            ckpt   = torch.load(mp, map_location=device, weights_only=False)
            config = ckpt.get('config', {
                'n_params': 3, 'n_fourier': 64, 'sigma': 1.0,
                'fusion_hidden': 128, 'latent_dim': 64, 'decoder_hidden': 64,
                'dropout': 0.3, 'n_knots': 8, 'n_timepoints': N_TIMEPOINTS,
                'total_population': N_POP,
            })
            sd = ckpt['model_state_dict']
            sd.pop('temporal_decoder.t_grid', None)
            model = create_ablation_model(cond, config).to(device)
            model.load_state_dict(sd, strict=True)
            model.eval()

            all_preds = []
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    all_preds.append(
                        model(batch, n_timesteps=N_TIMEPOINTS).cpu().numpy()
                    )
            preds = np.concatenate(all_preds, axis=0)   # (n_test, T, 3)

            err = np.abs(preds.sum(axis=-1) - N_POP) / N_POP * 100  # (n_test, T)
            per_rep_curves.append(err.mean(axis=0))                  # (T,)

        stacked = np.stack(per_rep_curves, axis=0)   # (n_reps, T)
        mean_t  = stacked.mean(axis=0)
        std_t   = stacked.std(axis=0)
        curves[cond] = (mean_t, std_t)

    # Panel a) all conditions on full scale 
    for cond, (mean_t, std_t) in curves.items():
        colour = COLOURS[cond]
        label  = CONDITIONS[cond]
        ax_all.plot(t, mean_t, color=colour, linewidth=2.0, label=label)
        ax_all.fill_between(t, mean_t - std_t, mean_t + std_t,
                            color=colour, alpha=0.15)

    ax_all.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.4)
    ax_all.set_xlabel('Time step', fontweight='bold')
    ax_all.set_ylabel('|S+I+R−N| / N  (%)', fontweight='bold')
    ax_all.set_title('All conditions')
    ax_all.legend(fontsize=10, framealpha=0.85, edgecolor='lightgrey')
    ax_all.grid(True, alpha=0.3)
    ax_all.text(0.02, 0.98, 'a)', transform=ax_all.transAxes,
                fontsize=13, fontweight='bold', va='top', ha='left')

    # Panel b) zoom: Full model & No RFF only 
    for cond in ('full', 'no_rff'):
        if cond not in curves:
            continue
        mean_t, std_t = curves[cond]
        colour = COLOURS[cond]
        label  = CONDITIONS[cond]
        ax_zoom.plot(t, mean_t, color=colour, linewidth=2.0, label=label)
        ax_zoom.fill_between(t, mean_t - std_t, mean_t + std_t,
                             color=colour, alpha=0.20)

    ax_zoom.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.4)
    ax_zoom.set_xlabel('Time step', fontweight='bold')
    ax_zoom.set_ylabel('|S+I+R−N| / N  (%)', fontweight='bold')
    ax_zoom.set_title('Full model & No RFF  (zoomed)')
    ax_zoom.legend(fontsize=10, framealpha=0.85, edgecolor='lightgrey')
    ax_zoom.grid(True, alpha=0.3)
    ax_zoom.text(0.02, 0.98, 'b)', transform=ax_zoom.transAxes,
                 fontsize=13, fontweight='bold', va='top', ha='left')
    ax_zoom.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{x:.2e}')
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    all_histories = {}
    for cond, label in CONDITIONS.items():
        cond_dir = BASE_DIR / cond
        if not cond_dir.exists():
            print(f"Skipping {cond} — directory not found")
            continue
        hists = load_histories(cond_dir)
        if not hists:
            print(f"Skipping {cond} — no history files found")
            continue
        print(f"Loaded {len(hists)} replicates for '{cond}'")
        all_histories[cond] = hists
        plot_condition(label, hists, cond_dir / 'training_curves.png', COLOURS[cond])

    if len(all_histories) > 1:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        plot_combined(all_histories, OUT_DIR / 'fig_ablation_training_curves.png')
        plot_training_summary(all_histories, OUT_DIR / 'fig_ablation_training_summary.png')

    plot_conservation_error(OUT_DIR / 'fig_ablation_conservation_error.png')


if __name__ == "__main__":
    main()
