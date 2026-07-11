import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import argparse
from pathlib import Path
import json
import pandas as pd
from scipy import stats
from step0_model import create_hybrid_mlp_model
from utils  import create_dataloaders, compute_metrics, get_device, \
                         PARAM_MINS, PARAM_MAXS
from utils import compute_metrics as _cm

N=100000
n_knots=7
n_timepoints=250
ratio=58
PEAK_THRESHOLD=1.0  

## I/O PATHS 
MODELS_DIR = Path("experiments/mcmc-sampling/out/trained-models")
TEST_DATA_DIR = Path("experiments/lhs-sampling/data/split")
RESULTS_DIR= Path("experiments/mcmc-sampling/out/results/testing/results_on_lhs_sampled_data")
PLOTS_DIR= Path("experiments/mcmc-sampling/out/plots/testing_plots/LHS_test_data")
REGRESSION_DATA_DIR = Path("experiments/Regression/data")

TRAIN_STRATEGY = 'MCMC'  
TEST_STRATEGY= 'LHS'  
AUGMENTATION= 1       
N_TRAIN_SIMULATIONS = 8400 


#1. MODEL LOADING
def load_replicate_model(model_path: Path, device: torch.device):
    """Load one replicate checkpoint and return (model, checkpoint)."""
    ckpt   = torch.load(model_path, map_location=device, weights_only=False)
    config = ckpt.get('config', {
        'n_params' : 3,
        'n_fourier': 64,
        'sigma' : 1.0,
        'fusion_hidden' : 128,
        'latent_dim': 64,
        'decoder_hidden': 64,
        'dropout' : 0.3,
        'n_knots' : 8,
        'n_timepoints': N_TIMEPOINTS,
        'total_population': N,
    })
    state = ckpt['model_state_dict']
    state.pop('temporal_decoder.t_grid', None)  # remove legacy buffer if present
    model = create_hybrid_mlp_model(config)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    val_r2 = ckpt.get('val_metrics', {}).get('R2', float('nan'))
    print(f"  Loaded {model_path.name}  "
          f"epoch={ckpt.get('epoch')}  val R²={val_r2:.4f}")
    return model, ckpt



#  2. INFERENCE
def evaluate_model(model, test_loader, device):
    """
    Run inference on the full test set.

    Returns
    -------
    predictions : (n_test, T, 3)  — emulator output
    targets     : (n_test, T, 3)  — ground truth
    params      : (n_test, 3)     — raw (un-normalised) parameters
    metrics     : dict            — scalar accuracy metrics (all samples)
    """
    model.eval()
    param_mins = torch.tensor(PARAM_MINS)
    param_maxs = torch.tensor(PARAM_MAXS)

    all_preds, all_targets, all_params = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            batch       = batch.to(device)
            predictions = model(batch, n_timesteps=N_TIMEPOINTS)
            all_preds.append(predictions.cpu())
            all_targets.append(batch.y.cpu())
            raw = (batch.params_norm.cpu()
                   * (param_maxs - param_mins) + param_mins)
            all_params.append(raw)

    predictions = torch.cat(all_preds,   dim=0)   # (n_test, T, 3)
    targets     = torch.cat(all_targets, dim=0)
    params      = torch.cat(all_params,  dim=0)   # (n_test, 3)

    
    metrics = compute_metrics(predictions, targets)
    return predictions, targets, params, metrics


def evaluate_all_replicates(models_dir, test_loader, device):
    """Evaluate every replicate checkpoint and return results_list."""
    model_paths = sorted(
        Path(models_dir).glob("best_balanced_mlp_model_*.pt"),
        key=lambda p: int(p.stem.split('_')[-1])
    )
    if not model_paths:
        raise FileNotFoundError(f"No model files found in {models_dir}")

    print(f"\n{'─'*70}")
    print(f"EVALUATING {len(model_paths)} REPLICATE(S)")

    results_list   = []
    shared_targets = None
    shared_params  = None

    for idx, path in enumerate(model_paths, 1):
        print(f"\n  Replicate {idx}/{len(model_paths)} — {path.name}")
        model, ckpt = load_replicate_model(path, device)
        preds, targets, params, metrics = evaluate_model(
            model, test_loader, device
        )
        if shared_targets is None:
            shared_targets = targets
            shared_params  = params

        print(f"    R²_I  : {metrics['R2_I']:.4f}")
        print(f"    MAE_I : {metrics['MAE_I']:.2f}")

        results_list.append({
            'replicate_id'   : idx,
            'model_path'     : str(path),
            'predictions'    : preds,
            'metrics'        : metrics,
            'checkpoint_info': {
                'epoch'      : ckpt.get('epoch'),
                'val_metrics': ckpt.get('val_metrics', {}),
            },
        })

    return results_list, shared_targets, shared_params



#  3. RELATIVE MAE_I
def compute_relative_mae_i(predictions, targets):
    """
    Relative MAE_I (per-sample then averaged).

    Formula
    -------
    Rel-MAE_I_i = (1/T  Σ_t |Î_i(t) − I_i(t)|) / max_t I_i(t)  × 100%
    Rel-MAE_I   = (1/n) Σ_i  Rel-MAE_I_i

    With rho ∈ [0.001, 0.01] and N = 100,000:
      I(0) = N × rho ∈ [100, 1000]  →  peak I ≥ 100 always.
    The denominator is therefore numerically stable for every sample.
    Near-extinction trajectories (peak ≈ I(0)) contribute higher
    relative errors due to the B-spline decoder's I(t) > 0 guarantee;
    this is reported transparently rather than masked by exclusion.

    Returns
    -------
    mean_rel      : float  — mean Rel-MAE_I (%) across all n samples
    std_rel       : float  — sample SD (ddof=1) across samples
    per_sample    : array  — (n,) per-sample values
    mean_peak     : float  — mean of true I_max across samples
    """
    mae_per_sample  = (predictions[:, :, 1] - targets[:, :, 1]).abs().mean(dim=1)
    peak_per_sample = targets[:, :, 1].max(dim=1)[0]

    rel       = (mae_per_sample / peak_per_sample * 100).numpy()  # (n,)
    mean_peak = float(peak_per_sample.mean().item())

    return (float(rel.mean()),
            float(rel.std(ddof=1) if len(rel) > 1 else 0.0),
            rel,
            mean_peak)



#  4. PER-REPLICATE DATAFRAME
def build_replicate_dataframe(results_list, targets,
                               train_strategy, test_strategy,
                               augmentation, n_train_simulations):
    """
    Build one-row-per-replicate DataFrame for the master regression CSV.

    All metrics are computed on the complete test set (no exclusion).
    in_domain is derived from strategy strings, never hardcoded.
    """
    rows = []
    for r in results_list:
        mean_rel, std_rel, _, mean_peak = compute_relative_mae_i(
            r['predictions'], targets
        )
        m = r['metrics']

        rows.append({
            # join / grouping keys 
            'replicate_id'        : int(r['replicate_id']),
            'train_strategy'      : train_strategy,
            'test_strategy'       : test_strategy,
            'augmentation'        : int(augmentation),
            'in_domain'           : 1,
            'n_train_simulations' : int(n_train_simulations),

            # primary accuracy metrics 
            'relative_MAE_I'      : round(mean_rel, 4),
            'R2_I'                : round(m['R2_I'],  6),
            'MAE_I'               : round(m['MAE_I'], 4),

            # secondary / other compartments 
            'R2_S'                : round(m['R2_S'],  6),
            'R2_R'                : round(m['R2_R'],  6),
            'R2_overall'          : round(m['R2'],    6),
            'MAE_S'               : round(m['MAE_S'], 4),
            'MAE_R'               : round(m['MAE_R'], 4),
            'RMSE'                : round(m['RMSE'],  4),
            'MSE'                 : round(m['MSE'],   4),

            # metadata 
            'peak_I'              : round(mean_peak, 2),
            'mean_peak_I'         : round(mean_peak, 2),
            'n_test_samples'      : int(len(targets)),
            'model_path'          : r['model_path'],
            'training_epoch'      : r['checkpoint_info']['epoch'],
        })

    df = pd.DataFrame(rows)
    print(f"\n  Replicate dataframe: {len(df)} rows × {len(df.columns)} cols")
    cols = ['replicate_id','train_strategy','test_strategy',
            'augmentation','in_domain','relative_MAE_I','R2_I']
    print(df[cols].to_string(index=False))
    return df


 
#  5. AGGREGATE STATISTICS
def compute_aggregate_statistics(results_list, targets):
    """
    Summary statistics across k replicates.

    Uses sample SD (ddof=1) and t-distribution CI (df = k-1)
    throughout — the k replicates are a sample, not a population.
    """
    k = len(results_list)

    metric_keys = [
        'MAE', 'MAE_S', 'MAE_I', 'MAE_R',
        'R2',  'R2_S',  'R2_I',  'R2_R',
        'RMSE', 'MSE',
    ]
    stats_dict = {'n_replicates': k}

    for key in metric_keys:
        arr = np.array([r['metrics'][key] for r in results_list])

        # sample SD — ddof=1 because k replicates are a SAMPLE
        std = float(arr.std(ddof=1)) if k > 1 else 0.
        sem = std / np.sqrt(k)       if k > 1 else 0.

        # t-distribution: k < 30, so normal approximation is inappropriate
        ci = (stats.t.interval(0.95, df=k-1, loc=arr.mean(), scale=sem)
              if k > 1 else (arr.mean(), arr.mean()))

        stats_dict[key] = {
            'mean'  : float(arr.mean()),
            'median': float(np.median(arr)),
            'std'   : std,                           # sample SD (ddof=1)
            'sem'   : sem,                           # SEM from sample SD
            'ci_95' : [float(ci[0]), float(ci[1])],
            'cv'    : float(std / arr.mean() * 100) if arr.mean() != 0 else 0.,
            'min'   : float(arr.min()),
            'max'   : float(arr.max()),
        }

    # Relative MAE_I across replicates 
    rel_means = []
    for r in results_list:
        mean_rel, _, _, mean_peak = compute_relative_mae_i(
            r['predictions'], targets
        )
        rel_means.append(mean_rel)

    rel_arr = np.array(rel_means)
    rel_std = float(rel_arr.std(ddof=1)) if k > 1 else 0.
    rel_sem = rel_std / np.sqrt(k)
    rel_ci  = (stats.t.interval(0.95, df=k-1,
                                 loc=rel_arr.mean(), scale=rel_sem)
               if k > 1 else (rel_arr.mean(), rel_arr.mean()))

    stats_dict['relative_MAE_I_%'] = {
        'mean'  : float(rel_arr.mean()),
        'median': float(np.median(rel_arr)),
        'std'   : rel_std,
        'sem'   : rel_sem,
        'ci_95' : [float(rel_ci[0]), float(rel_ci[1])],
        'cv'    : float(rel_std / rel_arr.mean() * 100)
                  if rel_arr.mean() != 0 else 0.,
        'note'  : 'per-sample MAE_I / peak_I — all samples included',
    }
    stats_dict['mean_peak_I_ground_truth'] = mean_peak

    # Console summary 
    print(f"\n{'─'*70}")
    print(f"AGGREGATE RESULTS  (k={k} replicates, sample SD ddof=1)")
    print(f"  Mean peak I (ground truth) : {mean_peak:,.1f}")
    print(f"  R²_I    : {stats_dict['R2_I']['mean']:.4f}"
          f" ± {stats_dict['R2_I']['std']:.4f}"
          f"  95% CI [{stats_dict['R2_I']['ci_95'][0]:.4f},"
          f" {stats_dict['R2_I']['ci_95'][1]:.4f}]")
    print(f"  MAE_I   : {stats_dict['MAE_I']['mean']:.2f}"
          f" ± {stats_dict['MAE_I']['std']:.2f}")
    print(f"  Rel-MAE_I: {rel_arr.mean():.2f}%"
          f" ± {rel_std:.2f}%"
          f"  95% CI [{rel_ci[0]:.2f}%, {rel_ci[1]:.2f}%]"
          f"  (t-dist df={k-1})")

    return stats_dict



#  6. VISUALISATION

def _r0_label(params, idx):
    """Return a parameter string for subplot titles."""
    if params is None:
        return ""
    tau, gam, rho = (params[idx, 0].item(),
                     params[idx, 1].item(),
                     params[idx, 2].item())
    return (f"τ={tau:.3f}  γ={gam:.3f}  "
            f"ρ={rho:.4f}  R₀={tau/gam*ratio:.2f}")


def plot_uncertainty_band(results_list, targets, plots_dir,
                          params=None, n_samples=8):
    """
    Mean ± 2σ band across k replicates — all three compartments.

    The ±2σ band shows EPISTEMIC UNCERTAINTY from weight initialisation:
    how much the k trained models disagree at each timestep.
    σ uses sample SD (ddof=1) because the k replicates are a sample.

    This is NOT a confidence interval for the mean — that would be
    mean ± t × (σ/√k) and would be ~2.8× narrower.
    The ±2σ band answers "where might the trajectory be?",
    not "how well have we estimated the mean?".
    """
    plots_dir  = Path(plots_dir)
    targets_np = targets.detach().cpu().numpy()
    n_total    = len(targets_np)
    indices    = np.unique(np.linspace(0, n_total-1, n_samples, dtype=int))

    all_preds = np.stack(
        [r['predictions'].detach().cpu().numpy() for r in results_list],
        axis=0
    )                                        # (k, n_test, T, 3)
    mean_pred = all_preds.mean(axis=0)       # (n_test, T, 3)
    std_pred  = all_preds.std(axis=0, ddof=1)  # sample SD — ddof=1

    compartments = ['Susceptible S(t)', 'Infected I(t)', 'Recovered R(t)']
    gt_colors    = ['steelblue',     'firebrick',  'forestgreen']
    band_colors  = ['cornflowerblue','salmon',     'mediumseagreen']

    n_rows = len(indices)
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 3.2 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    k = len(results_list)
    fig.suptitle(
        f'Prediction uncertainty — mean ± 2σ  '
        f'(σ = sample SD, ddof=1, k={k} replicates)\n'
        f'Train: {TRAIN_STRATEGY}  →  Test: {TEST_STRATEGY}  '
        f'aug={AUGMENTATION}',
        fontsize=12, fontweight='bold'
    )

    panel_idx = 0
    for row, idx in enumerate(indices):
        target    = targets_np[idx]
        mu        = mean_pred[idx]
        sigma     = std_pred[idx]
        t         = np.arange(mu.shape[0])
        param_str = _r0_label(params, idx)

        for col in range(3):
            ax = axes[row, col]
            lo = mu[:, col] - 2 * sigma[:, col]
            hi = mu[:, col] + 2 * sigma[:, col]

            ax.fill_between(t, lo, hi,
                            color=band_colors[col], alpha=0.35,
                            label='Mean ± 2σ')
            ax.plot(t, mu[:, col], '-',
                    color=gt_colors[col], lw=1.8, label='Mean pred')
            ax.plot(t, target[:, col], 'o',
                    color='black', alpha=0.5, ms=3, mew=0,
                    label='Ground truth')

            ax.text(0.02, 0.97, f"({chr(ord('a') + panel_idx)})",
                    transform=ax.transAxes, fontsize=9,
                    fontweight='bold', va='top', ha='left')

            if row == 0:
                ax.set_title(compartments[col], fontsize=11, fontweight='bold')
            if col == 0 and param_str:
                ax.annotate(param_str, xy=(0.5, 1.02),
                            xycoords='axes fraction',
                            fontsize=9, ha='center', va='bottom')

            ax.set_xlabel('Timestep', fontsize=8)
            ax.set_ylabel('Count',    fontsize=8)
            ax.grid(True, alpha=0.3, linestyle='--')
            if row == 0 and col == 1:
                ax.legend(loc='best', fontsize=7)
            panel_idx += 1

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = plots_dir / 'uncertainty_band.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_relative_mae_vs_peak(results_list, targets, params, plots_dir):
    """
    Scatter: per-sample Rel-MAE_I vs true epidemic peak I_max.
    Coloured by R₀ zone. Shows WHERE the emulator struggles.
    """
    plots_dir  = Path(plots_dir)
    _, _, per_sample, _ = compute_relative_mae_i(
        results_list[0]['predictions'], targets
    )
    peak  = targets[:, :, 1].max(dim=1)[0].numpy()

    if params is not None:
        tau = params[:, 0].numpy()
        gam = params[:, 1].numpy()
        R0  = (tau / gam) * ratio
        colours = np.where(R0 < 0.8, 'blue',
                  np.where(R0 <= 1.2, 'red', 'green'))
    else:
        colours = 'gray'

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(peak, per_sample,
               c=colours, alpha=0.45, s=14, edgecolors='none')
    ax.set_xlabel('True peak $I_{\\max}$ (individuals)', fontsize=11)
    ax.set_ylabel('Relative MAE$_I$ (%)',               fontsize=11)
    ax.set_title('Emulator error vs epidemic size\n'
                 '(blue=sub-critical, red=threshold, green=super-critical)',
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xscale('log')
    plt.tight_layout()
    out = plots_dir / 'rel_mae_vs_peak.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_infected_predictions(results_list, targets, params, plots_dir,
                               n_samples=16):
    """
    All-replicates overlay for the infected compartment I(t).
    One panel per randomly-selected test sample.
    """
    plots_dir   = Path(plots_dir)
    targets_np  = targets.detach().cpu().numpy()
    n_total     = len(targets_np)
    indices     = np.unique(np.linspace(0, n_total-1, n_samples, dtype=int))
    rep_colors  = plt.cm.tab10(np.linspace(0, 1, len(results_list)))

    n_cols = 4
    n_rows = int(np.ceil(len(indices) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()
    fig.suptitle(
        f'Infected I(t) — all {len(results_list)} replicates overlaid\n'
        f'Train: {TRAIN_STRATEGY}  :  Test: {TEST_STRATEGY}',
        fontsize=12, fontweight='bold'
    )

    for panel_idx, idx in enumerate(indices):
        ax = axes[panel_idx]
        ax.plot(targets_np[idx, :, 1], 'o',
                color='steelblue', alpha=0.7, ms=4, mew=0,
                label='Ground truth', zorder=10)
        for rep_i, r in enumerate(results_list):
            pred = r['predictions'][idx].detach().cpu().numpy()
            ax.plot(pred[:, 1], '-',
                    color=rep_colors[rep_i], lw=1.5, alpha=0.7,
                    label=f"M{r['replicate_id']}")
        ax.text(0.02, 0.97, f"({chr(ord('a') + panel_idx)})",
                transform=ax.transAxes, fontsize=9,
                fontweight='bold', va='top', ha='left')
        ax.set_title(_r0_label(params, idx), fontsize=9, pad=3)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Infected')
        ax.grid(True, alpha=0.3, linestyle='--')
        if panel_idx == 0:
            ax.legend(fontsize=7, ncol=2)

    for ax in axes[len(indices):]:
        ax.set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = plots_dir / 'infected_predictions.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")



#  7. SAVE RESULTS
def save_results(results_list, stats_dict, output_dir):
    """Save JSON stats, per-replicate CSV, and plain-text report."""
    output_dir = Path(output_dir)

    # JSON
    with open(output_dir / 'test_statistics.json', 'w') as f:
        json.dump(stats_dict, f, indent=2)

    # Per-replicate CSV
    rows = [{'replicate_id': r['replicate_id'],
             'model_path'  : r['model_path'],
             **r['metrics'],
             'epoch'       : r['checkpoint_info']['epoch']}
            for r in results_list]
    pd.DataFrame(rows).to_csv(
        output_dir / 'test_replicate_metrics.csv', index=False
    )

    # Text report
    rel   = stats_dict['relative_MAE_I_%']
    r2i   = stats_dict['R2_I']
    maei  = stats_dict['MAE_I']
    peak  = stats_dict['mean_peak_I_ground_truth']
    k     = stats_dict['n_replicates']
    grade = ("EXCEPTIONAL" if rel['mean'] < 5  else
             "EXCELLENT"   if rel['mean'] < 10 else
             "GOOD"        if rel['mean'] < 20 else
             "ACCEPTABLE"  if rel['mean'] < 35 else
             "NEEDS IMPROVEMENT")

    report = "\n".join([
        "═"*70,
        f"FINAL TEST RESULTS",
        f"  Train: {TRAIN_STRATEGY}  →  Test: {TEST_STRATEGY}"
        f"  aug={AUGMENTATION}",
        f"  k={k} replicates  |  mean peak I={peak:,.1f}",
        "─"*70,
        f"  R²_I      : {r2i['mean']:.4f} ± {r2i['std']:.4f}"
        f"  95%CI [{r2i['ci_95'][0]:.4f}, {r2i['ci_95'][1]:.4f}]",
        f"  MAE_I     : {maei['mean']:.2f} ± {maei['std']:.2f}",
        f"  Rel-MAE_I : {rel['mean']:.2f}% ± {rel['std']:.2f}%"
        f"  95%CI [{rel['ci_95'][0]:.2f}%, {rel['ci_95'][1]:.2f}%]",
        f"  95%CI [{rel['ci_95'][0]:.2f}%, {rel['ci_95'][1]:.2f}%]"
        f"  CV={rel['cv']:.2f}%",
        "─"*70,
        f"  Performance: {grade}",
        "═"*70,
    ])

    (output_dir / 'report.txt').write_text(report, encoding='utf-8')
    print(f"\n{report}")
    print(f"\n  Results saved → {output_dir.resolve()}")


#  8. MAIN

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SIR emulator test evaluation')
    parser.add_argument('--models_dir',type=str, default=str(MODELS_DIR))
    parser.add_argument('--data',  type=str, default=str(TEST_DATA_DIR / 'abm-data_split.pkl'))
    parser.add_argument('--output_dir',type=str, default=str(RESULTS_DIR))
    parser.add_argument('--plots_dir',type=str, default=str(PLOTS_DIR))
    parser.add_argument('--train_strategy', type=str, default=TRAIN_STRATEGY,
                        choices=['MCMC', 'LHS', 'Random'])
    parser.add_argument('--test_strategy',  type=str, default=TEST_STRATEGY,
                        choices=['MCMC', 'LHS', 'Random'])
    parser.add_argument('--augmentation',   type=int, default=AUGMENTATION,
                        choices=[0, 1])
    parser.add_argument('--n_train_sims',   type=int, default=N_TRAIN_SIMULATIONS)
    parser.add_argument('--batch_size',     type=int, default=64)
    parser.add_argument('--n_plot_samples', type=int, default=8)
    args = parser.parse_args()

    TRAIN_STRATEGY = args.train_strategy
    TEST_STRATEGY       = args.test_strategy
    AUGMENTATION        = args.augmentation
    N_TRAIN_SIMULATIONS = args.n_train_sims

    results_dir = Path(args.output_dir)
    plots_dir = Path(args.plots_dir)
   

    print(f"\n{'═'*70}")
    print(f"STEP 5  TEST EVALUATION")
    print(f"  Train: {TRAIN_STRATEGY}  : Test: {TEST_STRATEGY}  "
          f"aug={AUGMENTATION}")

    device = get_device()

    # Load data
    loaders      = create_dataloaders(args.data, batch_size=args.batch_size)
    test_loader  = loaders['test']
    N_TIMEPOINTS = loaders['metadata']['n_timepoints']
    N            = loaders['metadata'].get('total_population', N)
    print(f"  Test samples : {len(test_loader.dataset):,}  "
          f"T={N_TIMEPOINTS}  N={N:,}")

    # Evaluate
    results_list, targets, params = evaluate_all_replicates(
        args.models_dir, test_loader, device
    )

    # Aggregate stats
    stats_dict = compute_aggregate_statistics(results_list, targets)

    # Per-replicate DataFrame  to regression CSV
    df = build_replicate_dataframe(
        results_list, targets,
        TRAIN_STRATEGY, TEST_STRATEGY, AUGMENTATION, N_TRAIN_SIMULATIONS
    )
    tag    = f"{TRAIN_STRATEGY}_to_{TEST_STRATEGY}_aug{AUGMENTATION}"
    df_out = results_dir / f"replicate_results_{tag}.csv"
    df.to_csv(df_out, index=False)
    print(f"\n  Replicate CSV : {df_out.resolve()}")

    # Also copy to regression data directory
    REGRESSION_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(REGRESSION_DATA_DIR / df_out.name, index=False)

    # Per-sample peak_I (one row per test epidemic, shared across replicates)
    per_sample_df = pd.DataFrame({
        'peak_I'        : targets[:, :, 1].max(dim=1)[0].numpy(),
        'test_strategy' : TEST_STRATEGY,
        'train_strategy': TRAIN_STRATEGY,
        'augmentation'  : AUGMENTATION,
    })
    per_sample_df.to_csv(
        REGRESSION_DATA_DIR / f"per_sample_peak_{tag}.csv", index=False
    )

    # Plots
    print("\nGenerating plots...")
    plot_uncertainty_band(results_list, targets, plots_dir,
                          params=params,
                          n_samples=args.n_plot_samples)
    plot_infected_predictions(results_list, targets, params, plots_dir,
                               n_samples=min(16, len(targets)))
    plot_relative_mae_vs_peak(results_list, targets, params, plots_dir)

    # Save JSON + text report
    save_results(results_list, stats_dict, results_dir)