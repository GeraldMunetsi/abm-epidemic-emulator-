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

N=100000
n_knots=8
n_timepoints=80

## FIXED I/O PATHS 
MODELS_DIR = Path("experiments/lhs-sampling/out/trained-models")
TEST_DATA_DIR = Path("experiments/mcmc-sampling/data/augmented")
RESULTS_DIR= Path("experiments/lhs-sampling/out/results/testing/mcmc_test_data_results")
PLOTS_DIR= Path("experiments/lhs-sampling/out/plots/testing_plots/mcmc_test_data_plots")

# MODEL LOADING
def load_replicate_model(model_path: Path, device: torch.device):
    """
    Load a single replicate checkpoint.
    Handles minor architecture mismatches (e.g. knot count off-by-one).
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    config = checkpoint.get('config', {
        'n_params'        : 3,
        'n_fourier'       : 64,
        'sigma'           : 1.0,
        'fusion_hidden'   : 128,
        'latent_dim'      : 64,
        'decoder_hidden'  : 64,
        'dropout'         : 0.3,
        'n_knots'         : 'n_knots',
        'n_timepoints'    : 'n_timepoints',
        'total_population': N,
    })

    state_dict = checkpoint['model_state_dict']
    state_dict.pop('temporal_decoder.t_grid', None)   # backward compat

    model = create_hybrid_mlp_model(config)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    N_ckpt = config.get('total_population')
    R2_val = checkpoint.get('val_metrics', {}).get('R2', float('nan'))
    print(f"  {model_path.name}  "
          f"epoch={checkpoint.get('epoch')}  "
          f"val R²={R2_val:.4f}  N={N_ckpt:,}")

    return model, checkpoint

# INFERENCE
def evaluate_model(model, test_loader, device, n_timesteps):
    """
    Run inference on the full test set.

    Returns
    predictions : (n_test, T, 3)
    targets     : (n_test, T, 3)
    params      : (n_test, 3)   raw (tau, gamma, rho)
    metrics     : dict
    """
    model.eval()
    all_preds, all_targets, all_params = [], [], []

    param_mins = torch.tensor(PARAM_MINS)
    param_maxs = torch.tensor(PARAM_MAXS)

    with torch.no_grad():
        for batch in test_loader:
            batch       = batch.to(device)
            predictions = model(batch, n_timesteps=n_timesteps)

            all_preds.append(predictions.cpu())
            all_targets.append(batch.y.cpu())

            # Denormalise [0,1] → raw parameter values
            raw = batch.params_norm.cpu() * (param_maxs - param_mins) + param_mins
            all_params.append(raw)

    predictions = torch.cat(all_preds,   dim=0)   # (n, T, 3)
    targets     = torch.cat(all_targets, dim=0)   # (n, T, 3)
    params      = torch.cat(all_params,  dim=0)   # (n, 3)
    metrics     = compute_metrics(predictions, targets)

    return predictions, targets, params, metrics


def evaluate_all_replicates(models_dir, test_loader, device, n_timesteps):
    """Evaluate every replicate checkpoint on the test set."""
    model_paths = sorted(
        Path(models_dir).glob("best_balanced_mlp_model_*.pt"),
        key=lambda p: int(p.stem.split('_')[-1])
    )

    print(f"\n{'-'*70}")
    print(f"FINAL TEST EVALUATION·{len(model_paths)} REPLICATE(S)")
    results_list = []
    shared_targets= None
    shared_params= None

    for idx, path in enumerate(model_paths, 1):
        print(f"Replicate {idx}/{len(model_paths)} : {path.name}")

        model, checkpoint = load_replicate_model(path, device)
        predictions, targets, params, metrics = evaluate_model(
            model, test_loader, device, n_timesteps
        )

        if shared_targets is None:
            shared_targets = targets
            shared_params  = params

        results_list.append({
            'replicate_id'   : idx,
            'model_path'     : str(path),
            'predictions'    : predictions,
            'metrics'        : metrics,
            'checkpoint_info': {
                'epoch'      : checkpoint.get('epoch', '?'),
                'val_metrics': checkpoint.get('val_metrics', {}),
                'param_names': checkpoint.get('param_names', ['tau','gamma','rho']),
            },
        })

        print(f"  MAE_I : {metrics['MAE_I']:.2f}   key metric")
        print(f"  R²    : {metrics['R2_I']:.4f}\n")

    print(f"Evaluated {len(results_list)} replicate(s)")
    return results_list, shared_targets, shared_params

# RELATIVE MAE — per-sample then averaged 
def compute_relative_mae_i(predictions, targets):
    """
    Compute relative MAE_I per sample, then return the mean and std.
    Formula:  rel_mae_i = mean_over_samples( MAE_I_i / peak_I_i )
    This is statistically correct because it normalises each sample
    by its own peak before averaging, avoiding the distortion caused
    by dividing a mean by a mean (Jensen's inequality).
    A sample where peak_I < 1 is excluded to avoid division by near-zero.

    Returns
    mean_rel(%), std_rel (%)
    per_sample: np.ndarray— one relative MAE per test sample
    mean_peak : float —mean ground-truth peak I 
    """
    # MAE per sample on I compartment
    mae_per_sample  = (predictions[:, :, 1] - targets[:, :, 1]).abs().mean(dim=1)  # (n,)
    peak_per_sample = targets[:, :, 1].max(dim=1)[0]                               # (n,)

    # Exclude near-zero peaks (sub-critical extinction)
    valid = peak_per_sample >= 1.0
    if valid.sum() == 0:
        return float('nan'), float('nan'), np.array([]), float('nan')

    rel = (mae_per_sample[valid]/peak_per_sample[valid] * 100).numpy()

    return float(rel.mean()), float(rel.std(ddof=1) if len(rel) > 1 else 0.0), \
           rel, float(peak_per_sample[valid].mean().item()) # rel.mean()-average relative accross valid test outbreaks

# AGGREGATE STATISTICS
def compute_aggregate_statistics(results_list, targets):
    """
    Compute mean, std, CV, 95% CI across replicates for all metrics.
    Includes both absolute and relative MAE_I.

    Parameters
    results_list : list of per-replicate result dicts
    targets      : (n_test, T, 3) ground-truth tensor
    """
    n = len(results_list)

    metric_keys = ['MAE', 'MAE_S', 'MAE_I', 'MAE_R', 'R2', 'RMSE', 'MSE','R2_S','R2_I','R2_R' ]
    stats_dict  = {'n_replicates': n}

    for key in metric_keys:
        arr = np.array([r['metrics'][key] for r in results_list])
        sem = arr.std() / np.sqrt(n)
        ci  = stats.t.interval(0.95, n-1, loc=arr.mean(), scale=sem) \
              if n > 1 else (arr.mean(), arr.mean())
        stats_dict[key] = {
            'mean'  : float(arr.mean()),
            'std'   : float(arr.std(ddof=1) if n > 1 else 0.),
            'sem'   : float(sem),
            'ci_95' : [float(ci[0]), float(ci[1])],
            'cv'    : float(arr.std() / arr.mean() * 100) if arr.mean() != 0 else 0.,
        }

    # Relative MAE_I (per-sample then averaged) 
    rel_means, rel_stds = [], []
    for r in results_list:
        mean_rel, std_rel, _, mean_peak = compute_relative_mae_i(
            r['predictions'], targets
        )
        rel_means.append(mean_rel)
        rel_stds.append(std_rel)

    rel_arr = np.array(rel_means)
    rel_sem = rel_arr.std() / np.sqrt(n)
    rel_ci  = stats.t.interval(0.95, n-1, loc=rel_arr.mean(), scale=rel_sem) \
              if n > 1 else (rel_arr.mean(), rel_arr.mean())

    stats_dict['relative_MAE_I_%'] = {
        'mean': float(rel_arr.mean()),
        'std': float(rel_arr.std(ddof=1) if n > 1 else 0.),
        'ci_95': [float(rel_ci[0]), float(rel_ci[1])],
        'note': 'per-sample MAE_I / peak_I, averaged — excludes peak_I < 1',
    }
    stats_dict['mean_peak_I_ground_truth'] = mean_peak

    print(f"\n  Mean peak I (ground truth) : {mean_peak:,.1f}")
    print(f"  Absolute MAE_I: {stats_dict['MAE_I']['mean']:.2f} "
          f"± {stats_dict['MAE_I']['std']:.2f}")
    print(f"  Relative MAE_I: {rel_arr.mean():.2f}% "
          f"± {rel_arr.std(ddof=1) if n>1 else 0.:.2f}%  "
          f"(per-sample, then averaged)")

    return stats_dict

# VISUALISATION
def plot_all_compartments(results_list, targets, plots_dir, n_samples=8):
    """S, I, R trajectories for a sample of test cases."""
    plots_dir  = Path(plots_dir)
    targets_np = targets.numpy()
    n_total    = len(targets_np)
    indices    = np.linspace(0, n_total-1, n_samples, dtype=int)

    compartments = ['Susceptible (S)', 'Infected (I)', 'Recovered (R)']
    gt_colors    = ['lightblue', 'lightcoral', 'lightgreen']
    n_reps       = len(results_list)
    pred_colors  = plt.cm.tab10(np.linspace(0, 1, n_reps))

    fig = plt.figure(figsize=(18, 3 * n_samples))
    gs  = GridSpec(n_samples, 3, figure=fig, hspace=0.35, wspace=0.30)
    fig.suptitle(
        'Test Set Predictions (All Replicates)',
        fontsize=14, fontweight='bold'
    )

    for row, idx in enumerate(indices):
        target = targets_np[idx]
        for col in range(3):
            ax = fig.add_subplot(gs[row, col])
            ax.plot(target[:, col], 'o', color=gt_colors[col],
                    alpha=0.6, markersize=4, markeredgewidth=0,
                    label='Ground Truth', zorder=10)
            for rep_i, result in enumerate(results_list):
                pred = result['predictions'][idx].numpy()
                ax.plot(pred[:, col], '-',
                        color=pred_colors[rep_i], linewidth=1.5, alpha=0.7,
                        label=f"M{result['replicate_id']}" if col == 1 else "")
            if row == 0:
                ax.set_title(compartments[col], fontsize=11, fontweight='bold')
            ax.set_xlabel('Time step', fontsize=8)
            ax.set_ylabel('Count', fontsize=8)
            if col == 1:
                ax.legend(loc='best', fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3, linestyle='--')

    out = plots_dir / 'test_comparison_plots.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


def plot_infected_only(results_list, targets, plots_dir, n_samples=5):
    """I compartment only — easier to read for the key metric."""
    plots_dir  = Path(plots_dir)
    targets_np = targets.detach().cpu().numpy()
    n_total    = len(targets_np)
    indices    = np.linspace(0, n_total-1, n_samples, dtype=int)

    n_reps      = len(results_list)
    pred_colors = plt.cm.tab10(np.linspace(0, 1, n_reps))

    fig = plt.figure(figsize=(10, 3 * n_samples))
    gs  = GridSpec(n_samples, 1, figure=fig, hspace=0.4)
    fig.suptitle('Test Set — Infected (I) Compartment',
                 fontsize=14, fontweight='bold')

    for row, idx in enumerate(indices):
        ax     = fig.add_subplot(gs[row, 0])
        target = targets_np[idx]
        ax.plot(target[:, 1], 'o', color='steelblue',
                alpha=0.7, markersize=4, label='Ground Truth', zorder=10)
        for rep_i, result in enumerate(results_list):
            pred = result['predictions'][idx].detach().cpu().numpy()
            ax.plot(pred[:, 1], '-',
                    color=pred_colors[rep_i], linewidth=1.5, alpha=0.7,
                    label=f"M{result['replicate_id']}")
        ax.set_xlabel('Time step')
        ax.set_ylabel('Infected count')
        if row == 0:
            ax.legend(loc='best', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3, linestyle='--')

    out = plots_dir / 'test_infected_predictions.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


def plot_relative_mae_distribution(results_list, targets, plots_dir):
    """
    Histogram of per-sample relative MAE_I across the test set.
    One curve per replicate + ensemble mean.
    """
    plots_dir = Path(plots_dir)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Relative MAE_I Distribution  (per-sample MAE_I / peak_I)',
                 fontsize=13, fontweight='bold')

    colors = plt.cm.tab10(np.linspace(0, 1, len(results_list)))

    all_rel = []
    for rep_i, result in enumerate(results_list):
        _, _, per_sample, _ = compute_relative_mae_i(result['predictions'], targets)
        all_rel.append(per_sample)
        axes[0].hist(per_sample, bins=30, alpha=0.5,
                     color=colors[rep_i], label=f"M{result['replicate_id']}")

    axes[0].set_xlabel('Relative MAE_I (%)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution per replicate')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].axvline(np.concatenate(all_rel).mean(), color='red',
                    linestyle='--', linewidth=2,
                    label=f"Mean: {np.concatenate(all_rel).mean():.1f}%")

    # R₀ vs relative MAE scatter
    # Reconstruct R₀ from params 
    ax2 = axes[1]
    if len(results_list[0]['predictions']) > 0:
        _, _, per_sample_r1, _ = compute_relative_mae_i(
            results_list[0]['predictions'], targets)
        peak_per_sample = targets[:, :, 1].max(dim=1)[0].numpy()
        valid = peak_per_sample >= 1.0
        ax2.scatter(peak_per_sample[valid], per_sample_r1,
                    alpha=0.4, s=12, color=colors[0])
        ax2.set_xlabel('Ground-truth peak I (per sample)')
        ax2.set_ylabel('Relative MAE_I (%)')
        ax2.set_title('Relative MAE_I vs epidemic size\n'
                      '(expected: smaller epidemics harder)')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = plots_dir / 'test_relative_mae_distribution.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


def plot_metrics_summary(stats_dict, plots_dir):
    """Summary panel: per-compartment MAE bars + text summary."""
    plots_dir = Path(plots_dir)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Test Metrics Summary  ·  3-Parameter SIR (τ, γ, ρ)',
                 fontsize=14, fontweight='bold')

    # Per-compartment MAE bar chart
    ax = axes[0]
    comps = ['S', 'I', 'R']
    means = [stats_dict[f'MAE_{c}']['mean'] for c in comps]
    stds  = [stats_dict[f'MAE_{c}']['std']  for c in comps]
    ax.bar([0, 1, 2], means, yerr=stds, capsize=5,
           color=['cornflowerblue', 'tomato', 'mediumseagreen'],
           alpha=0.75, edgecolor='black')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['S', 'I (key)', 'R'])
    ax.set_ylabel('MAE (absolute counts)')
    ax.set_title('Per-Compartment Absolute MAE')
    ax.grid(True, alpha=0.3, axis='y')

    # Relative MAE bar
    ax = axes[1]
    rel_mean = stats_dict['relative_MAE_I_%']['mean']
    rel_std  = stats_dict['relative_MAE_I_%']['std']
    ax.bar(['Relative\nMAE_I (%)'], [rel_mean], yerr=[rel_std],
           capsize=5, color='tomato', alpha=0.75, edgecolor='black', width=0.4)
    ax.axhline(10, color='orange', ls='--', lw=1.5, label='10% threshold')
    ax.axhline(5,  color='green',  ls='--', lw=1.5, label='5% threshold')
    ax.set_ylabel('% of peak I')
    ax.set_title('Relative MAE_I\n(per-sample, then averaged)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(rel_mean * 1.5, 15))

    # Text summary
    ax = axes[2]
    ax.axis('off')
    summary = (
        f"TEST RESULTS SUMMARY\n"
        f"{'-'*32}\n\n"
        f"n replicates   : {stats_dict['n_replicates']}\n\n"
        f"R²\n"
        f"  Mean  : {stats_dict['R2']['mean']:.4f} ± {stats_dict['R2']['std']:.4f}\n"
        f"  95%CI : [{stats_dict['R2']['ci_95'][0]:.4f}, "
        f"{stats_dict['R2']['ci_95'][1]:.4f}]\n\n"
        f"Absolute MAE_I : key\n"
        f"  Mean  : {stats_dict['MAE_I']['mean']:.2f} ± {stats_dict['MAE_I']['std']:.2f}\n"
        f"  95%CI : [{stats_dict['MAE_I']['ci_95'][0]:.2f}, "
        f"{stats_dict['MAE_I']['ci_95'][1]:.2f}]\n"
        f"  CV    : {stats_dict['MAE_I']['cv']:.1f}%\n\n"
        f"Relative MAE_I : Key metric\n"
        f"  Mean  : {rel_mean:.2f}% ± {rel_std:.2f}%\n"
        f"  95%CI : [{stats_dict['relative_MAE_I_%']['ci_95'][0]:.2f}, "
        f"{stats_dict['relative_MAE_I_%']['ci_95'][1]:.2f}]%\n\n"
        f"Mean peak I (GT): {stats_dict['mean_peak_I_ground_truth']:,.0f}"
    )
    ax.text(0.05, 0.5, summary, fontsize=10, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    out = plots_dir / 'test_metrics_summary.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# SAVE RESULTS
def save_results(results_list, stats_dict, output_dir):
    """Save CSV, JSON, and dissertation-ready plain-text report."""
    # CSV 
    rows = [{
        'replicate_id'  : r['replicate_id'],
        'model_path'    : r['model_path'],
        **r['metrics'],
        'training_epoch': r['checkpoint_info']['epoch'],
    } for r in results_list]
    pd.DataFrame(rows).to_csv(output_dir / 'test_replicate_results.csv', index=False)
    print(f"Saved: {output_dir / 'test_replicate_results.csv'}")

    # JSON 
    with open(output_dir / 'test_final_statistics.json', 'w', encoding='utf-8') as f:
        json.dump(stats_dict, f, indent=2)
    print(f"Saved: {output_dir / 'test_final_statistics.json'}")

    #Plain-text dissertation report 
    mae_i_mean = stats_dict['MAE_I']['mean']
    mae_i_ci   = stats_dict['MAI']['ci_95'] if 'MAI' in stats_dict \
                 else stats_dict['MAE_I']['ci_95']
    mae_i_ci   = stats_dict['MAE_I']['ci_95']
    r2_mean    = stats_dict['R2']['mean']
    r2_ci      = stats_dict['R2']['ci_95']
    cv         = stats_dict['MAE_I']['cv']
    rel_mean   = stats_dict['relative_MAE_I_%']['mean']
    rel_std    = stats_dict['relative_MAE_I_%']['std']
    rel_ci     = stats_dict['relative_MAE_I_%']['ci_95']
    mean_peak  = stats_dict['mean_peak_I_ground_truth']
    n_test     = len(results_list[0]['predictions'])

    r2_s_mean = stats_dict['R2_S']['mean']
    r2_s_ci = stats_dict['R2_S']['ci_95']
    r2_i_mean = stats_dict['R2_I']['mean']
    r2_i_ci   = stats_dict['R2_I']['ci_95']
    r2_r_mean = stats_dict['R2_R']['mean']
    r2_r_ci   = stats_dict['R2_R']['ci_95']

    performance = (
        "EXCEPTIONAL"       if rel_mean < 5   else
        "EXCELLENT"         if rel_mean < 10  else
        "GOOD"              if rel_mean < 20  else
        "ACCEPTABLE"        if rel_mean < 35  else
        "NEEDS IMPROVEMENT"
    )
    consistency = (
        "EXCELLENT (CV < 5%)"       if cv < 5  else
        "GOOD (CV < 10%)"           if cv < 10 else
        "ACCEPTABLE (CV < 15%)"     if cv < 15 else
        f"HIGH VARIABILITY (CV={cv:.1f}%)"
    )

    lines = [
        "-" * 70,
        "FINAL TEST RESULTS",
        "",
        f"  Replicates   : {stats_dict['n_replicates']}",
        f"  Test samples : {n_test}",
        f"  Mean peak I  : {mean_peak:,.1f} counts (ground truth)",
        "",
        "-" * 70,
        "OVERALL PERFORMANCE",
        "",
        f"  R²_I : {r2_i_mean:.4f} ± {stats_dict['R2']['std']:.4f}",
        f"  95% CI : [{ r2_i_ci [0]:.4f}, { r2_i_ci[1]:.4f}]",
        "",
        f"  MAE : {stats_dict['MAE']['mean']:.2f} ± {stats_dict['MAE']['std']:.2f}",
        f"  RMSE : {stats_dict['RMSE']['mean']:.2f} ± {stats_dict['RMSE']['std']:.2f}",
        "",
        "-" * 70,
        "PER-COMPARTMENT MAE (absolute counts)",
        "",
        f"  MAE_S : {stats_dict['MAE_S']['mean']:.2f} ± {stats_dict['MAE_S']['std']:.2f}",
        f"  MAE_I : {mae_i_mean:.2f} ± {stats_dict['MAE_I']['std']:.2f} : key metric",
        f"  95% CI: [{mae_i_ci[0]:.2f}, {mae_i_ci[1]:.2f}]",
        f"  CV   : {cv:.1f}%",
        f"  MAE_R : {stats_dict['MAE_R']['mean']:.2f} ± {stats_dict['MAE_R']['std']:.2f}",
        "",
        "-" * 70,
        "RELATIVE MAE_I  (per-sample MAE_I / peak_I, then averaged)",
        "",
        f"  Mean peak I  : {mean_peak:,.1f}  (ground truth, test set)",
        f"  Relative MAE : {rel_mean:.2f}% ± {rel_std:.2f}%",
        f"  95%CI        : [{rel_ci[0]:.2f}%, {rel_ci[1]:.2f}%]",
        f"  Interpretation : for the average epidemic in the test set,",
        f"  the emulator's I(t) prediction is {rel_mean:.1f}% away from",
        f"  the true peak — regardless of epidemic size.",
        "",
        "-" * 70,
        "PERFORMANCE ASSESSMENT",
        "",
        f"  Level       : {performance}  (based on relative MAE_I = {rel_mean:.1f}%)",
        f"  Consistency : {consistency}",
        "",
        "-" * 70,
        "REPORT",
        "",
        f'"The SIR NNE achieved a test set MAE_I of {mae_i_mean:.0f} counts ({mae_i_ci[0]:.0f}-{mae_i_ci[1]:.0f},',
        f'95% CI, n={stats_dict["n_replicates"]} replicates), corresponding to {rel_mean:.1f}% of the mean ground-truth peak infected count',
        f'({mean_peak:,.0f} individuals)',
        f"The R² of the Susceptibles, Infectious and Recovered is {np.round(r2_s_mean,3)}, {np.round(r2_i_mean,3)}, {np.round(r2_r_mean,3)} respectively",
        f"There confidence intervals are {np.round(r2_s_ci,3)},{np.round(r2_i_ci,3)} and {np.round(r2_r_ci,3)} respectively.",
        f'{stats_dict["R2"]["std"]:.4f}. Replicate consistency was {consistency.lower()},',
        f'with coefficient of variation {cv:.1f}%',
        f'across random initialisations. The relative MAE_I of {rel_mean:.1f}%',
        f'(95% CI [{rel_ci[0]:.1f}%, {rel_ci[1]:.1f}%]) provides a scale-invariant',
        f'measure of accuracy that is directly comparable across epidemic regimes',
        f'with different outbreak sizes."',
        "",

    ]

    report = "\n".join(lines)
    (output_dir / 'Report.txt').write_text(report, encoding='utf-8')
    print(f"Saved: {output_dir / 'Report.txt'}")
    print("\n" + report)



# ENTRY POINT
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Final test evaluation "
    )
    parser.add_argument('--models_dir', type=str, default=str(MODELS_DIR))
    parser.add_argument('--data',       type=str,
                        default=str(TEST_DATA_DIR /
                                    'epidemic_data_age_adaptive_sobol_split_augmented.pkl'))
    parser.add_argument('--output_dir', type=str, default=str(RESULTS_DIR))
    parser.add_argument('--plots_dir',  type=str, default=str(PLOTS_DIR))
    parser.add_argument('--n_samples',  type=int, default=8,
                        help='Trajectory samples to plot')
    parser.add_argument('--batch_size', type=int, default=35)
    args = parser.parse_args()

    results_dir = Path(args.output_dir)
    plots_dir   = Path(args.plots_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True,   exist_ok=True)

    print("\n" + "-" * 70)
    print("STEP 5: FINAL TEST ")
    print(f"\n  Models  : {Path(args.models_dir).resolve()}")
    print(f"  Data    : {args.data}")
    print(f"  Results : {results_dir.resolve()}")
    print(f"  Plots   : {plots_dir.resolve()}")

    device = get_device()

    # Load test set
    print(f"\nLoading test data: {args.data}")
    dataloaders = create_dataloaders(args.data, batch_size=args.batch_size)
    test_loader = dataloaders['test']
    n_timesteps = dataloaders['metadata']['n_timepoints']
    N           = dataloaders['metadata'].get('total_population', 100_000)
    print(f"  Test samples : {len(test_loader.dataset):,}")
    print(f"  n_timepoints : {n_timesteps}")
    print(f"  N            : {N:,}")

    # Evaluate 
    results_list, targets, params = evaluate_all_replicates(
        args.models_dir, test_loader, device, n_timesteps
    )

    # Statistics (pass targets for relative MAE) ─
    stats_dict = compute_aggregate_statistics(results_list, targets)

    # Plots 
    print("\n" + "-" * 70)
    print("GENERATING FIGURES")
    plot_all_compartments(results_list, targets, plots_dir, args.n_samples)
    plot_infected_only(results_list, targets, plots_dir, n_samples=5)
    plot_relative_mae_distribution(results_list, targets, plots_dir)
    plot_metrics_summary(stats_dict, plots_dir)

    #  Save 
    print("SAVING RESULTS")
    save_results(results_list, stats_dict, results_dir)

    print(f"\n{'-'*70}")
    print("TEST COMPLETE")
    print(f"  R²_I: {stats_dict['R2_I']['mean']:.4f} "
          f"± {stats_dict['R2_I']['std']:.4f}")
    print(f"  Absolute MAE_I  : {stats_dict['MAE_I']['mean']:.2f} "
          f"± {stats_dict['MAE_I']['std']:.2f}")
    print(f"  Relative MAE_I  : {stats_dict['relative_MAE_I_%']['mean']:.2f}% "
          f"± {stats_dict['relative_MAE_I_%']['std']:.2f}%")
   