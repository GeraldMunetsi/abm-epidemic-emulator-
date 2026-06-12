
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
from scipy import stats
import pandas as pd
from matplotlib.gridspec import GridSpec
from step0_model import create_hybrid_mlp_model          
from utils import create_dataloaders, compute_metrics, get_device, PARAM_MINS, PARAM_MAXS

n_timepoints = 80
N=100000
knots=8
 
DATA_DIR=Path("experiments/lhs-sampling/out/results/testing/lhs_no_augmentation/trained models")
SPLIT_DATA_DIR=Path("experiments/lhs-sampling/data/split")
MODEL_DIR=Path("experiments/lhs-sampling/out/results/testing/lhs_no_augmentation/validation")
PLOTS_DIR=Path("experiments/lhs-sampling/out/results/testing/lhs_no_augmentation/validation")

# CONSTANTS
PARAM_NAMES   = ['tau', 'gamma', 'rho']          # 3-parameter SIR
N_PARAMS      = 3
COMPARTMENTS  = ['Susceptible (S)', 'Infected (I)', 'Recovered (R)']
COMP_COLORS   = ['lightblue', 'lightcoral', 'lightgreen']

# MODEL LOADING
def load_model(model_path, device):
    """
    Load a single replicate model checkpoint.

    Args:
        model_path : Path to .pt checkpoint file
        device     : torch.device

    Returns:
        model      : loaded model in eval mode
        checkpoint : raw checkpoint dict
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    
    def load_model(model_path, device):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        config = {
            'n_params'        : N_PARAMS,
            'n_fourier'       : 64,
            'fourier_hidden'  : 32,
            'param_hidden'    : 16,
            'mlp_hidden'      : 32,
            'mlp_layers'      : 2,
            'temporal_hidden' : 64,
            'dropout'         : 0.3,
            'n_knots'         : knots,
            'n_timepoints'    : n_timepoints,
            'total_population': N,
        }

    state_dict = checkpoint['model_state_dict']
    state_dict.pop('temporal_decoder.t_grid', None)
    model = create_hybrid_mlp_model(config)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    return model, checkpoint

def find_replicate_models(models_dir):
    """
    Find all numbered replicate models inside *models_dir*.

    Pattern: best_balanced_mlp_model_*.pt

    Returns:
        List of Paths sorted by replicate number.
    """
    models_dir = Path(models_dir)
    model_files = sorted(
        models_dir.glob("best_balanced_mlp_model_*.pt"),
        key=lambda x: int(x.stem.split('_')[-1])
    )

    return model_files



# EVALUATION
def evaluate_model(model, val_loader, device, n_timesteps):
    """
    Run forward pass for every batch in *val_loader*.

    Returns:
        predictions : Tensor [N, T, 3]
        targets     : Tensor [N, T, 3]
        params      : Tensor [N, 3]  
        metrics     : dict
    """
    model.eval()

    all_predictions = []
    all_targets     = []
    all_params      = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            #forward pass
            predictions = model(batch, n_timesteps=n_timesteps)
            targets = batch.y
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
            params_norm_cpu = batch.params_norm.cpu() #batch.params_norm is expected to be normalized between 0 and 1 #So the model does not need raw tau, gamma, rho — it expects normalized values.
            param_mins_t= torch.tensor(PARAM_MINS)
            param_maxs_t= torch.tensor(PARAM_MAXS)
            params_raw= params_norm_cpu * (param_maxs_t - param_mins_t) + param_mins_t
            all_params.append(params_raw)

    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets,dim=0)
    params= torch.cat(all_params,dim=0)
    metrics = compute_metrics(predictions, targets)
    return predictions, targets, params, metrics


def evaluate_all_replicates(models_dir, val_loader, device, n_timesteps):
    """
    Loop over every replicate model and collect predictions + metrics.

    Returns:
        results_list : list of dicts (one per replicate)
        targets      : Tensor – ground truth (identical across replicates)
        params       : Tensor – (tau, gamma, rho) for each validation sample
    """
    model_paths = find_replicate_models(models_dir)

    print(f"\n{'-'*70}")
    print(f"EVALUATING {len(model_paths)} REPLICATES")
    print(f"Models directory : {models_dir}")
    for mp in model_paths:
        print(f"  · {mp.name}")
    print()

    results_list=[]
    targets= None
    params= None

    for idx, model_path in enumerate(model_paths, 1):
        print(f"Replicate {idx}/{len(model_paths)}: {model_path.name}")
        model, checkpoint = load_model(model_path, device)
        predictions, targets_rep, params_rep, metrics = evaluate_model(
            model, val_loader, device, n_timesteps
        )

        if targets is None:
            targets = targets_rep
            params  = params_rep
        result = {
            'replicate_id': checkpoint.get('replicate_id', idx),
            'seed': checkpoint.get('seed', None),
            'model_filename': model_path.name,
            'predictions': predictions.numpy(),
            'metrics': metrics,
        }

        results_list.append(result)
        print(f"R² = {metrics['R2']:.4f} | MAE_I = {metrics['MAE_I']:.2f}\n")

    return results_list, targets, params

# AGGREGATE STATISTICS
def compute_aggregate_statistics(results_list):
    """
    Compute mean, std, 95 % CI, CV across replicates for each metric.

    Returns:
        stats_dict : nested dict keyed by metric name
    """
    metric_names = ['MAE', 'RMSE', 'R2', 'MAE_S', 'MAE_I', 'MAE_R']
    stats_dict   = {}

    for metric_name in metric_names:
        values = np.array([r['metrics'][metric_name] for r in results_list])
        n= len(values)

        mean=np.mean(values)
        std=np.std(values, ddof=1) if n > 1 else 0.0
        sem=stats.sem(values) if n > 1 else 0.0
        ci=stats.t.interval(0.95, n - 1, loc=mean, scale=sem) if n > 1 else (mean, mean)

        stats_dict[metric_name] = {
            'values'  : values.tolist(),
            'n'       : int(n),
            'mean'    : float(mean),
            'std'     : float(std),
            'sem'     : float(sem),
            'ci_lower': float(ci[0]),
            'ci_upper': float(ci[1]),
            'min'     : float(np.min(values)),
            'max'     : float(np.max(values)),
            'median'  : float(np.median(values)),
            'cv'      : float(std / mean * 100) if mean != 0 else None,
        }

    return stats_dict

# VISUALISATION
def load_training_histories(models_dir):
    """Load training_history_*.npy files (one per replicate)."""
    models_dir=Path(models_dir)
    history_files=sorted(
        models_dir.glob("training_history_*.npy"),
        key=lambda x: int(x.stem.split('_')[-1])
    )

    if len(history_files) == 0:
        return None

    histories = []
    for hist_file in history_files:
        try:
            history = np.load(hist_file, allow_pickle=True).item()
            histories.append(history)
        except Exception:
            histories.append(None)

    return histories

#Training curves for all replicate models."""
def plot_training_curves(results_list, models_dir, output_dir):
    output_dir= Path(output_dir)
    histories = load_training_histories(models_dir)    
    fig,axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('TRAINING CURVES (All Replicate Models)',fontsize=20, fontweight='bold')
    n_replicates = len(results_list)
    colors = plt.cm.tab10(np.linspace(0, 1, n_replicates))
    
    # MAE_S 
    for i, (result, history) in enumerate(zip(results_list, histories)):
        if history is not None and 'val_mae_s' in history:
            axes[0].plot(range(1, len(history['val_mae_s']) + 1), history['val_mae_s'],
                     color=colors[i], alpha=0.7, linewidth=1)
    axes[0].set_xlabel('Epoch', fontweight='bold')
    axes[0].set_ylabel('Validation MAE_S', fontweight='bold')
    axes[0].set_title('Susceptible (S) Error Evolution')
    axes[0].grid(True, alpha=0.3)

    # MAE_R 
    for i, (result, history) in enumerate(zip(results_list, histories)):
        if history is not None and 'val_mae_r' in history:
            axes[1].plot(range(1, len(history['val_mae_r']) + 1), history['val_mae_r'],
                     color=colors[i], alpha=0.7, linewidth=1)
    axes[1].set_xlabel('Epoch', fontweight='bold')
    axes[1].set_ylabel('Validation MAE_R', fontweight='bold')
    axes[1].set_title('Recovered (R) Error')
    axes[1].grid(True, alpha=0.3)

    # Convergence last 10 epochs 
    for i, (result, history) in enumerate(zip(results_list, histories)):
        if history is not None and 'val_r2' in history:
            last_10 = history['val_r2'][-10:]
            epochs_last = range(len(history['val_r2'])-9,len(history['val_r2']) + 1)
            axes[2].plot(epochs_last, last_10,color=colors[i], alpha=0.7, linewidth=1, marker='o')
    axes[2].set_xlabel('Epoch', fontweight='bold')
    axes[2].set_ylabel('Validation R²', fontweight='bold')
    axes[2].set_title('Convergence (Last 10 Epochs)')
    axes[2].grid(True, alpha=0.3)
    out_path = output_dir / 'training_curves_all_replicates.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f" Saved: {out_path}")


#Training vs validation loss + MAE_I and R² evolution for all replicates
    fig,axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('TRAINING CURVES (All Replicate Models)',fontsize=20, fontweight='bold')
    n_replicates = len(results_list)
    colors = plt.cm.tab10(np.linspace(0, 1, n_replicates))
    
    # Training vs Validation Loss
    for i, (result, history) in enumerate(zip(results_list, histories)):
        if history is not None and 'train_loss' in history and 'val_loss' in history:
            epochs= range(1, len(history['train_loss']) + 1)
            axes[0].plot(epochs, history['train_loss'],
                     color=colors[i], alpha=0.3, linewidth=1, linestyle='--')
            axes[0].plot(epochs, history['val_loss'],
                     color=colors[i], alpha=0.3, linewidth=1)
    axes[0].set_xlabel('Epoch',fontweight='bold')
    axes[0].set_ylabel('Loss', fontweight='bold')
    axes[0].set_title('Training (dashed) vs Validation (solid) Loss')
    axes[0].grid(True, alpha=0.3)
    #axes[0].set_yscale('log')

    # Validation MAE_I
    for i, (result, history) in enumerate(zip(results_list, histories)):
        if history is not None and 'val_mae_i' in history:
            axes[1].plot(range(1, len(history['val_mae_i']) + 1), history['val_mae_i'],
                     color=colors[i], alpha=0.7, linewidth=1)
    axes[1].set_xlabel('Epoch', fontweight='bold')
    axes[1].set_ylabel('Validation MAE_I', fontweight='bold')
    axes[1].set_title('MAE_I Evolution')
    axes[1].grid(True, alpha=0.3)

    # Validation R²
    for i, (result, history) in enumerate(zip(results_list, histories)):
        if history is not None and 'val_r2' in history:
            axes[2].plot(range(1, len(history['val_r2']) + 1), history['val_r2'],
                     color=colors[i], alpha=0.7, linewidth=1,
                     label=f"Model {result['replicate_id']}")
    axes[2].set_xlabel('Epoch', fontweight='bold')
    axes[2].set_ylabel('Validation R²', fontweight='bold')
    axes[2].set_title('R² Evolution')
    axes[2].legend(fontsize=8, ncol=2, loc='lower right')
    axes[2].grid(True, alpha=0.3)
    out_path = output_dir/'training_curves.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    #plt.show()
    print(f"Saved: {out_path}")

    #Training vs Validation Loss single plot
    for i, (result, history) in enumerate(zip(results_list, histories)):
        if history is not None and 'train_loss' in history and 'val_loss' in history:
            epochs= range(1, len(history['train_loss']) + 1)
            plt.plot(epochs, history['train_loss'],
                     color=colors[i], alpha=0.3, linewidth=1, linestyle='--')
            plt.plot(epochs, history['val_loss'],
                     color=colors[i], alpha=0.3, linewidth=1)
    plt.xlabel('Epoch',fontweight='bold')
    plt.ylabel('log(Loss)', fontweight='bold')
    plt.title('Training (dashed) vs Validation (solid) Loss')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    out_path = output_dir/'training_curve_only.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    #plt.show()
    print(f"Saved: {out_path}")
    


def plot_prediction_samples(results_list, targets, params, output_dir, n_samples=6):
    """
    Plot sample SIR predictions vs ground truth for all replicates.
    """
    output_dir=Path(output_dir)
    targets=targets.numpy()
    params=params.numpy()

    # Select evenly spaced samples from validation set
    n_total = len(targets)
    indices = np.linspace(0, n_total - 1, n_samples, dtype=int)
    fig = plt.figure(figsize=(18, 3 * n_samples))
    gs=GridSpec(n_samples, 3, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle('VALIDATION: PREDICTED VS GROUND TRUTH',fontsize=16, fontweight='bold')
    n_replicates= len(results_list)
    pred_colors= plt.cm.tab10(np.linspace(0, 1, n_replicates))

    for row, idx in enumerate(indices):
        target= targets[idx]           # shape [T, 3]
        param=params[idx]            # shape [3] 
        tau_val= param[0]
        gamma_val= param[1]
        rho_val= param[2]

        for col in range(3):
            ax = fig.add_subplot(gs[row, col])

            # Ground truth
            ax.plot(target[:, col], 'o',
                    color=COMP_COLORS[col], label='Ground Truth',
                    alpha=0.6, markersize=6, markeredgewidth=0, zorder=10)

            # Each replicate
            for rep_idx, result in enumerate(results_list):
                pred = result['predictions'][idx]       # [T, 3]
                ax.plot(pred[:, col], '-',
                        color=pred_colors[rep_idx], linewidth=1.5, alpha=0.6,
                        label=f"Model {result['replicate_id']}" if col == 1 else "")

            if row == 0:
                ax.set_title(COMPARTMENTS[col], fontsize=12, fontweight='bold')

            ax.set_xlabel('Time', fontsize=10)
            ax.set_ylabel('Counts', fontsize=10)

            if col == 1:
                ax.legend(loc='best', fontsize=7, ncol=2)

            ax.grid(True, alpha=0.2, linestyle='--')

            # Parameter annotation 
            if col == 1:
                info_text = (
                    f'tau={tau_val:.3f}\n'
                    f'gamma={gamma_val:.3f}\n'
                    f'rho={rho_val:.3f}'
                )
                ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                        fontsize=6, verticalalignment='top')

    out_path = output_dir / 'predictions_all_replicates.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f" Saved: {out_path}")


    s_r2   = stats_dict['R2']
    s_maei = stats_dict['MAE_I']

    summary_text = (
        f"VALIDATION SUMMARY\n"
        f"{'='*32}\n\n"
        f"Model: SIR NNE\n"
        f"Parameters: tau, gamma, rho\n"
        f"n replicates: {s_r2['n']}\n\n"
        f"R²:\n"
        f" Mean: {s_r2['mean']:.4f}\n"
        f"Std: {s_r2['std']:.4f}\n"
        f" 5% CI: [{s_r2['ci_lower']:.4f}, {s_r2['ci_upper']:.4f}]\n\n"
        f"MAE_I:\n"
        f"Mean: {s_maei['mean']:.2f}\n"
        f"Std: {s_maei['std']:.2f}\n"
        f"CV: {s_maei['cv']:.2f}%\n"
        f"95% CI: [{s_maei['ci_lower']:.2f}, {s_maei['ci_upper']:.2f}]\n\n"
        f"Consistency: "
        f"{'EXCELLENT' if s_maei['cv'] < 5 else 'GOOD' if s_maei['cv'] < 10 else 'ACCEPTABLE'}"
    )


# REPORTING
def convert_to_python_types(obj):
    """Recursively convert numpy scalars → Python native types for JSON."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_python_types(i) for i in obj)
    return obj


def save_results(results_list, stats_dict, output_dir):
    """Persist results as JSON, CSV, and plain-text report."""
    output_dir = Path(output_dir)

    # JSON 
    results_data = {
        'model_description'   : '3-Parameter SIR Emulator (tau, gamma, rho)',
        'individual_results'  : [
            {
                'replicate_id': int(r['replicate_id']) if r['replicate_id'] is not None else None,
                'seed':int(r['seed'])if r['seed'] is not None else None,
                'model_filename': str(r['model_filename']),
                'metrics':convert_to_python_types(r['metrics']),
            }
            for r in results_list
        ],
        'aggregate_statistics': convert_to_python_types(stats_dict),
    }

    json_path = output_dir / 'validation_results.json'
    with open(json_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"Saved: {json_path}")

    # CSV 
    rows = [
        {'replicate_id': r['replicate_id'],
         'seed'        : r['seed'],
         'model_filename': r['model_filename'],
         **r['metrics']}
        for r in results_list
    ]
    csv_path = output_dir / 'validation_results.csv'
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Plain-text report 
    lines = [
        "-" * 70,
        "REPORT",
        "Parameters: tau, gamma, rho",
        "",
        f"Number of replicates: {len(results_list)}",
        f"Model pattern: best_balanced_mlp_model_*.pt",
        "",
        "-" * 70,
        "AGGREGATE STATISTICS",
        "",
        "PRIMARY METRICS:",
      
    ]

    for metric_name in ['R2', 'MAE_I']:
        s   = stats_dict[metric_name]
        fmt = ".4f" if metric_name == 'R2' else ".2f"
        lines += [
            f"\n{metric_name}:",
            f" Mean:{s['mean']:{fmt}}",
            f"Std: {s['std']:{fmt}}",
            f"95% CI: [{s['ci_lower']:{fmt}},  {s['ci_upper']:{fmt}}]",
            f" CV: {s['cv']:.2f}%" if s['cv'] else "  CV      : N/A",
            f"Range : [{s['min']:{fmt}},  {s['max']:{fmt}}]",
        ]

    lines += [
        "",
        "\nPER-COMPARTMENT MAE:",
        "-" * 70,
    ]
    for comp in ['S', 'I', 'R']:
        s = stats_dict[f'MAE_{comp}']
        lines.append(
            f"  MAE_{comp}: {s['mean']:7.2f} ± {s['std']:5.2f}"
            f"  (95% CI: [{s['ci_lower']:.2f}, {s['ci_upper']:.2f}])"
        )

    lines += [
        "",
        "\nINDIVIDUAL REPLICATE RESULTS:",
        "-" * 70,
        f"{'ID':>3} {'Seed':>6} {'R²':>8} {'MAE_I':>8}  {'Model File':<40}",
        "-" * 70,
    ]
    for r in results_list:
        lines.append(
            f"{r['replicate_id']:3d} "
            f"{str(r['seed']) if r['seed'] else 'N/A':>6} "
            f"{r['metrics']['R2']:8.4f} "
            f"{r['metrics']['MAE_I']:8.2f}  "
            f"{r['model_filename']:<40}"
        )

    # Interpretation
    cv = stats_dict['MAE_I']['cv']
    consistency = (
        "EXCELLENT" if cv and cv < 5  else
        "GOOD"   if cv and cv < 10 else
        "ACCEPTABLE"   if cv and cv < 15 else
        "HIGH VARIABILITY"
    )
    performance = (
        "EXCEPTIONAL"if stats_dict['MAE_I']['mean'] < 150 else
        "EXCELLENT"if stats_dict['MAE_I']['mean'] < 200 else
        "GOOD" if stats_dict['MAE_I']['mean'] < 250 else
        "NEEDS IMPROVEMENT"
    )

    lines += [
        "",
        "INTERPRETATION",
        f"\nModel Consistency (CV) : {consistency}",
        f"Overall Performance    : {performance}",
        "",
        "-" * 70,
    ]

    report_text="\n".join(lines)
    report_path= output_dir/'VALIDATION_REPORT.txt'
    with open(report_path, 'w', encoding="utf-8") as f:
        f.write(report_text)
    print(f"Saved: {report_path}")
    print("\n" + report_text)

# ENTRY POINT
if __name__ == "__main__":
    # Default filename 
    DATA_FILENAME = 'epidemic_data_age_adaptive_sobol_split.pkl'
    parser = argparse.ArgumentParser(description="Validate replicate models")
    parser.add_argument('--models_dir',type=str,default=str(DATA_DIR),help='Directory containing trained replicate models')
    parser.add_argument('--data',type=str,default=str(SPLIT_DATA_DIR / DATA_FILENAME), help='Full path to split dataset .pkl file')
    parser.add_argument('--output_dir',type=str,default=str(MODEL_DIR),help='Output directory for results (JSON, CSV, report)')
    parser.add_argument('--plots_dir',type=str,default=str(PLOTS_DIR),help='Output directory for plots')
    args = parser.parse_args()

    # paths 
    models_dir=Path(args.models_dir)
    data_path=Path(args.data)
    results_dir=Path(args.output_dir)
    plots_dir=Path(args.plots_dir)


    print("STEP 4: VALIDATION ")
    print(f"\nModels dir: {models_dir.resolve()}")
    print(f"Data file: {data_path.resolve()}")       
    print(f"Results dir : {results_dir.resolve()}")
    print(f"Plots dir: {plots_dir.resolve()}")
    print(f"Pattern: best_balanced_mlp_model_*.pt")

    device = get_device()

    #  Load validation data 
    print(f"\nLoading data: {data_path}")
    dataloaders=create_dataloaders(str(data_path), batch_size=40) 
    val_loader=dataloaders['val']
    n_timesteps=dataloaders['metadata']['n_timepoints']
    print(f"Validation samples : {len(val_loader.dataset)}")

    #  Evaluate all replicates 
    results_list, targets, params = evaluate_all_replicates(
        models_dir, val_loader, device, n_timesteps
    )

    #  Aggregate statistics 
    print("AGGREGATE STATISTICS")
    stats_dict = compute_aggregate_statistics(results_list)
    print(f"\n Statistics over {len(results_list)} replicate(s)")
    print(f"Mean R²: {stats_dict['R2']['mean']:.4f} ± {stats_dict['R2']['std']:.4f}")
    print(f"Mean MAE_I: {stats_dict['MAE_I']['mean']:.2f} ± {stats_dict['MAE_I']['std']:.2f}")
    print(f"CV (MAE_I): {stats_dict['MAE_I']['cv']:.2f}%")

    #  Visualisations 
    plot_training_curves(results_list, models_dir, plots_dir)
    plot_prediction_samples(results_list, targets, params, plots_dir, n_samples=6)
    

