import torch
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')
from torch.utils.data import Dataset, DataLoader

# NORMALISATION CONSTANTS 
PARAM_MINS = np.array([0.00025, 0.03,  0.001], dtype=np.float32)   # [tau, gamma, rho]
PARAM_MAXS = np.array([0.17,  1.0,  0.01], dtype=np.float32)

def normalise_params(params_raw: np.ndarray) -> np.ndarray:
    """
    Min-max normalise raw [tau, gamma, rho] to [0, 1]^3.

    Args:
        params_raw : (3,) or (N, 3) array of raw parameter values
    Returns:
        params_norm : same shape, each dimension in [0, 1]
    """
    return (params_raw - PARAM_MINS) / (PARAM_MAXS - PARAM_MINS + 1e-8)

# DATASET
class EpidemicDatasetSIR(Dataset):
    """
    PyTorch Dataset for the 3-parameter SIR emulator.

    Each item returns a dict with:
        params_norm : (3,)    normalised [tau, gamma, rho] in [0,1]
                              fed into the Fourier encoder
        rho_raw     : scalar  raw (un-normalised) rho in [0.001, 0.010]
                              used by decoder to pin S(0)=N(1-rho), I(0)=N*rho
        y           : (T, 3)  target [S(t), I(t), R(t)] trajectories
    """

    def __init__(self, simulations: list, n_timepoints: int):
        super().__init__()
        self.simulations  = simulations
        self.n_timepoints = n_timepoints

    def __len__(self) -> int:
        return len(self.simulations)

    def __getitem__(self, idx: int) -> dict:
        sim = self.simulations[idx]

        #  Raw parameters 
        params_raw = np.array([
            sim['params']['tau'],
            sim['params']['gamma'],
            sim['params']['rho'],
        ], dtype=np.float32)                                   # (3,)
        params_norm = normalise_params(params_raw)             # (3,) in [0, 1]

        # Raw rho for decoder initial conditions 
        # S(0) = N*(1-rho) and I(0) = N*rho are hard constraints, not predictions.
        # The decoder needs RAW rho to compute actual counts in people.
        rho_raw = params_raw[2]                                # scalar float32

        # SIR trajectories 
        S = sim['output']['S']
        I = sim['output']['I']
        R = sim['output']['R']
        y = np.stack([S, I, R], axis=1).astype(np.float32)    # (T, 3)

        return {
            'params_norm': params_norm,
            'rho_raw'    : rho_raw,
            'y'          : y,
        }



# BATCH WRAPPER
class BatchWrapper:
    """
    Thin wrapper for attribute-style batch access.

    Attributes:
        params_norm : (B, 3)    normalised params  → Fourier encoder
        rho_raw     : (B,)      raw rho            → decoder initial conditions
        y           : (B, T, 3) target trajectories
    """

    def __init__(self, params_norm, rho_raw, y):
        self.params_norm = params_norm
        self.rho_raw     = rho_raw
        self.y           = y

    def to(self, device):
        self.params_norm = self.params_norm.to(device)
        self.rho_raw     = self.rho_raw.to(device)
        self.y           = self.y.to(device)
        return self

# COLLATE FUNCTION
def collate_sir(batch_list: list) -> BatchWrapper:
    """
    Custom collate: stacks dicts from EpidemicDatasetSIR into a BatchWrapper.

    Args:
        batch_list : list of dicts (one per sample)
    Returns:
        BatchWrapper
    """
    params_norm = torch.FloatTensor(
        np.stack([item['params_norm'] for item in batch_list])  # (B, 3)
    )
    rho_raw = torch.FloatTensor(
        np.array([item['rho_raw'] for item in batch_list])       # (B,)
    )
    y = torch.FloatTensor(
        np.stack([item['y'] for item in batch_list])             # (B, T, 3)
    )
    return BatchWrapper(params_norm, rho_raw, y)



# DATA LOADERS
def create_dataloaders(dataset_path: str, batch_size: int = 32,
                       num_workers: int = 0) -> dict:
    """
    Load the SIR dataset pickle and return train/val/test DataLoaders.

    Expected pickle structure
    
    {
      'train'   : {'simulations': [...]},
      'val'     : {'simulations': [...]},
      'test'    : {'simulations': [...]},
      'metadata': {'n_timepoints': int, ...}
    }

    Each simulation dict must contain:
      sim['params']  → {'tau': float, 'gamma': float, 'rho': float}
      sim['output']  → {'t': array, 'S': array, 'I': array, 'R': array}

    Args:
        dataset_path : path to .pkl file
        batch_size   : samples per batch
        num_workers  : DataLoader worker processes

    Returns:
        dict with keys: 'train', 'val', 'test', 'metadata', 'n_timepoints'
    """
    print(f"\nLoading dataset : {dataset_path}")

    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)

    # Infer number of time points from first simulation 
    first_sim    = data['train']['simulations'][0]
    n_timepoints = len(first_sim['output']['t'])
    print(f"  n_timepoints  : {n_timepoints}")

    #  Build datasets 
    train_dataset = EpidemicDatasetSIR(data['train']['simulations'], n_timepoints)
    val_dataset   = EpidemicDatasetSIR(data['val']['simulations'],   n_timepoints)
    test_dataset  = EpidemicDatasetSIR(data['test']['simulations'],  n_timepoints)

    print(f"  Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size= batch_size,
        shuffle = True,
        drop_last = True,          # prevents BatchNorm crash on 1-sample tail batch
        num_workers = num_workers,
        collate_fn = collate_sir,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size = batch_size,
        shuffle = False,
        drop_last= False,
        num_workers= num_workers,
        collate_fn=collate_sir,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = batch_size,
        shuffle= False,
        drop_last= False,
        num_workers=num_workers,
        collate_fn=collate_sir,
    )

    metadata = data.get('metadata', {})
    metadata['n_timepoints'] = n_timepoints

    return {
        'train'       : train_loader,
        'val'         : val_loader,
        'test'        : test_loader,
        'metadata'    : metadata,
        'n_timepoints': n_timepoints,
    }



# METRICS
def compute_metrics(predictions, targets, prefix: str = '') -> dict:
    """
    Compute regression metrics for SIR trajectory predictions.

    Args:
        predictions : (N, T, 3) tensor or array  — predicted [S, I, R]
        targets     : (N, T, 3) tensor or array  — ground truth [S, I, R]
    Returns:
        dict of floats 
    """
    if torch.is_tensor(predictions):
        predictions = predictions.detach().cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()

    def _r2(pred, true):
        ss_r = np.sum((true - pred) ** 2)
        ss_t = np.sum((true - true.mean()) ** 2)
        return float(1.0 - ss_r / (ss_t + 1e-8))
    
    #Global MAE -MeasureS total emulator quality over full trajectory tensor.
    mae  = float(np.abs(predictions - targets).mean())
    #Global MSE
    mse  = float(((predictions - targets) ** 2).mean())
    #Global RMSE-
    rmse = float(np.sqrt(mse))
    #Global R2-measures how much of the total variation in all true outputs is explained by the emulator predictions across every sample, every time point, and every compartment combined.
    r2   = _r2(predictions, targets)

    mae_s = float(np.abs(predictions[:, :, 0] - targets[:, :, 0]).mean())
    mae_i = float(np.abs(predictions[:, :, 1] - targets[:, :, 1]).mean())
    mae_r = float(np.abs(predictions[:, :, 2] - targets[:, :, 2]).mean())

    #Compartmebtal R2-
    r2_s = _r2(predictions[:, :, 0], targets[:, :, 0]) # Does model capture susceptible depletion?
    r2_i = _r2(predictions[:, :, 1], targets[:, :, 1]) #Does model capture infected peaks? (usually hardest)
    r2_r = _r2(predictions[:, :, 2], targets[:, :, 2]) #Does model capture recovery accumulation?

    p = prefix
    return {
        f'{p}MAE'  : mae,   f'{p}MSE'  : mse,   f'{p}RMSE' : rmse,
        f'{p}R2'   : r2,
        f'{p}MAE_S': mae_s, f'{p}MAE_I': mae_i,  f'{p}MAE_R': mae_r,
        f'{p}R2_S' : r2_s,  f'{p}R2_I' : r2_i,   f'{p}R2_R' : r2_r,
    }



def compute_zone_r2i(predictions, targets, params=None) -> dict:
    """
    R^2_I stratified by epidemic severity zones (tertiles of true peak I/N).

    Zones are defined by the 33rd and 66th percentiles of each sample's true
    peak I, so each zone contains ~1/3 of the validation set. Because all
    ablation conditions share the same validation set, the tertile thresholds
    are identical across conditions — comparisons are fair.

    Args:
        predictions : (N, T, 3) tensor/array  [S, I, R]
        targets     : (N, T, 3) tensor/array  [S, I, R]
        params      : unused; kept for API compatibility
    Returns:
        dict with R2_I_low, R2_I_med, R2_I_high and n_low, n_med, n_high
    """
    if torch.is_tensor(predictions): predictions = predictions.detach().cpu().numpy()
    if torch.is_tensor(targets):     targets     = targets.detach().cpu().numpy()

    def _r2(pred, true):
        ss_r = np.sum((true - pred) ** 2)
        ss_t = np.sum((true - true.mean()) ** 2)
        return float(1.0 - ss_r / (ss_t + 1e-8))

    peak_I_true = targets[:, :, 1].max(axis=1)   # true peak I per sample (N,)
    lo, hi = np.percentile(peak_I_true, [33, 66])

    masks = {
        'low' : peak_I_true <  lo,
        'med' : (peak_I_true >= lo) & (peak_I_true < hi),
        'high': peak_I_true >= hi,
    }

    result = {}
    for zone, mask in masks.items():
        n = int(mask.sum())
        result[f'R2_I_{zone}'] = _r2(predictions[mask, :, 1], targets[mask, :, 1]) if n > 1 else float('nan')
        result[f'n_{zone}']    = n
    return result


# ABLATION METRICS
def compute_ablation_metrics(predictions, targets, total_population: int = 100_000) -> dict:
    """
    Compute Rel-MAE_I and conservation error for ablation table reporting.

    Args:
        predictions       : (N, T, 3) array or tensor  [S, I, R]
        targets           : (N, T, 3) array or tensor  [S, I, R]
        total_population  : N (used to normalise conservation error)
    Returns:
        dict with 'rel_mae_i' (%) and 'conservation_error' (%)
    """
    if torch.is_tensor(predictions):
        predictions = predictions.detach().cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()

    I_true   = targets[:, :, 1]
    I_pred   = predictions[:, :, 1]
    mean_I   = np.abs(I_true).mean()
    rel_mae_i = float(np.abs(I_pred - I_true).mean() / (mean_I + 1e-8) * 100.0)

    S_pred = predictions[:, :, 0]
    R_pred = predictions[:, :, 2]
    conservation_error = float(
        np.abs(S_pred + I_pred + R_pred - total_population).mean()
        / total_population * 100.0
    )

    return {
        'rel_mae_i'          : rel_mae_i,
        'conservation_error' : conservation_error,
    }


# DEVICE HELPER
def get_device() -> torch.device:
    """Return CUDA GPU if available, otherwise CPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"\nUsing GPU : {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("\n Using CPU")
    return device



# EARLY STOPPING
class EarlyStopping:
    """
    Stop training when a monitored metric stops improving.

    Correct usage — monitor val R², stop when it plateaus:

        stopper = EarlyStopping(patience=15, min_delta=1e-4, mode='max')

        for epoch in range(max_epochs):
            train_one_epoch(...)
            val_metrics = evaluate(model, val_loader)

            # Save checkpoint if this is the best epoch
            if val_metrics['r2'] > best_r2:
                best_r2 = val_metrics['r2']
                torch.save(model.state_dict(), 'best_model.pt')

            # Check stopping criterion
            if stopper(val_metrics['r2']):
                print(f"Early stopping at epoch {epoch}")
                break

    Args:
        patience  : epochs to wait after last improvement before stopping
                    15 is appropriate for this model (converges ~20-40 epochs)
        min_delta : minimum absolute improvement to reset the counter
                    1e-4 prevents stopping on numerical noise in R²
        mode      : 'max' for R² (higher is better)
                    'min' for loss (lower is better)
    """

    def __init__(self, patience: int = 15, min_delta: float = 1e-4,
                 mode: str = 'max'):
        assert mode in ('min', 'max'), "mode must be 'min' or 'max'"
        self.patience   = patience
        self.min_delta  = min_delta
        self.mode       = mode
        self.counter    = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Args:
            score : current epoch's monitored metric (e.g. val_metrics['r2'])
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter    = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def reset(self):
        """Reset state — use when resuming training from a checkpoint."""
        self.counter    = 0
        self.best_score = None
        self.early_stop = False
