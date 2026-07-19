import torch
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')
from torch.utils.data import Dataset, DataLoader


PARAM_MINS = np.array([0.0003, 0.03,  0.001], dtype=np.float32)  # [tau, gamma, rho]
PARAM_MAXS = np.array([0.02,   1.0,   0.01 ], dtype=np.float32)


def normalise_params(params_raw: np.ndarray) -> np.ndarray:
    """
    Min-max normalise raw [tau, gamma, rho] to [0, 1]^3.

    Args:
        params_raw : (3,) or (N, 3) array of raw parameter values
    Returns:
        params_norm : same shape, each dimension in [0, 1]
    """
    return (params_raw - PARAM_MINS) / (PARAM_MAXS - PARAM_MINS + 1e-8)


# ── DATASET ───────────────────────────────────────────────────────────────────
class EpidemicDatasetSIR(Dataset):
    """
    PyTorch Dataset for the 3-parameter SIR emulator.

    Each item returns a dict with:
        params_norm : (3,)    normalised [tau, gamma, rho] in [0, 1]
                              fed into the RFF encoder
        rho_raw     : scalar  raw (un-normalised) rho in [0.001, 0.010]
                              used by B-spline decoder to pin
                              S(0) = N(1-rho) and I(0) = N×rho exactly
        y           : (T, 3)  target trajectories [S(t), I(t), R(t)]
    """

    def __init__(self, simulations: list, n_timepoints: int):
        super().__init__()
        self.simulations  = simulations
        self.n_timepoints = n_timepoints

    def __len__(self) -> int:
        """Number of simulations in this split."""
        return len(self.simulations)

    def __getitem__(self, idx: int) -> dict:
        """Build one training sample: normalised params, raw rho, and the (T,3) SIR target."""
        sim = self.simulations[idx]

        # Raw parameters
        params_raw = np.array([
            sim['params']['tau'],
            sim['params']['gamma'],
            sim['params']['rho'],
        ], dtype=np.float32)                                    # (3,)
        params_norm = normalise_params(params_raw)              # (3,) in [0, 1]

        # Raw rho for decoder initial conditions.
        # S(0) = N*(1-rho) and I(0) = N*rho are architectural guarantees,
        # not learned outputs. The decoder requires raw rho to compute counts.
        rho_raw = params_raw[2]                                 # scalar float32

        # SIR trajectories
        S = sim['output']['S']
        I = sim['output']['I']
        R = sim['output']['R']
        y = np.stack([S, I, R], axis=1).astype(np.float32)     # (T, 3)

        return {
            'params_norm': params_norm,
            'rho_raw'    : rho_raw,
            'y'          : y,
        }


# ── BATCH WRAPPER ─────────────────────────────────────────────────────────────
class BatchWrapper:
    """
    Thin wrapper providing attribute-style access to batched tensors.

    Attributes:
        params_norm : (B, 3)    normalised params  → RFF encoder
        rho_raw     : (B,)      raw rho            → decoder initial conditions
        y           : (B, T, 3) target trajectories
    """

    def __init__(self, params_norm, rho_raw, y):
        self.params_norm = params_norm
        self.rho_raw     = rho_raw
        self.y           = y

    def to(self, device):
        """Move all batched tensors to the given device in place, return self for chaining."""
        self.params_norm = self.params_norm.to(device)
        self.rho_raw     = self.rho_raw.to(device)
        self.y           = self.y.to(device)
        return self


# ── COLLATE FUNCTION ──────────────────────────────────────────────────────────
def collate_sir(batch_list: list) -> BatchWrapper:
    """
    Custom collate: stacks dicts from EpidemicDatasetSIR into a BatchWrapper.

    Args:
        batch_list : list of dicts (one per sample)
    Returns:
        BatchWrapper with stacked tensors
    """
    params_norm = torch.FloatTensor(
        np.stack([item['params_norm'] for item in batch_list])  # (B, 3)
    )
    rho_raw = torch.FloatTensor(
        np.array([item['rho_raw'] for item in batch_list])      # (B,)
    )
    y = torch.FloatTensor(
        np.stack([item['y'] for item in batch_list])            # (B, T, 3)
    )
    return BatchWrapper(params_norm, rho_raw, y)


# DATA LOADERS 
def create_dataloaders(dataset_path: str, batch_size: int = 32,
                       num_workers: int = 0) -> dict:
    """
    Load the SIR dataset pickle and return train/val/test DataLoaders.
    Args:
        dataset_path,batch_size, num_workers  

    Returns:
        dict with keys: 'train', 'val', 'test', 'metadata', 'n_timepoints'
    """
    print(f"\nLoading dataset : {dataset_path}")

    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)

    # Infer number of time points from first simulation
    first_sim = data['train']['simulations'][0]
    n_timepoints = len(first_sim['output']['t'])
    print(f"  n_timepoints : {n_timepoints}")

    # Build datasets
    train_dataset = EpidemicDatasetSIR(data['train']['simulations'], n_timepoints)
    val_dataset = EpidemicDatasetSIR(data['val']['simulations'],   n_timepoints)
    test_dataset  = EpidemicDatasetSIR(data['test']['simulations'],  n_timepoints)
    print(f"  Train={len(train_dataset)}, Val={len(val_dataset)}, "
          f"Test={len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        shuffle  = True,
        drop_last   = True,       # prevents BatchNorm crash on 1-sample tail batch
        num_workers = num_workers,
        collate_fn  = collate_sir,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        drop_last   = False,
        num_workers = num_workers,
        collate_fn  = collate_sir,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        drop_last   = False,
        num_workers = num_workers,
        collate_fn  = collate_sir,
    )

    metadata = data.get('metadata', {})
    metadata['n_timepoints'] = n_timepoints

    return {
        'train' : train_loader,
        'val' : val_loader,
        'test' : test_loader,
        'metadata': metadata,
        'n_timepoints': n_timepoints,
    }


# ── METRICS ─
def compute_metrics(predictions, targets, prefix: str = '') -> dict:
    """
    Compute regression metrics for SIR trajectory predictions.

    Args:
        predictions ,targets,prefix   

    Returns:
        dict of scalar floats
    """
    if torch.is_tensor(predictions):
        predictions = predictions.detach().cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()

    def _r2(pred, true):
        """Pooled R² over all (sample, timestep) pairs."""
        ss_res = np.sum((true - pred) ** 2)
        ss_tot = np.sum((true - true.mean()) ** 2)
        return float(1.0 - ss_res / (ss_tot + 1e-8))

    # Global metrics — all compartments, all samples 
    mae  = float(np.abs(predictions - targets).mean())
    mse  = float(((predictions - targets) ** 2).mean())
    rmse = float(np.sqrt(mse))
    r2   = _r2(predictions, targets)

    #  Per-compartment MAE
    mae_s = float(np.abs(predictions[:, :, 0] - targets[:, :, 0]).mean())
    mae_i = float(np.abs(predictions[:, :, 1] - targets[:, :, 1]).mean())
    mae_r = float(np.abs(predictions[:, :, 2] - targets[:, :, 2]).mean())

    # Per-compartment R^2
    r2_s = _r2(predictions[:, :, 0], targets[:, :, 0])
    r2_i = _r2(predictions[:, :, 1], targets[:, :, 1])
    r2_r = _r2(predictions[:, :, 2], targets[:, :, 2])

    p = prefix
    return {
        f'{p}MAE'  : mae,
        f'{p}MSE'  : mse,
        f'{p}RMSE' : rmse,
        f'{p}R2'   : r2,
        f'{p}MAE_S': mae_s,
        f'{p}MAE_I': mae_i,
        f'{p}MAE_R': mae_r,
        f'{p}R2_S' : r2_s,
        f'{p}R2_I' : r2_i,
        f'{p}R2_R' : r2_r,
    }


# DEVICE HELPER 
def get_device() -> torch.device:
    """Return CPU device."""
    device = torch.device("cpu")
    print("\nUsing CPU")
    return device


# EARLY STOPPING ─
class EarlyStopping:
    """
    Stop training when a monitored metric stops improving.

    Args:
        patience, min_delta, mode   
    """

    def __init__(self, patience: int = 35, min_delta: float = 1e-4,
                 mode: str = 'min'):
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
            score : current epoch's monitored metric
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
        self.counter = 0
        self.best_score = None
        self.early_stop = False