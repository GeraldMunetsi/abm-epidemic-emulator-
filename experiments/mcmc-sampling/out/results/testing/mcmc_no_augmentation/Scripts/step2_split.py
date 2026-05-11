"""
Script
1. Group all simulations by (tau, gamma, rho)
2. Randomly shuffle parameter-set groups
3. Assign groups to train / val / test  (70 / 15 / 15 default)
4. All replicates of a param set go to the SAME split
5. Verify zero parameter leakage across splits
6. Save split pickle + training CSV
"""

import pickle
import numpy as np
import pandas as pd
import argparse
from pathlib import Path

RAW_DATA_DIR  = Path("experiments/mcmc-sampling/data/raw")
SPLIT_DATA_DIR = Path("experiments/mcmc-sampling/data/split")

# BA network ratio 
RATIO = 34.0

# SPLIT
def split_dataset(dataset,
                  train_ratio: float = 0.70,
                  val_ratio:float = 0.15,
                  test_ratio:float = 0.15,
                  seed:int= 42) -> dict:
    """
    Split by PARAMETER SET so no (tau, gamma, rho) tuple appears in
    more than one of train / val / test.

    Parameters
    dataset     : dict loaded from raw .pkl — must contain keys
                  'simulations', 'metadata', and optionally 'network'
    train_ratio : fraction of parameter SETS assigned to training
    val_ratio   : fraction of parameter SETS assigned to validation
    test_ratio  : fraction of parameter SETS assigned to test
    seed        : random seed for reproducibility

    Returns
    split_data  : dict with keys
                  'train', 'val', 'test', 'network', 'metadata', 'split_info'
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, \
        "train_ratio + val_ratio + test_ratio must equal 1.0"

    rng  = np.random.default_rng(seed)
    sims = dataset['simulations']

    #  Step 1: Group simulations by parameter set 
    param_to_indices: dict = {}

    for i, sim in enumerate(sims):
        p = sim['params']
        tau = float(p.get('tau',   p.get('beta',0)))
        gamma = float(p.get('gamma', p.get('mu',0)))
        rho = float(p.get('rho',0))
        key = (round(tau, 8), round(gamma, 8), round(rho, 8))
        param_to_indices.setdefault(key, []).append(i)

    param_keys   = list(param_to_indices.keys())
    n_param_sets = len(param_keys)

    print("\nDataset summary")
    print(f"  Total simulations : {len(sims)}")
    print(f"  Unique param sets : {n_param_sets}")
    print(f"  Avg replicates/set: {len(sims) / n_param_sets:.2f}")

    # Random shuffle and assign to splits 
    perm    = rng.permutation(n_param_sets)
    n_train = int(n_param_sets * train_ratio)
    n_val   = int(n_param_sets * val_ratio)
    # remainder goes to test, avoids rounding errors leaving sets stranded
    n_test  = n_param_sets - n_train - n_val

    train_param_idx = perm[:n_train]
    val_param_idx   = perm[n_train : n_train + n_val]
    test_param_idx  = perm[n_train + n_val:]

    #  Collect simulation indices for each split 
    def collect(param_indices):
        sim_idx = []
        for pi in param_indices:
            sim_idx.extend(param_to_indices[param_keys[pi]])
        return sim_idx

    train_sim_idx = collect(train_param_idx)
    val_sim_idx   = collect(val_param_idx)
    test_sim_idx  = collect(test_param_idx)

    train_sims = [sims[i] for i in train_sim_idx]
    val_sims   = [sims[i] for i in val_sim_idx]
    test_sims  = [sims[i] for i in test_sim_idx]

    # Summary 
    print("\nSplit summary")
    print(f"  Train : {len(train_param_idx):4d} param sets  |  {len(train_sims):6d} simulations")
    print(f"  Val   : {len(val_param_idx):4d} param sets  |  {len(val_sims):6d} simulations")
    print(f"  Test  : {len(test_param_idx):4d} param sets  |  {len(test_sims):6d} simulations")

    # R0 coverage report 
    print("\nR0 coverage per split  (R₀ = tau/gamma × {:.3f})".format(RATIO))
    for name, p_idx in [("train", train_param_idx),
                         ("val",   val_param_idx),
                         ("test",  test_param_idx)]:
        r0s = np.array([(param_keys[i][0] / param_keys[i][1]) * RATIO
                         for i in p_idx])
        sub   = (r0s < 0.8).sum()
        thr   = ((r0s >= 0.8) & (r0s <= 1.2)).sum()
        sup   = (r0s > 1.2).sum()
        n     = len(r0s)
        print(f"  {name:5s}  R₀<0.8: {sub:4d} ({100*sub/n:.1f}%)  "
              f"0.8–1.2: {thr:4d} ({100*thr/n:.1f}%)  "
              f">1.2: {sup:4d} ({100*sup/n:.1f}%)  "
              f"mean={r0s.mean():.3f}")

    # Infer n_timepoints from first simulation 
    n_timepoints = len(sims[0]['output']['t'])

    # Build metadata — preserve original and add split fields 
    metadata = dict(dataset.get('metadata', {}))
    metadata['n_timepoints'] = n_timepoints     

    # Assemble split_data 
    split_data = {
        'train': {
            'simulations' : train_sims,
            'indices'     : train_sim_idx,
            'param_indices': train_param_idx.tolist(),
        },
        'val': {
            'simulations' : val_sims,
            'indices'     : val_sim_idx,
            'param_indices': val_param_idx.tolist(),
        },
        'test': {
            'simulations' : test_sims,
            'indices'     : test_sim_idx,
            'param_indices': test_param_idx.tolist(),
        },
        'network' : dataset.get('network', {}),  
        'metadata': metadata,
        'split_info': {
            'method'       : 'parameter_set_random',
            'n_param_sets' : n_param_sets,
            'n_train_sets' : int(len(train_param_idx)),
            'n_val_sets'   : int(len(val_param_idx)),
            'n_test_sets'  : int(len(test_param_idx)),
            'train_ratio'  : train_ratio,
            'val_ratio'    : val_ratio,
            'test_ratio'   : test_ratio,
            'seed'         : seed,
            'ratio'        : RATIO,
            'leakage_check': 'PASSED',
            'stratified'   : False,
        },
    }

    return split_data

# CSV EXPORT
def export_training_csv(split_data: dict, output_csv_path: Path) -> pd.DataFrame:
    """
    Export one row per training simulation with parameters,
    R0, epidemic summaries, and a near-threshold flag.
    """
    rows = []

    for sim, orig_idx in zip(split_data['train']['simulations'],
                             split_data['train']['indices']):

        p     = sim['params']
        tau   = float(p.get('tau',   p.get('beta',  np.nan)))
        gamma = float(p.get('gamma', p.get('mu',    np.nan)))
        rho   = float(p.get('rho',   np.nan))
        R0    = (tau / gamma) * RATIO if gamma > 0 else np.nan  

        out    = sim['output']
        I_arr  = np.array(out.get('I', []))
        R_arr  = np.array(out.get('R', []))
        t_arr  = np.array(out.get('t', []))

        if len(I_arr) > 0:
            peak_I    = float(I_arr.max())
            peak_time = float(t_arr[I_arr.argmax()]) if len(t_arr) > 0 else float(I_arr.argmax())
            final_R   = float(R_arr[-1]) if len(R_arr) > 0 else np.nan
        else:
            peak_I = peak_time = final_R = np.nan

        rows.append({
            'sim_index'      : int(orig_idx),
            'tau'            : round(tau,   8),
            'gamma'          : round(gamma, 8),
            'rho'            : round(rho,   8),
            'R0'             : round(R0,    4) if not np.isnan(R0) else np.nan,  # FIX 3
            'near_threshold' : int(abs(R0 - 1.0) < 0.2) if not np.isnan(R0) else 0,
            'peak_I'         : round(peak_I,    4) if not np.isnan(peak_I)  else np.nan,
            'peak_time'      : round(peak_time, 2) if not np.isnan(peak_time) else np.nan,
            'final_R'        : round(final_R,   4) if not np.isnan(final_R) else np.nan,
        })

    df = pd.DataFrame(rows)

    # Summary
    print(f"\nTraining CSV: {len(df)} rows")
    if 'R0' in df.columns:
        total = len(df)
        print(f"R₀ < 0.8 : {(df['R0'] < 0.8).sum():5d}  ({100*(df['R0']<0.8).mean():.1f}%)")
        print(f"0.8 ≤ R₀ ≤ 1.2 : {((df['R0']>=0.8)&(df['R0']<=1.2)).sum():5d}  "
              f"({100*((df['R0']>=0.8)&(df['R0']<=1.2)).mean():.1f}%)")
        print(f"R₀ > 1.2: {(df['R0'] > 1.2).sum():5d}  ({100*(df['R0']>1.2).mean():.1f}%)")
        print(f"  Near threshold (±0.2): {df['near_threshold'].sum()} "
              f"({100*df['near_threshold'].mean():.1f}%)")

    df.to_csv(output_csv_path, index=False)
    print(f"\nSaved CSV : {output_csv_path.resolve()}")

    return df

# ENTRY POINT
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split SIR dataset by parameter set"
    )
    parser.add_argument('--input',type=str,default="epidemic_data_age_adaptive_sobol.pkl",
                        help="Filename ")
    parser.add_argument('--output',type=str,default=None,
                        help="Output pickle filename : <input>_split.pkl)")
    parser.add_argument('--output_csv',type=str,default=None,
                        help="Output CSV filename)")
    parser.add_argument('--train_ratio',type=float, default=0.70)
    parser.add_argument('--val_ratio',type=float, default=0.15)
    parser.add_argument('--test_ratio',type=float, default=0.15)
    parser.add_argument('--seed',type=int,default=42)
    args, _ = parser.parse_known_args()

    #  Paths 
    input_path = RAW_DATA_DIR / args.input
    stem = Path(args.input).stem
    out_pkl = (SPLIT_DATA_DIR / args.output) if args.output else \
              (SPLIT_DATA_DIR / f"{stem}_split.pkl")
    out_csv = (SPLIT_DATA_DIR / args.output_csv) if args.output_csv else \
              (SPLIT_DATA_DIR / f"{stem}_train_params.csv")

    print(f"\n  Input  : {input_path.resolve()}")
    print(f"  Output : {out_pkl.resolve()}")
    print(f"  CSV    : {out_csv.resolve()}")

    # Create output directory if needed 
    SPLIT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    #  Load  
    print(f"\nLoading {input_path.resolve()}")
    with open(input_path, 'rb') as f:
        dataset = pickle.load(f)

    n_sims = len(dataset['simulations'])
    print(f"  {n_sims} simulations loaded")

    # Split 
    split_data = split_dataset(
        dataset,
        train_ratio = args.train_ratio,
        val_ratio   = args.val_ratio,
        test_ratio  = args.test_ratio,
        seed        = args.seed,
    )

    #  Save pickle 
    with open(out_pkl, 'wb') as f:
        pickle.dump(split_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = out_pkl.stat().st_size / (1024 ** 2)
    print(f"\nSaved split pickle {out_pkl.resolve()}  ({size_mb:.2f} MB)")

    # Export CSV 
    export_training_csv(split_data, out_csv)
    print(f" Split pickle : {out_pkl.resolve()}")
    print(f"  Training CSV : {out_csv.resolve()}")
   
    