
import pickle
import numpy as np
import pandas as pd
import argparse
from pathlib import Path


DATA_DIR = Path("experiments/lhs-sampling/data/split")
# SPLIT DATASET BY PARAMETER SET (RANDOM, NO STRATIFICATION)


def split_dataset(dataset,
                  train_ratio=0.70,
                  val_ratio=0.15,
                  test_ratio=0.15,
                  seed=42):

    """
    Split dataset by PARAMETER SET.

    Steps
    1. Group simulations by (tau, gamma, rho)
    2. Randomly shuffle parameter sets
    3. Assign param sets to train / val / test
    4. All replicates stay in the same split
    5. Verify zero parameter leakage
    """

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, \
        "Ratios must sum to 1"

    rng = np.random.default_rng(seed)

    sims = dataset['simulations']

    # Step 1: Group simulations by parameter set
  

    param_to_indices = {}

    for i, sim in enumerate(sims):

        p = sim['params']

        tau   = float(p.get('tau', p.get('beta', 0)))
        gamma = float(p.get('gamma', p.get('mu', 0)))
        rho   = float(p.get('rho', 0))

        key = (round(tau,8), round(gamma,8), round(rho,8))

        param_to_indices.setdefault(key, []).append(i)

    param_keys = list(param_to_indices.keys())
    n_param_sets = len(param_keys)

    print("\nDataset summary")
    print("----------------------")
    print(f"Total simulations: {len(sims)}")
    print(f"Unique parameter sets: {n_param_sets}")
    print(f"Average replicates per set: {len(sims)/n_param_sets:.2f}")

  
    # Step 2: Random shuffle parameter sets
   

    perm = rng.permutation(n_param_sets)

    n_train = int(n_param_sets * train_ratio)
    n_val   = int(n_param_sets * val_ratio)

    train_param_idx = perm[:n_train]
    val_param_idx   = perm[n_train:n_train+n_val]
    test_param_idx  = perm[n_train+n_val:]

  
    # Step 3: Collect simulations for each split
  

    def collect_simulations(param_indices):

        sim_indices = []

        for pi in param_indices:
            sim_indices.extend(param_to_indices[param_keys[pi]])

        return sim_indices

    train_sim_idx = collect_simulations(train_param_idx)
    val_sim_idx   = collect_simulations(val_param_idx)
    test_sim_idx  = collect_simulations(test_param_idx)

    train_sims = [sims[i] for i in train_sim_idx]
    val_sims   = [sims[i] for i in val_sim_idx]
    test_sims  = [sims[i] for i in test_sim_idx]

    # Step 4: Verify zero parameter leakage
  

    train_params = set(param_keys[i] for i in train_param_idx)
    val_params   = set(param_keys[i] for i in val_param_idx)
    test_params  = set(param_keys[i] for i in test_param_idx)

    leak_tr_te = train_params & test_params
    leak_tr_va = train_params & val_params
    leak_va_te = val_params & test_params

    print("\nLeakage check")

    print(f"Train ∩ Test param sets: {len(leak_tr_te)}")
    print(f"Train ∩ Val  param sets: {len(leak_tr_va)}")
    print(f"Val   ∩ Test param sets: {len(leak_va_te)}")

    if len(leak_tr_te) or len(leak_tr_va) or len(leak_va_te):
        raise RuntimeError("Parameter leakage detected!")

    print("Zero leakage confirmed")

   
    # Step 5: Print split summary
   

    print("\nSplit summary")
    print("----------------------")
    print(f"Train: {len(train_param_idx)} param sets | {len(train_sim_idx)} simulations")
    print(f"Val  : {len(val_param_idx)} param sets | {len(val_sim_idx)} simulations")
    print(f"Test : {len(test_param_idx)} param sets | {len(test_sim_idx)} simulations")

    split_data = {
        'train': {
            'simulations': train_sims,
            'indices': train_sim_idx,
            'param_indices': train_param_idx.tolist()
        },
        'val': {
            'simulations': val_sims,
            'indices': val_sim_idx,
            'param_indices': val_param_idx.tolist()
        },
        'test': {
            'simulations': test_sims,
            'indices': test_sim_idx,
            'param_indices': test_param_idx.tolist()
        },
        'metadata': dataset['metadata'],
        'split_info': {
            'method': 'parameter_set_random',
            'n_param_sets': n_param_sets,
            'seed': seed,
            'leakage_check': 'PASSED'
        }
    }

    return split_data



# EXPORT TRAINING CSV


def export_training_csv(split_data, output_csv_path):

    rows = []

    for sim, orig_idx in zip(split_data['train']['simulations'],
                             split_data['train']['indices']):

        p = sim['params']

        tau   = float(p.get('tau', p.get('beta', np.nan)))
        gamma = float(p.get('gamma', p.get('mu', np.nan)))
        rho   = float(p.get('rho', np.nan))

        out = sim['output']

        I_mean = np.array(out.get('I_mean', out.get('I', [])))
        R_mean = np.array(out.get('R_mean', out.get('R', [])))

        if len(I_mean) > 0:

            peak_I = float(I_mean.max())
            peak_time = int(I_mean.argmax())
            final_R = float(R_mean[-1]) if len(R_mean) > 0 else np.nan

        else:

            peak_I = np.nan
            peak_time = np.nan
            final_R = np.nan

        rows.append({
            "sim_index": int(orig_idx),
            "tau": tau,
            "gamma": gamma,
            "rho": rho,
            "peak_I": peak_I,
            "peak_time": peak_time,
            "final_R": final_R
        })

    df = pd.DataFrame(rows)

    print(f"\nTraining CSV rows: {len(df)}")

    df.to_csv(output_csv_path, index=False)

    print(f"Saved CSV → {output_csv_path}")

    return df



# ENTRY POINT


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Random split by PARAMETER SET (replicates grouped)"
    )

    parser.add_argument('--input',
                        type=str,
                        default="epidemic_data_age_adaptive_sobol.pkl")

    parser.add_argument('--output',
                        type=str,
                        default=None)

    parser.add_argument('--output_csv',
                        type=str,
                        default=None)

    parser.add_argument('--train_ratio',
                        type=float,
                        default=0.70)

    parser.add_argument('--val_ratio',
                        type=float,
                        default=0.15)

    parser.add_argument('--test_ratio',
                        type=float,
                        default=0.15)

    parser.add_argument('--seed',
                        type=int,
                        default=42)

    args, unknown = parser.parse_known_args()

    print("="*65)
    print("STEP 2: RANDOM PARAMETER SPLIT (LEAKAGE-FREE)")
    print("="*65)

    with open(args.input, 'rb') as f:
        dataset = pickle.load(f)

    split_data = split_dataset(
        dataset,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    # Save split pickle
    out_path = Path(args.output) if args.output else \
        Path(args.input).parent / (Path(args.input).stem + "_split_augmented.pkl")

    with open(out_path, 'wb') as f:
        pickle.dump(split_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nSaved split dataset → {out_path}")

    # Export training CSV
    csv_path = Path(args.output_csv) if args.output_csv else \
        DATA_DIR/(Path(args.input).stem + "_train_params.csv")

    export_training_csv(split_data, csv_path)

    print("\nDone.")


