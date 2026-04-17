import numpy as np
import pickle
from copy import deepcopy
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import pickle


DATA_DIR = Path("experiments/mcmc-sampling/data/split")
AUGMENTED_DATA_DIR =Path("experiments/mcmc-sampling/data/augmented")
PLOTS_DIR = Path("experiments/mcmc-sampling/out/plots/augmentation_plots")

INPUT_PKL=DATA_DIR/"epidemic_data_age_adaptive_sobol_split.pkl"
AUGMENTED_PKL=AUGMENTED_DATA_DIR/"epidemic_data_age_adaptive_sobol_split_augmented.pkl"
AUGMENTED_CSV=AUGMENTED_DATA_DIR/"epidemic_data_age_adaptive_sobol_split_augmented.csv"

class SIRAugmenter:
    def __init__(self, param_noise=0.01, comp_noise=0.01,  #0.001
                 n_param_aug=10, n_comp_aug=1): #was 10, 1
        self.param_noise = param_noise
        self.comp_noise  = comp_noise
        self.n_param_aug = n_param_aug
        self.n_comp_aug  = n_comp_aug

    # This is the method your code expects
    def augment_simulation(self, sim):
        sims = [sim]
        for _ in range(self.n_param_aug):
            sims.append(self.augment_params(sim))
        for _ in range(self.n_comp_aug):
            sims.append(self.augment_compartments(sim))
        return sims

    def augment_params(self, sim):
        sim_new = deepcopy(sim)
        params = sim_new['params']
        for k in ['tau', 'gamma', 'rho']:
            params[k] *= 1 + np.random.normal(0, self.param_noise)
        return sim_new

    def augment_compartments(self, sim):
        sim_new = deepcopy(sim)
        out = sim_new['output']
        S, I, R = np.array(out['S']), np.array(out['I']), np.array(out['R'])
        S2 = np.maximum(S * (1 + np.random.normal(0, self.comp_noise, S.shape)), 0)
        I2 = np.maximum(I * (1 + np.random.normal(0, self.comp_noise, I.shape)), 0)
        R2 = np.maximum(R * (1 + np.random.normal(0, self.comp_noise, R.shape)), 0)
        N = S[0] + I[0] + R[0]
        factor = N / (S2 + I2 + R2 + 1e-8)
        sim_new['output'] = {'t': out['t'],
                             'S': (S2 * factor).tolist(),
                             'I': (I2 * factor).tolist(),
                             'R': (R2 * factor).tolist()}
        return sim_new


# Apply augmentation
    
def augment_train_split(split_data, augmenter):

    augmented = deepcopy(split_data)

    original = split_data["train"]["simulations"]
    new_sims = []

    for sim in original:
        new_sims.extend(augmenter.augment_simulation(sim))

    augmented["train"]["simulations"] = new_sims

    return augmented

# Export CSV of parameters + R0 for training set
def export_params_with_R0(split_data, csv_path, split_name="train", ratio=34.0):
    sims = split_data[split_name]['simulations']
    rows = []
    for sim in sims:
        tau   = sim['params']['tau']
        gamma = sim['params']['gamma']
        rho   = sim['params']['rho']
        R0    = (tau / gamma) * ratio  
        rows.append({'tau': tau, 'gamma': gamma, 'rho': rho, 'R0': R0})

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"CSV saved → {csv_path}  shape={df.shape}")
    return df

# Run augmentation

if __name__ == "__main__":

    #  Step 1: Load split data from DATA_DIR 
    print("=" * 60)
    print("SIR DATA AUGMENTATION")
    print("=" * 60)
    print(f"\n  Input   : {INPUT_PKL.resolve()}")
    print(f"  Output  : {AUGMENTED_PKL.resolve()}")
    print(f"  CSV     : {AUGMENTED_CSV.resolve()}")
    print(f"  Plots   : {PLOTS_DIR.resolve()}\n")

    with open(INPUT_PKL, "rb") as f:        
        data = pickle.load(f)

    print(f"  Train simulations (before): "
          f"{len(data['train']['simulations'])}")

    #  Step 2: Augment 
    augmenter = SIRAugmenter(
        param_noise  = 0.001,
        comp_noise   = 0.01,
        n_param_aug  = 10,
        n_comp_aug   = 1,
    )

    augmented = augment_train_split(data, augmenter)

    print(f"  Train simulations (after) : "
          f"{len(augmented['train']['simulations'])}")

    # Step 3: Save augmented pickle 
    with open(AUGMENTED_PKL, "wb") as f:   
        pickle.dump(augmented, f)
    print(f"\nSaved pickle → {AUGMENTED_PKL}")

    #  Step 4: Export CSV 
    df = export_params_with_R0(
        augmented,
        csv_path   = AUGMENTED_CSV,        
        split_name = "train",
        ratio      = 34.0,
    )

    #  Step 5: Explore 
    augmented_data = pd.read_csv(AUGMENTED_CSV)   

    print(f"\nColumns      : {list(augmented_data.columns)}")
    print(f"Total rows   : {len(augmented_data)}")
    print(f"Missing      :\n{augmented_data.isnull().sum()}")
    print(f"\n{augmented_data.describe(include='all')}")

    total = len(augmented_data)
    greater   = augmented_data[augmented_data['R0'] > 1.2]
    between   = augmented_data[(augmented_data['R0'] >= 0.7) &
                               (augmented_data['R0'] <= 1.2)]
    less_than = augmented_data[augmented_data['R0'] < 0.8]

    print(f"\nR₀ > 1.2       : {len(greater):5d}  ({len(greater)/total*100:.1f}%)")
    print(f"0.7 ≤ R₀ ≤ 1.2 : {len(between):5d}  ({len(between)/total*100:.1f}%)")
    print(f"R₀ < 0.8       : {len(less_than):5d}  ({len(less_than)/total*100:.1f}%)")

    # Step 6: Scatter plot 
    slope = 1 / 34.0

    plt.figure(figsize=(10, 10))
    plt.scatter(augmented_data['gamma'], augmented_data['tau'],
                alpha=0.1, label='Augmented samples')

    x_vals = np.linspace(augmented_data['gamma'].min(),
                         augmented_data['gamma'].max(), 100)
    y_vals = slope * x_vals
    plt.plot(x_vals, y_vals, color='red', linestyle='--',
             label=f'R₀=1 boundary  (slope={slope:.5f})')

    plt.xlabel('gamma (γ)')
    plt.ylabel('tau (τ)')
    plt.title('Augmented Posterior Samples — τ vs γ\n'
              '(Points above red line: R₀ > 1)')
    plt.legend()
    plt.grid(True)

    scatter_path = PLOTS_DIR / "tau_gamma_scatter_augmented.png" 
    plt.savefig(scatter_path, dpi=200, bbox_inches='tight')
    plt.show()
    print(f"Saved: {scatter_path}")

    print(f"\n  Data  → {AUGMENTED_DATA_DIR.resolve()}")
    print(f"  Plots → {PLOTS_DIR.resolve()}")
    print("\nDone.")
    





