
import numpy as np
import networkx as nx
from scipy.stats import qmc
import EoN
from pathlib import Path
import pickle
from scipy.stats import qmc
from pathlib import Path
from tqdm import tqdm
from pathlib import Path
import csv
import pymc as pm 
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd

import warnings
warnings.filterwarnings("ignore")
 
# PARAMETERS
N = 1000                 # network size
m = 10                      # Barabasi–Albert attachment parameter

tmax=80
n_timepoints=80
initial_samples=500   # initial Sobol samples #500
sigma = 0.30            # width of R0 target distribution
n_replicates=1  # replicates of parameter sets 

PARAM_RANGES = {
    'tau'  : (0.0005,0.024),  # Expected range: R₀ ∈ [0.12, 4.98] #   recovery rate
    'gamma': (0.01,0.5),     # Expected range: R₀ ∈ [0.12, 4.98] #   transmission rate
    'rho'  : (0.001,0.010),
}

PARAM_NAMES = ['tau', 'gamma', 'rho']
output_path = Path('epidemic_data_age_adaptive_sobol.pkl')


ratio=34.0 # calculated for 100000 graphs

net_stats = {
    'k_avg': 9.9988,
    'k2_avg': 266.020,
    'ratio': ratio,
    'k_std': 1.5,
    'k_max': 20
}



# R0 COMPUTATION
def compute_R0(samples,ratio):
    """
    Compute epidemic reproduction number.

    R0 = (tau/gamma) * <k²>/<k>
    """

    tau = samples[:, 0]
    gamma = samples[:, 1]

    R0 = (tau/gamma) * ratio  

    return R0

# Using MCMC to sample from the target distribution  with Uniform distribution

#thinning    = int(mcmc_steps / initial_samples) 

with pm.Model() as model:
    
    # Priors: Uniform over plausible ranges
    tau   = pm.Uniform("tau", lower=PARAM_RANGES['tau'][0], upper=PARAM_RANGES['tau'][1])
    gamma = pm.Uniform("gamma", lower=PARAM_RANGES['gamma'][0], upper=PARAM_RANGES['gamma'][1])
    rho   = pm.Uniform("rho", lower=PARAM_RANGES['rho'][0], upper=PARAM_RANGES['rho'][1])

    # Compute R0 deterministically
    R0 = pm.Deterministic("R0", (tau / gamma) * ratio)
    
    # Target density: Gaussian around R0 ≈ 1
    logp = -0.5 * ((R0 - 1.0) / sigma) ** 2
    pm.Potential("R0_target", logp)
    
    # Sample with NUTS
   
    trace = pm.sample(  
        draws=initial_samples,
        tune=2000, # thinning,
        chains=2,  # thus Total_sample=samples*chains
        cores=1,
        target_accept=0.95,
        random_seed=42
    )

# Convert trace to (n_samples, 3) array: tau, gamma, rho
posterior_samples = np.vstack([
    trace.posterior['tau'].values.flatten(),
    trace.posterior['gamma'].values.flatten(),
    trace.posterior['rho'].values.flatten()
]).T


az.summary(trace, var_names=["tau", "gamma", "rho"])

az.plot_trace(
    trace,
    var_names=["tau", "gamma", "rho"],
    compact=False,
    figsize=(12, 8)
)

plt.tight_layout()
plt.show()


#posterior sample dimensions 
print(posterior_samples.shape) #1000,3


def run_batch(G, posterior_samples, n_replicates, tmax, n_timepoints, seed=None):
    
    if seed is not None:
        np.random.seed(seed)

    t_fixed = np.linspace(0, tmax, n_timepoints)
    all_sims = []

    for tau, gamma, rho in tqdm(posterior_samples, desc="Running batch simulations"):

        for rep in range(n_replicates):

            t, S, I, R = EoN.fast_SIR(G, tau, gamma, rho=rho, tmax=tmax)

            all_sims.append({
                'params': {
                    'tau': float(tau),
                    'gamma': float(gamma),
                    'rho': float(rho),
                },
                'output': {
                    't': t_fixed,
                    'S': np.interp(t_fixed, t, S),
                    'I': np.interp(t_fixed, t, I),
                    'R': np.interp(t_fixed, t, R),
                },
                'replicate_id': rep,
            })

    print(f"Generated {len(all_sims)} simulations "
          f"({len(posterior_samples)} param sets × {n_replicates} replicates)")

    return all_sims



# Build dataset structure

def build_dataset(all_sims, G, net_stats, m, n_replicates, param_ranges):

    n_timepoints = len(all_sims[0]['output']['t'])

    dataset = {

        'simulations': all_sims,

        'network': {
            'type': 'barabasi_albert',
            'N': G.number_of_nodes(),
            'm': m,
            'ratio': net_stats['ratio'],
            'graph': G,
        },

        'metadata': {
            'n_samples': len(all_sims),
            'n_replicates': n_replicates,
            'n_timepoints': n_timepoints,
            'param_ranges': param_ranges,
            'R0_formula': 'R0 = (tau/gamma) * <k²>/<k>',
            'sampling_strategy': 'MCMC',
        }
    }

    return dataset



# Aggregate simulations for plotting


def summarise_for_plot(dataset):

    sims = dataset['simulations']
    t_fixed = sims[0]['output']['t']

    all_I = np.array([s['output']['I'] for s in sims])
    all_S = np.array([s['output']['S'] for s in sims])
    all_R = np.array([s['output']['R'] for s in sims])

    return {
        't': t_fixed,
        'all_I': all_I,
        'all_S': all_S,
        'all_R': all_R,
        'I_mean': all_I.mean(axis=0),
        'I_p10': np.percentile(all_I, 10, axis=0),
        'I_p25': np.percentile(all_I, 25, axis=0),
        'I_p75': np.percentile(all_I, 75, axis=0),
        'I_p90': np.percentile(all_I, 90, axis=0),
    }


# Plot epidemic uncertainty


def plot_sir_uncertainty(full_results, N):

    t = full_results['t']
    all_I = full_results['all_I']
    all_R = full_results['all_R']

    final_R = all_R[:, -1]

    extinct_mask = final_R < 0.01 * N
    outbreak_mask = ~extinct_mask

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    fig.suptitle(
        "SIR Epidemic Trajectories — Posterior + Stochastic Uncertainty",
        fontsize=13,
        fontweight='bold'
    )

    for ax, mask, label, color in [

        (axes[0], extinct_mask, "Extinction / near-threshold", "steelblue"),
        (axes[1], outbreak_mask, "Outbreak", "firebrick"),

    ]:

        subset = all_I[mask]
        n = mask.sum()

        if n == 0:
            ax.set_title(f"{label}\n(no trajectories)")
            continue

        mean_I = subset.mean(axis=0)
        p10 = np.percentile(subset, 10, axis=0)
        p25 = np.percentile(subset, 25, axis=0)
        p75 = np.percentile(subset, 75, axis=0)
        p90 = np.percentile(subset, 90, axis=0)

        ax.fill_between(t, p10, p90, color=color, alpha=0.15)
        ax.fill_between(t, p25, p75, color=color, alpha=0.30)

        ax.plot(t, mean_I, color=color, linewidth=2)

        ax.set_xlabel("Time")
        ax.set_ylabel("Number infectious")
        ax.set_title(f"{label} (n={n})")
        ax.set_ylim(bottom=0)

        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("trajectories_split.png", dpi=200, bbox_inches='tight')
    plt.show()



# Save dataset (pickle)


def save_dataset(dataset, filepath):

    with open(filepath, 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Dataset saved → {filepath} ({len(dataset['simulations'])} sims)")



# Save summary CSV


def save_csv(dataset, filepath):

    sims = dataset['simulations']
    ratio = dataset['network']['ratio']
    N = dataset['network']['N']

    fields = [
        'sim_id', 'replicate_id',
        'tau', 'gamma', 'rho',
        'R0', 'peak_I', 'peak_time',
        'final_R', 'attack_rate', 'near_threshold'
    ]

    with open(filepath, 'w', newline='') as f:

        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for sim_id, sim in enumerate(sims):

            tau = sim['params']['tau']
            gamma = sim['params']['gamma']
            rho = sim['params']['rho']

            R0 = (tau / gamma) * ratio

            I = sim['output']['I']
            R = sim['output']['R']
            t = sim['output']['t']

            peak_I = float(I.max())

            peak_time = float(t[I.argmax()]) if peak_I > 0 else np.nan

            writer.writerow({

                'sim_id': sim_id,
                'replicate_id': sim.get('replicate_id', 0),

                'tau': tau,
                'gamma': gamma,
                'rho': rho,

                'R0': round(R0, 4),

                'peak_I': peak_I,
                'peak_time': peak_time,

                'final_R': float(R[-1]),
                'attack_rate': float(R[-1] / N),

                'near_threshold': int(abs(R0 - 1) < 0.2),
            })

    print(f"CSV saved to {filepath} ({len(sims)} rows)")


# Entry point

if __name__ == "__main__":
    import argparse
    from experiment_paths import get_paths, makedirs, RAW_PKL, RAW_CSV

    parser = argparse.ArgumentParser(
        description="Step 1 — MCMC Sampling (NUTS near R0=1) data generation"
    )
    parser.add_argument('--experiment', type=str, default='mcmc',
                        choices=['random', 'lhs', 'mcmc'],
                        help="Experiment to generate data for (default: mcmc).")
    args = parser.parse_args()

    # Resolve and create output paths
    paths = makedirs(args.experiment)
    raw_pkl = paths['data_raw'] / RAW_PKL
    raw_csv = paths['data_raw'] / RAW_CSV

    print(f"\n  Experiment : {args.experiment}")
    print(f"  Output pkl : {raw_pkl}")
    print(f"  Output csv : {raw_csv}")

    G = nx.barabasi_albert_graph(N, m)

    all_sims = run_batch(
        G,
        posterior_samples,
        n_replicates,
        tmax,
        n_timepoints,
    )

    dataset = build_dataset(
        all_sims,
        G,
        net_stats,
        m,
        n_replicates,
        PARAM_RANGES,
    )

    full_results = summarise_for_plot(dataset)
    plot_sir_uncertainty(full_results, N=G.number_of_nodes())

    save_dataset(dataset, raw_pkl)
    save_csv(dataset, raw_csv)

    print(f"\n  Raw data saved to : {paths['data_raw']}")
    print(f"  Next step         : run step2 split notebook, then step2A augmentation, then:")
    print(f"    python src/step3_train.py --experiment {args.experiment}")

#Data Exploration 

data=pd.read_csv('epidemic_data_age_adaptive_sobol.csv')

print(data.columns)

print(data.head(20))
print(len(data))
print(data.isnull().sum()) # no missing values

print(data.shape)
print(ratio)


print(data.describe(include='all'))

total_samples=len(data)
print(total_samples)
greater=data[data['R0']>1.2]
pc1=(len(greater)/total_samples)*100
print(f"greater than 1.2: {len(greater),pc1}")
between = data[(data['R0'] >= 0.8) & (data['R0'] <= 1.2)]
pc2=(len(between)/total_samples)*100
print(f"between 0.8 and 1.2: {len(between),pc2}")

#print(pc2)

less_than=data[data['R0']<0.8]
pc3=(len(less_than)/total_samples)*100
print(f"less than 0.8: {len(less_than),pc3}")




print(ratio)
slope=1/ratio
print(slope)


slope = 1/ratio

plt.figure(figsize=(10,10))
plt.scatter(data['gamma'], data['tau'], alpha=0.2)

x_vals = np.linspace(data['gamma'].min(), data['gamma'].max(), 100)
y_vals = slope * x_vals

plt.plot(x_vals, y_vals, color='red', linestyle='--', label=f'slope={slope}')

plt.xlabel('gamma')
plt.ylabel('tau')
plt.title('Scatter plot of tau vs gamma')
plt.legend()
plt.grid(True)
plt.show()

