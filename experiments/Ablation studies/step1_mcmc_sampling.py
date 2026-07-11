
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
import time
import datetime
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path("experiments/mcmc-sampling/data/raw")
PLOTS_DIR = Path("experiments/mcmc-sampling/out/plots/mcmc_sampling_plots")
MCMC_DIR = Path("experiments/mcmc-sampling/data/raw/mcmc_posterior_results")
RESULTS_DIR = Path("experiments/mcmc-sampling/out/results")
OUTPUT_PKL = DATA_DIR / "abm-data.pkl"
OUTPUT_CSV = DATA_DIR / "abm-data.csv"
TIMING_FILE = RESULTS_DIR / "timing.txt"


# PARAMETERS
N=100000              # network size
m=10                      # Barabasi–Albert attachment parameter

tmax=250
n_timepoints=250
initial_samples=10000 # initial Sobol samples #500
sigma = 2# width of R0 target distribution
n_replicates=1  # replicates of parameter sets 

PARAM_RANGES = {  
    'tau':(0.00025,0.17),
    'gamma':(0.03,1),
    'rho':(0.001,0.01)

}



PARAM_NAMES = ['tau', 'gamma', 'rho']
output_path = Path('abm-data.pkl')

# results_k_avg = []
# results_k2_avg = []
# results_ratio = []

# for i in range(1000):
    
#     G = nx.barabasi_albert_graph(N, m)
    
#     degrees = np.array([d for _, d in G.degree()])
    
#     k_avg = degrees.mean()
#     k2_avg = (degrees**2).mean()
#     ratio = (k2_avg-k_avg)/ k_avg   # R0 = (tau/gamma) * (<k²> - <k>) / <k> that is our BA network ratio for R0 computation 
    
#     results_k_avg.append(k_avg)
#     results_k2_avg.append(k2_avg)
#     results_ratio.append(ratio)

# # converting to arrays
# results_k_avg = np.array(results_k_avg)
# results_k2_avg = np.array(results_k2_avg)
# results_ratio = np.array(results_ratio)

# # means
# mean_k_avg = results_k_avg.mean()
# mean_k2_avg = results_k2_avg.mean()
# mean_ratio = results_ratio.mean()

# # standard errors 
# se_k_avg = results_k_avg.std(ddof=1)/np.sqrt(len(results_k_avg))
# se_k2_avg = results_k2_avg.std(ddof=1)/np.sqrt(len(results_k2_avg))
# se_ratio = results_ratio.std(ddof=1)/np.sqrt(len(results_ratio))

# # 95% CI  (mean ± 1.96 * standard error)
# ci_k_avg = (mean_k_avg - 1.96*se_k_avg, mean_k_avg + 1.96*se_k_avg)
# ci_k2_avg = (mean_k2_avg - 1.96*se_k2_avg, mean_k2_avg + 1.96*se_k2_avg)
# ci_ratio = (mean_ratio - 1.96*se_ratio, mean_ratio + 1.96*se_ratio)

# print("Mean <k>:", mean_k_avg, "CI:", ci_k_avg)
# print("Mean <k^2>:", mean_k2_avg, "CI:", ci_k2_avg)
# print("Mean <k^2>/<k>:", mean_ratio, "CI:", ci_ratio)






#ratio=58.979# calculated for 1000 graphs
ratio = 58# calculated for 1000 graphs
net_stats = {
    'k_avg': 19.99,
    'k2_avg': 1199.47,
    'ratio': ratio,
    
}



# R0 COMPUTATION
def compute_R0(samples,ratio):
    """
    Epidemic reproduction number.

    R0 = (tau/gamma) * (<k²> - <k>) / <k> 
    """

    tau = samples[:, 0]
    gamma = samples[:, 1]

    R0 = (tau/gamma) * ratio  

    return R0

#PRIOR SPECIFICATION
# Using MCMC-NUTS to sample from the target distribution  with Uniform distribution

t0_data_gen = time.perf_counter()

with pm.Model() as model:

    # Priors: Uniform over plausible ranges
    tau= pm.Uniform("tau", lower=PARAM_RANGES['tau'][0], upper=PARAM_RANGES['tau'][1])
    gamma= pm.Uniform("gamma", lower=PARAM_RANGES['gamma'][0], upper=PARAM_RANGES['gamma'][1])
    rho = pm.Uniform("rho", lower=PARAM_RANGES['rho'][0], upper=PARAM_RANGES['rho'][1])

    # Computing R0 deterministically
    R0 = pm.Deterministic("R0", (tau/gamma) * ratio)

    # Target density: Gaussian near the epidemic threshold (R0=1) with width sigma that controls the concentration of samples around R0=1
    logp = -0.5 * ((R0 - 1.0) / sigma) ** 2
    pm.Potential("R0_target", logp)

    # Sampling with NUTS
    trace = pm.sample(
        draws=initial_samples,
        tune=3000,
        chains=4 ,  # thus Total sample=samples*chains
        cores=1,
        thin=10,
        target_accept=0.95,
        random_seed=43
    )
#Thinning after sampling.
trace_thinned = trace.sel(draw=slice(None, None, 10))

data_gen_time = time.perf_counter() - t0_data_gen
print(f"Data generation time (uniform MCMC): {data_gen_time:.2f}s")

# Converting trace_thinned to (n_samples, 3) array: tau, gamma, rho
posterior_samples=np.vstack([
    trace_thinned.posterior['tau'].values.flatten(),
    trace_thinned.posterior['gamma'].values.flatten(),
    trace_thinned.posterior['rho'].values.flatten()
]).T

# Saving posterior samples to CSV
summary_df = az.summary(trace_thinned, var_names=["tau", "gamma", "rho", "R0"], hdi_prob=0.95)
print(summary_df)
summary_df.to_csv(MCMC_DIR/ "mcmc_summary.csv")

r0_samples = trace_thinned.posterior["R0"].values.flatten()
r0_lo, r0_hi = np.percentile(r0_samples, [5, 95])
print(f"\n95% posterior interval for R0: [{r0_lo:.4f}, {r0_hi:.4f}]")
print(f"Median R0: {np.median(r0_samples):.4f}  :  Mean R0: {r0_samples.mean():.4f}")

#Plotting trace plots for tau, gamma, rho
az.plot_trace(
    trace,
    var_names=["tau", "gamma", "rho"],
    compact=False,
    figsize=(12, 8)
)

plt.tight_layout()
trace_plot_path = PLOTS_DIR / "mcmc_trace_plot.png"
plt.savefig(trace_plot_path, dpi=200, bbox_inches='tight')
#plt.show()
print(f"Saved: {trace_plot_path}")


#posterior sample dimensions 
print(posterior_samples.shape) #4000,3

# Function to run batch simulations
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



# Building dataset structure
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

# Aggregating simulations for plotting
def summarise_for_plot(dataset):

    sims = dataset['simulations']
    ratio = dataset['network']['ratio']
    t_fixed = sims[0]['output']['t']

    all_I = np.array([s['output']['I'] for s in sims])
    all_S = np.array([s['output']['S'] for s in sims])
    all_R = np.array([s['output']['R'] for s in sims])

    R0_vals = np.array([(s['params']['tau'] / s['params']['gamma']) * ratio for s in sims])

    return {
        't': t_fixed,
        'all_I': all_I,
        'all_S': all_S,
        'all_R': all_R,
        'R0': R0_vals,
        'I_mean': all_I.mean(axis=0),
        'I_p10': np.percentile(all_I, 10, axis=0),
        'I_p25': np.percentile(all_I, 25, axis=0),
        'I_p75': np.percentile(all_I, 75, axis=0),
        'I_p90': np.percentile(all_I, 90, axis=0),
    }


# Plot for epidemic uncertainty — three R0 regimes
def plot_sir_uncertainty(full_results, output_dir=PLOTS_DIR):

    t = full_results['t']
    all_I = full_results['all_I']
    R0_vals = full_results['R0']
    total = len(R0_vals)

    sub_mask  = R0_vals < 0.1
    near_mask = (R0_vals >= 0.1) & (R0_vals <= 2)
    out_mask  = R0_vals > 2

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    fig.suptitle(
        "Epidemic Trajectories by R₀ Regime",
        fontsize=13,
        fontweight='bold'
    )

    for ax, mask, label, color in [
        (axes[0], sub_mask,  "Sub-threshold (R₀ < 0.1)",          "steelblue"),
        (axes[1], near_mask, "Near-threshold (0.1 ≤ R₀ ≤ 2)", "goldenrod"),
        (axes[2], out_mask,  "Outbreak (R₀ > 2)",                  "firebrick"),
    ]:
        subset = all_I[mask]
        n = mask.sum()

        if n == 0:
            ax.set_title(f"{label}\n(no trajectories)")
            ax.grid(True, alpha=0.3)
            continue

        mean_I = subset.mean(axis=0)
        p10 = np.percentile(subset, 10, axis=0)
        p25 = np.percentile(subset, 25, axis=0)
        p75 = np.percentile(subset, 75, axis=0)
        p90 = np.percentile(subset, 90, axis=0)

        ax.fill_between(t, p10, p90, color=color, alpha=0.15, label='10–90th pct')
        ax.fill_between(t, p25, p75, color=color, alpha=0.30, label='25–75th pct')
        ax.plot(t, mean_I, color=color, linewidth=2, label='Mean')

        ax.set_xlabel("Time")
        ax.set_ylabel("Number infectious")
        ax.set_title(f"{label}\n(n={n}, {n/total*100:.1f}%)")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    output_dir = Path(output_dir)
    out_path   = output_dir / "trajectories_split.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    #plt.show()
    print(f"Saved: {out_path}")



# Save dataset (pickle)
def save_dataset(dataset, filepath):

    with open(filepath, 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Dataset saved {filepath} ({len(dataset['simulations'])} sims)")



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

                'near_threshold': int(0.1 <= R0 <= 2),
            })

    print(f"CSV saved to {filepath} ({len(sims)} rows)")

# Entry point
G = nx.barabasi_albert_graph(N, m)

t0_sim = time.perf_counter()
all_sims = run_batch(
    G,
    posterior_samples,
    n_replicates,
    tmax,
    n_timepoints,
)
sim_time = time.perf_counter() - t0_sim
print(f"Simulation time: {sim_time:.2f}s")

dataset = build_dataset(
    all_sims,
    G,
    net_stats,
    m,
    n_replicates,
    PARAM_RANGES,
)

full_results = summarise_for_plot(dataset)

# Plots
plot_sir_uncertainty(full_results, output_dir=PLOTS_DIR)

# Data
save_dataset(dataset, OUTPUT_PKL)   
save_csv(dataset, OUTPUT_CSV)       

# Exploration 
data = pd.read_csv(OUTPUT_CSV)      

print(data.columns)
print(data.head(20))
print(f"\nTotal rows   : {len(data)}")
print(f"Missing values:\n{data.isnull().sum()}")
print(f"Shape: {data.shape}")
print(f"\n{data.describe(include='all')}")

summary = data.describe(include='all')

summary.to_csv(DATA_DIR / "summary.csv")

total_samples = len(data)
greater= data[data['R0'] > 2]
between = data[(data['R0'] >= 0.1) & (data['R0'] <= 2)]
less_than = data[data['R0'] < 0.1]

print(f"\nR0 > 2      : {len(greater)}  ({len(greater)/total_samples*100:.1f}%)")
print(f"0.1 ≤ R0 ≤ 2: {len(between)} ({len(between)/total_samples*100:.1f}%)")
print(f"R0 < 0.1    : {len(less_than)}  ({len(less_than)/total_samples*100:.1f}%)")

print(f"\nData : {DATA_DIR}")
print(f"Plots : {PLOTS_DIR}")

# Write timing report
n_samples = len(posterior_samples)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
with open(TIMING_FILE, "w") as f:
    f.write("=" * 50 + "\n")
    f.write("MCMC Sampling — Timing Report\n")
    f.write(f"Run date : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Samples generated : {n_samples}\n")
    f.write(f"Replicates        : {n_replicates}\n")
    f.write(f"Network size (N)  : {N}\n\n")
    f.write(f"Data generation time (uniform MCMC) : {data_gen_time:.2f} s  ({data_gen_time/60:.2f} min)\n")
    f.write(f"Simulation time                     : {sim_time:.2f} s  ({sim_time/60:.2f} min)\n")
    f.write(f"Total time                          : {(data_gen_time + sim_time):.2f} s  ({(data_gen_time + sim_time)/60:.2f} min)\n")
print(f"Timing saved: {TIMING_FILE}")


#Saves to PLOTS_DIR
# R0 = (tau/gamma) * ratio  =>  tau = R0 * gamma / ratio
slope = 1 / ratio  # R0=1 line slope
plt.figure(figsize=(10, 10))
plt.scatter(data['gamma'], data['tau'], alpha=0.3, s=10, color='steelblue', label='MCMC samples')

x_vals = np.linspace(data['gamma'].min(), data['gamma'].max(), 300)

# Near-threshold shaded region: 0.1 <= R0 <= 2.0
y_lo = (0.1 / ratio) * x_vals
y_hi = (2.0 / ratio) * x_vals
plt.fill_between(x_vals, y_lo, y_hi,
                 alpha=0.20, color='gold', label='Near threshold (0.1 ≤ R₀ ≤ 2.0)')
plt.plot(x_vals, y_lo, color='goldenrod', linestyle=':', linewidth=1.5, label='R₀ = 0.1')
plt.plot(x_vals, y_hi, color='darkorange', linestyle=':', linewidth=1.5, label='R₀ = 2.0')

# Epidemic threshold: R0 = 1
y_r0_1 = slope * x_vals
plt.plot(x_vals, y_r0_1, color='red', linestyle='--', linewidth=2, label='R₀ = 1 (epidemic threshold)')

tau_max = data['tau'].max() * 1.1
plt.ylim(0, tau_max)
plt.xlim(data['gamma'].min() * 0.95, data['gamma'].max() * 1.02)

plt.xlabel('gamma', fontsize=13)
plt.ylabel('tau', fontsize=13)
plt.title('Scatter plot of tau vs gamma — MCMC Posterior Samples', fontsize=13)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

scatter_path = PLOTS_DIR / "tau_gamma_scatter.png"
plt.savefig(scatter_path, dpi=200, bbox_inches='tight')
#plt.show()
print(f"Saved: {scatter_path}")



data = data

params = {
    'R0':('R0', 'R\u2080','steelblue'),
    'tau':('tau','\u03C4 (transmission rate)', 'darkorange'),
    'rho':('rho','\u03C1 (initial infected)',  'mediumpurple'),
    'gamma':('gamma','\u03C1 (recovery rate)',  'red'),
    'attack_rate': ('attack_rate', '(Attack Rate)','firebrick'),
    'final_R':('final_R','Final Recovered (R)','seagreen'),
    'near_threshold': ('near_threshold', 'Near Threshold (R\u2080\u22481)', 'goldenrod'),
    'peak_I': ('peak_I','(peak of infected compartment)','green'),
}

fig, axes = plt.subplots(2, 4, figsize=(16, 9))
axes = axes.flatten()

for ax, (col, (key, label, color)) in zip(axes, params.items()):
    values = data[key].dropna()

    if key == 'near_threshold':
        counts = values.value_counts().sort_index()
        ax.bar(counts.index.astype(str), counts.values, color=[color, 'lightgray'][:len(counts)],
               alpha=0.85, edgecolor='white', linewidth=0.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Not near (0)', 'Near (1)'], fontsize=10)
        pct = values.mean() * 100
        ax.set_title(f'{label}\n({pct:.1f}% near threshold)', fontsize=11, fontweight='bold')
    else:
        counts, bin_edges = np.histogram(values, bins=30)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        width = bin_edges[1] - bin_edges[0]
        ax.bar(bin_centers, counts, width=width * 0.85, color=color, alpha=0.85,
               edgecolor='white', linewidth=0.4)
        ax.axvline(values.mean(), color='black', linestyle='--', linewidth=1.5,
                   label=f'Mean = {values.mean():.4f}')
        ax.legend(fontsize=9)
        ax.set_title(f'Distribution of {label}', fontsize=11, fontweight='bold')

    ax.set_xlabel(label, fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

fig.suptitle('MCMC Posterior — Parameter & Outcome Distributions', fontsize=14, fontweight='bold')
plt.tight_layout()

out = PLOTS_DIR / "parameter_bar_plots.png"
plt.savefig(out, dpi=200, bbox_inches='tight')
plt.show()
print(f"Saved: {out}")
