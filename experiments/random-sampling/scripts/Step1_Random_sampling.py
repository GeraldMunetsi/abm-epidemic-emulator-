
import numpy as np
import networkx as nx
import EoN
import pickle
from pathlib import Path
from tqdm import tqdm
from numpy.random import default_rng
import csv
import pandas as pd
import matplotlib.pyplot as plt


# GLOBAL SETTINGS

DATA_DIR = Path("experiments/random-sampling/data/raw")
N = 100000
m = 10

n_timepoints = 80
tmax = 80
n_replicates = 2
n_samples = 1000

PARAM_NAMES = ['tau','gamma','rho']

PARAM_RANGES = {
    'tau':(0.0005,0.024),
    'gamma':(0.01,0.5),
    'rho':(0.001,0.01)
}

seed = 4849

# NETWORK STATISTICS

_NETWORK_STATS_CACHE = {
    "k_avg": 10.0,
    "k2_avg": 272.6,
    "ratio": 34.0,
    "k_std": 9.49,
    "k_max": 734
}

# RANDOM SAMPLING


def random_sampling(n_samples,param_ranges=PARAM_RANGES,seed=None):

    rng = default_rng(seed)

    samples = np.zeros((n_samples,len(PARAM_NAMES)))

    for i,name in enumerate(PARAM_NAMES):

        low,high = param_ranges[name]

        samples[:,i] = rng.uniform(low,high,n_samples)

    return samples

# NETWORK GENERATION
import networkx as nx

_NETWORK_STATS_CACHE = {
    'k_avg': 10,
    'k2_avg': 340,
    'ratio': 34,
    'k_std': 9.49,
    'k_max': 30
}

def generate_network(N=N, m=m, seed=42):

    print(f"\nBuilding BA network (N={N:,}, m={m})")

    G = nx.barabasi_albert_graph(N, m, seed=seed)

    print(f"{G.number_of_nodes():,} nodes")
    print(f"{G.number_of_edges():,} edges")

    stats = _NETWORK_STATS_CACHE

    print("\nUsing cached network statistics")
    print(f"<k>       = {stats['k_avg']:.2f}")
    print(f"<k²>      = {stats['k2_avg']:.2f}")
    print(f"<k²>/<k>  = {stats['ratio']:.2f}")
    print(f"k_std     = {stats['k_std']:.2f}")
    print(f"k_max     = {stats['k_max']}")

    return G, stats


G, net_stats = generate_network()
ratio = net_stats["ratio"]

# SIR SIMULATION


def run_sir_replicates(G,tau,gamma,rho,
                       n_replicates=n_replicates,
                       tmax=tmax,
                       n_timepoints=n_timepoints):

    t_fixed = np.linspace(0,tmax,n_timepoints)

    S_runs=[]
    I_runs=[]
    R_runs=[]

    try:

        for _ in range(n_replicates):

            t,S,I,R = EoN.fast_SIR(G,tau,gamma,rho=rho,tmax=tmax)

            S_runs.append(np.interp(t_fixed,t,S))
            I_runs.append(np.interp(t_fixed,t,I))
            R_runs.append(np.interp(t_fixed,t,R))

        return {
            't':t_fixed,
            'S':np.mean(S_runs,axis=0),
            'I':np.mean(I_runs,axis=0),
            'R':np.mean(R_runs,axis=0),
            'S_std':np.std(S_runs,axis=0),
            'I_std':np.std(I_runs,axis=0),
            'R_std':np.std(R_runs,axis=0)
        }

    except Exception as e:

        print("Simulation failed:",e)

        zeros=np.zeros(n_timepoints)

        return {
            't':t_fixed,
            'S':zeros,
            'I':zeros,
            'R':zeros,
            'S_std':zeros,
            'I_std':zeros,
            'R_std':zeros
        }


# RUN BATCH

def run_batch_with_replicates(G, params_array, n_replicates=n_replicates):
    results = []

    for row in tqdm(params_array, desc="Simulating"):
        tau, gamma, rho = row
        for rep in range(n_replicates):
            output = run_sir_replicates(G, tau, gamma, rho, n_replicates=1)
            results.append({
                'params': {
                    'tau': float(tau),
                    'gamma': float(gamma),
                    'rho': float(rho),
                    'replicate': rep
                },
                'output': output
            })
    return results


# DATASET GENERATION
def generate_dataset():

    print("\nGenerating dataset")

    # build network + get cached stats
    G, net = generate_network()

    params_array = random_sampling(n_samples, seed=seed)

    sims = run_batch_with_replicates(G, params_array)

    tau_arr = np.array([s['params']['tau'] for s in sims])
    gamma_arr = np.array([s['params']['gamma'] for s in sims])

    R0_arr = (tau_arr / gamma_arr) * net['ratio']

    print("\nR0 distribution")
    print("min :", R0_arr.min())
    print("max :", R0_arr.max())
    print("mean:", R0_arr.mean())

    dataset = {

        'simulations': sims,

        'network': {
            'N': N,
            'm': m,
            'k_avg': net['k_avg'],
            'k2_avg': net['k2_avg'],
            'ratio': net['ratio']
        },

        'metadata': {
            'n_samples': len(sims),
            'n_replicates': n_replicates,
            'param_names': PARAM_NAMES,
            'param_ranges': PARAM_RANGES,
            'tmax': tmax,
            'n_timepoints': n_timepoints
        }
    }

    return dataset



# SAVE DATASET


def save_dataset(dataset,filepath):

    filepath=Path(filepath)

    with open(filepath,'wb') as f:

        pickle.dump(dataset,f,pickle.HIGHEST_PROTOCOL)

    size_mb=filepath.stat().st_size/(1024**2)

    print("\nDataset saved")
    print(filepath)
    print(f"{size_mb:.2f} MB")


# SAVE CSV SUMMARY

def save_csv(dataset,filepath):

    sims = dataset['simulations']

    ratio = dataset['network']['ratio']
    N = dataset['network']['N']

    fields=[
        'sim_id',
        'tau','gamma','rho',
        'R0','peak_I','peak_time',
        'final_R','attack_rate','near_threshold'
    ]

    with open(filepath,'w',newline='') as f:

        writer=csv.DictWriter(f,fieldnames=fields)

        writer.writeheader()

        for sim_id,sim in enumerate(sims):

            tau=sim['params']['tau']
            gamma=sim['params']['gamma']
            rho=sim['params']['rho']

            R0=(tau/gamma)*ratio

            I=sim['output']['I']
            R=sim['output']['R']
            t=sim['output']['t']

            peak_I=float(I.max())

            peak_time=float(t[I.argmax()])

            writer.writerow({

                'sim_id':sim_id,
                'tau':tau,
                'gamma':gamma,
                'rho':rho,
                'R0':round(R0,4),
                'peak_I':peak_I,
                'peak_time':peak_time,
                'final_R':float(R[-1]),
                'attack_rate':float(R[-1]/N),
                'near_threshold':int(abs(R0-1)<0.2)
            })

    print("CSV saved:",filepath)

# MAIN

if __name__=="__main__":

    dataset = generate_dataset()

    save_dataset(dataset,DATA_DIR/"epidemic_data_age_adaptive_sobol.pkl")

    save_csv(dataset,DATA_DIR/"epidemic_data_age_adaptive_sobol.csv")



random_sampling_data=pd.read_csv(DATA_DIR / 'epidemic_data_age_adaptive_sobol.csv')

print(random_sampling_data.columns)
print(len(random_sampling_data))
print(random_sampling_data.isnull().sum()) # no missing values
random_sampling_data.describe(include='all')

print(random_sampling_data.head())


total_samples=len(random_sampling_data)
print(total_samples)
greater=random_sampling_data[random_sampling_data['R0']>1.2]
pc1=(len(greater)/total_samples)*100
print(f"greater than 1.2: {len(greater),pc1}")
between = random_sampling_data[(random_sampling_data['R0'] >= 0.7) & (random_sampling_data['R0'] <= 1.2)]
pc2=(len(between)/total_samples)*100
print(f"between 0.7 and 1.2: {len(between),pc2}")

#print(pc2)

less_than=random_sampling_data[random_sampling_data['R0']<0.8]
pc3=(len(less_than)/total_samples)*100
print(f"less than 0.8: {len(less_than),pc3}")


#Plotting

slope=1/34
plt.figure(figsize=(10,10))
plt.scatter(random_sampling_data['gamma'], random_sampling_data['tau'], alpha=0.1)

x_vals = np.linspace(random_sampling_data['gamma'].min(), random_sampling_data['gamma'].max(), 100)
y_vals = slope * x_vals

plt.plot(x_vals, y_vals, color='red', linestyle='--', label=f'slope={slope}')

plt.xlabel('gamma')
plt.ylabel('tau')
plt.title('Scatter plot of tau vs gamma')
plt.legend()
plt.grid(True)
plt.show()
    




