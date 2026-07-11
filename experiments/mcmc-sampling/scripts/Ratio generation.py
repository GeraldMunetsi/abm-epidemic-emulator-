import networkx as nx
import numpy as np

N=100000              # network size
m=10                      # Barabasi–Albert attachment parameter
 # replicates of parameter sets 

PARAM_RANGES = {
   'tau':(0.0006,0.17),
    'gamma':(0.03,1),
    'rho':(0.001,0.01)
}



PARAM_NAMES = ['tau', 'gamma', 'rho']


results_k_avg = []
results_k2_avg = []
results_ratio = []

for i in range(100):
    
    G = nx.barabasi_albert_graph(N, m)
    
    degrees = np.array([d for _, d in G.degree()])
    
    k_avg = degrees.mean()
    k2_avg = (degrees**2).mean()
    ratio = (k2_avg-k_avg)/ k_avg   # R0 = (tau/gamma) * (<k²> - <k>) / <k> that is our BA network ratio for R0 computation 
    
    results_k_avg.append(k_avg)
    results_k2_avg.append(k2_avg)
    results_ratio.append(ratio)

# converting to arrays
results_k_avg = np.array(results_k_avg)
results_k2_avg = np.array(results_k2_avg)
results_ratio = np.array(results_ratio)

# means
mean_k_avg = results_k_avg.mean()
mean_k2_avg = results_k2_avg.mean()
mean_ratio = results_ratio.mean()

# standard errors 
se_k_avg = results_k_avg.std(ddof=1)/np.sqrt(len(results_k_avg))
se_k2_avg = results_k2_avg.std(ddof=1)/np.sqrt(len(results_k2_avg))
se_ratio = results_ratio.std(ddof=1)/np.sqrt(len(results_ratio))

# 95% CI  (mean ± 1.96 * standard error)
ci_k_avg = (mean_k_avg - 1.96*se_k_avg, mean_k_avg + 1.96*se_k_avg)
ci_k2_avg = (mean_k2_avg - 1.96*se_k2_avg, mean_k2_avg + 1.96*se_k2_avg)
ci_ratio = (mean_ratio - 1.96*se_ratio, mean_ratio + 1.96*se_ratio)

print("Mean <k>:", mean_k_avg, "CI:", ci_k_avg)
print("Mean <k^2>:", mean_k2_avg, "CI:", ci_k2_avg)
print("Mean <k^2>/<k>:", mean_ratio, "CI:", ci_ratio)