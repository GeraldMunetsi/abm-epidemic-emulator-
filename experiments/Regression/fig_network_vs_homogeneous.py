"""
fig_network_vs_homogeneous.py  —  Section 2.3
───────────────────────────────────────────────
Compares SIR epidemic dynamics on a Barabasi-Albert (BA) contact
network against homogeneous-mixing SIR at the SAME R0 = 2.5.

Key point: the two models use the SAME effective R0 but the
network model achieves it with a much SMALLER tau because the
hub structure amplifies transmission (R0_net = tau/gamma * k2/k).
This shows how identical parameter sets behave differently
across model structures.

Two approaches shown side by side:
  (a) Homogeneous mixing  — tau set so R0_homo = 2.5
  (b) BA network          — tau set so R0_net  = 2.5 (much smaller tau)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
import EoN

np.random.seed(7)

N_NET  = 5_000
M_BA   = 5
GAMMA  = 0.16
N_REAL = 20
T_END  = 100
I0     = 5
R0_TARGET = 2.5

# Build fixed BA network
G       = nx.barabasi_albert_graph(N_NET, M_BA, seed=42)
degrees = np.array([d for _, d in G.degree()])
mean_k  = degrees.mean()
mean_k2 = (degrees**2).mean()
ratio   = mean_k2 / mean_k

# tau for EACH model to achieve R0 = 2.5
TAU_HOMO = R0_TARGET * GAMMA               # homogeneous mixing
TAU_NET  = R0_TARGET * GAMMA / ratio       # network (smaller tau)

print(f"BA: N={N_NET}, m={M_BA}, ratio={ratio:.2f}")
print(f"Homogeneous: tau={TAU_HOMO:.4f}  R0={TAU_HOMO/GAMMA:.2f}")
print(f"Network:     tau={TAU_NET:.4f}   R0_net={TAU_NET/GAMMA*ratio:.2f}")


def gillespie_homo(N, tau, gamma, I0, t_end, seed=None):
    rng = np.random.default_rng(seed)
    S, I = N - I0, I0
    t, ts, Is = 0.0, [0.0], [I0]
    while t < t_end and I > 0:
        lam = tau * S * I / N + gamma * I
        if lam == 0: break
        t += rng.exponential(1.0 / lam)
        if t > t_end: break
        if rng.random() < (tau * S * I / N) / lam:
            S -= 1; I += 1
        else:
            I -= 1
        ts.append(t); Is.append(I)
    ts.append(t_end); Is.append(Is[-1])
    return np.array(ts), np.array(Is, dtype=float) / N


def network_sir_eon(G, tau, gamma, I0_count, t_end, seed=None):
    if seed is not None:
        np.random.seed(seed)
    init = np.random.choice(list(G.nodes()), I0_count, replace=False)
    t, S, I, R = EoN.fast_SIR(G, tau, gamma,
                                initial_infecteds=init,
                                tmax=t_end, return_full_data=False)
    return t, I / G.number_of_nodes()


fig = plt.figure(figsize=(14, 5.0))
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38)

# Panel (a) — Homogeneous mixing
ax1 = fig.add_subplot(gs[0])
print(f"\nHomogeneous ({N_REAL} realisations) ...")
for seed in range(N_REAL):
    ts, Is = gillespie_homo(N_NET, TAU_HOMO, GAMMA, I0, T_END, seed)
    ax1.step(ts, Is, '-', color='#2a78d6', lw=0.9, alpha=0.30,
             where='post')
ax1.set_title(
    r'$\mathbf{(a)}$  Homogeneous mixing  (Gillespie SSA)'
    f'\n$N={N_NET:,}$  '
    r'$\tau=$'+f'{TAU_HOMO:.3f}  '
    r'$\gamma=$'+f'{GAMMA}  '
    r'$\mathcal{{R}}_0=$'+f'{R0_TARGET:.1f}',
    fontsize=11)
ax1.set_xlabel('Time (days)', fontsize=11)
ax1.set_ylabel(r'Fraction infected $I(t)/N$', fontsize=11)
ax1.set_xlim(0, T_END); ax1.set_ylim(bottom=-0.003)
ax1.grid(True, alpha=0.25, linestyle='--')
ax1.spines[['top','right']].set_visible(False)

# Panel (b) — BA network
ax2 = fig.add_subplot(gs[1])
print(f"BA network ({N_REAL} realisations) ...")
for seed in range(N_REAL):
    ts, Is = network_sir_eon(G, TAU_NET, GAMMA, I0, T_END, seed)
    ax2.step(ts, Is, '-', color='#e24b4a', lw=0.9, alpha=0.30,
             where='post')
ax2.set_title(
    r'$\mathbf{(b)}$  Barabási–Albert network  (EoN fast\_SIR)'
    f'\n$N={N_NET:,}$  $m={M_BA}$  '
    r'$\tau=$'+f'{TAU_NET:.4f}  '
    r'$\mathcal{{R}}_0^{{\rm net}}=$'+f'{R0_TARGET:.1f}  '
    r'($\langle k^2\rangle/\langle k\rangle=$'+f'{ratio:.1f})',
    fontsize=11)
ax2.set_xlabel('Time (days)', fontsize=11)
ax2.set_ylabel(r'Fraction infected $I(t)/N$', fontsize=11)
ax2.set_xlim(0, T_END); ax2.set_ylim(bottom=-0.003)
ax2.grid(True, alpha=0.25, linestyle='--')
ax2.spines[['top','right']].set_visible(False)
ax2.text(0.97, 0.97,
         r'Same $\mathcal{R}_0$ but 27$\times$ smaller $\tau$' '\n'
         r'— hub structure amplifies spread',
         transform=ax2.transAxes, ha='right', va='top', fontsize=9,
         bbox=dict(boxstyle='round,pad=0.3', fc='white',
                   alpha=0.85, ec='grey'))

fig.suptitle(
    r'Homogeneous mixing vs network-structured SIR  '
    r'(matched $\mathcal{R}_0 = 2.5$)',
    fontsize=12, fontweight='bold', y=1.02)
fig.text(
    0.5, -0.04,
    r'Both panels achieve $\mathcal{R}_0=2.5$ but via different $\tau$.  '
    r'On the BA network $\mathcal{R}_0^{\rm net}=(\tau/\gamma)'
    r'\langle k^2\rangle/\langle k\rangle$; '
    r'degree heterogeneity amplifies transmission requiring '
    r'a 27$\times$ smaller $\tau$ to reach the same effective threshold.',
    ha='center', fontsize=9, style='italic', color='#444')

for ext in ('pdf','png'):
    plt.savefig(f'/mnt/user-data/outputs/fig_network_vs_homogeneous.{ext}',
                dpi=250, bbox_inches='tight')
plt.close()
print("Saved fig_network_vs_homogeneous.pdf / .png")
