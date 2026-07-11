"""
fig_sampling_comparison.py  —  Section 2.6.1
─────────────────────────────────────────────
Single figure comparing three parameter-space sampling strategies:
  (a) Random uniform sampling
  (b) Latin Hypercube Sampling (LHS)
  (c) Sobol quasi-random sequence

All three sample the 2D space (tau, gamma) with n=200 points.
The comparison shows visually how each strategy covers the space.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import qmc

np.random.seed(42)

N_SAMPLES = 200

# Parameter bounds  [tau, gamma]
TAU_LO,   TAU_HI   = 0.0003, 0.02
GAMMA_LO, GAMMA_HI = 0.03,   1.0

def scale(samples):
    """Scale unit-hypercube samples to [tau, gamma] bounds."""
    tau   = TAU_LO   + samples[:, 0] * (TAU_HI   - TAU_LO)
    gamma = GAMMA_LO + samples[:, 1] * (GAMMA_HI - GAMMA_LO)
    return tau, gamma


# ── 1. Random uniform ─────────────────────────────────────────────────
rng = np.random.default_rng(42)
rand_raw = rng.uniform(size=(N_SAMPLES, 2))
tau_rand, gamma_rand = scale(rand_raw)

# ── 2. Latin Hypercube Sampling ───────────────────────────────────────
sampler_lhs = qmc.LatinHypercube(d=2, seed=42)
lhs_raw = sampler_lhs.random(N_SAMPLES)
tau_lhs, gamma_lhs = scale(lhs_raw)

# ── 3. Sobol sequence ─────────────────────────────────────────────────
# n must be a power of 2 for proper Sobol; use 256 and take first 200
sampler_sobol = qmc.Sobol(d=2, scramble=True, seed=42)
sobol_raw = sampler_sobol.random(256)[:N_SAMPLES]
tau_sobol, gamma_sobol = scale(sobol_raw)


# ── Discrepancy (lower = more uniform coverage) ───────────────────────
disc_rand  = qmc.discrepancy(rand_raw)
disc_lhs   = qmc.discrepancy(lhs_raw)
disc_sobol = qmc.discrepancy(sobol_raw)
print(f"Discrepancy: Random={disc_rand:.4f}  "
      f"LHS={disc_lhs:.4f}  Sobol={disc_sobol:.4f}")


# ── Figure ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 4.8))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

configs = [
    (tau_rand,  gamma_rand,  '#2a78d6', disc_rand,  'Random uniform',
     r'$\mathbf{(a)}$  Random uniform sampling'),
    (tau_lhs,   gamma_lhs,   '#e24b4a', disc_lhs,   'LHS',
     r'$\mathbf{(b)}$  Latin Hypercube Sampling (LHS)'),
    (tau_sobol, gamma_sobol, '#1baf7a', disc_sobol, 'Sobol',
     r'$\mathbf{(c)}$  Sobol quasi-random sequence'),
]

for col, (tau, gamma, colour, disc, label, title) in enumerate(configs):
    ax = fig.add_subplot(gs[col])

    ax.scatter(tau, gamma, c=colour, s=12, alpha=0.70, edgecolors='none')

    # Grid lines to reveal coverage
    for x in np.linspace(TAU_LO, TAU_HI, 6):
        ax.axvline(x, color='grey', lw=0.4, alpha=0.4)
    for y in np.linspace(GAMMA_LO, GAMMA_HI, 6):
        ax.axhline(y, color='grey', lw=0.4, alpha=0.4)

    ax.set_title(title + f'\n$n={N_SAMPLES}$', fontsize=11)
    ax.set_xlabel(r'Transmission rate $\tau$', fontsize=11)
    if col == 0:
        ax.set_ylabel(r'Recovery rate $\gamma$', fontsize=11)
    ax.set_xlim(TAU_LO, TAU_HI)
    ax.set_ylim(GAMMA_LO, GAMMA_HI)
    ax.spines[['top','right']].set_visible(False)

    # Discrepancy badge
    ax.text(0.97, 0.04,
            f'Discrepancy = {disc:.4f}',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', fc='white',
                      alpha=0.85, ec='grey'))

    # Marginal rug plots
    ax.plot(tau,  [GAMMA_LO]*len(tau),  '|',
            color=colour, ms=4, alpha=0.3)
    ax.plot([TAU_LO]*len(gamma), gamma, '_',
            color=colour, ms=4, alpha=0.3)

fig.suptitle(
    r'Parameter-space coverage: random vs LHS vs Sobol  '
    r'($\tau \in [0.0003,\,0.02]$,  $\gamma \in [0.03,\,1.0]$)',
    fontsize=12, fontweight='bold', y=1.02)
fig.text(
    0.5, -0.04,
    r'Lower discrepancy indicates more uniform space-filling coverage.  '
    r'LHS guarantees one sample per row and column of a stratified grid.  '
    r'Sobol sequences are deterministic quasi-random designs with '
    r'provably lower discrepancy than random sampling.',
    ha='center', fontsize=9, style='italic', color='#444')

for ext in ('pdf','png'):
    plt.savefig(f'/mnt/user-data/outputs/fig_sampling_comparison.{ext}',
                dpi=250, bbox_inches='tight')
plt.close()
print("Saved fig_sampling_comparison.pdf / .png")
