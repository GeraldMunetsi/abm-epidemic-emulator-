"""
fig_threshold.py  —  Section 2.4
─────────────────────────────────
Stochastic SIR realisations just above and just below the epidemic
threshold R0 = 1, demonstrating bimodal outcomes near criticality.

Three R0 values shown:
  R0 = 0.80  (sub-critical)   → all epidemics extinguish
  R0 = 1.05  (just above)     → bimodal: extinction or outbreak
  R0 = 1.50  (super-critical) → all epidemics produce outbreaks

Uses the Gillespie SSA from sir_ode_vs_gillespie.py — copy that
function here so this script is self-contained.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp

np.random.seed(2024)

N      = 10000
GAMMA  = 0.16
I0     = 10
S0     = N - I0
T_END  = 200
N_REAL = 40        # realisations per panel

SCENARIOS = [
    (0.80 * GAMMA, r'$\mathcal{R}_0 = 0.80$  (sub-critical)',  'blue'),
    (1.05 * GAMMA, r'$\mathcal{R}_0 = 1.05$  (just above threshold)', 'red'),
    (1.50 * GAMMA, r'$\mathcal{R}_0 = 1.50$  (super-critical)', 'green'),
]


# ── Gillespie SSA ──────────────────────────────────────────────────────
def gillespie_sir(beta, gamma, N, S0, I0, t_end, seed=None):
    rng = np.random.default_rng(seed)
    S, I, R = S0, I0, N - S0 - I0
    t = 0.0
    ts, Is = [t], [I]
    while t < t_end and I > 0:
        lam_inf = beta * S * I / N
        lam_rec = gamma * I
        lam_tot = lam_inf + lam_rec
        if lam_tot == 0:
            break
        t += rng.exponential(1.0 / lam_tot)
        if t > t_end:
            break
        if rng.random() < lam_inf / lam_tot:
            S -= 1; I += 1
        else:
            I -= 1; R += 1
        ts.append(t); Is.append(I)
    # pad to t_end at final value
    if ts[-1] < t_end:
        ts.append(t_end); Is.append(Is[-1])
    return np.array(ts), np.array(Is, dtype=float) / N


# ── ODE solution ───────────────────────────────────────────────────────
def ode_I(beta, gamma, N, S0, I0, t_end):
    def rhs(t, y):
        S, I, _ = y
        return [-beta*S*I/N, beta*S*I/N - gamma*I, gamma*I]
    t_eval = np.linspace(0, t_end, 800)
    sol = solve_ivp(rhs, [0, t_end], [S0, I0, 0],
                    t_eval=t_eval, method='RK45', rtol=1e-9)
    return sol.t, sol.y[1] / N


# ── Figure ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 4.5))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

for col, (beta, title, colour) in enumerate(SCENARIOS):
    ax  = fig.add_subplot(gs[col])
    R0  = beta / GAMMA

    # Stochastic realisations
    n_extinct = 0
    for seed in range(N_REAL):
        ts, Is = gillespie_sir(beta, GAMMA, N, S0, I0, T_END, seed=seed)
        peak   = Is.max()
        extinct = peak < 5 / N          # trivial epidemic
        n_extinct += int(extinct)
        ax.step(ts, Is, '-',
                color=colour, lw=0.8,
                alpha=0.25 if not extinct else 0.15,
                where='post')

    # ODE
    t_ode, I_ode = ode_I(beta, GAMMA, N, S0, I0, T_END)
    ax.plot(t_ode, I_ode, '--', color='black', lw=2.2,
            label='ODE', zorder=8)

    # Annotations
    ax.set_title(
        rf'$\mathbf{{({chr(97+col)})}}$  {title}',
        fontsize=11, pad=5
    )
    ax.text(0.97, 0.95,
            f'{N_REAL - n_extinct}/{N_REAL} outbreaks',
            transform=ax.transAxes,
            ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3',
                      fc='white', alpha=0.8, ec='grey'))
    ax.set_xlabel('Time (days)', fontsize=11)
    if col == 0:
        ax.set_ylabel(r'Fraction infected $I(t)/N$', fontsize=11)
    ax.set_xlim(0, T_END)
    ax.set_ylim(bottom=-0.003)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.spines[['top', 'right']].set_visible(False)
    if col == 0:
        ax.legend(fontsize=9)

fig.suptitle(
    rf'Stochastic SIR dynamics near the epidemic threshold  '
    rf'($N={N:,}$,  $\gamma={GAMMA}$,  $I_0={I0}$)',
    fontsize=12, fontweight='bold', y=1.02
)
fig.text(
    0.5, -0.04,
    r'Thin coloured lines: individual Gillespie SSA realisations.  '
    r'Black dashed: deterministic ODE.  '
    r'Fraction of realisations producing a major outbreak shown in each panel.',
    ha='center', fontsize=9, style='italic', color='#444'
)

for ext in ('pdf', 'png'):
    plt.savefig(f'/mnt/user-data/outputs/fig_threshold.{ext}',
                dpi=250, bbox_inches='tight')
plt.close()
print("Saved fig_threshold.pdf / .png")
