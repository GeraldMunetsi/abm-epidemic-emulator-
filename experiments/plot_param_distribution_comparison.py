"""
Compare the distributions of the sampled parameters (tau, gamma, rho) across
sampling strategies' raw simulation outputs:
  - experiments/lhs-sampling/data/raw/abm-data.csv
  - experiments/mcmc-sampling/data/raw/abm-data.csv
  - experiments/random-sampling/data/raw/abm-data_sobol.csv
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

OUT_DIR = Path("experiments/comparison_plots")

STRATEGIES = {
    "LHS": Path("experiments/lhs-sampling/data/raw/abm-data.csv"),
    "NTS": Path("experiments/mcmc-sampling/data/raw/abm-data.csv"),
    "Random": Path("experiments/random-sampling/data/raw/abm-data_sobol.csv"),
}

PARAMS = ["tau", "gamma", "rho"]

dfs = {name: pd.read_csv(path) for name, path in STRATEGIES.items()}

# Each parameter is bounded and similarly ranged across strategies, so use
# shared linear bin edges per parameter (row) for a fair visual comparison.
bin_edges = {
    param: np.linspace(
        min(df[param].min() for df in dfs.values()),
        max(df[param].max() for df in dfs.values()),
        41,
    )
    for param in PARAMS
}

fig, axes = plt.subplots(len(PARAMS), len(STRATEGIES), figsize=(13, 10))

for row, param in enumerate(PARAMS):
    for col, (name, df) in enumerate(dfs.items()):
        ax = axes[row, col]
        values = df[param].values
        ax.hist(values, bins=bin_edges[param], color="blue", alpha=0.6,

                edgecolor="white", linewidth=0.4)
        ax.tick_params(axis="x", labelbottom=True)
        ax.grid(True, alpha=0.3)
        if row == 0:
            ax.set_title(f"{name}", fontsize=12, fontweight="bold")
        if col == 0:
            ax.set_ylabel(f"{param}\nCount", fontweight="bold", fontsize=11)
        if row == len(PARAMS) - 1:
            ax.set_xlabel(param, fontweight="bold", fontsize=11)
        ax.text(0.97, 0.95, f"n = {len(values)}\nmean = {values.mean():.4g}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8)

fig.suptitle("Parameter Distribution Comparison Across Sampling Strategies", fontsize=16, fontweight="bold", y=0.95)
fig.tight_layout(rect=(0, 0, 1, 0.95))

out_path = OUT_DIR / "param_distribution_comparison_panel.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path.resolve()}")
