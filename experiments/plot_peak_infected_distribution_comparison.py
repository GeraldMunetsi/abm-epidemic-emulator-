"""
Compare the distribution of peak infected counts (peak_I) across sampling
strategies' raw simulation outputs:
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


dfs = {name: pd.read_csv(path) for name, path in STRATEGIES.items()}
peak_max = max(df["peak_I"].max() for df in dfs.values())
bin_edges = np.linspace(0, peak_max, 41)

fig, axes = plt.subplots(3, 1, figsize=(8, 11), sharex=True)

for ax, (name, df) in zip(axes, dfs.items()):
    peak_i = df["peak_I"].values
    ax.hist(peak_i, bins=bin_edges, color="blue", alpha=0.6, edgecolor="white",
            linewidth=0.4, label="peak_I distribution (count)")

    ax.set_yscale("log")
    ax.set_xlabel("Peak Infected (I)", fontweight="bold", fontsize=12)
    ax.tick_params(axis="x", labelbottom=True)
    ax.set_title(
        f"{name} Raw Simulation Set\n"
        f"n = {len(peak_i)}   median peak_I = {np.median(peak_i):.0f}   "
        f"max peak_I = {peak_i.max():.0f}",
        fontsize=11
    )
    ax.grid(True, alpha=0.3)

for ax in axes:
    ax.set_ylabel("Count (log scale)", fontweight="bold", fontsize=12)
axes[0].legend(fontsize=9, framealpha=0.85)
fig.suptitle("Peak Infected (I) Distribution Comparison Across Sampling Strategies", fontsize=16, fontweight="bold", y=0.95)
fig.tight_layout(rect=(0, 0, 1, 0.95))

out_path = OUT_DIR / "peak_infected_distribution_comparison_panel.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path.resolve()}")
