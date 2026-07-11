"""
Combine the R0-distribution plots produced by each sampling strategy's
augmentation script into a single panel figure for comparison:
  - experiments/lhs-sampling/scripts/step2_data_augmentation.py
  - experiments/mcmc-sampling/scripts/step2A_augmented.py
  - experiments/random-sampling/scripts/step2A_augmented.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

OUT_DIR = Path("experiments/comparison_plots")


STRATEGIES = {
    "LHS": Path("experiments/lhs-sampling/data/augmented/abm-data_augmented.csv"),
    "NTS": Path("experiments/mcmc-sampling/data/augmented/abm-data_split_augmented.csv"),
    "Random": Path("experiments/random-sampling/data/augmented/abm-data_split_augmented.csv"),
}

# R0 has a long right tail (up to ~34 for LHS/Random), while the bulk of the
# mass sits below ~5. A linear y-axis with few bins hides the tail entirely,
# so use shared bin edges across panels plus a log-scaled y-axis to make the
# sparse high-R0 bars visible next to the dense low-R0 peak.
dfs = {name: pd.read_csv(path) for name, path in STRATEGIES.items()}
r0_max = max(df["R0"].max() for df in dfs.values())
bin_edges = np.linspace(0, r0_max, 41)

fig, axes = plt.subplots(3, 1, figsize=(8, 11), sharex=True)

for ax, (name, df) in zip(axes, dfs.items()):
    r0 = df["R0"].values
    n_sub = (r0 < 1).sum()
    n_sup = (r0 >= 1).sum()
    n_band = ((r0 >= 0.1) & (r0 <= 2)).sum()

    ax.hist(r0, bins=bin_edges, color="blue", alpha=0.6, edgecolor="white",
            linewidth=0.4, label="R₀ distribution (count)")
    ax.axvline(1.0, color="crimson", linewidth=2.0, linestyle="--",
               label="R₀ = 1 (epidemic threshold)")
    

    ax.set_yscale("log")
    ax.set_xlabel("R₀", fontweight="bold", fontsize=12)
    ax.tick_params(axis="x", labelbottom=True)
    ax.set_title(
        f"{name} Augmented Training Set\n"
        f"Sub-critical: {n_sub} ({n_sub/len(r0)*100:.1f}%)   "
        f"Super-critical: {n_sup} ({n_sup/len(r0)*100:.1f}%)\n"
        f"NT : {n_band} (R₀ : 0.1-2)",
        fontsize=11
    )
    ax.grid(True, alpha=0.3)

for ax in axes:
    ax.set_ylabel("Count (log scale)", fontweight="bold", fontsize=12)
axes[0].legend(fontsize=9, framealpha=0.85)
fig.suptitle("R₀ Distribution Comparison Across Sampling Strategies", fontsize=16, fontweight="bold", y=0.95)
fig.tight_layout(rect=(0, 0, 1, 0.95))

out_path = OUT_DIR / "r0_distribution_comparison_panel.png"
fig.savefig(out_path, dpi=600, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path.resolve()}")
