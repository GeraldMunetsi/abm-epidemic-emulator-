"""
fig_generalization_delta.py  —  Section 2.6 (generalization gap)
──────────────────────────────────────────────────────────────────
Generalization gap for each sampling-strategy emulator: the change
in relative MAE_I when tested out-of-domain (OOD -- on data generated
by a *different* sampling strategy) versus tested in-domain
(self-test: train_strategy == test_strategy).

  gap_pct = mean_relative_MAE_I(OOD) - mean_relative_MAE_I(self-test)

Positive gap  -> emulator is worse outside its training domain (expected).
Negative gap  -> emulator generalizes *better* out-of-domain than on its
                 own training domain (flags a strategy whose self-test
                 score may be optimistic relative to general use).

Reads replicate_results_*.csv from ./data and reproduces the same
train/test/augmentation aggregation used in Results_Combined.ipynb
(see the "Rank stability: self-test vs OOD" cell).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

HERE     = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

STRATEGY_ORDER  = ["LHS", "MCMC", "UNIFORM_RANDOM"]
STRATEGY_LABELS = {"LHS": "LHS", "MCMC": "NTS", "UNIFORM_RANDOM": "Uniform-Random"}


# ── Load data ────────────────────────────────────────────────────────
csv_files = sorted(DATA_DIR.glob("replicate_results_*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No replicate_results_*.csv files found in {DATA_DIR}")

dfs = []
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    df["source_file"] = csv_file.name
    dfs.append(df)
master_df = pd.concat(dfs, ignore_index=True)


# ── Self-test vs OOD means ──────────────────────────────────────────
selftest_mask = master_df["train_strategy"] == master_df["test_strategy"]

def strategy_means(mask):
    return (master_df[mask]
            .groupby(["train_strategy", "augmentation"])["relative_MAE_I"]
            .mean()
            .reset_index())

self_means = strategy_means(selftest_mask).rename(columns={"relative_MAE_I": "relative_MAE_I_self"})
ood_means  = strategy_means(~selftest_mask).rename(columns={"relative_MAE_I": "relative_MAE_I_ood"})

gap = self_means.merge(ood_means, on=["train_strategy", "augmentation"])
gap["gap_pct"] = gap["relative_MAE_I_ood"] - gap["relative_MAE_I_self"]

print(gap[["train_strategy", "augmentation", "relative_MAE_I_self",
           "relative_MAE_I_ood", "gap_pct"]].to_string(index=False))


# ── Figure ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)

x      = np.arange(len(STRATEGY_ORDER))
width  = 0.35
colors = {0: "#8fb8e0", 1: "#2a78d6"}

for i, aug in enumerate([0, 1]):
    vals = [gap.loc[(gap["train_strategy"] == s) & (gap["augmentation"] == aug),
                     "gap_pct"].values[0]
            for s in STRATEGY_ORDER]
    xpos = x + (i - 0.5) * width
    ax.bar(xpos, vals, width, color=colors[aug], edgecolor="white",
           label=f"Augmentation = {aug}")
    for xi, v in zip(xpos, vals):
        ax.text(xi, v, f"{v:+.2f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=9)

ax.axhline(0, color="black", lw=1)
ax.set_xticks(x)
ax.set_xticklabels([STRATEGY_LABELS[s] for s in STRATEGY_ORDER])
ax.set_ylabel(r"Generalization gap: $\Delta$ relative MAE$_I$ (OOD $-$ self-test), pp",
              fontsize=11)
ax.set_title("Generalization Gap by Sampling Strategy", fontsize=14, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=9)

fig.text(
    0.5, -0.04,
    r"Positive bars: emulator performs worse out-of-domain than on its own "
    r"training domain (expected). Negative bars (MCMC/NTS) generalize better "
    r"OOD than on self-test.",
    ha="center", fontsize=9, style="italic", color="#444")

for ext in ("pdf", "png"):
    fig.savefig(HERE / f"fig_generalization_delta.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved fig_generalization_delta.pdf / .png to {HERE}")
