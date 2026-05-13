"""Visualize the classified benchmark dataset (seaborn).

Produces a multi-panel PNG covering:
  1. Required-API frequency
  2. Actual-complexity breakdown
  3. Difficulty x actual_complexity (alignment check)
  4. Framing inflation by difficulty
  5. APIs-per-task histogram
  6. API co-occurrence heatmap
"""
import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a benchmark CSV that has been processed by classify_benchmark_apis.py."
    )
    parser.add_argument("--input", default="benchmark_dataset_classified.csv")
    parser.add_argument("--output", default="benchmark_classification_viz.png")
    parser.add_argument("--top-cooccurrence", type=int, default=14)
    return parser.parse_args()


args = parse_args()
CSV = Path(args.input)
OUT = Path(args.output)

sns.set_theme(style="whitegrid", context="talk", palette="viridis")

df = pl.read_csv(CSV)
total = len(df)


def split_apis(s: str | None) -> list[str]:
    if not s:
        return []
    return [a.strip() for a in s.split(",")]


api_lists = [split_apis(s) for s in df["required_apis"]]
api_counter = Counter()
for lst in api_lists:
    api_counter.update(lst)
apis_sorted = [a for a, _ in api_counter.most_common()]

fig = plt.figure(figsize=(22, 16), constrained_layout=True)
gs = fig.add_gridspec(3, 3)

# Panel 1: API frequency horizontal bar
ax1 = fig.add_subplot(gs[0:2, 0])
api_freq_df = pd.DataFrame(
    {"api": apis_sorted, "count": [api_counter[a] for a in apis_sorted]}
)
sns.barplot(
    data=api_freq_df, y="api", x="count", hue="api", legend=False,
    palette="viridis_r", ax=ax1,
)
ax1.set_title(f"Required API frequency (multi-label, n={total})",
              fontweight="bold", fontsize=14)
ax1.set_xlabel(f"# tasks needing API")
ax1.set_ylabel("")
for i, c in enumerate(api_freq_df["count"]):
    ax1.text(c + 1, i, f"{c}  ({c/total*100:.0f}%)",
             va="center", fontsize=10)
ax1.set_xlim(0, api_freq_df["count"].max() * 1.18)

# Panel 2: actual_complexity breakdown
ax2 = fig.add_subplot(gs[0, 1])
comp_order = ["trivial_array_ops", "routine_numpy", "nontrivial_algorithm"]
comp_pdf = (
    df["actual_complexity"]
    .value_counts()
    .to_pandas()
    .set_index("actual_complexity")
    .reindex(comp_order, fill_value=0)
    .reset_index()
)
sns.barplot(
    data=comp_pdf, x="actual_complexity", y="count",
    hue="actual_complexity", legend=False,
    palette={"trivial_array_ops": "#6BAED6",
             "routine_numpy": "#FD8D3C",
             "nontrivial_algorithm": "#74C476"},
    order=comp_order, ax=ax2,
)
ax2.set_title("Actual complexity\n(skeptical judge)",
              fontweight="bold", fontsize=13)
ax2.set_xlabel("")
ax2.set_ylabel("# tasks")
for i, v in enumerate(comp_pdf["count"]):
    ax2.text(i, v + 1, f"{v}\n({v/total*100:.1f}%)",
             ha="center", fontsize=10)
ax2.tick_params(axis="x", labelrotation=15)
ax2.set_ylim(0, comp_pdf["count"].max() * 1.20)

# Panel 3: difficulty x actual_complexity stacked
ax3 = fig.add_subplot(gs[0, 2])
diffs = ["Easy", "Medium", "Hard"]
matrix_pdf = (
    df.group_by(["difficulty", "actual_complexity"]).len()
    .to_pandas()
    .pivot(index="difficulty", columns="actual_complexity", values="len")
    .reindex(index=diffs, columns=comp_order)
    .fillna(0)
    .astype(int)
)
matrix_pdf.plot(
    kind="bar", stacked=True,
    color=["#6BAED6", "#FD8D3C", "#74C476"],
    ax=ax3, width=0.7, edgecolor="white",
)
ax3.set_title("Difficulty label vs actual complexity\n(misalignment check)",
              fontweight="bold", fontsize=13)
ax3.set_xlabel("")
ax3.set_ylabel("# tasks")
ax3.tick_params(axis="x", labelrotation=0)
ax3.legend(fontsize=9, loc="upper left", framealpha=0.9)
# overlay counts
for i, (_, row) in enumerate(matrix_pdf.iterrows()):
    cum = 0
    for v in row:
        if v > 0:
            ax3.text(i, cum + v / 2, str(v),
                     ha="center", va="center",
                     fontsize=10, color="white", fontweight="bold")
        cum += v

# Panel 4: framing inflation by difficulty
ax4 = fig.add_subplot(gs[1, 1])
inf_pdf = (
    df.group_by(["difficulty", "framing_inflation"]).len()
    .to_pandas()
    .pivot(index="difficulty", columns="framing_inflation", values="len")
    .reindex(index=diffs)
    .fillna(0)
    .astype(int)
)
inf_pdf = inf_pdf.reindex(columns=[True, False])
inf_pdf.plot(
    kind="bar", stacked=True,
    color=["#E6550D", "#9ECAE1"],
    ax=ax4, width=0.7, edgecolor="white",
)
ax4.set_title("Framing inflation by difficulty",
              fontweight="bold", fontsize=13)
ax4.set_xlabel("")
ax4.set_ylabel("# tasks")
ax4.tick_params(axis="x", labelrotation=0)
ax4.legend(["inflated", "honest"], fontsize=10)
for i, (_, row) in enumerate(inf_pdf.iterrows()):
    cum = 0
    for v in row:
        if v > 0:
            ax4.text(i, cum + v / 2, str(v),
                     ha="center", va="center",
                     fontsize=11, color="white", fontweight="bold")
        cum += v

# Panel 5: APIs-per-task histogram
ax5 = fig.add_subplot(gs[1, 2])
api_counts_per_task = [len(lst) for lst in api_lists if lst]
sns.histplot(
    api_counts_per_task, bins=range(min(api_counts_per_task),
                                    max(api_counts_per_task) + 2),
    color="#807DBA", edgecolor="white", ax=ax5,
)
mean_val = float(np.mean(api_counts_per_task))
ax5.axvline(mean_val, ls="--", color="black", lw=1.5)
ax5.text(mean_val + 0.1, ax5.get_ylim()[1] * 0.9,
         f"mean={mean_val:.2f}", fontsize=11)
ax5.set_xlabel("# distinct APIs per task")
ax5.set_ylabel("# tasks")
ax5.set_title("API count per task", fontweight="bold", fontsize=13)

# Panel 6: API co-occurrence heatmap (top-N)
top_n = args.top_cooccurrence
top_apis = apis_sorted[:top_n]
co = np.zeros((top_n, top_n), dtype=int)
api_to_idx = {a: i for i, a in enumerate(top_apis)}
for lst in api_lists:
    indices = [api_to_idx[a] for a in lst if a in api_to_idx]
    for i in indices:
        for j in indices:
            co[i, j] += 1
co_df = pd.DataFrame(co, index=top_apis, columns=top_apis)
ax6 = fig.add_subplot(gs[2, :])
sns.heatmap(
    co_df, annot=True, fmt="d",
    cmap="YlGnBu", linewidths=0.5, linecolor="white",
    cbar_kws={"label": "# tasks", "shrink": 0.6},
    annot_kws={"fontsize": 9},
    ax=ax6,
)
ax6.set_title(
    f"API co-occurrence (top-{top_n}; diagonal = marginal frequency)",
    fontweight="bold", fontsize=13,
)
ax6.tick_params(axis="x", labelrotation=35)
ax6.tick_params(axis="y", labelrotation=0)
plt.setp(ax6.get_xticklabels(), ha="right")

fig.suptitle(
    f"Benchmark dataset classification — {total} tasks (numrs2, 2026-04-29)",
    fontsize=17, fontweight="bold",
)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=130, bbox_inches="tight")
print(f"Saved: {OUT}")
