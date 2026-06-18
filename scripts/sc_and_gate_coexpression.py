"""AND-gate per-cell co-expression: how often do POSTN and ADAM12 fire
together at the per-cell level in keloid mesenchymal fibroblasts?

This is the empirical hit-rate of the team's SynNotch AND gate on the
disease cell. The answer matters for the iGEM poster.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed" / "scRNA"
FIG = ROOT / "figures" / "sc_extended"
FIG.mkdir(parents=True, exist_ok=True)

sns.set_context("talk", font_scale=0.85)
sns.set_style("white")

GENES = ["POSTN", "ADAM12"]
THR = 0.0  # log-norm expression > 0 = "expressing"


def get_expr(a, gene):
    if gene not in a.var_names:
        return None
    x = a[:, gene].X
    return np.asarray(x.todense()).ravel() if hasattr(x, "todense") else np.asarray(x).ravel()


def per_cell_logic(a, cond_col):
    p = get_expr(a, "POSTN")
    d = get_expr(a, "ADAM12")
    df = pd.DataFrame({
        "condition": a.obs[cond_col].values,
        "sample": a.obs["sample"].values,
        "POSTN_pos": p > THR,
        "ADAM12_pos": d > THR,
    })
    df["both_pos"] = df["POSTN_pos"] & df["ADAM12_pos"]
    df["postn_only"] = df["POSTN_pos"] & ~df["ADAM12_pos"]
    df["adam12_only"] = ~df["POSTN_pos"] & df["ADAM12_pos"]
    df["neither"] = ~df["POSTN_pos"] & ~df["ADAM12_pos"]
    return df


# Load datasets
print("Loading...")
wound = sc.read_h5ad(PROC / "wound_fibroblasts.h5ad")
deng  = sc.read_h5ad(PROC / "deng_fibroblasts_processed.h5ad")
dire  = sc.read_h5ad(PROC / "direder_fibroblasts_processed.h5ad")

# Per-cell logic, per dataset
wf = per_cell_logic(wound, "Condition"); wf["dataset"] = "Wound"
df = per_cell_logic(deng,  "condition"); df["dataset"] = "Deng"
rf = per_cell_logic(dire,  "condition"); rf["dataset"] = "Direder"

# Restrict Deng analysis to the keloid-enriched cluster (mfb_cluster=True if present)
if "mfb_cluster" in deng.obs.columns:
    mfb_mask = deng.obs["mfb_cluster"].astype(str).isin(["True", "1", "True"])
    df_mfb = df.iloc[mfb_mask.values]
    df_mfb["dataset"] = "Deng (MFB cluster only)"
    print(f"Deng MFB cluster: {len(df_mfb)} cells")

allf = pd.concat([wf, df, rf], ignore_index=True)

# ---------- Per-condition summary ----------
def summarize(allf, label_col="condition"):
    out = []
    for (ds, cond), sub in allf.groupby(["dataset", label_col]):
        n = len(sub)
        out.append({
            "dataset": ds,
            "condition": cond,
            "n_cells": n,
            "pct_POSTN": sub["POSTN_pos"].mean() * 100,
            "pct_ADAM12": sub["ADAM12_pos"].mean() * 100,
            "pct_BOTH (gate ON)": sub["both_pos"].mean() * 100,
            "pct_postn_only": sub["postn_only"].mean() * 100,
            "pct_adam12_only": sub["adam12_only"].mean() * 100,
            "pct_neither": sub["neither"].mean() * 100,
        })
    return pd.DataFrame(out)

summary = summarize(allf)
summary.to_csv(FIG / "and_gate_per_cell_summary.csv", index=False)
print("\n=== AND-gate per-cell summary ===")
print(summary.round(1).to_string(index=False))

# ---------- Figure 1: stacked bars of gate state per condition ----------
print("\nFigure: stacked bars per condition")

cond_order = [
    ("Wound", "Skin"), ("Wound", "Wound1"), ("Wound", "Wound7"), ("Wound", "Wound30"),
    ("Direder", "Healthy skin"), ("Direder", "Normal scar"), ("Direder", "Keloid"),
    ("Deng", "Normal scar"), ("Deng", "Keloid"),
]
summary["xlabel"] = summary.apply(lambda r: f"{r['dataset']}\n{r['condition']}", axis=1)
key2lab = {f"{d}\n{c}": (d, c) for d, c in cond_order}
xorder = [f"{d}\n{c}" for d, c in cond_order if f"{d}\n{c}" in summary["xlabel"].values]
summary = summary.set_index("xlabel").reindex(xorder).reset_index()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))

# Stacked bars of cell states
labels = ["pct_BOTH (gate ON)", "pct_postn_only", "pct_adam12_only", "pct_neither"]
colors = ["#C0392B", "#3498DB", "#F39C12", "#BDC3C7"]
display_labels = ["BOTH (AND-gate ON)", "POSTN only", "ADAM12 only", "Neither"]

bottoms = np.zeros(len(summary))
for lab, col, dl in zip(labels, colors, display_labels):
    vals = summary[lab].values
    ax1.bar(summary["xlabel"], vals, bottom=bottoms, color=col, label=dl, edgecolor="black", linewidth=0.5)
    bottoms += vals

ax1.set_ylabel("% of cells")
ax1.set_title("Per-cell co-expression state of AND-gate inputs\n(gray = gate silent; red = gate active)",
              fontsize=12)
ax1.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=10)
ax1.tick_params(axis='x', rotation=30)
for lbl in ax1.get_xticklabels(): lbl.set_horizontalalignment("right")
sns.despine(ax=ax1)

# Just the AND-gate-on percentage
ax2.bar(summary["xlabel"], summary["pct_BOTH (gate ON)"], color="#C0392B", edgecolor="black")
for i, (xl, v) in enumerate(zip(summary["xlabel"], summary["pct_BOTH (gate ON)"])):
    ax2.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=11, fontweight="bold")
ax2.set_ylabel("% of cells with AND gate ON\n(POSTN+ AND ADAM12+)")
ax2.set_title("AND-gate empirical hit rate per condition", fontsize=12)
ax2.tick_params(axis='x', rotation=30)
ax2.set_ylim(0, max(summary["pct_BOTH (gate ON)"]) * 1.15)
for lbl in ax2.get_xticklabels(): lbl.set_horizontalalignment("right")
sns.despine(ax=ax2)

fig.suptitle("AND-gate (POSTN ∩ ADAM12) per-cell hit rate across all conditions",
             fontsize=14, y=1.02)
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(FIG / f"fig_andgate_per_cell.{ext}", dpi=180, bbox_inches="tight")
plt.close(fig)


# ---------- Figure 2: 4-quadrant scatter (POSTN vs ADAM12 per cell) ----------
print("Figure: 4-quadrant scatter")

# For visual clarity: plot a few key conditions
plot_conds = [
    ("Wound", "Skin"),
    ("Wound", "Wound30"),
    ("Direder", "Healthy skin"),
    ("Direder", "Keloid"),
    ("Deng", "Normal scar"),
    ("Deng", "Keloid"),
]

fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True, sharey=True)
axes = axes.flatten()

for ax, (ds, cond) in zip(axes, plot_conds):
    if ds == "Wound":
        a = wound; col = "Condition"
    elif ds == "Deng":
        a = deng; col = "condition"
    else:
        a = dire; col = "condition"
    mask = (a.obs[col] == cond).values
    p = get_expr(a, "POSTN")[mask]
    d = get_expr(a, "ADAM12")[mask]
    n = len(p)
    pct_both = ((p > THR) & (d > THR)).mean() * 100

    # Hex 2D density
    hb = ax.hexbin(p, d, gridsize=40, cmap="magma", mincnt=1, bins="log")
    ax.axvline(THR, color="white", linestyle="--", alpha=0.6, linewidth=1)
    ax.axhline(THR, color="white", linestyle="--", alpha=0.6, linewidth=1)
    ax.set_title(f"{ds}: {cond}\nn={n}, gate-ON = {pct_both:.0f}%", fontsize=11)
    ax.set_xlabel("POSTN log-norm")
    ax.set_ylabel("ADAM12 log-norm")
    sns.despine(ax=ax)

fig.suptitle("Per-cell POSTN × ADAM12 scatter (top-right quadrant = AND gate ON)",
             fontsize=13, y=1.00)
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(FIG / f"fig_andgate_scatter.{ext}", dpi=180, bbox_inches="tight")
plt.close(fig)


# ---------- Figure 3: per-sample AND-gate hit rate (variability check) ----------
print("Figure: per-sample variability")

per_sample = (
    allf.groupby(["dataset", "condition", "sample"])["both_pos"].mean().reset_index()
)
per_sample.rename(columns={"both_pos": "frac_gate_on"}, inplace=True)
per_sample["pct"] = per_sample["frac_gate_on"] * 100
per_sample["xlabel"] = per_sample.apply(lambda r: f"{r['dataset']}\n{r['condition']}", axis=1)
per_sample = per_sample[per_sample["xlabel"].isin(xorder)].copy()

fig, ax = plt.subplots(figsize=(14, 6))
palette = {"Wound": "#5DADE2", "Deng": "#C0392B", "Direder": "#E67E22"}
for ds in per_sample["dataset"].unique():
    sub = per_sample[per_sample["dataset"] == ds]
    ax.scatter(sub["xlabel"], sub["pct"], s=180, color=palette[ds],
                edgecolor="black", linewidth=0.8, label=ds, zorder=3)
ax.set_ylabel("% AND-gate ON cells (per sample)")
ax.set_title("Per-sample variability of AND-gate hit rate\n(each dot = one biological sample)",
              fontsize=12)
ax.tick_params(axis='x', rotation=30)
for lbl in ax.get_xticklabels(): lbl.set_horizontalalignment("right")
ax.legend(loc="upper left", fontsize=10)
sns.despine(ax=ax)
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(FIG / f"fig_andgate_per_sample.{ext}", dpi=180, bbox_inches="tight")
plt.close(fig)

print(f"\nDone. Outputs in {FIG}")
