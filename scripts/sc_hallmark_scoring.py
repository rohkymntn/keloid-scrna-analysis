"""Hallmark MSigDB signature scoring per cell.

Score each fibroblast for canonical Hallmark gene sets relevant to keloid
biology: EMT, TGF-beta signaling, myogenesis (myofibroblast), hypoxia,
glycolysis, inflammatory response, complement, oxphos, hedgehog, Wnt.
Compare across conditions.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import gseapy as gp

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed" / "scRNA"
FIG = ROOT / "figures" / "sc_extended"
FIG.mkdir(parents=True, exist_ok=True)

sns.set_context("talk", font_scale=0.85)
sns.set_style("white")

# Pull hallmark gene sets from gseapy/MSigDB cache
print("Fetching MSigDB Hallmark gene sets via gseapy...")
HALLMARKS_OF_INTEREST = [
    "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
    "HALLMARK_TGF_BETA_SIGNALING",
    "HALLMARK_MYOGENESIS",
    "HALLMARK_HYPOXIA",
    "HALLMARK_GLYCOLYSIS",
    "HALLMARK_INFLAMMATORY_RESPONSE",
    "HALLMARK_COMPLEMENT",
    "HALLMARK_OXIDATIVE_PHOSPHORYLATION",
    "HALLMARK_HEDGEHOG_SIGNALING",
    "HALLMARK_WNT_BETA_CATENIN_SIGNALING",
    "HALLMARK_IL6_JAK_STAT3_SIGNALING",
    "HALLMARK_APOPTOSIS",
]
SHORT = {
    "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION": "EMT",
    "HALLMARK_TGF_BETA_SIGNALING": "TGF-β",
    "HALLMARK_MYOGENESIS": "Myogenesis (MFB)",
    "HALLMARK_HYPOXIA": "Hypoxia",
    "HALLMARK_GLYCOLYSIS": "Glycolysis",
    "HALLMARK_INFLAMMATORY_RESPONSE": "Inflammatory",
    "HALLMARK_COMPLEMENT": "Complement",
    "HALLMARK_OXIDATIVE_PHOSPHORYLATION": "OxPhos",
    "HALLMARK_HEDGEHOG_SIGNALING": "Hedgehog",
    "HALLMARK_WNT_BETA_CATENIN_SIGNALING": "Wnt-βcat",
    "HALLMARK_IL6_JAK_STAT3_SIGNALING": "IL6/JAK/STAT3",
    "HALLMARK_APOPTOSIS": "Apoptosis",
}

hallmark_df = gp.get_library(name="MSigDB_Hallmark_2020", organism="Human")
gene_sets = {h: hallmark_df.get(h.replace("HALLMARK_", "").replace("_", " ").title(), [])
              for h in HALLMARKS_OF_INTEREST}

# Try alternative MSigDB key formats - the gseapy lib uses display names
hallmark_lib_keys = list(hallmark_df.keys())
print(f"Got {len(hallmark_lib_keys)} hallmark sets, examples: {hallmark_lib_keys[:5]}")

# Match heuristically
def match_hallmark(target):
    target_clean = target.replace("HALLMARK_", "").replace("_", " ").lower()
    for k in hallmark_lib_keys:
        if k.lower().replace("-", " ").replace("_", " ") == target_clean:
            return k
    # partial match
    for k in hallmark_lib_keys:
        if target_clean in k.lower() or k.lower() in target_clean:
            return k
    return None

gene_sets = {}
for h in HALLMARKS_OF_INTEREST:
    k = match_hallmark(h)
    if k is not None:
        gene_sets[h] = hallmark_df[k]
        print(f"  {SHORT[h]:20s}: {len(hallmark_df[k])} genes from '{k}'")
    else:
        print(f"  {SHORT[h]:20s}: NOT FOUND")


# Score per dataset
def score_anndata(a, name, cond_col):
    print(f"\nScoring {name}...")
    for h, genes in gene_sets.items():
        present = [g for g in genes if g in a.var_names]
        if len(present) < 5:
            print(f"  Skip {SHORT[h]} (only {len(present)} genes present)")
            continue
        sc.tl.score_genes(a, gene_list=present, score_name=f"score_{SHORT[h]}", random_state=0)


print("\nLoading AnnDatas...")
wound = sc.read_h5ad(PROC / "wound_fibroblasts.h5ad")
deng  = sc.read_h5ad(PROC / "deng_fibroblasts_processed.h5ad")
dire  = sc.read_h5ad(PROC / "direder_fibroblasts_processed.h5ad")

score_anndata(wound, "wound", "Condition")
score_anndata(deng, "deng", "condition")
score_anndata(dire, "direder", "condition")


# Build combined long-form table
def long_scores(a, cond_col, dataset):
    score_cols = [c for c in a.obs.columns if c.startswith("score_")]
    df = a.obs[[cond_col] + score_cols].copy()
    df.columns = ["condition"] + [c.replace("score_", "") for c in score_cols]
    df["dataset"] = dataset
    df["sample"] = a.obs["sample"].values
    return df.melt(id_vars=["dataset", "condition", "sample"], var_name="hallmark", value_name="score")


long_df = pd.concat([
    long_scores(wound, "Condition", "Wound"),
    long_scores(deng,  "condition", "Deng"),
    long_scores(dire,  "condition", "Direder"),
], ignore_index=True)

# Per-sample mean (for clean stats and figure)
per_sample = (long_df.groupby(["dataset", "condition", "sample", "hallmark"])
              ["score"].mean().reset_index())

# ---------- Figure 1: per-condition mean per hallmark, all conditions ----------
print("\nFigure: heatmap of mean hallmark scores per condition")

cond_order = [
    ("Wound", "Skin"), ("Wound", "Wound1"), ("Wound", "Wound7"), ("Wound", "Wound30"),
    ("Direder", "Healthy skin"), ("Direder", "Normal scar"), ("Direder", "Keloid"),
    ("Deng", "Normal scar"), ("Deng", "Keloid"),
]

mean_per_cond = (long_df.groupby(["dataset", "condition", "hallmark"])
                  ["score"].mean().reset_index())
mean_per_cond["xlabel"] = mean_per_cond["dataset"] + "\n" + mean_per_cond["condition"]
xorder = [f"{d}\n{c}" for d, c in cond_order if (mean_per_cond["xlabel"] == f"{d}\n{c}").any()]
hallmark_order = [SHORT[h] for h in HALLMARKS_OF_INTEREST if SHORT[h] in mean_per_cond["hallmark"].unique()]

heatmap = mean_per_cond.pivot_table(index="hallmark", columns="xlabel", values="score")
heatmap = heatmap.reindex(index=hallmark_order, columns=xorder)

fig, ax = plt.subplots(figsize=(13, 7))
sns.heatmap(heatmap, cmap="RdBu_r", center=0, ax=ax, cbar_kws={"label": "mean signature score"},
             linewidths=0.4, annot=True, fmt=".2f", annot_kws={"fontsize": 9})
ax.set_title("Hallmark MSigDB signature scores by condition\n(mean per cell, per condition)",
             fontsize=12)
ax.set_xlabel(""); ax.set_ylabel("")
plt.xticks(rotation=30, ha="right")
plt.yticks(rotation=0)
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(FIG / f"fig_hallmark_heatmap.{ext}", dpi=180, bbox_inches="tight")
plt.close(fig)


# ---------- Figure 2: per-sample boxplots for top fibrotic-relevant hallmarks ----------
print("Figure: per-sample boxplots for key hallmarks")

key_hallmarks = ["EMT", "TGF-β", "Myogenesis (MFB)", "Hypoxia", "Glycolysis",
                  "Inflammatory", "Hedgehog", "Wnt-βcat"]

fig, axes = plt.subplots(2, 4, figsize=(22, 11), sharex=True)
palette = {
    "Wound": "#5DADE2", "Deng": "#C0392B", "Direder": "#E67E22",
}
for ax, h in zip(axes.flat, key_hallmarks):
    sub = per_sample[per_sample["hallmark"] == h].copy()
    sub["xlabel"] = sub["dataset"] + "\n" + sub["condition"]
    sub = sub[sub["xlabel"].isin(xorder)]
    sub["xlabel"] = pd.Categorical(sub["xlabel"], categories=xorder, ordered=True)
    sub = sub.sort_values("xlabel")

    for ds in sub["dataset"].unique():
        sds = sub[sub["dataset"] == ds]
        ax.scatter(sds["xlabel"], sds["score"], s=130, color=palette[ds],
                    edgecolor="black", linewidth=0.7, zorder=3, label=ds)
    means = sub.groupby("xlabel")["score"].mean().reindex(xorder)
    ax.plot(range(len(xorder)), means.values, "k-", linewidth=1, alpha=0.4, zorder=1)
    ax.set_title(h, fontweight="bold", fontsize=12)
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_ylabel("score (per sample mean)")
    ax.tick_params(axis='x', rotation=30)
    for lbl in ax.get_xticklabels(): lbl.set_horizontalalignment("right")
    sns.despine(ax=ax)

# Legend on first axis only
handles, labels = axes[0, 0].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=11,
           bbox_to_anchor=(0.99, 0.99))

fig.suptitle("Hallmark signature scores per biological sample",
             fontsize=14, y=1.00)
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(FIG / f"fig_hallmark_per_sample.{ext}", dpi=180, bbox_inches="tight")
plt.close(fig)


# Save raw data
per_sample.to_csv(FIG / "hallmark_per_sample.csv", index=False)
mean_per_cond.to_csv(FIG / "hallmark_mean_per_condition.csv", index=False)

print(f"\nDone. Outputs in {FIG}")
