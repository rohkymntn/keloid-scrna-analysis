"""decoupler PROGENy + DoRothEA pathway/TF activity scoring per cell.

PROGENy = pathway activity inference from a footprint of consensus
gene-target-response signatures.
DoRothEA = TF activity inference from regulons (TF + targets, weighted by
confidence A-E).

Goal: empirically score per-cell activity of pathways and TFs central to
the team's circuit -- TGF-beta, JAK-STAT, MAPK, Hippo/YAP, NF-kB, Wnt --
and TFs SMAD3, TEAD1-4, RELA, JUN, FOXO etc. across keloid vs healing
vs healthy.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import decoupler as dc

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed" / "scRNA"
FIG = ROOT / "figures" / "sc_extended"
FIG.mkdir(parents=True, exist_ok=True)

sns.set_context("talk", font_scale=0.85)
sns.set_style("white")

# -------- Get prior knowledge networks --------
print("Fetching PROGENy + DoRothEA networks...")
try:
    progeny_net = dc.op.progeny(organism='human', top=500)
    print(f"  PROGENy: {progeny_net.shape}")
except Exception as e:
    print(f"  PROGENy fetch failed: {e}")
    progeny_net = None

try:
    dorothea_net = dc.op.collectri(organism='human')
    print(f"  CollecTRI (DoRothEA successor): {dorothea_net.shape}")
except Exception as e:
    print(f"  CollecTRI fetch failed: {e}")
    dorothea_net = None


def score_one(a, name, cond_col):
    print(f"\n=== {name} ===")
    if progeny_net is not None:
        print("  PROGENy ULM...")
        dc.mt.ulm(data=a, net=progeny_net, verbose=False)
        # Pull out scores stored in obsm
        if 'score_ulm' in a.obsm:
            scores = a.obsm['score_ulm']
            a.obsm['progeny'] = scores
        elif 'ulm_estimate' in a.obsm:
            a.obsm['progeny'] = a.obsm['ulm_estimate']
    if dorothea_net is not None:
        print("  CollecTRI ULM (TF activity)...")
        # Filter to a manageable set of TFs of interest
        tfs_of_interest = [
            "TEAD1", "TEAD2", "TEAD3", "TEAD4",
            "SMAD2", "SMAD3", "SMAD4",
            "RELA", "NFKB1",
            "JUN", "FOS", "ATF3",
            "FOXO1", "FOXO3",
            "STAT1", "STAT3",
            "MYC", "TP53",
            "SRF",
            "TWIST1", "SNAI1", "SNAI2", "ZEB1", "ZEB2",
            "RUNX1", "RUNX2",
            "NRF2", "NFE2L2",
            "HIF1A", "EPAS1",
            "CREB1", "ATF4",
        ]
        net_filt = dorothea_net[dorothea_net["source"].isin(tfs_of_interest)].copy()
        print(f"    Using {net_filt['source'].nunique()} TFs of interest")
        dc.mt.ulm(data=a, net=net_filt, verbose=False)
        if 'score_ulm' in a.obsm:
            a.obsm['dorothea'] = a.obsm['score_ulm']
        elif 'ulm_estimate' in a.obsm:
            a.obsm['dorothea'] = a.obsm['ulm_estimate']

    return a


print("Loading...")
wound = sc.read_h5ad(PROC / "wound_fibroblasts.h5ad")
deng  = sc.read_h5ad(PROC / "deng_fibroblasts_processed.h5ad")
dire  = sc.read_h5ad(PROC / "direder_fibroblasts_processed.h5ad")

wound = score_one(wound, "wound", "Condition")
deng = score_one(deng, "deng", "condition")
dire = score_one(dire, "direder", "condition")


# -------- Build per-condition mean activity tables --------
def per_cond_means(a, cond_col, dataset_label, kind):
    if kind not in a.obsm:
        return None
    df = pd.DataFrame(a.obsm[kind].copy() if not isinstance(a.obsm[kind], pd.DataFrame) else a.obsm[kind])
    df["condition"] = a.obs[cond_col].values
    df["sample"] = a.obs["sample"].astype(str).values
    df["dataset"] = dataset_label
    long = df.melt(id_vars=["dataset", "condition", "sample"], var_name="source", value_name="activity")
    return long


progeny_long = pd.concat([
    per_cond_means(wound, "Condition", "Wound", "progeny"),
    per_cond_means(deng, "condition", "Deng", "progeny"),
    per_cond_means(dire, "condition", "Direder", "progeny"),
], ignore_index=True) if progeny_net is not None else None

dorothea_long = pd.concat([
    per_cond_means(wound, "Condition", "Wound", "dorothea"),
    per_cond_means(deng, "condition", "Deng", "dorothea"),
    per_cond_means(dire, "condition", "Direder", "dorothea"),
], ignore_index=True) if dorothea_net is not None else None


# -------- Figures --------
cond_order = [
    ("Wound", "Skin"), ("Wound", "Wound1"), ("Wound", "Wound7"), ("Wound", "Wound30"),
    ("Direder", "Healthy skin"), ("Direder", "Normal scar"), ("Direder", "Keloid"),
    ("Deng", "Normal scar"), ("Deng", "Keloid"),
]


def heatmap_condition(long, kind):
    if long is None: return
    print(f"Figure: {kind} heatmap")
    long["xlabel"] = long["dataset"] + "\n" + long["condition"]
    xorder = [f"{d}\n{c}" for d, c in cond_order if (long["xlabel"] == f"{d}\n{c}").any()]
    cell_means = (long.groupby(["xlabel", "source"])["activity"].mean().reset_index())
    pivot = cell_means.pivot(index="source", columns="xlabel", values="activity")
    pivot = pivot.reindex(columns=xorder)
    # Sort rows by max activity in keloid columns
    keloid_cols = [c for c in xorder if "Keloid" in c]
    if keloid_cols:
        pivot = pivot.reindex(pivot[keloid_cols].mean(axis=1).sort_values(ascending=False).index)
    fig, ax = plt.subplots(figsize=(13, max(7, 0.32 * len(pivot))))
    sns.heatmap(pivot, cmap="RdBu_r", center=0, ax=ax,
                 cbar_kws={"label": "ULM activity score"}, linewidths=0.4,
                 annot=True, fmt=".2f", annot_kws={"fontsize": 7})
    ax.set_title(f"{kind.upper()} activity by condition (per-cell mean)", fontsize=12)
    ax.set_xlabel(""); ax.set_ylabel(kind)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"fig_{kind}_heatmap.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


heatmap_condition(progeny_long, "progeny")
heatmap_condition(dorothea_long, "dorothea")


# -------- Per-sample scatter for key sources --------
def per_sample_panel(long, sources_of_interest, kind, title):
    if long is None: return
    print(f"Figure: per-sample {kind} scatter for selected sources")
    long_filt = long[long["source"].isin(sources_of_interest)].copy()
    long_filt["xlabel"] = long_filt["dataset"] + "\n" + long_filt["condition"]
    xorder = [f"{d}\n{c}" for d, c in cond_order if (long_filt["xlabel"] == f"{d}\n{c}").any()]
    per_sample = (long_filt.groupby(["dataset", "condition", "sample", "source"])
                            ["activity"].mean().reset_index())
    per_sample["xlabel"] = per_sample["dataset"] + "\n" + per_sample["condition"]

    n = len(sources_of_interest)
    cols = min(4, n); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 5 * rows))
    if rows * cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    palette = {"Wound": "#5DADE2", "Deng": "#C0392B", "Direder": "#E67E22"}

    for ax, src in zip(axes, sources_of_interest):
        sub = per_sample[per_sample["source"] == src]
        if len(sub) == 0: ax.set_visible(False); continue
        sub = sub[sub["xlabel"].isin(xorder)].copy()
        sub["xlabel"] = pd.Categorical(sub["xlabel"], categories=xorder, ordered=True)
        sub = sub.sort_values("xlabel")
        for ds in sub["dataset"].unique():
            sds = sub[sub["dataset"] == ds]
            ax.scatter(sds["xlabel"], sds["activity"], s=130, color=palette[ds],
                        edgecolor="black", linewidth=0.7, zorder=3, label=ds)
        ax.axhline(0, color="grey", linestyle="--", alpha=0.5)
        ax.set_title(src, fontweight="bold")
        ax.set_ylabel("ULM activity")
        ax.tick_params(axis='x', rotation=30)
        for lbl in ax.get_xticklabels(): lbl.set_horizontalalignment("right")
        sns.despine(ax=ax)
    for ax in axes[len(sources_of_interest):]: ax.set_visible(False)
    fig.suptitle(title, fontsize=13, y=1.00)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"fig_{kind}_per_sample.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


# Key PROGENy pathways
progeny_keys = ["TGFb", "Hippo", "JAK-STAT", "MAPK", "NFkB", "WNT", "PI3K", "Hypoxia"]
per_sample_panel(progeny_long, progeny_keys, "progeny",
                 "PROGENy pathway activity per biological sample")

# Key TFs
tf_keys = ["TEAD1", "TEAD4", "SMAD3", "RELA", "JUN", "STAT3", "TWIST1", "ZEB1", "RUNX2"]
per_sample_panel(dorothea_long, tf_keys, "dorothea",
                 "Transcription factor activity per biological sample (CollecTRI)")


if progeny_long is not None:
    progeny_long.to_csv(FIG / "progeny_long.csv", index=False)
if dorothea_long is not None:
    dorothea_long.to_csv(FIG / "dorothea_long.csv", index=False)

print(f"\nDone. Outputs in {FIG}")
