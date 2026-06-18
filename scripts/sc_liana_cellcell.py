"""LIANA cell-cell communication on direder_all (full keloid scRNA atlas).

Uses LIANA's consensus method (rank_aggregate) which combines CellChat,
NATMI, SingleCellSignalR, etc. across the multiple LR resources. Compares
keloid vs healthy skin to identify ligand-receptor pairs that:
  1. Are most active in keloid cells
  2. Most differentially active between keloid and healthy

For the iGEM project, we focus on:
  - Senders/receivers involving Fibroblast (the disease cell)
  - LR pairs involving POSTN, ADAM12, WISP1, TGFB family, TWEAK/Fn14, etc.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import liana as li

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed" / "scRNA"
FIG = ROOT / "figures" / "sc_extended"
FIG.mkdir(parents=True, exist_ok=True)

sns.set_context("talk", font_scale=0.85)
sns.set_style("white")


print("Loading direder_all...")
a = sc.read_h5ad(PROC / "direder_all.h5ad")
print(f"  shape: {a.shape}, cell types: {a.obs.cell_type.value_counts().to_dict()}")

# Filter out very rare cell types for stable LR inference
keep_types = ["Fibroblast", "Endothelial", "Keratinocyte", "Pericyte_SMC", "Myeloid", "T_cell"]
a = a[a.obs.cell_type.isin(keep_types)].copy()
print(f"After filter: {a.shape}")

# Subsample within each cell type per condition for tractability
np.random.seed(0)
keep_idx = []
for (ct, cond), sub in a.obs.groupby(["cell_type", "condition"], observed=True):
    n = min(len(sub), 1500)
    if n == 0: continue
    keep_idx.extend(np.random.choice(sub.index, size=n, replace=False))

a = a[keep_idx].copy()
print(f"After subsample: {a.shape}")
print(f"  cell type x condition counts:")
print(a.obs.groupby(["cell_type", "condition"], observed=True).size().unstack(fill_value=0))

# Run LIANA per condition separately, then compare
def run_one_cond(adata, cond_label):
    print(f"\nRunning LIANA on {cond_label}...")
    sub = adata[adata.obs.condition == cond_label].copy()
    if sub.n_obs < 100:
        print("  too few cells, skipping")
        return None
    li.mt.rank_aggregate(
        sub,
        groupby="cell_type",
        resource_name="consensus",
        verbose=False,
        use_raw=False,
        n_perms=100,
        seed=0,
    )
    df = sub.uns.get("liana_res")
    if df is None:
        print(f"  no liana_res returned for {cond_label}")
        return None
    df = df.copy()
    df["condition"] = cond_label
    return df


keloid_res = run_one_cond(a, "Keloid")
healthy_res = run_one_cond(a, "Healthy skin")
scar_res = run_one_cond(a, "Normal scar")

results = pd.concat([df for df in [keloid_res, healthy_res, scar_res] if df is not None],
                     ignore_index=True)
results.to_csv(FIG / "liana_results.csv", index=False)
print(f"\nLIANA combined results: {results.shape}")
print(f"  columns: {list(results.columns)[:20]}")

# Get top LR pairs in keloid (using magnitude_rank — lower = stronger)
print("\nTop LR pairs in keloid (by magnitude_rank):")
top_keloid = (keloid_res.sort_values("magnitude_rank")
              [["source", "target", "ligand_complex", "receptor_complex",
                 "magnitude_rank", "specificity_rank"]]
              .head(40))
print(top_keloid.to_string())
top_keloid.to_csv(FIG / "liana_top40_keloid.csv", index=False)

# ---------- Figure 1: Top fibroblast-as-source LR pairs in keloid ----------
print("\nFigure: top fibroblast-source LR pairs in keloid")
fb_source = keloid_res[keloid_res["source"] == "Fibroblast"].copy()
fb_source = fb_source.sort_values("magnitude_rank").head(30)
fb_source["lr_label"] = (fb_source["ligand_complex"].astype(str) + " → " +
                          fb_source["receptor_complex"].astype(str) + "\n(" +
                          fb_source["target"].astype(str) + ")")

fig, ax = plt.subplots(figsize=(11, 11))
ax.barh(fb_source["lr_label"][::-1], -np.log10(fb_source["magnitude_rank"][::-1] + 1e-3),
        color="#C0392B", edgecolor="black", linewidth=0.4)
ax.set_xlabel("-log10(magnitude_rank)  →  stronger interaction")
ax.set_title("Top 30 Fibroblast-secreted ligand-receptor pairs in keloid\n"
             "(LIANA consensus rank across CellChat, NATMI, SingleCellSignalR)",
             fontsize=11)
sns.despine(ax=ax)
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(FIG / f"fig_liana_fibroblast_source_keloid.{ext}", dpi=180, bbox_inches="tight")
plt.close(fig)


# ---------- Figure 2: Top fibroblast-as-target LR pairs in keloid ----------
print("Figure: top fibroblast-target LR pairs in keloid (incoming signals)")
fb_target = keloid_res[keloid_res["target"] == "Fibroblast"].copy()
fb_target = fb_target.sort_values("magnitude_rank").head(30)
fb_target["lr_label"] = (fb_target["source"].astype(str) + ": " +
                          fb_target["ligand_complex"].astype(str) + " → " +
                          fb_target["receptor_complex"].astype(str))

fig, ax = plt.subplots(figsize=(11, 11))
ax.barh(fb_target["lr_label"][::-1], -np.log10(fb_target["magnitude_rank"][::-1] + 1e-3),
        color="#3498DB", edgecolor="black", linewidth=0.4)
ax.set_xlabel("-log10(magnitude_rank)  →  stronger incoming signal")
ax.set_title("Top 30 incoming ligand-receptor pairs ON keloid Fibroblasts\n"
             "(who is signaling TO the disease cell?)",
             fontsize=11)
sns.despine(ax=ax)
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(FIG / f"fig_liana_fibroblast_target_keloid.{ext}", dpi=180, bbox_inches="tight")
plt.close(fig)


# ---------- Figure 3: Differential keloid vs healthy ----------
print("Figure: keloid vs healthy differential LR pairs")
if healthy_res is not None:
    # Merge by (source, target, ligand, receptor)
    key_cols = ["source", "target", "ligand_complex", "receptor_complex"]
    k = keloid_res[key_cols + ["magnitude_rank", "lr_means"]].copy()
    k = k.rename(columns={"magnitude_rank": "rank_keloid", "lr_means": "lrm_keloid"})
    h = healthy_res[key_cols + ["magnitude_rank", "lr_means"]].copy()
    h = h.rename(columns={"magnitude_rank": "rank_healthy", "lr_means": "lrm_healthy"})
    merged = k.merge(h, on=key_cols, how="outer")
    merged["delta_rank"] = merged["rank_healthy"].fillna(1.0) - merged["rank_keloid"].fillna(1.0)
    # Restrict to fibroblast involvement
    merged_fb = merged[(merged["source"] == "Fibroblast") | (merged["target"] == "Fibroblast")].copy()
    # Top 25 keloid-up
    keloid_up = merged_fb.nlargest(25, "delta_rank").copy()
    keloid_up["lr_label"] = (keloid_up["source"].astype(str) + " → " +
                              keloid_up["target"].astype(str) + ": " +
                              keloid_up["ligand_complex"].astype(str) + " ↔ " +
                              keloid_up["receptor_complex"].astype(str))

    fig, ax = plt.subplots(figsize=(13, 11))
    ax.barh(keloid_up["lr_label"][::-1], keloid_up["delta_rank"][::-1],
             color="#C0392B", edgecolor="black", linewidth=0.4)
    ax.set_xlabel("ΔRank (keloid - healthy);  positive = keloid-enriched")
    ax.set_title("Top 25 keloid-enriched fibroblast LR pairs\n"
                  "(compared to healthy skin)", fontsize=11)
    sns.despine(ax=ax)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"fig_liana_keloid_vs_healthy_diff.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    keloid_up.to_csv(FIG / "liana_keloid_vs_healthy_top25.csv", index=False)


# ---------- Figure 4: Where do specific genes show up? ----------
print("Figure: project-specific genes in LR network")
project_genes = ["POSTN", "WISP1", "CCN4", "ADAM12", "TNFRSF12A", "TNFSF12",
                  "TGFB1", "TGFB3", "ITGA11", "ITGB1", "WNT5A", "FN1"]
mask = (
    keloid_res["ligand_complex"].astype(str).isin(project_genes) |
    keloid_res["receptor_complex"].astype(str).isin(project_genes)
)
project_lr = keloid_res[mask].sort_values("magnitude_rank").copy()
project_lr.to_csv(FIG / "liana_project_genes_keloid.csv", index=False)
print(f"  {len(project_lr)} LR pairs in keloid involve project genes")
print(project_lr[["source", "target", "ligand_complex", "receptor_complex",
                   "magnitude_rank"]].head(20).to_string())

# Heatmap of top project-gene LR pairs across conditions
if len(project_lr) > 0:
    top_proj = project_lr.head(20)[["source", "target", "ligand_complex", "receptor_complex"]]
    project_lr_full = results[
        results.set_index(["source", "target", "ligand_complex", "receptor_complex"]).index.isin(
            top_proj.set_index(["source", "target", "ligand_complex", "receptor_complex"]).index
        )
    ].copy()

    project_lr_full["lr_label"] = (project_lr_full["source"].astype(str) + " → " +
                                     project_lr_full["target"].astype(str) + ": " +
                                     project_lr_full["ligand_complex"].astype(str) + " ↔ " +
                                     project_lr_full["receptor_complex"].astype(str))

    pivot = project_lr_full.pivot_table(index="lr_label", columns="condition",
                                          values="magnitude_rank", aggfunc="first")
    pivot = -np.log10(pivot.fillna(1.0) + 1e-3)

    fig, ax = plt.subplots(figsize=(8, max(5, 0.3 * len(pivot))))
    sns.heatmap(pivot, cmap="Reds", ax=ax, cbar_kws={"label": "-log10(rank)  → stronger"},
                 linewidths=0.4, annot=True, fmt=".2f", annot_kws={"fontsize": 9})
    ax.set_title("Project-gene LR pairs across conditions", fontsize=11)
    ax.set_xlabel(""); ax.set_ylabel("")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"fig_liana_project_genes_heatmap.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


print(f"\nDone. Outputs in {FIG}")
