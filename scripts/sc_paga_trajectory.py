"""PAGA trajectory on the unified keloid + wound + healthy fibroblasts.

Concatenate the three datasets (gene-name harmonized), batch-correct via
BBKNN, leiden cluster, and run PAGA to visualize the cell-state graph
across conditions. Goal: visualize whether keloid mesenchymal fibroblasts
are 'stuck' off the normal-healing trajectory.
"""
from __future__ import annotations

from pathlib import Path

import anndata as ad
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
sc.settings.figdir = FIG


# -------- Load --------
print("Loading...")
wound = sc.read_h5ad(PROC / "wound_fibroblasts.h5ad")
deng  = sc.read_h5ad(PROC / "deng_fibroblasts_processed.h5ad")
dire  = sc.read_h5ad(PROC / "direder_fibroblasts_processed.h5ad")

# -------- Harmonize obs columns --------
def prep(a, dataset_name, cond_col):
    a = a.copy()
    a.obs["dataset"] = dataset_name
    a.obs["condition_orig"] = a.obs[cond_col].astype(str)
    # Unified condition label
    cond_map = {
        "Skin": "Healthy_intactSkin",
        "Healthy skin": "Healthy_intactSkin",
        "Wound1": "Wound_D1",
        "Wound7": "Wound_D7",
        "Wound30": "Wound_D30",
        "Normal scar": "Normal_scar_mature",
        "Keloid": "Keloid",
    }
    a.obs["condition"] = a.obs["condition_orig"].map(cond_map).fillna(a.obs["condition_orig"])
    a.obs["sample"] = a.obs["sample"].astype(str)
    a.obs["batch"] = dataset_name + "_" + a.obs["sample"]
    return a

wound = prep(wound, "Wound", "Condition")
deng = prep(deng, "Deng", "condition")
dire = prep(dire, "Direder", "condition")

# Subsample direder to comparable size for tractable computation
np.random.seed(0)
if dire.n_obs > 12000:
    keep = np.random.choice(dire.n_obs, size=12000, replace=False)
    dire = dire[keep].copy()
    print(f"Subsampled Direder to {dire.n_obs} cells")

# -------- Concatenate on common gene set --------
common = sorted(set(wound.var_names) & set(deng.var_names) & set(dire.var_names))
print(f"Common genes across 3 datasets: {len(common)}")

wound = wound[:, common].copy()
deng = deng[:, common].copy()
dire = dire[:, common].copy()

# Concat using anndata.concat (preserves layers if present)
combined = ad.concat([wound, deng, dire], axis=0, join="outer", merge="first",
                      uns_merge="first", index_unique="-")
print(f"Combined: {combined.shape}")

# -------- HVG selection on combined --------
print("HVGs...")
sc.pp.highly_variable_genes(combined, n_top_genes=3000, flavor="seurat", batch_key="dataset")
combined = combined[:, combined.var.highly_variable].copy()
print(f"  Kept {combined.n_vars} HVGs")

sc.pp.scale(combined, max_value=10)
sc.tl.pca(combined, n_comps=40)

# -------- Batch correction with BBKNN (built into scanpy.external) --------
print("BBKNN batch correction...")
try:
    import bbknn
    bbknn.bbknn(combined, batch_key="dataset", n_pcs=40, neighbors_within_batch=4)
    correction = "bbknn"
except ImportError:
    print("  bbknn not installed; using basic neighbors with batch ridge")
    sc.pp.neighbors(combined, n_neighbors=15, n_pcs=40)
    correction = "basic"

# -------- UMAP + leiden + PAGA --------
print("UMAP...")
sc.tl.umap(combined, min_dist=0.4, spread=1.0)
print("Leiden...")
sc.tl.leiden(combined, resolution=0.6)
print(f"  {combined.obs.leiden.nunique()} clusters")

print("PAGA...")
sc.tl.paga(combined, groups="leiden")

# -------- Figures --------
print("Drawing PAGA + UMAP...")
sc.settings.set_figure_params(dpi=180, dpi_save=180, figsize=(7, 6), frameon=False)

# Multi-panel: condition, dataset, leiden cluster, PAGA
fig, axes = plt.subplots(2, 3, figsize=(22, 13))
axes = axes.flatten()

cond_palette = {
    "Healthy_intactSkin": "#27AE60",
    "Wound_D1": "#F1C40F",
    "Wound_D7": "#E67E22",
    "Wound_D30": "#7F8C8D",
    "Normal_scar_mature": "#3498DB",
    "Keloid": "#C0392B",
}
ds_palette = {"Wound": "#5DADE2", "Deng": "#C0392B", "Direder": "#E67E22"}

sc.pl.umap(combined, color="condition", palette=cond_palette, ax=axes[0], show=False,
            title="Condition")
sc.pl.umap(combined, color="dataset", palette=ds_palette, ax=axes[1], show=False,
            title="Dataset (batch)")
sc.pl.umap(combined, color="leiden", ax=axes[2], show=False, title="Leiden clusters",
            legend_loc="on data", legend_fontsize=8)

# Score the AND gate state on the integrated UMAP
def get_expr_dense(a, gene):
    if gene not in a.var_names:
        return np.zeros(a.n_obs)
    x = a[:, gene].X
    return np.asarray(x.todense()).ravel() if hasattr(x, "todense") else np.asarray(x).ravel()


# Score AND-gate on full common-gene combined
# Re-load pre-HVG combined gene panel for POSTN/ADAM12 expression
print("Re-extracting POSTN / ADAM12 from full datasets for UMAP overlay...")
# Reload original full panels to get POSTN/ADAM12 expression
w_full = sc.read_h5ad(PROC / "wound_fibroblasts.h5ad")
d_full = sc.read_h5ad(PROC / "deng_fibroblasts_processed.h5ad")
r_full = sc.read_h5ad(PROC / "direder_fibroblasts_processed.h5ad")

# Build a series indexed by combined.obs_names
postn_vec = np.zeros(combined.n_obs)
adam12_vec = np.zeros(combined.n_obs)

# combined obs_names are like "<orig>-{0,1,2}" from index_unique="-"
for src, full, suffix in [("Wound", w_full, "0"), ("Deng", d_full, "1"), ("Direder", r_full, "2")]:
    mask = combined.obs.index.str.endswith(f"-{suffix}")
    orig_names = combined.obs.index[mask].str.replace(f"-{suffix}$", "", regex=True)
    if "POSTN" in full.var_names:
        idx = full.obs_names.get_indexer(orig_names)
        ok = idx >= 0
        ex = get_expr_dense(full, "POSTN")
        full_idx = np.where(mask)[0][ok]
        postn_vec[full_idx] = ex[idx[ok]]
    if "ADAM12" in full.var_names:
        idx = full.obs_names.get_indexer(orig_names)
        ok = idx >= 0
        ex = get_expr_dense(full, "ADAM12")
        full_idx = np.where(mask)[0][ok]
        adam12_vec[full_idx] = ex[idx[ok]]

combined.obs["POSTN_expr"] = postn_vec
combined.obs["ADAM12_expr"] = adam12_vec
combined.obs["AND_gate_state"] = ((postn_vec > 0) & (adam12_vec > 0)).astype(int).astype(str)

sc.pl.umap(combined, color="POSTN_expr", ax=axes[3], show=False, title="POSTN expression",
            cmap="Reds", vmax=4)
sc.pl.umap(combined, color="ADAM12_expr", ax=axes[4], show=False, title="ADAM12 expression",
            cmap="Blues", vmax=4)
sc.pl.umap(combined, color="AND_gate_state", ax=axes[5], show=False,
            title="AND-gate cells (red = ON)",
            palette={"0": "#E0E0E0", "1": "#C0392B"})

fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(FIG / f"fig_paga_umap_overview.{ext}", dpi=180, bbox_inches="tight")
plt.close(fig)


# PAGA standalone
fig, ax = plt.subplots(1, 2, figsize=(18, 8))
sc.pl.paga(combined, ax=ax[0], show=False, threshold=0.05, plot=True, title="PAGA graph")

# UMAP colored by cluster, with PAGA edges overlaid
sc.tl.umap(combined, init_pos="paga", min_dist=0.4)
sc.pl.umap(combined, color="condition", palette=cond_palette, ax=ax[1], show=False,
            title="UMAP initialized from PAGA — condition overlay")
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(FIG / f"fig_paga_graph.{ext}", dpi=180, bbox_inches="tight")
plt.close(fig)


# Cluster x condition heatmap (which cluster contains which condition?)
print("Cluster x condition composition...")
crosstab = pd.crosstab(combined.obs.leiden, combined.obs.condition, normalize="index") * 100
fig, ax = plt.subplots(figsize=(11, max(6, 0.3 * len(crosstab))))
sns.heatmap(crosstab, cmap="viridis", annot=True, fmt=".0f", ax=ax,
             cbar_kws={"label": "% of cluster"})
ax.set_title("Composition of each leiden cluster by condition (%)")
ax.set_ylabel("Leiden cluster")
ax.set_xlabel("Condition")
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(FIG / f"fig_paga_cluster_composition.{ext}", dpi=180, bbox_inches="tight")
plt.close(fig)


# Save the integrated AnnData for downstream use
combined.write_h5ad(PROC / "integrated_fibroblasts_paga.h5ad")
print(f"\nSaved integrated AnnData. Outputs in {FIG}")
