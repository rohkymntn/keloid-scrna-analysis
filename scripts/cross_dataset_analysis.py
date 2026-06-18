"""Unbiased cross-dataset fibroblast characterization.

Datasets:
  - GSE163973 (Deng 2021)   -- 3 keloid + 3 normal scar  (12,177 fib)
  - GSE241132 (Liu 2024)    -- 3 donors x intact/D1/D7/D30 (12,259 fib)
  - GSE181316 (Direder 2022) -- 4 keloid + 3 normal scar + 1 healthy skin (~37k fib)

For each dataset, identify fibroblast subsets and compute DE for all condition
contrasts. Then compare the top-marker landscape across datasets. 
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed" / "scRNA"
PROC.mkdir(parents=True, exist_ok=True)


def de_with_pts(adata, groupby, group, reference, n_top=2000):
    sc.tl.rank_genes_groups(
        adata, groupby=groupby, groups=[group], reference=reference,
        method="wilcoxon", n_genes=adata.n_vars, pts=True,
    )
    df = sc.get.rank_genes_groups_df(adata, group=group)
    df = df.rename(columns={
        "names": "gene", "scores": "score", "logfoldchanges": "logfc",
        "pvals": "pval", "pvals_adj": "padj", "pct_nz_group": "pct_grp",
    })
    pts = adata.uns["rank_genes_groups"]["pts"]
    if reference in pts.columns:
        df["pct_ref"] = df["gene"].map(pts[reference])
    df["delta_pct"] = df["pct_grp"] - df["pct_ref"]
    return df.head(n_top)


# ---------- 1. Deng (already done; just reload) ----------
print("=== Deng 2021 (keloid + normal scar) ===")
deng = sc.read_h5ad(PROC / "deng_fibroblasts_processed.h5ad")
print(f"  {deng.n_obs} fibroblasts; clusters x condition:")
print(pd.crosstab(deng.obs["seurat_clusters"], deng.obs["condition"]))

# Prior DE files exist; just re-load.
deng_de = pd.read_csv(PROC / "de_keloid_mfb_vs_normal_scar_fib.csv")

# ---------- 2. Wound (Liu/Landén 2024) ----------
print("\n=== Liu/Landén 2024 wound healing ===")
wound = sc.read_h5ad(PROC / "wound_fibroblasts.h5ad")
print(f"  {wound.n_obs} fibroblasts")
print(pd.crosstab(wound.obs["Condition"], wound.obs["newCellTypes"]))

# Two key contrasts:
# A) Wound D7 fibroblasts vs intact Skin -> active healing markers
# B) Wound D30 vs Skin -> maturing/persistent markers
# C) Wound D7 FB-I (activated) vs intact Skin all-FB -> the closest analog
#    to the keloid pathological-MFB question

print("DE: Wound D7 fibroblasts vs intact Skin fibroblasts")
wound_d7_vs_skin = de_with_pts(
    wound[wound.obs["Condition"].isin(["Wound7", "Skin"])].copy(),
    groupby="Condition", group="Wound7", reference="Skin",
)
wound_d7_vs_skin.to_csv(PROC / "de_wound_d7_vs_skin.csv", index=False)

print("DE: Wound D30 vs intact Skin")
wound_d30_vs_skin = de_with_pts(
    wound[wound.obs["Condition"].isin(["Wound30", "Skin"])].copy(),
    groupby="Condition", group="Wound30", reference="Skin",
)
wound_d30_vs_skin.to_csv(PROC / "de_wound_d30_vs_skin.csv", index=False)

print("DE: Wound FB-I (activated) vs intact Skin all-FB")
wound2 = wound.copy()
wound2.obs["wound_subset"] = np.where(
    (wound2.obs["Condition"] == "Wound7") & (wound2.obs["newCellTypes"] == "FB-I"),
    "WoundD7_FBI",
    np.where(wound2.obs["Condition"] == "Skin", "Skin_allFB", "other"),
)
wound_fbi_vs_skin = de_with_pts(
    wound2[wound2.obs["wound_subset"].isin(["WoundD7_FBI", "Skin_allFB"])].copy(),
    groupby="wound_subset", group="WoundD7_FBI", reference="Skin_allFB",
)
wound_fbi_vs_skin.to_csv(PROC / "de_wound_fbi_vs_skin.csv", index=False)

# ---------- 3. Direder (keloid + scar + healthy) ----------
print("\n=== Direder 2022 (keloid + normal scar + healthy skin) ===")
direder = sc.read_h5ad(PROC / "direder_fibroblasts.h5ad")
print(f"  {direder.n_obs} fibroblasts")
print(pd.crosstab(direder.obs["condition"], direder.obs["sample"]))

# Run UMAP on Direder fibroblasts only
print("Embedding Direder fibroblasts...")
sc.pp.highly_variable_genes(direder, n_top_genes=3000, flavor="seurat",
                             batch_key="sample")
direder_h = direder[:, direder.var["highly_variable"]].copy()
sc.pp.scale(direder_h, max_value=10)
sc.tl.pca(direder_h, n_comps=40, random_state=0)
direder.obsm["X_pca"] = direder_h.obsm["X_pca"]
sc.pp.neighbors(direder, n_neighbors=15, use_rep="X_pca", random_state=0)
sc.tl.umap(direder, min_dist=0.4, random_state=0)
sc.tl.leiden(direder, resolution=0.4, random_state=0)

direder.write_h5ad(PROC / "direder_fibroblasts_processed.h5ad", compression="gzip")

print("DE: Direder Keloid fibroblasts vs Healthy skin fibroblasts")
dir_kelheal = de_with_pts(
    direder[direder.obs["condition"].isin(["Keloid", "Healthy skin"])].copy(),
    groupby="condition", group="Keloid", reference="Healthy skin",
)
dir_kelheal.to_csv(PROC / "de_direder_keloid_vs_healthy.csv", index=False)

print("DE: Direder Keloid vs Normal scar fibroblasts")
dir_kelscar = de_with_pts(
    direder[direder.obs["condition"].isin(["Keloid", "Normal scar"])].copy(),
    groupby="condition", group="Keloid", reference="Normal scar",
)
dir_kelscar.to_csv(PROC / "de_direder_keloid_vs_normalscar.csv", index=False)

print("DE: Direder Normal scar vs Healthy skin (controls 'wound vs healthy')")
dir_scarheal = de_with_pts(
    direder[direder.obs["condition"].isin(["Normal scar", "Healthy skin"])].copy(),
    groupby="condition", group="Normal scar", reference="Healthy skin",
)
dir_scarheal.to_csv(PROC / "de_direder_scar_vs_healthy.csv", index=False)

# ---------- 4. Build the cross-dataset specificity matrix ----------
print("\n=== Cross-dataset specificity matrix ===")
# For each gene, store logFC (or NaN) across all comparisons
contrasts = {
    "Deng:KeloidC3_MFB_vs_NSfib": deng_de.set_index("gene")["logfc"],
    "Direder:Keloid_vs_Healthy":  dir_kelheal.set_index("gene")["logfc"],
    "Direder:Keloid_vs_NormalScar": dir_kelscar.set_index("gene")["logfc"],
    "Direder:NormalScar_vs_Healthy": dir_scarheal.set_index("gene")["logfc"],
    "Wound:D7_vs_Skin":           wound_d7_vs_skin.set_index("gene")["logfc"],
    "Wound:D30_vs_Skin":          wound_d30_vs_skin.set_index("gene")["logfc"],
    "Wound:D7_FBI_vs_Skin":       wound_fbi_vs_skin.set_index("gene")["logfc"],
}
matrix = pd.DataFrame(contrasts)
matrix.to_csv(PROC / "cross_dataset_logfc_matrix.csv")
print(f"Saved cross-dataset matrix: {matrix.shape}")
print(matrix.head())

# Genes UP in keloid (Deng + Direder) but NOT in wound healing -> keloid-restricted
def is_high(s, thr=1.0): return s.fillna(0) > thr
def is_low(s, thr=0.5): return s.fillna(0).abs() < thr

keloid_restricted = matrix[
    is_high(matrix["Deng:KeloidC3_MFB_vs_NSfib"], 2) &
    is_high(matrix["Direder:Keloid_vs_Healthy"], 1) &
    is_low(matrix["Wound:D7_vs_Skin"], 1) &
    is_low(matrix["Wound:D30_vs_Skin"], 1)
].copy()
keloid_restricted["mean_keloid_lfc"] = keloid_restricted[[
    "Deng:KeloidC3_MFB_vs_NSfib", "Direder:Keloid_vs_Healthy"
]].mean(axis=1)
keloid_restricted = keloid_restricted.sort_values("mean_keloid_lfc", ascending=False)
keloid_restricted.to_csv(PROC / "keloid_restricted_genes.csv")

# Genes UP in wound + UP in keloid -> shared activation programme
wound_and_keloid = matrix[
    is_high(matrix["Deng:KeloidC3_MFB_vs_NSfib"], 2) &
    is_high(matrix["Wound:D7_vs_Skin"], 1)
].copy()
wound_and_keloid["mean_active"] = wound_and_keloid[[
    "Deng:KeloidC3_MFB_vs_NSfib", "Wound:D7_vs_Skin"
]].mean(axis=1)
wound_and_keloid = wound_and_keloid.sort_values("mean_active", ascending=False)
wound_and_keloid.to_csv(PROC / "shared_activation_genes.csv")

print(f"\nKeloid-restricted (up in keloid, NOT in wound): {len(keloid_restricted)}")
print(keloid_restricted.head(15)[
    ["Deng:KeloidC3_MFB_vs_NSfib", "Direder:Keloid_vs_Healthy",
     "Wound:D7_vs_Skin", "Wound:D30_vs_Skin"]
].to_string())

print(f"\nShared with wound activation: {len(wound_and_keloid)}")
print(wound_and_keloid.head(15)[
    ["Deng:KeloidC3_MFB_vs_NSfib", "Wound:D7_vs_Skin", "Wound:D30_vs_Skin"]
].to_string())

print("\nDone.")
