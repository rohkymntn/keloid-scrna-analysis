"""Cross-dataset figure pack — descriptive, no sensor bias.

Outputs to figures/cross/:
  fig_c1_umap_triptych.png         -- UMAPs of all three datasets
  fig_c2_wound_timecourse.png      -- fibroblast subset proportions over wound time
  fig_c3_landscape_heatmap.png     -- top genes across Healthy->D1->D7->D30->Scar->Keloid
  fig_c4_specificity_scatter.png   -- keloid logFC vs wound logFC (separates spec vs shared)
  fig_c5_keloid_restricted.png     -- top genes UP in keloid AND silent in wound
  fig_c6_gwas_combined.png         -- GWAS x scRNA combined panel (already exists in gwas/)
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
FIG = ROOT / "figures" / "cross"
FIG.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


sns.set_context("talk", font_scale=0.85)
sns.set_style("white")

# ---------- Load all three datasets ----------
print("Loading...")
deng = sc.read_h5ad(PROC / "deng_fibroblasts_processed.h5ad")
wound = sc.read_h5ad(PROC / "wound_fibroblasts.h5ad")
direder = sc.read_h5ad(PROC / "direder_fibroblasts_processed.h5ad")

# Compute UMAP for wound if not present
if "X_umap" not in wound.obsm:
    print("  computing wound UMAP...")
    sc.pp.highly_variable_genes(wound, n_top_genes=3000, flavor="seurat",
                                 batch_key="Patient")
    h = wound[:, wound.var["highly_variable"]].copy()
    sc.pp.scale(h, max_value=10)
    sc.tl.pca(h, n_comps=40, random_state=0)
    wound.obsm["X_pca"] = h.obsm["X_pca"]
    sc.pp.neighbors(wound, n_neighbors=15, use_rep="X_pca", random_state=0)
    sc.tl.umap(wound, min_dist=0.4, random_state=0)

# ---------- FIG C1: UMAP triptych ----------
print("Fig C1: UMAP triptych")
fig, axes = plt.subplots(2, 3, figsize=(20, 13))

sc.pl.umap(deng, color="condition", ax=axes[0, 0], show=False,
            palette={"Keloid": "#C0392B", "Normal scar": "#34495E"},
            title="Deng 2021\nKeloid + Normal scar (n=12,177 fib)", frameon=False)
sc.pl.umap(wound, color="Condition", ax=axes[0, 1], show=False,
            palette={"Skin": "#27AE60", "Wound1": "#F39C12",
                     "Wound7": "#E67E22", "Wound30": "#7F8C8D"},
            title="Liu/Landén 2024 wound healing\nIntact / D1 / D7 / D30 (n=12,259 fib)",
            frameon=False)
sc.pl.umap(direder, color="condition", ax=axes[0, 2], show=False,
            palette={"Healthy skin": "#27AE60", "Normal scar": "#34495E",
                     "Keloid": "#C0392B"},
            title="Direder 2022\nKeloid + Normal scar + Healthy (n=37,713 fib)",
            frameon=False)

sc.pl.umap(deng, color="seurat_clusters", ax=axes[1, 0], show=False,
            legend_loc="on data", title="Deng author clusters (C3 = keloid-enriched)",
            frameon=False)
sc.pl.umap(wound, color="newCellTypes", ax=axes[1, 1], show=False,
            title="Wound fibroblast subsets\n(FB-I = activated, dominates D7+D30)",
            frameon=False)
sc.pl.umap(direder, color="leiden", ax=axes[1, 2], show=False,
            legend_loc="on data", title="Direder leiden clusters (this analysis)",
            frameon=False)

save(fig, "fig_c1_umap_triptych")

# ---------- FIG C2: Wound time course ----------
print("Fig C2: Wound time course of fibroblast subsets")
order_cond = ["Skin", "Wound1", "Wound7", "Wound30"]
order_sub = ["FB-I", "FB-II", "FB-III", "FB-prolif"]
prop = (
    pd.crosstab(wound.obs["Condition"], wound.obs["newCellTypes"], normalize="index")
    .reindex(index=order_cond, columns=order_sub)
)
fig, axes = plt.subplots(1, 2, figsize=(18, 6.5))
prop.plot(kind="bar", stacked=True, ax=axes[0],
           color=["#C0392B", "#3498DB", "#2ECC71", "#F39C12"], width=0.7)
axes[0].set_ylabel("Fraction of fibroblasts")
axes[0].set_xlabel("")
axes[0].set_title("Fibroblast subset composition over wound healing")
axes[0].legend(title="Subset", frameon=False, loc="lower left",
                bbox_to_anchor=(0.0, -0.32), ncol=4)
axes[0].tick_params(axis="x", rotation=0)

# absolute counts
counts = pd.crosstab(wound.obs["Condition"], wound.obs["newCellTypes"]).reindex(
    index=order_cond, columns=order_sub)
counts.plot(kind="bar", ax=axes[1],
             color=["#C0392B", "#3498DB", "#2ECC71", "#F39C12"], width=0.75)
axes[1].set_ylabel("# fibroblasts")
axes[1].set_xlabel("")
axes[1].set_title("Absolute fibroblast counts by subset")
axes[1].legend(title="Subset", frameon=False, loc="lower left",
                bbox_to_anchor=(0.0, -0.32), ncol=4)
axes[1].tick_params(axis="x", rotation=0)
sns.despine()
save(fig, "fig_c2_wound_timecourse")

# ---------- FIG C3: Landscape heatmap across all conditions ----------
print("Fig C3: Cross-condition expression heatmap")
# Pull a panel of "interesting" genes from cross-dataset matrix
matrix = pd.read_csv(PROC / "cross_dataset_logfc_matrix.csv", index_col=0)
keloid_restricted = pd.read_csv(PROC / "keloid_restricted_genes.csv", index_col=0)
shared = pd.read_csv(PROC / "shared_activation_genes.csv", index_col=0)

# Pick a curated panel: top 8 keloid-restricted + top 8 shared + classic markers
panel_genes = (
    list(keloid_restricted.head(10).index)
    + list(shared.head(10).index)
    + ["TIMP1", "TIMP3", "MMP1", "MMP2", "MMP14", "POSTN", "ADAM12",
       "ITGA11", "NEDD4", "NRG1", "GPC1", "ACTA2", "COL1A1", "CCN2",
       "TGFB1", "TGFB3"]
)
panel_genes = list(dict.fromkeys(panel_genes))

# Build per-condition mean expression
def mean_expr(adata, genes, group_col):
    present = [g for g in genes if g in adata.var_names]
    out = {}
    for cond in adata.obs[group_col].unique():
        sub = adata[adata.obs[group_col] == cond]
        x = sub[:, present].X
        if hasattr(x, "toarray"):
            x = x.toarray()
        out[cond] = pd.Series(np.asarray(x.mean(axis=0)).ravel(), index=present)
    return pd.DataFrame(out)

deng_means = mean_expr(deng, panel_genes, "condition")
wound_means = mean_expr(wound, panel_genes, "Condition")
direder_means = mean_expr(direder, panel_genes, "condition")

# Combine into one dataframe with consistent ordering
combined = pd.DataFrame(index=panel_genes)
combined["Healthy skin\n(Direder)"] = direder_means.get("Healthy skin")
combined["Intact skin\n(Wound D0)"] = wound_means.get("Skin")
combined["Wound D1"] = wound_means.get("Wound1")
combined["Wound D7"] = wound_means.get("Wound7")
combined["Wound D30"] = wound_means.get("Wound30")
combined["Normal scar\n(Direder)"] = direder_means.get("Normal scar")
combined["Normal scar\n(Deng)"] = deng_means.get("Normal scar")
combined["Keloid all-fib\n(Deng)"] = deng_means.get("Keloid")
combined["Keloid all-fib\n(Direder)"] = direder_means.get("Keloid")

# Add Deng C3 keloid-only column
deng_c3 = deng[(deng.obs["seurat_clusters"] == "3") &
               (deng.obs["condition"] == "Keloid")]
present = [g for g in panel_genes if g in deng_c3.var_names]
x = deng_c3[:, present].X
if hasattr(x, "toarray"):
    x = x.toarray()
combined["Keloid C3 MFB\n(Deng pathological)"] = pd.Series(
    np.asarray(x.mean(axis=0)).ravel(), index=present
)

combined = combined.dropna(how="all").fillna(0)
# Z-score across rows
zmat = combined.sub(combined.mean(axis=1), axis=0).div(
    combined.std(axis=1).replace(0, 1), axis=0)

fig, ax = plt.subplots(figsize=(13, 14))
sns.heatmap(zmat, cmap="RdBu_r", center=0, ax=ax,
             linewidths=0.4, cbar_kws={"label": "z-score (row-normalized)"})
ax.set_title("Cross-dataset expression landscape\n"
              "(Healthy → Wound D1/D7/D30 → Normal scar → Keloid all-fib → Keloid C3 pathological MFB)")
ax.set_xlabel("")
ax.set_ylabel("")
plt.xticks(rotation=30, ha="right")
save(fig, "fig_c3_landscape_heatmap")

# ---------- FIG C4: Specificity scatter ----------
print("Fig C4: Specificity scatter")
sub = matrix[
    matrix["Deng:KeloidC3_MFB_vs_NSfib"].notna()
    & matrix["Wound:D7_vs_Skin"].notna()
].copy()
sub["category"] = "shared"
sub.loc[(sub["Deng:KeloidC3_MFB_vs_NSfib"] > 2)
        & (sub["Wound:D7_vs_Skin"].abs() < 1), "category"] = "keloid-specific"
sub.loc[(sub["Wound:D7_vs_Skin"] > 2)
        & (sub["Deng:KeloidC3_MFB_vs_NSfib"].abs() < 1), "category"] = "wound-specific"
sub.loc[(sub["Deng:KeloidC3_MFB_vs_NSfib"] > 2)
        & (sub["Wound:D7_vs_Skin"] > 2), "category"] = "shared activation"

fig, ax = plt.subplots(figsize=(10, 9))
palette = {"shared": "#BDC3C7", "keloid-specific": "#C0392B",
            "wound-specific": "#27AE60", "shared activation": "#9B59B6"}
for cat in ["shared", "wound-specific", "shared activation", "keloid-specific"]:
    s = sub[sub["category"] == cat]
    ax.scatter(s["Wound:D7_vs_Skin"], s["Deng:KeloidC3_MFB_vs_NSfib"],
                s=14, alpha=0.65, c=palette[cat], label=f"{cat} (n={len(s)})",
                rasterized=True)

# Highlight: poster markers + GWAS hits + nominated sensors
highlight = ["POSTN", "ADAM12", "TIMP1", "MMP1", "WISP1", "FBN2", "ITGA11",
              "NEDD4", "NRG1", "GPC1", "COL11A1", "ASPN", "ACTA2", "CCN2",
              "TGFBI", "LOXL2", "TNFRSF12A", "MMP14"]
for g in highlight:
    if g in sub.index:
        r = sub.loc[g]
        ax.annotate(g, (r["Wound:D7_vs_Skin"], r["Deng:KeloidC3_MFB_vs_NSfib"]),
                    fontsize=10, fontweight="bold",
                    xytext=(4, 4), textcoords="offset points")
ax.axhline(0, c="black", lw=0.4)
ax.axvline(0, c="black", lw=0.4)
ax.axhline(2, ls="--", c="gray", lw=0.7)
ax.axvline(2, ls="--", c="gray", lw=0.7)
ax.set_xlabel("Wound D7 vs Intact Skin (log2FC, GSE241132)")
ax.set_ylabel("Keloid C3 MFB vs Normal scar fib (log2FC, GSE163973)")
ax.set_title("Keloid- vs wound-fibroblast activation specificity\n"
              "(top-left = keloid-restricted; top-right = shared 'active fibroblast' programme)")
ax.legend(loc="lower right", frameon=False, fontsize=10)
ax.set_xlim(-3, 9)
ax.set_ylim(-3, 9)
sns.despine()
save(fig, "fig_c4_specificity_scatter")

# ---------- FIG C5: Top keloid-restricted genes ----------
print("Fig C5: Top keloid-restricted genes")
kr = keloid_restricted.head(20).copy()
kr = kr.sort_values("Deng:KeloidC3_MFB_vs_NSfib", ascending=True)
fig, ax = plt.subplots(figsize=(10, 9))
y = np.arange(len(kr))
ax.barh(y - 0.2, kr["Deng:KeloidC3_MFB_vs_NSfib"], 0.4,
         label="Keloid C3 MFB vs NS fib (Deng)", color="#C0392B")
ax.barh(y + 0.2, kr["Direder:Keloid_vs_Healthy"].clip(upper=10), 0.4,
         label="Keloid vs Healthy (Direder)", color="#E67E22")
# wound bar (small or zero)
wd7 = kr["Wound:D7_vs_Skin"].fillna(0)
ax.barh(y + 0.6, wd7, 0.4,
         label="Wound D7 vs Skin (low/silent in wound)", color="#7F8C8D")
ax.set_yticks(y)
ax.set_yticklabels(kr.index)
ax.set_xlabel("log2FC")
ax.set_title("Top 20 keloid-RESTRICTED fibroblast genes\n"
              "(replicated in Deng + Direder, NOT activated in normal wound healing)")
ax.legend(frameon=False, loc="lower right")
ax.axvline(0, c="black", lw=0.4)
sns.despine()
save(fig, "fig_c5_keloid_restricted")

# ---------- FIG C6: GWAS combined panel ----------
print("Fig C6: GWAS combined panel (load existing)")
# This figure is already produced in figures/gwas/. Symlink/copy combined image
import shutil
src = ROOT / "figures" / "gwas" / "fig_g3_gwas_x_scrna.png"
if src.exists():
    shutil.copy(src, FIG / "fig_c6_gwas_x_scrna.png")
    shutil.copy(src.with_suffix(".svg"), FIG / "fig_c6_gwas_x_scrna.svg")

print(f"\nAll cross-dataset figures saved under {FIG}")
