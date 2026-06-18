"""Analyze Deng 2021 keloid fibroblasts: identify keloid-enriched MFB cluster,
run DE, classify markers as secreted/surface/receptor, nominate sensor and
output-target candidates for the iGEM patch design.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed" / "scRNA"
ANN = ROOT / "data" / "raw" / "annotations"
OUT = PROC
OUT.mkdir(parents=True, exist_ok=True)

POSTER_MARKERS = {
    "sensors_proposed": ["TIMP1", "TIMP2", "TIMP3"],
    "outputs_proposed": ["MMP1", "MMP2", "MMP9", "TLN2"],
    "fibrosis_markers": ["POSTN", "COL1A1", "COL3A1", "ACTA2", "CCN2", "CCN1",
                         "FN1", "TNC", "LOX", "LOXL2", "TGFB1", "TGFB3"],
    "yap_taz_targets": ["CCN2", "CCN1", "ANKRD1", "AMOTL2", "CTGF",
                        "CYR61", "BIRC5", "AXL"],
}


def load_annotation_sets() -> dict[str, set[str]]:
    secretome = pd.read_csv(ANN / "hpa_secretome.tsv", sep="\t")["Gene"].dropna().tolist()
    membrane = pd.read_csv(ANN / "hpa_membrane.tsv", sep="\t")["Gene"].dropna().tolist()
    return {"secreted": set(secretome), "membrane": set(membrane)}


def main() -> None:
    sc.settings.verbosity = 1
    sc.settings.figdir = ROOT / "figures"

    print("Loading AnnData...")
    adata = sc.read_h5ad(PROC / "deng_fibroblasts.h5ad")
    print(f"  {adata.n_obs} cells x {adata.n_vars} genes")

    # Cluster labels from authors (Seurat). Cluster 3 is the keloid-enriched MFB.
    clusters_kf_enrichment = (
        pd.crosstab(adata.obs["seurat_clusters"], adata.obs["condition"])
    )
    clusters_kf_enrichment["KF_enrichment"] = (
        clusters_kf_enrichment["Keloid"] / clusters_kf_enrichment["Normal scar"]
    )
    print("\nKF enrichment by cluster:")
    print(clusters_kf_enrichment.sort_values("KF_enrichment", ascending=False))

    # ----- Embedding for figures -----
    print("\nComputing HVGs / PCA / UMAP...")
    sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor="seurat", batch_key="sample")
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata_hvg, max_value=10)
    sc.tl.pca(adata_hvg, n_comps=40, random_state=0)
    adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"]
    sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca", random_state=0)
    sc.tl.umap(adata, min_dist=0.4, random_state=0)

    # Label the keloid-enriched cluster
    adata.obs["mfb_cluster"] = np.where(
        adata.obs["seurat_clusters"] == "3",
        "Pathological MFB (C3)",
        "Other fibroblasts",
    )

    # ----- DE: pathological MFB vs other fibroblasts (within keloid samples) -----
    print("\nDE: cluster 3 vs all other fibroblasts (within Keloid)...")
    keloid = adata[adata.obs["condition"] == "Keloid"].copy()
    sc.tl.rank_genes_groups(
        keloid,
        groupby="mfb_cluster",
        groups=["Pathological MFB (C3)"],
        reference="Other fibroblasts",
        method="wilcoxon",
        n_genes=adata.n_vars,
        pts=True,
    )
    de_mfb = sc.get.rank_genes_groups_df(keloid, group="Pathological MFB (C3)")
    de_mfb = de_mfb.rename(columns={
        "names": "gene", "scores": "score", "logfoldchanges": "logfc",
        "pvals": "pval", "pvals_adj": "padj",
        "pct_nz_group": "pct_grp",
    })
    pts_ref = keloid.uns["rank_genes_groups"]["pts"]["Other fibroblasts"]
    de_mfb["pct_ref"] = de_mfb["gene"].map(pts_ref)
    de_mfb["delta_pct"] = de_mfb["pct_grp"] - de_mfb["pct_ref"]

    # ----- DE: Keloid pathological MFB (C3) vs Normal scar fibroblasts (all NS clusters) -----
    print("\nDE: Keloid C3 vs Normal-scar fibroblasts (any cluster)...")
    adata.obs["disease_vs_normal"] = np.where(
        (adata.obs["condition"] == "Keloid") & (adata.obs["seurat_clusters"] == "3"),
        "Keloid_MFB_C3",
        np.where(adata.obs["condition"] == "Normal scar",
                 "NS_fibroblast",
                 "other_keloid"),
    )
    sub = adata[adata.obs["disease_vs_normal"].isin(["Keloid_MFB_C3", "NS_fibroblast"])].copy()
    sc.tl.rank_genes_groups(
        sub,
        groupby="disease_vs_normal",
        groups=["Keloid_MFB_C3"],
        reference="NS_fibroblast",
        method="wilcoxon",
        n_genes=adata.n_vars,
        pts=True,
    )
    de_dis = sc.get.rank_genes_groups_df(sub, group="Keloid_MFB_C3")
    de_dis = de_dis.rename(columns={
        "names": "gene", "scores": "score", "logfoldchanges": "logfc",
        "pvals": "pval", "pvals_adj": "padj",
        "pct_nz_group": "pct_grp",
    })
    pts_ref_d = sub.uns["rank_genes_groups"]["pts"]["NS_fibroblast"]
    de_dis["pct_ref"] = de_dis["gene"].map(pts_ref_d)
    de_dis["delta_pct"] = de_dis["pct_grp"] - de_dis["pct_ref"]

    # ----- Localization annotation -----
    annots = load_annotation_sets()
    for df in (de_mfb, de_dis):
        df["secreted"] = df["gene"].isin(annots["secreted"])
        df["membrane"] = df["gene"].isin(annots["membrane"])

    de_mfb.to_csv(OUT / "de_mfb_vs_other_fibroblasts.csv", index=False)
    de_dis.to_csv(OUT / "de_keloid_mfb_vs_normal_scar_fib.csv", index=False)

    # ----- Sensor candidates: secreted + upregulated + specific to MFB -----
    sensors = de_mfb[
        (de_mfb["secreted"]) &
        (de_mfb["logfc"] > 1.0) &
        (de_mfb["padj"] < 1e-10) &
        (de_mfb["pct_grp"] > 0.30) &
        (de_mfb["delta_pct"] > 0.10)
    ].copy()
    sensors["specificity_score"] = sensors["logfc"] * sensors["delta_pct"]
    sensors = sensors.sort_values("specificity_score", ascending=False)
    sensors_top = sensors.head(40)
    sensors_top.to_csv(OUT / "sensor_candidates.csv", index=False)
    print(f"\nTop 15 SENSOR candidates (secreted, MFB-specific):")
    print(sensors_top[["gene", "logfc", "pct_grp", "pct_ref", "delta_pct", "specificity_score"]].head(15).to_string(index=False))

    # ----- Output target candidates: surface receptors on MFB -----
    targets = de_mfb[
        (de_mfb["membrane"]) &
        (~de_mfb["secreted"]) &
        (de_mfb["logfc"] > 0.5) &
        (de_mfb["padj"] < 1e-5) &
        (de_mfb["pct_grp"] > 0.20)
    ].copy()
    targets["target_score"] = targets["logfc"] * targets["pct_grp"]
    targets = targets.sort_values("target_score", ascending=False)
    targets_top = targets.head(40)
    targets_top.to_csv(OUT / "output_target_candidates.csv", index=False)
    print(f"\nTop 15 OUTPUT TARGET candidates (membrane on MFB):")
    print(targets_top[["gene", "logfc", "pct_grp", "pct_ref", "target_score"]].head(15).to_string(index=False))

    # ----- Cross-check poster markers -----
    print("\nPoster-proposed markers in DE results:")
    for category, genes in POSTER_MARKERS.items():
        rows = []
        for g in genes:
            for label, df in [("MFB_vs_other", de_mfb), ("Keloid_C3_vs_NS_fib", de_dis)]:
                m = df[df["gene"] == g]
                if not m.empty:
                    r = m.iloc[0]
                    rows.append({
                        "gene": g,
                        "comparison": label,
                        "logfc": round(r["logfc"], 2),
                        "padj": f"{r['padj']:.1e}",
                        "pct_grp": round(r["pct_grp"], 2),
                        "pct_ref": round(r["pct_ref"], 2),
                        "secreted": bool(r["secreted"]),
                    })
        if rows:
            print(f"\n  {category}:")
            print(pd.DataFrame(rows).to_string(index=False))

    adata.write_h5ad(PROC / "deng_fibroblasts_processed.h5ad", compression="gzip")
    print(f"\nWrote processed h5ad with UMAP + cluster labels.")


if __name__ == "__main__":
    main()
