"""Load GSE181316 (Direder 2022) — keloid + normal scar + healthy skin.

8 samples: 4 keloid + 3 normal scar + 1 healthy skin.
No author metadata provided; we do our own QC + cluster + annotate.
"""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import gzip
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "GSE181316"
OUT = ROOT / "data" / "processed" / "scRNA"
OUT.mkdir(parents=True, exist_ok=True)

SAMPLES = {
    "skin_7":   "Healthy skin",
    "scar_1":   "Normal scar",
    "scar_2":   "Normal scar",
    "scar_3":   "Normal scar",
    "keloid_1": "Keloid",
    "keloid_2": "Keloid",
    "keloid_3L": "Keloid",
    "keloid_3R": "Keloid",
}


def find_files(sample: str):
    """Find barcodes/features/matrix files for a sample (handles GSM prefix)."""
    bc = list(RAW.glob(f"GSM*_{sample}_barcodes.tsv.gz"))
    ft = list(RAW.glob(f"GSM*_{sample}_features.tsv.gz"))
    mt = list(RAW.glob(f"GSM*_{sample}_matrix.mtx.gz"))
    return bc[0], ft[0], mt[0]


def load_sample(sample: str, condition: str) -> ad.AnnData:
    bc_path, ft_path, mt_path = find_files(sample)
    with gzip.open(mt_path) as f:
        mtx = scipy.io.mmread(f).tocsr().T
    bcs = pd.read_csv(bc_path, sep="\t", header=None)[0].tolist()
    feat = pd.read_csv(ft_path, sep="\t", header=None)
    feat.columns = ["ensembl", "symbol", "type"][:feat.shape[1]]
    a = ad.AnnData(X=mtx)
    a.obs_names = [f"{sample}_{bc}" for bc in bcs]
    a.var_names = feat["symbol"].astype(str).values
    a.var_names_make_unique()
    a.obs["sample"] = sample
    a.obs["condition"] = condition
    return a


def main():
    sc.settings.verbosity = 1
    print("Loading samples...")
    adatas = []
    for sample, condition in SAMPLES.items():
        a = load_sample(sample, condition)
        print(f"  {sample} ({condition}): {a.n_obs} cells x {a.n_vars} genes")
        adatas.append(a)

    print("Concatenating...")
    adata = ad.concat(adatas, join="outer", merge="same")
    print(f"  combined: {adata.n_obs} x {adata.n_vars}")

    # QC
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None,
                                log1p=False, inplace=True)
    print(f"  pre-QC: {adata.n_obs}")
    sc.pp.filter_cells(adata, min_genes=300)
    sc.pp.filter_genes(adata, min_cells=3)
    adata = adata[adata.obs["pct_counts_mt"] < 25].copy()
    adata = adata[adata.obs["n_genes_by_counts"] < 7500].copy()
    print(f"  post-QC: {adata.n_obs}")

    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # HVG + PCA + neighbors + UMAP + leiden
    print("Embedding...")
    sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor="seurat",
                                 batch_key="sample")
    a_hvg = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(a_hvg, max_value=10)
    sc.tl.pca(a_hvg, n_comps=40, random_state=0)
    adata.obsm["X_pca"] = a_hvg.obsm["X_pca"]
    sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca", random_state=0)
    sc.tl.umap(adata, min_dist=0.4, random_state=0)
    sc.tl.leiden(adata, resolution=0.5, random_state=0)

    # Quick cell-type annotation by marker score
    markers = {
        "Fibroblast": ["COL1A1", "COL1A2", "DCN", "LUM", "PDGFRA"],
        "Keratinocyte": ["KRT5", "KRT14", "KRT10", "KRT1"],
        "Endothelial": ["PECAM1", "VWF", "CDH5"],
        "T_cell": ["CD3D", "CD3E", "CD2"],
        "Myeloid": ["CD68", "LYZ", "AIF1", "C1QA"],
        "Mast": ["TPSAB1", "TPSB2", "KIT"],
        "Schwann": ["MPZ", "S100B", "PLP1"],
        "Pericyte_SMC": ["ACTA2", "MYH11", "TAGLN", "RGS5"],
        "Melanocyte": ["MLANA", "TYR", "DCT"],
    }
    for ct, genes in markers.items():
        present = [g for g in genes if g in adata.var_names]
        sc.tl.score_genes(adata, present, score_name=f"score_{ct}")

    # Per-cluster top score
    cluster_top_ct = {}
    for c in sorted(adata.obs["leiden"].unique()):
        sub = adata[adata.obs["leiden"] == c]
        scores = {ct: float(sub.obs[f"score_{ct}"].mean()) for ct in markers}
        top = max(scores, key=scores.get)
        cluster_top_ct[c] = top
        print(f"  cluster {c} ({sub.n_obs} cells) -> {top}")
    adata.obs["cell_type"] = adata.obs["leiden"].map(cluster_top_ct)

    print(f"\nFinal counts:")
    print(pd.crosstab(adata.obs["condition"], adata.obs["cell_type"]))

    out = OUT / "direder_all.h5ad"
    adata.write_h5ad(out, compression="gzip")
    print(f"Saved {out}")

    fib = adata[adata.obs["cell_type"] == "Fibroblast"].copy()
    print(f"\nFibroblast subset: {fib.n_obs}")
    fib.write_h5ad(OUT / "direder_fibroblasts.h5ad", compression="gzip")


if __name__ == "__main__":
    main()
