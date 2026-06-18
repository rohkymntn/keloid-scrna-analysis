"""Load Deng 2021 keloid scRNA-seq (GSE163973) and write fibroblast AnnData.

Source: GSE163973 — 6 samples (3 keloid KL1-3, 3 normal scar NS1-3).
Subsets to the 12,177 fibroblasts that the authors curated and clustered.
"""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "GSE163973"
OUT = ROOT / "data" / "processed" / "scRNA"
OUT.mkdir(parents=True, exist_ok=True)

# Author cluster labels live in fib_main_clusters_metadata.csv:
# barcodes look like "AAACCCAAGATCCTAC_1" but tagged by sample via orig.ident column.
SAMPLE_DIRS = {
    "KL1": RAW / "mtx" / "KL1" / "KF1_matrix",
    "KL2": RAW / "mtx" / "KL2" / "KF2_matrix",
    "KL3": RAW / "mtx" / "KL3" / "KF3_matrix",
    "NS1": RAW / "mtx" / "NS1" / "NF1_matrix",
    "NS2": RAW / "mtx" / "NS2" / "NF2_matrix",
    "NS3": RAW / "mtx" / "NS3" / "NF3_matrix",
}

# Per fib metadata, authors used orig.ident = KF1/KF2/KF3/NF1/NF2/NF3
SAMPLE_TO_ORIG = {
    "KL1": "KF1", "KL2": "KF2", "KL3": "KF3",
    "NS1": "NF1", "NS2": "NF2", "NS3": "NF3",
}


def load_sample(sample: str, path: Path) -> ad.AnnData:
    a = sc.read_10x_mtx(path, var_names="gene_symbols", cache=False)
    a.var_names_make_unique()
    a.obs["sample"] = sample
    a.obs["orig.ident"] = SAMPLE_TO_ORIG[sample]
    a.obs["condition"] = "Keloid" if sample.startswith("KL") else "Normal scar"
    a.obs_names = [f"{bc}_{sample}" for bc in a.obs_names]
    return a


def main() -> None:
    sc.settings.verbosity = 1

    print("Loading 6 samples...")
    adatas = {s: load_sample(s, p) for s, p in SAMPLE_DIRS.items()}
    for s, a in adatas.items():
        print(f"  {s}: {a.n_obs:>6} cells x {a.n_vars} genes")

    print("Concatenating...")
    adata = ad.concat(adatas, join="outer", label="batch", merge="same")
    print(f"  combined: {adata.n_obs} cells x {adata.n_vars} genes")

    print("Loading author fibroblast metadata...")
    meta = pd.read_csv(RAW / "fib_main_clusters_metadata.csv", index_col=0)
    print(f"  fib metadata: {meta.shape}")

    # Author barcodes are like "AAACCCAAGATCCTAC_1" where _1.._6 indexes samples
    # Map orig.ident -> sample suffix used in metadata
    orig_to_sample = {v: k for k, v in SAMPLE_TO_ORIG.items()}
    meta_keys = []
    for bc, row in meta.iterrows():
        sample = orig_to_sample[row["orig.ident"]]
        prefix = bc.split("_")[0]
        meta_keys.append(f"{prefix}-1_{sample}")
    meta.index = meta_keys

    # Subset to fibroblasts
    common = adata.obs_names.intersection(meta.index)
    print(f"  matched fibroblast barcodes: {len(common)} / {len(meta)}")
    adata = adata[common].copy()

    for col in ["seurat_clusters", "integrated_snn_res.0.4", "integrated_snn_res.0.45"]:
        adata.obs[col] = meta.loc[adata.obs_names, col].astype(str).values

    # QC
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    print(f"  median genes/cell: {int(np.median(adata.obs['n_genes_by_counts']))}")
    print(f"  median counts/cell: {int(np.median(adata.obs['total_counts']))}")
    print(f"  median pct.mt: {np.median(adata.obs['pct_counts_mt']):.2f}")

    # Save raw counts before normalization
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    out = OUT / "deng_fibroblasts.h5ad"
    adata.write_h5ad(out, compression="gzip")
    print(f"Saved {out} ({adata.n_obs} cells x {adata.n_vars} genes)")
    print("Cluster x condition:")
    print(pd.crosstab(adata.obs["seurat_clusters"], adata.obs["condition"]))


if __name__ == "__main__":
    main()
