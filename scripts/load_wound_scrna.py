"""Load GSE241132 wound healing scRNA-seq (Liu/Landén 2024).

3 patients (PWH26/27/28) x 4 timepoints (D0=intact skin, D1, D7, D30 wound).
Author has provided cell type annotations in cell_metadata.txt.
Fibroblast subset: 12,259 cells.
"""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "GSE241132"
OUT = ROOT / "data" / "processed" / "scRNA"
OUT.mkdir(parents=True, exist_ok=True)


def load_sample(sample_dir: Path, sample_name: str) -> ad.AnnData:
    inner = sample_dir / sample_name
    files = list(inner.glob("*.mtx.gz"))
    if not files:
        return None
    # 10x with custom prefixes
    import scipy.io
    import scipy.sparse as sp
    with __import__("gzip").open(inner / f"{sample_name}_matrix.mtx.gz") as f:
        mtx = scipy.io.mmread(f).tocsr().T  # cells x genes
    bcs = pd.read_csv(inner / f"{sample_name}_barcodes.tsv.gz", sep="\t",
                      header=None)[0].tolist()
    feat = pd.read_csv(inner / f"{sample_name}_features.tsv.gz", sep="\t",
                        header=None)
    feat.columns = ["ensembl", "symbol", "type"][:feat.shape[1]]
    a = ad.AnnData(X=mtx)
    a.obs_names = [f"{sample_name}_{bc}" for bc in bcs]
    a.var_names = feat["symbol"].astype(str).values
    a.var_names_make_unique()
    a.obs["sample"] = sample_name
    return a


def main():
    sc.settings.verbosity = 1
    print("Loading samples...")
    sample_dirs = sorted([p for p in RAW.glob("GSM*") if p.is_dir()])
    adatas = []
    for sd in sample_dirs:
        sample = sd.name.split("_", 1)[1]  # GSM7717079_PWH26D0 -> PWH26D0
        a = load_sample(sd, sample)
        if a is None:
            print(f"  SKIP {sample}")
            continue
        print(f"  {sample}: {a.n_obs} cells x {a.n_vars} genes")
        adatas.append(a)

    print("Concatenating...")
    adata = ad.concat(adatas, join="outer", merge="same")
    print(f"  combined: {adata.n_obs} x {adata.n_vars}")

    print("Loading author cell metadata...")
    meta = pd.read_csv(RAW / "cell_metadata.txt", sep="\t", index_col=0)
    print(f"  meta: {meta.shape}")

    common = adata.obs_names.intersection(meta.index)
    print(f"  matched: {len(common)} cells")
    adata = adata[common].copy()
    for col in ["Patient", "Condition", "newCellTypes", "newMainCellTypes",
                 "seurat_clusters"]:
        adata.obs[col] = meta.loc[adata.obs_names, col].values

    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None,
                                log1p=False, inplace=True)

    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    out = OUT / "wound_all.h5ad"
    adata.write_h5ad(out, compression="gzip")
    print(f"Saved {out} ({adata.n_obs} cells)")
    print("\nCondition x main cell type:")
    print(pd.crosstab(adata.obs["Condition"], adata.obs["newMainCellTypes"]))

    # Subset to fibroblasts only
    fib = adata[adata.obs["newMainCellTypes"] == "Fibroblast"].copy()
    print(f"\nFibroblast subset: {fib.n_obs}")
    print(pd.crosstab(fib.obs["Condition"], fib.obs["newCellTypes"]))
    fib.write_h5ad(OUT / "wound_fibroblasts.h5ad", compression="gzip")
    print(f"Saved {OUT/'wound_fibroblasts.h5ad'}")


if __name__ == "__main__":
    main()
