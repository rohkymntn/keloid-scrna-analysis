"""Build a properly-pooled fibroblast AnnData with unified condition labels
across Deng + Direder + Wound datasets. Used for SIGnature and any other
cross-dataset analyses where we need keloid-vs-normal cell-state comparison.

Output: data/processed/scRNA/pooled_fibroblasts.h5ad
"""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed" / "scRNA"
OUT = PROC / "pooled_fibroblasts.h5ad"


def prep(adata, dataset_label, cond_col):
    """Standardize obs columns and add unified condition + state."""
    a = adata.copy()
    a.obs = a.obs.copy()
    a.obs["dataset"] = dataset_label
    a.obs["condition_orig"] = a.obs[cond_col].astype(str)

    # Unified condition labels across datasets
    cond_map = {
        # Healthy/intact
        "Skin": "Healthy_intactSkin",
        "Healthy skin": "Healthy_intactSkin",
        # Wound time course
        "Wound1": "Wound_D1",
        "Wound7": "Wound_D7",
        "Wound30": "Wound_D30",
        # Scar
        "Normal scar": "Normal_scar_mature",
        # Disease
        "Keloid": "Keloid",
    }
    a.obs["condition"] = a.obs["condition_orig"].map(cond_map).fillna(a.obs["condition_orig"])

    # Keloid vs normal binary state (for SIGnature comparison)
    state_map = {
        "Healthy_intactSkin": "Normal",
        "Normal_scar_mature": "Normal",
        "Wound_D1": "Wound",
        "Wound_D7": "Wound",
        "Wound_D30": "Wound",
        "Keloid": "Keloid",
    }
    a.obs["state"] = a.obs["condition"].map(state_map).fillna("Other")

    # Per-cell sample (donor) ID
    a.obs["sample"] = a.obs.get("sample", pd.Series(["unk"] * a.n_obs, index=a.obs.index)).astype(str)

    # Keep only essential obs cols
    keep_cols = ["dataset", "condition_orig", "condition", "state", "sample"]
    a.obs = a.obs[keep_cols].copy()
    return a


# -------- Load datasets --------
print("Loading...")
wound = sc.read_h5ad(PROC / "wound_fibroblasts.h5ad")
deng = sc.read_h5ad(PROC / "deng_fibroblasts_processed.h5ad")
dire = sc.read_h5ad(PROC / "direder_fibroblasts_processed.h5ad")
print(f"  wound: {wound.shape}, deng: {deng.shape}, direder: {dire.shape}")

# Standardize
wound = prep(wound, "Wound", "Condition")
deng = prep(deng, "Deng", "condition")
dire = prep(dire, "Direder", "condition")

# Subsample Direder to balance (it's by far the largest)
np.random.seed(0)
if dire.n_obs > 15000:
    keep = np.random.choice(dire.n_obs, size=15000, replace=False)
    dire = dire[keep].copy()
    print(f"  subsampled direder to {dire.n_obs}")

# Find common genes
common = sorted(set(wound.var_names) & set(deng.var_names) & set(dire.var_names))
print(f"Common genes: {len(common)}")

# Restrict each to common genes
wound = wound[:, common].copy()
deng = deng[:, common].copy()
dire = dire[:, common].copy()

# Concat keeping all obs columns (since they are now identical)
combined = ad.concat(
    [wound, deng, dire],
    axis=0,
    join="outer",
    merge="same",
    uns_merge=None,
    label="dataset_orig",
    keys=["Wound", "Deng", "Direder"],
    index_unique="-",
)

# Check
print(f"\nCombined: {combined.shape}")
print(f"Datasets: {combined.obs.dataset.value_counts().to_dict()}")
print(f"Conditions: {combined.obs.condition.value_counts().to_dict()}")
print(f"States: {combined.obs.state.value_counts().to_dict()}")

# Drop unused dataset_orig column added by concat
if "dataset_orig" in combined.obs.columns:
    combined.obs = combined.obs.drop(columns="dataset_orig")

# Verify key genes are present
key_genes = ["POSTN", "ADAM12", "WISP1", "CCN4", "COL11A1", "ASPN",
              "ITGA11", "TNFRSF12A", "ACTA2", "COMP"]
print(f"\nKey gene presence in pooled dataset:")
for g in key_genes:
    print(f"  {g}: {'YES' if g in combined.var_names else 'NO'}")

# Verify X is what we expect (should be log-normalized from upstream processing)
print(f"\nX type: {type(combined.X).__name__}")
print(f"X.max: {float(combined.X.max()):.2f}, mean: {float(combined.X.mean()):.4f}")

# Save
combined.write_h5ad(OUT)
print(f"\nSaved {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
