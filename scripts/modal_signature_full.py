"""FULL SIGnature attribution run on all pooled fibroblast cells.
Run: modal run scripts/modal_signature_full.py
"""
import io
import tempfile
import os as _os
import time
from pathlib import Path

import modal

app = modal.App("signature-keloid-full")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget", "git")
    .pip_install(
        "sc-signature==1.0.0",
        "scanpy==1.10.4",
        "anndata==0.10.9",
        "numpy<2",
        "scipy",
        "pandas",
        "torch==2.4.0",
        "captum==0.9.0",
        "tiledb==0.36.1",
        "scimilarity",
    )
)

model_volume = modal.Volume.from_name("scimilarity-model", create_if_missing=False)


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/models": model_volume},
    timeout=7200,
    memory=32768,
)
def run_full_attributions(adata_bytes: bytes):
    import numpy as np
    import anndata as ad
    import torch
    from SIGnature.models.scimilarity import SCimilarityWrapper

    print("CUDA:", torch.cuda.is_available())
    print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")

    # Load AnnData via temp file
    print("Loading AnnData...")
    with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as tmp:
        tmp.write(adata_bytes)
        tmp_path = tmp.name
    adata = ad.read_h5ad(tmp_path)
    _os.unlink(tmp_path)
    print(f"  shape: {adata.shape}")
    print(f"  states: {adata.obs['state'].value_counts().to_dict()}")
    print(f"  conditions: {adata.obs['condition'].value_counts().to_dict()}")
    print(f"  datasets: {adata.obs['dataset'].value_counts().to_dict()}")

    # Load model from cached volume
    print("\nLoading SCimilarity model from /models/scimilarity...")
    t0 = time.time()
    scim = SCimilarityWrapper(model_path="/models/scimilarity", use_gpu=True)
    print(f"  loaded in {time.time() - t0:.1f}s")

    # Preprocess
    print("Preprocessing (gene alignment + normalization)...")
    t0 = time.time()
    tadata = scim.preprocess_adata(adata)
    print(f"  shape after preprocess: {tadata.shape}")
    print(f"  preprocess time: {time.time() - t0:.1f}s")

    # Compute attributions — full run
    print(f"\nComputing IG attributions on {tadata.n_obs} cells (this is the main run)...")
    t0 = time.time()
    atts = scim.calculate_attributions(
        tadata.X,
        method="ig",
        batch_size=200,  # Higher than smoke to use GPU better
    )
    elapsed = time.time() - t0
    print(f"  attributions: {atts.shape}")
    print(f"  compute time: {elapsed:.1f}s = {elapsed/60:.1f} min")
    print(f"  per-cell time: {elapsed/tadata.n_obs*1000:.2f} ms/cell")

    # Convert to dense float32 for transport
    atts_dense = np.asarray(atts.todense()) if hasattr(atts, "todense") else np.asarray(atts)
    print(f"  memory: {atts_dense.nbytes / 1e9:.2f} GB")

    return {
        "attributions": atts_dense.astype(np.float32),
        "gene_names": list(tadata.var_names),
        "obs": {
            "dataset": list(tadata.obs["dataset"].astype(str)),
            "condition": list(tadata.obs["condition"].astype(str)),
            "state": list(tadata.obs["state"].astype(str)),
            "sample": list(tadata.obs["sample"].astype(str)),
        },
    }


@app.local_entrypoint()
def main():
    import scanpy as sc
    import numpy as np

    print("=" * 60)
    print("Loading pooled dataset...")
    print("=" * 60)
    pooled = sc.read_h5ad("data/processed/scRNA/pooled_fibroblasts.h5ad")
    print(f"Pooled: {pooled.shape}")
    print(f"States: {pooled.obs.state.value_counts().to_dict()}")
    print(f"Conditions: {pooled.obs.condition.value_counts().to_dict()}")

    # Subsample within each condition to keep things tractable but balanced
    # Target: ~5000 per major state, all cells per smaller groups
    np.random.seed(42)
    target_per_state = 5000
    keep_idx = []
    for state in pooled.obs.state.unique():
        idx = np.where(pooled.obs.state == state)[0]
        n = min(len(idx), target_per_state)
        sel = np.random.choice(idx, size=n, replace=False)
        keep_idx.append(sel)
        print(f"  {state}: keeping {n}/{len(idx)}")
    keep_idx = np.concatenate(keep_idx)
    sub = pooled[keep_idx].copy()
    print(f"\nTotal subset: {sub.shape}")

    # Also keep stratification by condition for downstream analysis
    print(f"Final condition breakdown: {sub.obs.condition.value_counts().to_dict()}")

    # Serialize via temp file
    print("\nSerializing AnnData for Modal transfer...")
    with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as tmp:
        tmp_path = tmp.name
    sub.write_h5ad(tmp_path)
    with open(tmp_path, "rb") as f:
        adata_bytes = f.read()
    _os.unlink(tmp_path)
    print(f"  serialized: {len(adata_bytes) / 1e6:.1f} MB")

    print("\n" + "=" * 60)
    print("Running SCimilarity attributions on Modal...")
    print("=" * 60)
    t0 = time.time()
    result = run_full_attributions.remote(adata_bytes)
    print(f"\nTotal wall time: {(time.time() - t0)/60:.1f} min")

    print(f"\nAttributions shape: {result['attributions'].shape}")
    print(f"Genes: {len(result['gene_names'])}")

    # Save locally
    out_dir = Path("data/processed/scRNA/signature_full")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "full_attributions.npz"
    np.savez_compressed(
        out_path,
        attributions=result["attributions"],
        gene_names=np.array(result["gene_names"]),
        dataset=np.array(result["obs"]["dataset"]),
        condition=np.array(result["obs"]["condition"]),
        state=np.array(result["obs"]["state"]),
        sample=np.array(result["obs"]["sample"]),
    )
    print(f"\nSaved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    # Quick sanity check
    atts = result["attributions"]
    states = np.array(result["obs"]["state"])
    genes = result["gene_names"]
    print("\nSanity check — our candidate genes (mean attribution per state):")
    print(f"{'Gene':10s} {'Keloid':>8s} {'Normal':>8s} {'Wound':>8s} {'Δ(K-N)':>8s}")
    for g in ["POSTN", "ADAM12", "ASPN", "COL11A1", "TNC", "TGFBI", "ACTA2", "WISP1", "CCN4"]:
        if g in genes:
            gi = genes.index(g)
            k = atts[states == "Keloid", gi].mean()
            n = atts[states == "Normal", gi].mean()
            w = atts[states == "Wound", gi].mean()
            print(f"{g:10s} {k:>8.3f} {n:>8.3f} {w:>8.3f} {k-n:>+8.3f}")
        else:
            print(f"{g:10s} not in SCimilarity vocabulary")
