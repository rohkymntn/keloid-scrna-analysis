"""Generate publication-quality figures for the Deng 2021 keloid scRNA analysis.

Outputs to figures/scrna/:
  fig_s1_umap.png/.svg                  -- UMAP by cluster + condition + MFB call
  fig_s2_marker_dotplot.png/.svg        -- top markers x cluster dotplot
  fig_s3_volcano_mfb.png/.svg           -- volcano: pathological MFB vs other fib
  fig_s4_sensor_candidates.png/.svg     -- ranked secreted sensor candidates
  fig_s5_output_targets.png/.svg        -- ranked surface output-target candidates
  fig_s6_poster_vs_data.png/.svg        -- TIMP1/POSTN/CTGF reality check
  fig_s7_bulk_vs_sc_validation.png/.svg -- bulk GSE188952 vs sc GSE163973
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
FIG = ROOT / "figures" / "scrna"
FIG.mkdir(parents=True, exist_ok=True)


def save(fig, name: str) -> None:
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    sns.set_context("talk", font_scale=0.85)
    sns.set_style("white")

    print("Loading processed AnnData...")
    adata = sc.read_h5ad(PROC / "deng_fibroblasts_processed.h5ad")
    de_mfb = pd.read_csv(PROC / "de_mfb_vs_other_fibroblasts.csv")
    de_dis = pd.read_csv(PROC / "de_keloid_mfb_vs_normal_scar_fib.csv")
    sensors = pd.read_csv(PROC / "sensor_candidates.csv")
    targets = pd.read_csv(PROC / "output_target_candidates.csv")

    # ----- FIG 1: UMAP triptych -----
    print("Fig 1: UMAP triptych")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.4))
    sc.pl.umap(adata, color="seurat_clusters", ax=axes[0], show=False,
               legend_loc="on data", title="Author clusters", frameon=False)
    sc.pl.umap(adata, color="condition", ax=axes[1], show=False,
               palette={"Keloid": "#C0392B", "Normal scar": "#2C3E50"},
               title="Condition", frameon=False)
    sc.pl.umap(adata, color="mfb_cluster", ax=axes[2], show=False,
               palette={"Pathological MFB (C3)": "#C0392B",
                        "Other fibroblasts": "#BDC3C7"},
               title="Pathological MFB call", frameon=False)
    save(fig, "fig_s1_umap")

    # ----- FIG 2: Marker dotplot -----
    print("Fig 2: Dotplot of top markers per cluster")
    top_per_cluster = []
    keloid = adata[adata.obs["condition"] == "Keloid"].copy()
    sc.tl.rank_genes_groups(keloid, "seurat_clusters", method="wilcoxon",
                             n_genes=10, pts=True)
    for c in sorted(keloid.obs["seurat_clusters"].unique()):
        try:
            df = sc.get.rank_genes_groups_df(keloid, group=c).head(5)
            top_per_cluster.extend(df["names"].tolist())
        except Exception:
            pass
    top_per_cluster = list(dict.fromkeys(top_per_cluster))[:40]
    fig = sc.pl.dotplot(adata, top_per_cluster, groupby="seurat_clusters",
                       standard_scale="var", return_fig=True, show=False,
                       figsize=(14, 6))
    fig.savefig(FIG / "fig_s2_marker_dotplot.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIG / "fig_s2_marker_dotplot.svg", bbox_inches="tight")
    plt.close()

    # ----- FIG 3: Volcano -----
    print("Fig 3: Volcano of pathological MFB vs other")
    de = de_mfb.copy()
    de["nlogp"] = -np.log10(de["padj"].clip(lower=1e-300))
    de["category"] = "other"
    de.loc[(de["secreted"]) & (de["logfc"] > 1) & (de["padj"] < 1e-10),
           "category"] = "secreted (sensor)"
    de.loc[(de["membrane"]) & ~(de["secreted"]) & (de["logfc"] > 0.5)
           & (de["padj"] < 1e-5), "category"] = "membrane (target)"
    fig, ax = plt.subplots(figsize=(10, 7))
    palette = {"other": "#BDC3C7", "secreted (sensor)": "#2980B9",
               "membrane (target)": "#27AE60"}
    for cat in ["other", "membrane (target)", "secreted (sensor)"]:
        sub = de[de["category"] == cat]
        ax.scatter(sub["logfc"], sub["nlogp"], s=8, alpha=0.6,
                   c=palette[cat], label=cat, rasterized=True)
    label_genes = (
        list(sensors["gene"].head(8))
        + list(targets["gene"].head(6))
        + ["TIMP1", "TIMP3", "MMP1", "CTGF", "ACTA2", "POSTN"]
    )
    label_genes = list(dict.fromkeys(label_genes))
    for g in label_genes:
        m = de[de["gene"] == g]
        if not m.empty:
            r = m.iloc[0]
            ax.annotate(g, (r["logfc"], r["nlogp"]),
                        fontsize=10, fontweight="bold", ha="left", va="bottom",
                        xytext=(4, 4), textcoords="offset points")
    ax.axhline(-np.log10(1e-10), ls="--", c="gray", lw=1)
    ax.axvline(1, ls="--", c="gray", lw=1)
    ax.set_xlabel("log2FC (Pathological MFB vs other fibroblasts, in keloid)")
    ax.set_ylabel("-log10(padj)")
    ax.set_title("Volcano: keloid pathological MFB markers")
    ax.legend(loc="upper left", frameon=False)
    ax.set_xlim(-5, 6)
    sns.despine()
    save(fig, "fig_s3_volcano_mfb")

    # ----- FIG 4: Sensor candidate ranking -----
    print("Fig 4: Sensor candidate ranking")
    top_sensors = sensors.head(20).copy()
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(top_sensors["gene"][::-1], top_sensors["specificity_score"][::-1],
                   color="#2980B9", edgecolor="white")
    for bar, lfc, pct in zip(bars,
                              top_sensors["logfc"][::-1],
                              top_sensors["pct_grp"][::-1]):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"L2FC {lfc:.1f}, {int(pct*100)}%",
                va="center", fontsize=9, color="#2C3E50")
    ax.set_xlabel("Specificity score (logFC × Δpct)")
    ax.set_title("Top 20 secreted SENSOR candidates\n(MFB-specific, secreted; HPA secretome)")
    sns.despine()
    save(fig, "fig_s4_sensor_candidates")

    # ----- FIG 5: Output target ranking -----
    print("Fig 5: Output target ranking")
    top_targets = targets.head(20).copy()
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(top_targets["gene"][::-1], top_targets["target_score"][::-1],
                   color="#27AE60", edgecolor="white")
    for bar, lfc, pct in zip(bars,
                              top_targets["logfc"][::-1],
                              top_targets["pct_grp"][::-1]):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"L2FC {lfc:.1f}, {int(pct*100)}%",
                va="center", fontsize=9, color="#2C3E50")
    ax.set_xlabel("Target score (logFC × pct expressing)")
    ax.set_title("Top 20 surface OUTPUT-TARGET candidates\n(membrane on MFB; HPA membrane proteome)")
    sns.despine()
    save(fig, "fig_s5_output_targets")

    # ----- FIG 6: Poster vs data reality check -----
    print("Fig 6: Poster markers reality check")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel A: Per-cell expression of poster markers across clusters
    poster_genes = ["TIMP1", "TIMP2", "TIMP3", "MMP1", "MMP2", "MMP9", "CTGF",
                    "POSTN", "COL11A1", "ADAM12", "ASPN", "TGFBI"]
    poster_genes = [g for g in poster_genes if g in adata.var_names]
    df_long = []
    for g in poster_genes:
        x = adata[:, g].X.toarray().ravel() if hasattr(adata.X, "toarray") else adata[:, g].X.ravel()
        for cluster, exp in zip(adata.obs["seurat_clusters"], x):
            df_long.append({"gene": g, "cluster": cluster, "expr": exp})
    long = pd.DataFrame(df_long)
    summary = long.groupby(["gene", "cluster"]).agg(
        mean_expr=("expr", "mean"),
        pct_expr=("expr", lambda v: (v > 0).mean()),
    ).reset_index()
    pivot_mean = summary.pivot(index="gene", columns="cluster", values="mean_expr")
    pivot_mean = pivot_mean.reindex(poster_genes)
    pivot_mean.columns = [f"C{c}" for c in pivot_mean.columns]
    sns.heatmap(pivot_mean, cmap="RdBu_r", ax=axes[0],
                cbar_kws={"label": "mean log-norm expr"}, linewidths=0.4)
    axes[0].set_title("Poster + nominated markers across fibroblast clusters\n(C3 = keloid-enriched pathological MFB)")
    axes[0].set_xlabel("Author cluster")
    axes[0].set_ylabel("")

    # Panel B: TIMP1 reality check — % expressing across condition x cluster
    fig_genes = ["TIMP1", "TIMP3", "POSTN", "COL11A1", "ADAM12"]
    rows = []
    for g in fig_genes:
        if g not in adata.var_names:
            continue
        x = adata[:, g].X.toarray().ravel() if hasattr(adata.X, "toarray") else adata[:, g].X.ravel()
        for cond, exp in zip(adata.obs["condition"], x):
            rows.append({"gene": g, "cond": cond, "expr": exp})
    bar_df = pd.DataFrame(rows)
    pct_df = bar_df.groupby(["gene", "cond"]).agg(
        pct=("expr", lambda v: (v > 0).mean() * 100),
        mean=("expr", "mean"),
    ).reset_index()
    sns.barplot(pct_df, x="gene", y="pct", hue="cond",
                palette={"Keloid": "#C0392B", "Normal scar": "#2C3E50"},
                ax=axes[1], order=fig_genes)
    axes[1].set_ylabel("% fibroblasts expressing (per-cell, scRNA-seq)")
    axes[1].set_title("Per-cell expression: poster sensors vs data-nominated\n(TIMP1 is 99% in BOTH conditions = useless as discriminator)")
    axes[1].set_xlabel("")
    axes[1].legend(frameon=False, title="")
    axes[1].set_ylim(0, 105)
    for p in axes[1].patches:
        if p.get_height() > 0:
            axes[1].text(p.get_x() + p.get_width() / 2, p.get_height() + 1,
                         f"{p.get_height():.0f}%", ha="center", fontsize=9)
    sns.despine(ax=axes[1])
    save(fig, "fig_s6_poster_vs_data")

    # ----- FIG 7: Bulk vs sc validation -----
    print("Fig 7: Bulk GSE188952 vs sc GSE163973 validation")
    bulk = pd.read_csv(ROOT / "data" / "processed" / "expression_log2.csv", index_col=0)
    bulk_meta = pd.read_csv(ROOT / "data" / "processed" / "sample_metadata.csv", index_col=0)
    bulk_meta["group"] = bulk_meta["group"].fillna("Unknown")
    keloid_samples = bulk_meta.index[bulk_meta["group"] == "Keloid"]
    normal_samples = bulk_meta.index[bulk_meta["group"].isin(["Normal scar"])]

    # Pick top sensors and check bulk concordance
    check_genes = (list(sensors["gene"].head(15))
                   + ["TIMP1", "TIMP3", "MMP1", "CTGF", "POSTN", "ACTA2"])
    check_genes = list(dict.fromkeys(check_genes))
    rows = []
    for g in check_genes:
        if g not in bulk.index:
            continue
        bk = bulk.loc[g, keloid_samples].astype(float).mean()
        bn = bulk.loc[g, normal_samples].astype(float).mean()
        bulk_lfc = bk - bn  # already log2
        sc_match = de_dis[de_dis["gene"] == g]
        sc_lfc = float(sc_match["logfc"].iloc[0]) if not sc_match.empty else np.nan
        rows.append({"gene": g, "bulk_log2FC": bulk_lfc, "sc_log2FC": sc_lfc})
    cmp_df = pd.DataFrame(rows).dropna()
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.axhline(0, c="gray", lw=0.6); ax.axvline(0, c="gray", lw=0.6)
    ax.plot([-3, 9], [-3, 9], ls="--", c="lightgray", lw=1)
    ax.scatter(cmp_df["bulk_log2FC"], cmp_df["sc_log2FC"],
               s=70, alpha=0.85, c="#34495E", edgecolor="white")
    for _, r in cmp_df.iterrows():
        ax.annotate(r["gene"], (r["bulk_log2FC"], r["sc_log2FC"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=10)
    # Highlight TIMP1 and POSTN
    for g, color in [("TIMP1", "#C0392B"), ("POSTN", "#27AE60")]:
        m = cmp_df[cmp_df["gene"] == g]
        if not m.empty:
            ax.scatter(m["bulk_log2FC"], m["sc_log2FC"], s=180, edgecolor=color,
                       facecolor="none", linewidth=2.4)
    ax.set_xlabel("Bulk GSE188952  (Keloid - Normal scar)  log2FC")
    ax.set_ylabel("scRNA GSE163973  (Keloid C3 MFB - NS fib)  log2FC")
    ax.set_title("Bulk vs single-cell concordance\n(TIMP1: bulk-elevated but per-cell flat → cell-number artifact)")
    sns.despine()
    save(fig, "fig_s7_bulk_vs_sc_validation")

    print(f"\nAll figures saved under {FIG}")


if __name__ == "__main__":
    main()
