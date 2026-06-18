"""Analyze keloid GWAS catalog associations and cross-reference with our scRNA-seq.

Outputs:
- GWAS hit table with mapped/reported genes
- Manhattan-style plot
- Gene-level summary with cross-cohort replication
- Overlay of GWAS gene expression in keloid pathological MFB (Deng 2021)
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
GWAS = ROOT / "data" / "raw" / "GWAS"
OUT_PROC = ROOT / "data" / "processed" / "GWAS"
OUT_FIG = ROOT / "figures" / "gwas"
OUT_PROC.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)


def explode_genes(s):
    """Split mapped-gene strings like 'NEDD4' or 'HYAL1 - HYAL2' or 'ABC, DEF' into a list."""
    if pd.isna(s):
        return []
    parts = re.split(r"[,;\-]| - ", str(s))
    return [p.strip() for p in parts if p.strip() and p.strip() != "NR"]


def main():
    g = pd.read_csv(GWAS / "gwas_catalog_keloid.tsv", sep="\t")
    g["pval_num"] = pd.to_numeric(g["P-VALUE"], errors="coerce")
    g["nlogp"] = -np.log10(g["pval_num"].clip(lower=1e-300))
    g["chr"] = pd.to_numeric(g["CHR_ID"], errors="coerce")
    g["pos"] = pd.to_numeric(g["CHR_POS"], errors="coerce")
    g = g.dropna(subset=["pval_num", "chr", "pos"])
    g["chr_int"] = g["chr"].astype(int)

    # Determine ancestry tag from sample size text
    def ancestry(s):
        s = str(s).lower()
        out = []
        if "japanese" in s: out.append("Japanese")
        if "han taiwanese" in s: out.append("Han Taiwanese")
        if "east asian" in s: out.append("East Asian")
        if "european" in s: out.append("European")
        if "african" in s: out.append("African")
        if "hispanic" in s or "latin american" in s: out.append("Hispanic/Latin")
        return ",".join(out) if out else "Unknown"
    g["ancestry"] = g["INITIAL SAMPLE SIZE"].apply(ancestry)

    # Gene-level summary: count cohorts, min p-value, etc.
    g["genes"] = g["MAPPED_GENE"].apply(explode_genes)
    rows = []
    for _, r in g.iterrows():
        for gene in r["genes"]:
            rows.append({
                "gene": gene, "snp": r["STRONGEST SNP-RISK ALLELE"],
                "chr": r["chr_int"], "pos": int(r["pos"]),
                "pval": r["pval_num"], "ancestry": r["ancestry"],
                "study": r["FIRST AUTHOR"], "year": r["DATE"][:4],
            })
    by_gene = pd.DataFrame(rows)
    gene_summary = by_gene.groupby("gene").agg(
        min_pval=("pval", "min"),
        n_hits=("snp", "nunique"),
        n_studies=("study", "nunique"),
        ancestries=("ancestry", lambda x: ",".join(sorted(set(",".join(x).split(","))))),
        chr=("chr", "first"),
        top_snp=("snp", lambda x: x.iloc[x.values.argmin() if len(x) else 0]),
    ).reset_index().sort_values("min_pval")
    gene_summary["nlogp"] = -np.log10(gene_summary["min_pval"].clip(lower=1e-300))
    gene_summary.to_csv(OUT_PROC / "keloid_gwas_gene_summary.csv", index=False)
    print(f"Gene-level summary: {len(gene_summary)} unique mapped genes")
    print("\nTop GWAS genes:")
    print(gene_summary.head(15).to_string(index=False))

    # ----- Cross-reference with scRNA expression -----
    de_dis = pd.read_csv(ROOT / "data" / "processed" / "scRNA" /
                         "de_keloid_mfb_vs_normal_scar_fib.csv")
    cross = gene_summary.merge(de_dis[["gene", "logfc", "padj", "pct_grp", "pct_ref"]],
                                on="gene", how="left")
    cross.to_csv(OUT_PROC / "keloid_gwas_x_scrna.csv", index=False)
    print(f"\nGenes with both GWAS hit AND keloid C3 MFB expression data:")
    have_expr = cross.dropna(subset=["logfc"])
    print(have_expr.head(20).to_string(index=False))

    # ----- FIG 1: Manhattan plot -----
    sns.set_context("talk", font_scale=0.85)
    sns.set_style("white")

    # Use approximate hg38 chromosome lengths (Mb)
    chr_lens = {1:248,2:242,3:198,4:190,5:182,6:171,7:159,8:145,9:138,10:134,
                11:135,12:133,13:114,14:107,15:102,16:90,17:83,18:80,19:59,
                20:64,21:47,22:51}
    chr_offset = {}
    cum = 0
    for c in range(1, 23):
        chr_offset[c] = cum
        cum += chr_lens[c] * 1e6 + 5e6
    g["x"] = g["chr_int"].map(chr_offset) + g["pos"]

    fig, ax = plt.subplots(figsize=(17, 6.5))
    palette = ["#2C3E50", "#7F8C8D"]
    for c in range(1, 23):
        sub = g[g["chr_int"] == c]
        if len(sub):
            ax.scatter(sub["x"], sub["nlogp"], s=44, alpha=0.85,
                       c=palette[c % 2], rasterized=True, edgecolor="white", linewidth=0.4)
    top_for_label = g.sort_values("pval_num").drop_duplicates("MAPPED_GENE").head(10)
    for _, r in top_for_label.iterrows():
        gene = r["MAPPED_GENE"].split(" - ")[0].split(",")[0]
        ax.annotate(gene, (r["x"], r["nlogp"]),
                    fontsize=10, fontweight="bold", ha="left", va="bottom",
                    xytext=(5, 5), textcoords="offset points",
                    color="#C0392B" if r["nlogp"] > 30 else "#2C3E50")
    ax.axhline(-np.log10(5e-8), color="#C0392B", ls="--", lw=1.2,
               label="GWAS-significant (p < 5e-8)")
    ax.set_xticks([chr_offset[c] + chr_lens[c]*5e5 for c in range(1, 23)])
    ax.set_xticklabels(range(1, 23))
    ax.set_xlabel("Chromosome")
    ax.set_ylabel("-log10(p)")
    ax.set_title("Keloid GWAS Catalog (66 genome-wide-significant associations across 12 studies, multi-ancestry)")
    ax.set_xlim(0, cum)
    ax.legend(loc="upper right", frameon=False)
    sns.despine()
    fig.savefig(OUT_FIG / "fig_g1_manhattan.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT_FIG / "fig_g1_manhattan.svg", bbox_inches="tight")
    plt.close(fig)

    # ----- FIG 2: Top genes bar with replication tally -----
    top = gene_summary.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(top["gene"], top["nlogp"],
                    color="#3498DB", edgecolor="white")
    for bar, nstudies, anc in zip(bars, top["n_studies"], top["ancestries"]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{int(nstudies)} study/ancestry: {anc}",
                va="center", fontsize=9, color="#2C3E50")
    ax.set_xlabel("-log10(min p-value)")
    ax.set_title("Top keloid GWAS genes (cross-ancestry replication tagged)")
    sns.despine()
    fig.savefig(OUT_FIG / "fig_g2_top_genes.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT_FIG / "fig_g2_top_genes.svg", bbox_inches="tight")
    plt.close(fig)

    # ----- FIG 3: GWAS gene x scRNA expression in keloid MFB -----
    have = cross.dropna(subset=["logfc"]).copy()
    if len(have):
        have = have.sort_values("min_pval").head(25)
        fig, ax = plt.subplots(figsize=(11, 7))
        sc_color = ["#C0392B" if x > 1 else "#7F8C8D" if x > 0 else "#2980B9"
                    for x in have["logfc"]]
        ax.barh(have["gene"], have["logfc"], color=sc_color, edgecolor="white")
        for i, r in enumerate(have.itertuples()):
            ax.text(0.02 if r.logfc > 0 else -0.02,
                     i, f"GWAS p={r.min_pval:.0e}", va="center",
                     ha="left" if r.logfc > 0 else "right",
                     fontsize=8, color="#2C3E50")
        ax.set_xlabel("scRNA log2FC (Keloid C3 MFB vs normal scar fibroblasts)")
        ax.axvline(0, color="black", lw=0.6)
        ax.set_title("GWAS-supported genes vs keloid pathological MFB expression\n(red = up in keloid MFB; blue = down; gray = small effect)")
        sns.despine()
        fig.savefig(OUT_FIG / "fig_g3_gwas_x_scrna.png", dpi=180, bbox_inches="tight")
        fig.savefig(OUT_FIG / "fig_g3_gwas_x_scrna.svg", bbox_inches="tight")
        plt.close(fig)

    print(f"\nSaved figures under {OUT_FIG}")


if __name__ == "__main__":
    main()
