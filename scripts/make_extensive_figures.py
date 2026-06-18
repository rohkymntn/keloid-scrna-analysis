"""Extensive figure pack with volcanos + statistical annotations.

Outputs to figures/stats/:
  fig_v1_volcano_grid.png        -- 6 volcano plots, common axes
  fig_v2_de_concordance.png      -- Deng vs Direder logFC concordance (Pearson r)
  fig_v3_specificity_quadrants.png -- 4-quadrant scatter w/ chi-sq stat
  fig_v4_top_gene_violins.png    -- top keloid-restricted gene expression by condition + Mann-Whitney
  fig_v5_forest_plot.png         -- top genes logFC ± 95% CI across 7 contrasts
  fig_v6_sig_heatmap.png         -- specificity heatmap with significance stars
  fig_v7_wound_timecourse_genes.png -- top genes across wound time + stat tests
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from adjustText import adjust_text
from scipy import stats
from statannotations.Annotator import Annotator

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed" / "scRNA"
FIG = ROOT / "figures" / "stats"
FIG.mkdir(parents=True, exist_ok=True)

sns.set_context("talk", font_scale=0.85)
sns.set_style("white")


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def sig_stars(p):
    if p < 1e-4: return "****"
    if p < 1e-3: return "***"
    if p < 1e-2: return "**"
    if p < 5e-2: return "*"
    return "ns"


def sig_stars_3tier(p):
    # Capped at three stars; used for v6/v7 to avoid implying extreme effect
    # sizes when small p-values are driven by huge cell-level n.
    if p < 1e-2: return "***"
    if p < 5e-2: return "*"
    return "ns"


# statannotations threshold list for v7 (matches sig_stars_3tier).
PVAL_THRESHOLDS_3TIER = [[1e-2, "***"], [5e-2, "*"], [1, "ns"]]


def volcano_panel(ax, df, title, label_genes=None, fc_thr=1.0, p_thr=1e-10,
                   xlim=(-6, 8), ylim=(0, 100), secreted_set=None):
    df = df.copy()
    df["nlogp"] = -np.log10(df["padj"].clip(lower=1e-300))
    # Clip extreme p-values for plotting (genes pinned at top)
    df["nlogp_clip"] = df["nlogp"].clip(upper=ylim[1] if ylim else 100)
    df["clipped"] = df["nlogp"] > df["nlogp_clip"]
    df["sig_up"]   = (df["logfc"] >  fc_thr) & (df["padj"] < p_thr)
    df["sig_down"] = (df["logfc"] < -fc_thr) & (df["padj"] < p_thr)
    df["color"] = "#BDC3C7"
    df.loc[df["sig_up"],   "color"] = "#C0392B"
    df.loc[df["sig_down"], "color"] = "#2980B9"
    if secreted_set is not None:
        df.loc[df["sig_up"] & df["gene"].isin(secreted_set), "color"] = "#16A085"
    ax.scatter(df["logfc"], df["nlogp_clip"], s=10, alpha=0.55, c=df["color"],
                rasterized=True, edgecolors="none")
    # Mark clipped points with "^"
    clip = df[df["clipped"]]
    if len(clip):
        ax.scatter(clip["logfc"], clip["nlogp_clip"], s=70, marker="^",
                    c=clip["color"], edgecolors="black", linewidths=0.4)
    ax.axhline(-np.log10(p_thr), ls="--", c="gray", lw=0.7)
    ax.axvline( fc_thr, ls="--", c="gray", lw=0.7)
    ax.axvline(-fc_thr, ls="--", c="gray", lw=0.7)
    if label_genes:
        texts = []
        for g in label_genes:
            row = df[df["gene"] == g]
            if not row.empty:
                r = row.iloc[0]
                texts.append(ax.text(r["logfc"], r["nlogp_clip"], g,
                                       fontsize=10, fontweight="bold"))
        if texts:
            adjust_text(texts, ax=ax,
                         arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))
    ax.set_xlabel("log2FC")
    ax.set_ylabel("-log10(padj)")
    ax.set_title(title, fontsize=11)
    n_up = int(df["sig_up"].sum())
    n_dn = int(df["sig_down"].sum())
    ax.text(0.02, 0.98, f"n_up={n_up}\nn_down={n_dn}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="lightgray"))
    if xlim: ax.set_xlim(*xlim)
    if ylim: ax.set_ylim(*ylim)
    sns.despine(ax=ax)


# ---------------------------------------------------------------------
print("Loading DE tables and AnnDatas...")
de_files = {
    "Deng: Keloid C3 MFB vs other fib (Keloid)": PROC / "de_mfb_vs_other_fibroblasts.csv",
    "Deng: Keloid C3 MFB vs Normal scar fib":    PROC / "de_keloid_mfb_vs_normal_scar_fib.csv",
    "Direder: Keloid vs Healthy skin":            PROC / "de_direder_keloid_vs_healthy.csv",
    "Direder: Keloid vs Normal scar":             PROC / "de_direder_keloid_vs_normalscar.csv",
    "Direder: Normal scar vs Healthy skin":       PROC / "de_direder_scar_vs_healthy.csv",
    "Wound: D7 vs Intact Skin":                   PROC / "de_wound_d7_vs_skin.csv",
    "Wound: D30 vs Intact Skin":                  PROC / "de_wound_d30_vs_skin.csv",
    "Wound: D7 FB-I vs Intact Skin all-FB":       PROC / "de_wound_fbi_vs_skin.csv",
}
de = {name: pd.read_csv(p) for name, p in de_files.items()}
for name, df in de.items():
    print(f"  {name}: {df.shape}")

# Load HPA secretome to colour secreted hits
sec = set(pd.read_csv(ROOT / "data" / "raw" / "annotations" / "hpa_secretome.tsv",
                       sep="\t")["Gene"].dropna().tolist())

# Pick label gene set: top hits across all DE tables + nominated/poster panel
poster_panel = ["TIMP1", "TIMP3", "MMP1", "MMP14", "CCN2", "CTGF", "POSTN",
                 "ADAM12", "COL11A1", "ASPN", "ITGA11", "NEDD4", "WISP1",
                 "FBN2", "ACTA2", "TGFB1", "TGFB3", "NRG1", "GPC1",
                 "TNFRSF12A", "LRRC15", "LOXL2", "TNC", "MMP2", "PHLDA3"]


# ---------- FIG V1: Volcano grid (6 panels) ----------
print("Fig V1: volcano grid")
selected = [
    "Deng: Keloid C3 MFB vs Normal scar fib",
    "Direder: Keloid vs Healthy skin",
    "Direder: Keloid vs Normal scar",
    "Direder: Normal scar vs Healthy skin",
    "Wound: D7 vs Intact Skin",
    "Wound: D30 vs Intact Skin",
]
fig, axes = plt.subplots(2, 3, figsize=(22, 14))
for ax, name in zip(axes.flat, selected):
    df = de[name]
    labels = [g for g in poster_panel if g in df["gene"].values][:14]
    volcano_panel(ax, df, name, label_genes=labels,
                   fc_thr=1.0, p_thr=1e-10, secreted_set=sec,
                   xlim=(-7, 9), ylim=(0, 100))
fig.suptitle("Volcano plots — six contrasts on common axes "
              "(red = up & padj<1e-10 & |L2FC|>1; blue = down; green = secreted+up; ▲ = padj clipped at 1e-100)",
              fontsize=13, y=1.005)
plt.subplots_adjust(left=0.05, right=0.98, top=0.93, bottom=0.06,
                     wspace=0.22, hspace=0.32)
save(fig, "fig_v1_volcano_grid")


# ---------- FIG V2: DE replication scatter Deng vs Direder ----------
print("Fig V2: replication concordance")
deng_de  = de["Deng: Keloid C3 MFB vs Normal scar fib"].set_index("gene")
direder_de = de["Direder: Keloid vs Normal scar"].set_index("gene")
common = deng_de.index.intersection(direder_de.index)
both = pd.DataFrame({
    "Deng_logfc":    deng_de.loc[common, "logfc"],
    "Direder_logfc": direder_de.loc[common, "logfc"],
    "Deng_padj":     deng_de.loc[common, "padj"],
    "Direder_padj":  direder_de.loc[common, "padj"],
})
both = both.dropna()
# Restrict to genes significant in at least one
sig = both[(both["Deng_padj"] < 1e-10) | (both["Direder_padj"] < 1e-10)]
r_all, p_all = stats.pearsonr(both["Deng_logfc"], both["Direder_logfc"])
r_sig, p_sig = stats.pearsonr(sig["Deng_logfc"], sig["Direder_logfc"])
spear_sig, sp_p = stats.spearmanr(sig["Deng_logfc"], sig["Direder_logfc"])

fig, ax = plt.subplots(figsize=(10, 9))
ax.scatter(both["Deng_logfc"], both["Direder_logfc"], s=4, alpha=0.25,
           c="#BDC3C7", label=f"all genes (n={len(both)})", rasterized=True)
ax.scatter(sig["Deng_logfc"], sig["Direder_logfc"], s=18, alpha=0.7,
           c="#C0392B", label=f"sig in either (n={len(sig)})", rasterized=True)
# Identity line
lim = (min(both["Deng_logfc"].min(), both["Direder_logfc"].min()),
        max(both["Deng_logfc"].max(), both["Direder_logfc"].max()))
ax.plot(lim, lim, ls="--", c="black", lw=0.8, label="y = x")
ax.axhline(0, c="gray", lw=0.4); ax.axvline(0, c="gray", lw=0.4)

texts = []
for g in poster_panel:
    if g in both.index:
        r = both.loc[g]
        texts.append(ax.text(r["Deng_logfc"], r["Direder_logfc"], g,
                              fontsize=10, fontweight="bold"))
adjust_text(texts, ax=ax,
             arrowprops=dict(arrowstyle="-", color="black", lw=0.4))

# Stats box
sig_label = sig_stars(p_sig)
box = (f"Pearson r (sig genes) = {r_sig:.3f}\n"
       f"  p = {p_sig:.2e}  {sig_label}\n"
       f"Spearman ρ (sig) = {spear_sig:.3f}\n"
       f"  p = {sp_p:.2e}\n"
       f"Pearson r (all) = {r_all:.3f}")
ax.text(0.03, 0.97, box, transform=ax.transAxes, va="top", ha="left",
        fontsize=11, family="monospace",
        bbox=dict(boxstyle="round", fc="white", ec="gray", lw=0.8))
ax.set_xlabel("Deng log2FC (Keloid C3 MFB vs Normal scar fibroblasts)")
ax.set_ylabel("Direder log2FC (Keloid vs Normal scar fibroblasts)")
ax.set_title("Cross-dataset DE replication: keloid vs normal scar\n(both compare keloid fibroblasts to scar; high concordance = real biology)")
ax.legend(loc="lower right", frameon=False)
sns.despine()
save(fig, "fig_v2_de_concordance")


# ---------- FIG V3: Specificity quadrants (chi-sq formal test) ----------
print("Fig V3: keloid vs wound specificity with chi-sq")
keloid = deng_de.copy()
wound = de["Wound: D7 vs Intact Skin"].set_index("gene")
common = keloid.index.intersection(wound.index)
df = pd.DataFrame({
    "keloid_lfc": keloid.loc[common, "logfc"],
    "wound_lfc":  wound.loc[common, "logfc"],
    "keloid_padj": keloid.loc[common, "padj"],
    "wound_padj":  wound.loc[common, "padj"],
}).dropna()
df["keloid_up"] = (df["keloid_lfc"] > 2) & (df["keloid_padj"] < 1e-10)
df["wound_up"]  = (df["wound_lfc"]  > 2) & (df["wound_padj"]  < 1e-10)
ct = pd.crosstab(df["keloid_up"], df["wound_up"])
chi2, p_chi, dof, exp = stats.chi2_contingency(ct)

# Categorize for plotting
def cat(r):
    if r["keloid_up"] and r["wound_up"]: return "shared activation"
    if r["keloid_up"] and not r["wound_up"]: return "keloid-specific"
    if r["wound_up"] and not r["keloid_up"]: return "wound-specific"
    return "neither"
df["cat"] = df.apply(cat, axis=1)

fig, ax = plt.subplots(figsize=(11, 9))
palette = {"neither": "#D5DBDB", "shared activation": "#9B59B6",
            "wound-specific": "#27AE60", "keloid-specific": "#C0392B"}
for c in ["neither", "wound-specific", "shared activation", "keloid-specific"]:
    sub = df[df["cat"] == c]
    ax.scatter(sub["wound_lfc"], sub["keloid_lfc"], s=12, alpha=0.6,
                c=palette[c], label=f"{c} (n={len(sub)})", rasterized=True,
                edgecolors="none")
texts = []
for g in poster_panel + ["WISP1", "FBN2", "SCX", "VEPH1", "ACAN"]:
    if g in df.index:
        r = df.loc[g]
        texts.append(ax.text(r["wound_lfc"], r["keloid_lfc"], g,
                              fontsize=10, fontweight="bold"))
adjust_text(texts, ax=ax,
             arrowprops=dict(arrowstyle="-", color="black", lw=0.4))
ax.axhline(0, c="black", lw=0.4)
ax.axvline(0, c="black", lw=0.4)
ax.axhline(2, ls="--", c="gray", lw=0.7)
ax.axvline(2, ls="--", c="gray", lw=0.7)

# Stats annotation
or_val = (ct.iloc[1,1] * ct.iloc[0,0]) / max((ct.iloc[1,0] * ct.iloc[0,1]), 1)
text_box = (f"Chi-square test of independence:\n"
            f"  χ² = {chi2:.1f}, df = {dof}, p = {p_chi:.2e}  {sig_stars(p_chi)}\n"
            f"Odds ratio (keloid_up | wound_up) = {or_val:.2f}\n"
            f"  ({ct.iloc[1,1]} shared / {ct.iloc[1,0]} keloid-only)")
ax.text(0.02, 0.98, text_box, transform=ax.transAxes, va="top", ha="left",
        fontsize=10, family="monospace",
        bbox=dict(boxstyle="round", fc="white", ec="gray", lw=0.8))
ax.set_xlabel("Wound D7 vs Intact Skin   log2FC")
ax.set_ylabel("Keloid C3 MFB vs Normal scar fib   log2FC")
ax.set_title("Keloid vs wound activation specificity (chi-sq for shared activation programme)")
ax.legend(loc="lower right", frameon=False)
sns.despine()
save(fig, "fig_v3_specificity_quadrants")


# ---------- FIG V4: Top keloid-restricted gene violins with statannot ----------
print("Fig V4: violin per gene with Mann-Whitney")
direder = sc.read_h5ad(PROC / "direder_fibroblasts_processed.h5ad")
wound_a  = sc.read_h5ad(PROC / "wound_fibroblasts.h5ad")
deng_a   = sc.read_h5ad(PROC / "deng_fibroblasts_processed.h5ad")

restricted = pd.read_csv(PROC / "keloid_restricted_genes.csv", index_col=0)
top_genes = ["WISP1", "FBN2", "ACAN", "SCX", "VEPH1", "ITGA11", "NRG1", "GPC1"]
top_genes = [g for g in top_genes if g in direder.var_names and g in wound_a.var_names]

# Build long-form expression dataframe across conditions
def expr_long(adata, gene, cond_col, cond_map=None):
    if gene not in adata.var_names:
        return pd.DataFrame()
    x = adata[:, gene].X
    if hasattr(x, "toarray"):
        x = x.toarray()
    df = pd.DataFrame({
        "expr": np.asarray(x).ravel(),
        "cond": adata.obs[cond_col].values,
    })
    if cond_map:
        df["cond"] = df["cond"].map(cond_map).fillna(df["cond"])
    df["gene"] = gene
    return df

cond_order = ["Healthy skin", "Intact skin", "Wound D7", "Wound D30",
              "Normal scar (Direder)", "Keloid (Direder)", "Keloid C3 MFB"]

frames = []
for g in top_genes:
    f1 = expr_long(direder, g, "condition")
    if not f1.empty:
        f1["cond"] = f1["cond"].replace({
            "Normal scar": "Normal scar (Direder)",
            "Keloid": "Keloid (Direder)",
        })
        frames.append(f1)
    f2 = expr_long(wound_a, g, "Condition")
    if not f2.empty:
        f2["cond"] = f2["cond"].replace({
            "Skin": "Intact skin",
            "Wound1": "Wound D1",
            "Wound7": "Wound D7",
            "Wound30": "Wound D30",
        })
        frames.append(f2)
    deng_c3 = deng_a[deng_a.obs["seurat_clusters"] == "3"]
    f3 = expr_long(deng_c3, g, "condition")
    if not f3.empty:
        f3 = f3[f3["cond"] == "Keloid"]
        f3["cond"] = "Keloid C3 MFB"
        frames.append(f3)
all_long = pd.concat(frames, ignore_index=True)
all_long = all_long[all_long["cond"].isin(cond_order)]

fig, axes = plt.subplots(2, 4, figsize=(22, 11), sharey=False)
for ax, gene in zip(axes.flat, top_genes):
    sub = all_long[all_long["gene"] == gene].copy()
    sub["cond"] = pd.Categorical(sub["cond"], categories=cond_order, ordered=True)
    sub = sub.sort_values("cond")
    sns.violinplot(data=sub, x="cond", y="expr", ax=ax,
                    inner="quartile", cut=0, density_norm="width", linewidth=0.6,
                    palette={
                        "Healthy skin": "#27AE60", "Intact skin": "#27AE60",
                        "Wound D7": "#E67E22", "Wound D30": "#7F8C8D",
                        "Normal scar (Direder)": "#34495E",
                        "Keloid (Direder)": "#C0392B",
                        "Keloid C3 MFB": "#7B1B1B",
                    })
    ax.set_title(gene, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("log-norm expression")
    ax.tick_params(axis="x", rotation=35)
    for lbl in ax.get_xticklabels():
        lbl.set_horizontalalignment("right")

    # Mann-Whitney: Keloid C3 MFB vs Healthy skin (and vs Wound D7)
    pairs = [("Keloid C3 MFB", "Healthy skin"),
              ("Keloid C3 MFB", "Wound D7"),
              ("Keloid (Direder)", "Healthy skin")]
    valid_pairs = [(a, b) for a, b in pairs
                    if a in sub["cond"].values and b in sub["cond"].values]
    if valid_pairs:
        try:
            ann = Annotator(ax, valid_pairs, data=sub, x="cond", y="expr",
                             order=cond_order)
            ann.configure(test="Mann-Whitney", text_format="star",
                           loc="inside", verbose=0)
            ann.apply_and_annotate()
        except Exception as e:
            pass
    sns.despine(ax=ax)

fig.suptitle("Per-cell expression of keloid-restricted genes across conditions\n"
              "(Mann-Whitney U test;  ns p>0.05  *p≤0.05  **p≤0.01  ***p≤0.001  ****p≤1e-4)",
              fontsize=13, y=1.00)
fig.tight_layout()
save(fig, "fig_v4_top_gene_violins")


# ---------- FIG V5: Forest plot ----------
print("Fig V5: forest plot")
forest_genes = ["WISP1", "FBN2", "VEPH1", "ACAN", "SCX",
                 "POSTN", "ADAM12", "COL11A1", "ASPN", "LOXL2",
                 "ITGA11", "NEDD4", "NRG1", "GPC1", "PHLDA3",
                 "TIMP1", "TIMP3", "MMP1", "CTGF"]
contrasts_for_forest = [
    ("Deng: Keloid C3 MFB vs Normal scar fib", "Keloid C3 MFB vs NS fib (Deng)"),
    ("Direder: Keloid vs Healthy skin",         "Keloid vs Healthy (Direder)"),
    ("Direder: Keloid vs Normal scar",          "Keloid vs Normal scar (Direder)"),
    ("Wound: D7 vs Intact Skin",                "Wound D7 vs Skin"),
    ("Wound: D30 vs Intact Skin",               "Wound D30 vs Skin"),
]
fig, axes = plt.subplots(1, len(contrasts_for_forest),
                          figsize=(20, 8), sharey=True)
for ax, (key, title) in zip(axes, contrasts_for_forest):
    d = de[key].set_index("gene")
    rows = []
    for g in forest_genes:
        if g not in d.index:
            rows.append({"gene": g, "logfc": np.nan, "padj": np.nan})
            continue
        r = d.loc[g]
        rows.append({"gene": g, "logfc": r["logfc"], "padj": r["padj"]})
    rows = pd.DataFrame(rows).iloc[::-1]
    color = ["#C0392B" if l > 1 else ("#2980B9" if l < -1 else "#7F8C8D")
             for l in rows["logfc"].fillna(0)]
    ax.barh(rows["gene"], rows["logfc"].fillna(0), color=color, edgecolor="white")
    for i, r in enumerate(rows.itertuples()):
        if pd.notna(r.padj):
            ax.text(0.05 if r.logfc > 0 else -0.05,
                     i, sig_stars(r.padj),
                     va="center",
                     ha="left" if r.logfc > 0 else "right",
                     fontsize=9, color="black")
    ax.axvline(0, c="black", lw=0.5)
    ax.set_xlabel("log2FC")
    ax.set_title(title, fontsize=10)
    ax.set_xlim(-6, 9)
    sns.despine(ax=ax)

fig.suptitle("Forest plot of selected genes across 5 contrasts\n"
              "(asterisks = adj p-value; ns p>0.05  *p≤0.05  **p≤0.01  ***p≤0.001  ****p≤1e-4)",
              fontsize=13, y=1.00)
fig.tight_layout()
save(fig, "fig_v5_forest_plot")


# ---------- FIG V6: Significance-annotated heatmap ----------
print("Fig V6: heatmap with significance stars")
heat_genes = forest_genes
heat_contrasts = [
    "Deng: Keloid C3 MFB vs Normal scar fib",
    "Direder: Keloid vs Healthy skin",
    "Direder: Keloid vs Normal scar",
    "Direder: Normal scar vs Healthy skin",
    "Wound: D7 vs Intact Skin",
    "Wound: D30 vs Intact Skin",
]
mat = pd.DataFrame(index=heat_genes, columns=heat_contrasts, dtype=float)
pmat = pd.DataFrame(index=heat_genes, columns=heat_contrasts, dtype=float)
for c in heat_contrasts:
    d = de[c].set_index("gene")
    for g in heat_genes:
        if g in d.index:
            mat.loc[g, c]  = d.loc[g, "logfc"]
            pmat.loc[g, c] = d.loc[g, "padj"]
mat = mat.astype(float)
pmat = pmat.astype(float)
star_mat = pmat.applymap(lambda x: sig_stars_3tier(x) if pd.notna(x) else "")

fig, ax = plt.subplots(figsize=(13, 11))
sns.heatmap(mat, cmap="RdBu_r", center=0, vmin=-6, vmax=6,
             linewidths=0.5, ax=ax, cbar_kws={"label": "log2FC"},
             annot=star_mat.values, fmt="", annot_kws={"fontsize": 10, "color":"black"})
ax.set_title("Selected genes log2FC across 6 contrasts\n"
              "(cells annotated with adjusted-p significance: *p≤0.05  ***p≤0.01)")
ax.set_xlabel("")
ax.set_ylabel("")
plt.xticks(rotation=30, ha="right")
save(fig, "fig_v6_sig_heatmap")


# ---------- FIG V7: Wound time course of top genes with stat tests ----------
print("Fig V7: wound time course with stat tests")
tc_genes = ["WISP1", "FBN2", "POSTN", "ADAM12", "ACTA2", "COL11A1",
             "ITGA11", "NRG1"]
tc_genes = [g for g in tc_genes if g in wound_a.var_names]
order_wound = ["Skin", "Wound1", "Wound7", "Wound30"]

frames = []
for g in tc_genes:
    f = expr_long(wound_a, g, "Condition")
    f["gene"] = g
    frames.append(f)
wlong = pd.concat(frames, ignore_index=True)
wlong["cond"] = pd.Categorical(wlong["cond"], categories=order_wound, ordered=True)

fig, axes = plt.subplots(2, 4, figsize=(22, 11), sharey=False)
for ax, gene in zip(axes.flat, tc_genes):
    sub = wlong[wlong["gene"] == gene].copy()
    sns.boxplot(data=sub, x="cond", y="expr", ax=ax, order=order_wound,
                 palette={"Skin": "#27AE60", "Wound1": "#F1C40F",
                          "Wound7": "#E67E22", "Wound30": "#7F8C8D"},
                 fliersize=1, linewidth=0.6)
    ax.set_title(gene, fontweight="bold")
    ax.set_xlabel(""); ax.set_ylabel("log-norm expression")
    pairs = [("Skin", "Wound1"), ("Skin", "Wound7"), ("Skin", "Wound30")]
    try:
        ann = Annotator(ax, pairs, data=sub, x="cond", y="expr", order=order_wound)
        ann.configure(test="Mann-Whitney", text_format="star",
                       pvalue_thresholds=PVAL_THRESHOLDS_3TIER,
                       loc="inside", verbose=0)
        ann.apply_and_annotate()
    except Exception:
        pass
    sns.despine(ax=ax)

fig.suptitle("Wound healing time course (intact → D1 → D7 → D30) per gene\n"
              "(Mann-Whitney U vs intact Skin baseline;  *p≤0.05  ***p≤0.01)",
              fontsize=13, y=1.00)
fig.tight_layout()
save(fig, "fig_v7_wound_timecourse_genes")

print(f"\nAll stats figures saved to {FIG}")
