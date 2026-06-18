"""H5 test: does the keloid-associated mesenchymal-fibroblast (MF) program
RESOLVE in normal wound healing by D30, while persisting in keloid?

Outputs to figures/hypothesis/:
  fig_h5_mf_score_per_condition.png  -- mean MF signature score per condition (per-sample dots)
  fig_h5_mf_fraction_per_condition.png -- fraction of MF-positive cells per condition
  fig_h5_mf_gene_violins.png            -- POSTN/COL11A1/CCN4 across conditions
  fig_h5_mf_score_per_patient_paired.png -- paired wound time course per patient
  h5_mf_summary_stats.csv                -- numbers behind the figures
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed" / "scRNA"
FIG = ROOT / "figures" / "hypothesis"
FIG.mkdir(parents=True, exist_ok=True)

sns.set_context("talk", font_scale=0.85)
sns.set_style("white")

# Mesenchymal-fibroblast signature from Deng 2021 + Direder + commonly cited
# keloid mesenchymal markers. Genes confirmed present in ALL three datasets.
MF_SIG = ["POSTN", "COL11A1", "COL5A2", "COMP", "COL12A1", "ASPN", "TNC"]

# Score threshold for "MF-positive" cell. score_genes returns mean(set) -
# mean(matched-expression random control set). Calibrated below from Direder
# healthy-skin distribution; using a fixed 0.30 keeps the threshold
# interpretable and consistent across datasets.
MF_POS_THR = 0.30


def score(adata):
    sc.tl.score_genes(adata, gene_list=MF_SIG, score_name="mf_score", random_state=0)
    adata.obs["mf_pos"] = (adata.obs["mf_score"] > MF_POS_THR).astype(int)


def per_sample_table(a, cond_col, dataset):
    g = a.obs.groupby(["sample", cond_col], observed=True)
    out = pd.DataFrame({
        "n_cells": g.size(),
        "mf_score_mean": g["mf_score"].mean(),
        "mf_pos_frac": g["mf_pos"].mean(),
    }).reset_index().rename(columns={cond_col: "condition"})
    out["dataset"] = dataset
    return out


# ---------- Load and score ----------
print("Loading and scoring...")
wound = sc.read_h5ad(PROC / "wound_fibroblasts.h5ad")
deng  = sc.read_h5ad(PROC / "deng_fibroblasts_processed.h5ad")
dire  = sc.read_h5ad(PROC / "direder_fibroblasts_processed.h5ad")

for a in (wound, deng, dire):
    score(a)

# Calibration check: print per-condition score distributions to ensure
# threshold sits at a biologically meaningful breakpoint.
print("\nMF score quantiles per condition:")
for name, a, col in [("wound", wound, "Condition"),
                       ("deng", deng, "condition"),
                       ("direder", dire, "condition")]:
    q = a.obs.groupby(col, observed=True)["mf_score"].quantile([0.5, 0.75, 0.9])
    print(f"\n{name}\n{q}")

# ---------- Per-sample summary ----------
ws = per_sample_table(wound, "Condition", "Wound time-course")
ds = per_sample_table(deng,  "condition", "Deng (keloid)")
rs = per_sample_table(dire,  "condition", "Direder (keloid)")
summary = pd.concat([ws, ds, rs], ignore_index=True)
summary.to_csv(FIG / "h5_mf_summary_stats.csv", index=False)
print(f"\nSaved summary: {FIG/'h5_mf_summary_stats.csv'}")

# ---------- Figure 1: MF signature score per condition ----------
print("Fig: MF score per condition")
order = [
    ("Wound time-course", "Skin"),
    ("Wound time-course", "Wound1"),
    ("Wound time-course", "Wound7"),
    ("Wound time-course", "Wound30"),
    ("Direder (keloid)",   "Healthy skin"),
    ("Direder (keloid)",   "Normal scar"),
    ("Direder (keloid)",   "Keloid"),
    ("Deng (keloid)",      "Normal scar"),
    ("Deng (keloid)",      "Keloid"),
]
summary["xlabel"] = summary["dataset"].str.split(" ").str[0] + "\n" + summary["condition"]
key2lab = {(d, c): f"{d.split(' ')[0]}\n{c}" for d, c in order}
summary["xlabel"] = summary.apply(lambda r: key2lab.get((r["dataset"], r["condition"]),
                                                          f"{r['dataset']}\n{r['condition']}"),
                                    axis=1)
xorder = [key2lab[k] for k in order]

palette = {
    "Wound time-course": "#5DADE2",
    "Direder (keloid)":   "#E67E22",
    "Deng (keloid)":      "#C0392B",
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6.5))

for d in summary["dataset"].unique():
    sub = summary[summary["dataset"] == d]
    ax1.scatter(sub["xlabel"], sub["mf_score_mean"],
                 s=120, color=palette[d], edgecolor="black", linewidth=0.8,
                 label=d, zorder=3)
sub_means = summary.groupby("xlabel")["mf_score_mean"].mean().reindex(xorder)
ax1.plot(xorder, sub_means.values, color="#7F8C8D", linewidth=1, alpha=0.5, zorder=1)
ax1.axhline(MF_POS_THR, color="red", linestyle="--", linewidth=1, alpha=0.5,
             label=f"MF-positive threshold ({MF_POS_THR})")
ax1.set_ylabel("Mean mesenchymal-fibroblast\nsignature score (per sample)")
ax1.set_title("MF signature score across conditions\n(each dot = one biological sample)")
ax1.legend(loc="upper left", fontsize=10)
ax1.tick_params(axis='x', rotation=30)
for lbl in ax1.get_xticklabels(): lbl.set_horizontalalignment("right")
sns.despine(ax=ax1)

for d in summary["dataset"].unique():
    sub = summary[summary["dataset"] == d]
    ax2.scatter(sub["xlabel"], sub["mf_pos_frac"] * 100,
                 s=120, color=palette[d], edgecolor="black", linewidth=0.8, zorder=3)
ax2.set_ylabel(f"% MF-positive cells (score > {MF_POS_THR})")
ax2.set_title("Fraction of mesenchymal-fibroblast cells per sample")
ax2.tick_params(axis='x', rotation=30)
for lbl in ax2.get_xticklabels(): lbl.set_horizontalalignment("right")
sns.despine(ax=ax2)

fig.suptitle("H5 test — does the mesenchymal-fibroblast program resolve by D30?",
              fontsize=14, y=1.02)
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(FIG / f"fig_h5_mf_score_per_condition.{ext}", dpi=180, bbox_inches="tight")
plt.close(fig)

# ---------- Figure 2: paired wound time course per patient ----------
print("Fig: paired wound time course per patient")
ws_wide = (ws.pivot_table(index="sample", columns="condition",
                            values=["mf_score_mean", "mf_pos_frac"]))
patients = sorted({s[:-2] if s.endswith("D0") or s.endswith("D1") else
                    s[:-2] if s.endswith("D7") else s[:-3] if s.endswith("D30") else s
                    for s in ws["sample"].unique()})

# Build paired table by patient
ws["patient"] = ws["sample"].str.extract(r"(PWH\d+)")[0]
paired = ws.pivot_table(index="patient", columns="condition",
                          values="mf_score_mean")
paired = paired[["Skin", "Wound1", "Wound7", "Wound30"]]

paired_frac = ws.pivot_table(index="patient", columns="condition",
                                values="mf_pos_frac")
paired_frac = paired_frac[["Skin", "Wound1", "Wound7", "Wound30"]]

# Add keloid reference band (mean+/-sd of keloid samples from both datasets)
keloid_score = pd.concat([
    ds[ds["condition"] == "Keloid"]["mf_score_mean"],
    rs[rs["condition"] == "Keloid"]["mf_score_mean"],
])
keloid_frac = pd.concat([
    ds[ds["condition"] == "Keloid"]["mf_pos_frac"],
    rs[rs["condition"] == "Keloid"]["mf_pos_frac"],
])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
for ax, dat, refmean, refsd, ylab, title in [
    (ax1, paired, keloid_score.mean(), keloid_score.std(),
     "Mean MF signature score", "MF score: wound time course (paired) vs keloid band"),
    (ax2, paired_frac * 100, keloid_frac.mean() * 100, keloid_frac.std() * 100,
     "% MF-positive cells", "MF-positive fraction: wound time course vs keloid band"),
]:
    for pat in dat.index:
        ax.plot(dat.columns, dat.loc[pat].values, "o-", linewidth=2, markersize=10,
                 label=pat, alpha=0.9)
    ax.axhspan(refmean - refsd, refmean + refsd, color="#C0392B", alpha=0.18,
                label="Keloid mean ± SD")
    ax.axhline(refmean, color="#C0392B", linewidth=1.5, alpha=0.6)
    ax.set_ylabel(ylab)
    ax.set_xlabel("Wound time point")
    ax.set_title(title, fontsize=12)
    ax.legend(loc="best", fontsize=9)
    sns.despine(ax=ax)

fig.suptitle("H5 test, paired view — wound D7 transient vs keloid persistent?",
              fontsize=13, y=1.02)
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(FIG / f"fig_h5_mf_score_per_patient_paired.{ext}", dpi=180, bbox_inches="tight")
plt.close(fig)

# ---------- Figure 3: per-gene violin plots ----------
print("Fig: per-gene violin")
genes_to_plot = ["POSTN", "COL11A1", "CCN4", "COL5A2", "ASPN", "ITGA11"]


def expr_long(a, gene, cond_col, dataset_lab):
    if gene not in a.var_names:
        return None
    x = a[:, gene].X
    x = np.asarray(x.todense()).ravel() if hasattr(x, "todense") else np.asarray(x).ravel()
    return pd.DataFrame({
        "expr": x,
        "condition": a.obs[cond_col].values,
        "dataset": dataset_lab,
    })


frames = []
for g in genes_to_plot:
    for a, col, lab in [(wound, "Condition", "Wound time-course"),
                         (deng,  "condition", "Deng"),
                         (dire,  "condition", "Direder")]:
        f = expr_long(a, g, col, lab)
        if f is not None:
            f["gene"] = g
            frames.append(f)
expr_df = pd.concat(frames, ignore_index=True)
expr_df["xlabel"] = expr_df["dataset"].str.split(" ").str[0] + ":" + expr_df["condition"].astype(str)

xorder_g = ["Wound:Skin", "Wound:Wound1", "Wound:Wound7", "Wound:Wound30",
             "Direder:Healthy skin", "Direder:Normal scar", "Direder:Keloid",
             "Deng:Normal scar", "Deng:Keloid"]
xorder_g = [x for x in xorder_g if x in expr_df["xlabel"].unique()]

n = len(genes_to_plot)
fig, axes = plt.subplots(2, 3, figsize=(22, 11), sharey=False)
for ax, g in zip(axes.flat, genes_to_plot):
    sub = expr_df[expr_df["gene"] == g]
    if len(sub) == 0:
        ax.set_visible(False); continue
    sns.violinplot(data=sub, x="xlabel", y="expr", ax=ax, order=xorder_g,
                    inner="quartile", cut=0, linewidth=0.6,
                    palette=["#5DADE2"]*4 + ["#7DCEA0"] + ["#F39C12"]*2 + ["#E67E22"]*2)
    ax.set_title(g, fontweight="bold")
    ax.set_xlabel(""); ax.set_ylabel("log-norm expression")
    ax.tick_params(axis='x', rotation=30)
    for lbl in ax.get_xticklabels(): lbl.set_horizontalalignment("right")
    sns.despine(ax=ax)

fig.suptitle("Per-cell expression of mesenchymal-fibroblast markers across conditions",
              fontsize=13, y=1.00)
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(FIG / f"fig_h5_mf_gene_violins.{ext}", dpi=180, bbox_inches="tight")
plt.close(fig)

# ---------- Stats: paired test wound D7 vs D30, vs keloid ----------
print("\n=== H5 statistical tests ===")
# Paired Wilcoxon: D7 vs D30 within wound (per-patient)
d7 = paired["Wound7"].values
d30 = paired["Wound30"].values
w_stat, w_p = stats.wilcoxon(d7, d30, alternative="greater")
print(f"Paired Wilcoxon, MF score D7 > D30 (wound, n={len(d7)} patients): "
       f"W={w_stat}, p={w_p:.3g}")

# Mann-Whitney: keloid samples vs wound D30 samples
keloid_all = pd.concat([ds[ds.condition=="Keloid"]["mf_score_mean"],
                          rs[rs.condition=="Keloid"]["mf_score_mean"]]).values
d30_all = ws[ws.condition=="Wound30"]["mf_score_mean"].values
u_stat, u_p = stats.mannwhitneyu(keloid_all, d30_all, alternative="greater")
print(f"Mann-Whitney, MF score keloid > wound D30 "
       f"(n_keloid={len(keloid_all)}, n_d30={len(d30_all)}): "
       f"U={u_stat}, p={u_p:.3g}")

# Mann-Whitney: keloid vs wound D7
d7_all = ws[ws.condition=="Wound7"]["mf_score_mean"].values
u_stat2, u_p2 = stats.mannwhitneyu(keloid_all, d7_all, alternative="greater")
print(f"Mann-Whitney, MF score keloid > wound D7 "
       f"(n_keloid={len(keloid_all)}, n_d7={len(d7_all)}): "
       f"U={u_stat2}, p={u_p2:.3g}")

print(f"\nAll H5 figures saved to {FIG}")
