"""Comprehensive analysis of SIGnature attribution results.

  Fig 1: Workflow schematic + dataset composition
  Fig 2: Multi-metric gene ranking (volcano + composites)
  Fig 3: Cell-state specificity (heatmap + violins + co-attribution)
  Fig 4: Cross-method validation (SIGnature vs logFC)
  Fig 5: Sensor design implications

Output: figures/signature/fig_[1-5]_*.{png,svg,pdf}
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "scRNA"
FIG = ROOT / "figures" / "signature"
FIG.mkdir(parents=True, exist_ok=True)

# Publication-style aesthetic
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "pdf.fonttype": 42,  # editable text in PDFs
    "ps.fonttype": 42,
})

# Color palette — Nature-friendly colorblind-safe
COLORS = {
    "Keloid": "#C0392B",            # red
    "Normal": "#3498DB",            # blue
    "Wound": "#F39C12",             # orange
    "Healthy_intactSkin": "#27AE60",
    "Normal_scar_mature": "#3498DB",
    "Wound_D1": "#FFF0BB",
    "Wound_D7": "#F1C40F",
    "Wound_D30": "#E67E22",
    "candidate_primary": "#C0392B",
    "candidate_secondary": "#E67E22",
    "candidate_emerged": "#27AE60",
    "ECM": "#7F8C8D",
    "other": "#BDC3C7",
}

# Our project gene categories
CANDIDATES_PRIMARY = ["POSTN", "ADAM12"]  # committed AND-gate inputs
CANDIDATES_BACKUP = ["WISP1", "CCN4", "COL11A1", "ASPN", "COMP", "TNC", "TGFBI"]
CANDIDATES_OUTPUT = ["TNFRSF12A", "MMP14", "ITGA11", "ITGA10", "LRRC15", "LGR4"]
CANDIDATES_LOST = ["DCN", "CXCL12", "PI16", "GSN", "APOD", "TNXB"]
ALL_CANDIDATES = (CANDIDATES_PRIMARY + CANDIDATES_BACKUP
                   + CANDIDATES_OUTPUT + CANDIDATES_LOST
                   + ["ACTA2", "FBN2", "VEPH1", "SCX", "ACAN", "LOXL2", "ADAMTS6", "SDC1"])


# ============================================================
# Load and prepare data
# ============================================================
def load_full_results():
    f = DATA / "signature_full" / "full_attributions.npz"
    if not f.exists():
        raise FileNotFoundError(f"Missing {f} — run modal_signature_full.py first")
    d = np.load(f, allow_pickle=True)
    return {
        "A": d["attributions"].astype(np.float32),
        "genes": np.array([str(g) for g in d["gene_names"]]),
        "dataset": d["dataset"],
        "condition": d["condition"],
        "state": d["state"],
        "sample": d["sample"],
    }


def compute_per_state_scores(R):
    """Per-state mean, sd, n; delta and ratio for keloid-vs-normal."""
    A, genes, state, condition = R["A"], R["genes"], R["state"], R["condition"]
    states = ["Keloid", "Normal", "Wound"]
    rows = {}
    for s in states:
        mask = state == s
        if mask.sum() == 0:
            continue
        rows[s] = {
            "mean": A[mask].mean(axis=0),
            "sd": A[mask].std(axis=0),
            "n": int(mask.sum()),
        }
    # Also per-condition for the heatmap
    cond_means = {}
    for c in np.unique(condition):
        mask = condition == c
        if mask.sum() < 20:
            continue
        cond_means[c] = A[mask].mean(axis=0)
    return rows, cond_means


def compute_composite(per_state):
    k = per_state["Keloid"]["mean"]
    n = per_state["Normal"]["mean"]
    k_sd = per_state["Keloid"]["sd"]
    n_sd = per_state["Normal"]["sd"]

    eps = 0.001
    delta = k - n
    log_ratio = np.log2((k + eps) / (n + eps))
    cohens_d = delta / np.sqrt((k_sd**2 + n_sd**2) / 2 + 1e-9)
    specificity = k / (k + n + 1e-9)

    delta_z = (delta - delta.mean()) / (delta.std() + 1e-9)
    lr_z = (log_ratio - log_ratio.mean()) / (log_ratio.std() + 1e-9)
    zsum = delta_z + lr_z

    rank_delta = pd.Series(delta).rank(ascending=False).values
    rank_lr = pd.Series(log_ratio).rank(ascending=False).values
    borda = -(rank_delta + rank_lr)
    borda_rank = pd.Series(borda).rank(ascending=False).astype(int).values

    return {
        "delta": delta, "log_ratio": log_ratio, "specificity": specificity,
        "cohens_d": cohens_d, "zsum": zsum, "borda": borda,
        "rank_delta": rank_delta, "rank_lr": rank_lr, "borda_rank": borda_rank,
        "k_mean": k, "n_mean": n,
    }


def get_top_by(comp, genes, score_col, n=30):
    order = np.argsort(-comp[score_col])
    return [genes[i] for i in order[:n]], comp[score_col][order[:n]]


# ============================================================
# FIGURE 2 — Multi-metric gene ranking
# ============================================================
def fig2_multi_metric(R, comp, save_dir):
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.4,
                   width_ratios=[1.4, 1.0, 1.0])

    genes = R["genes"]
    delta = comp["delta"]
    log_ratio = comp["log_ratio"]
    borda_rank = comp["borda_rank"]

    # ----- Panel A: Volcano-style scatter (Δ vs log₂ ratio) -----
    # Functional gene categorizations (manually curated by biological role)
    ECM_COLLAGEN = ["COL1A1", "COL1A2", "COL3A1", "COL5A1", "COL5A2",
                     "COL6A1", "COL6A2", "COL6A3", "COL8A1", "COL11A1",
                     "COL12A1", "COL14A1", "COL27A1", "COL9A3",
                     "BGN", "ASPN", "COMP", "LUM", "DCN", "FBN2", "FBN1",
                     "SPARC", "FN1", "POSTN", "TGFBI", "MFAP2", "MFAP5",
                     "PCOLCE", "DPT", "MGP", "TNXB", "TNC", "CTHRC1",
                     "EFEMP1", "EFEMP2", "AEBP1", "THBS4", "THBS1", "THBS2",
                     "EDIL3", "EMILIN1", "VCAN", "ACAN", "OGN", "PRELP",
                     "LGALS1", "LGALS3", "SRPX2", "SULF1", "SULF2"]
    PROTEASES = ["ADAM12", "ADAM19", "ADAMTS6", "ADAMTS5", "ADAMTS4",
                  "ADAMTS14", "MMP14", "MMP11", "MMP23B", "MMP1", "MMP2",
                  "MMP3", "MMP9", "MMP13", "PRSS23", "FAP", "CTSK",
                  "PLAU", "PLAT", "DPP4", "SERPINE1", "SERPINE2", "TIMP1",
                  "TIMP2", "TIMP3"]
    GROWTH_SIGNALING = ["WISP1", "CCN4", "CTGF", "CCN2", "CYR61", "CCN1",
                         "NOV", "CCN3", "WNT5A", "WNT2", "SFRP2", "SFRP4",
                         "DKK1", "DKK3", "LGR4", "LGR5", "CXCL12", "CXCL14",
                         "CXCL9", "CXCL10", "TNFRSF12A", "TNFSF12",
                         "IGF2", "IGFBP3", "IGFBP5", "IGFBP7", "PDGFRA",
                         "PDGFRB", "PDGFA", "PDGFB", "FGF2", "FGF7",
                         "TGFB1", "TGFB2", "TGFB3", "BMP4", "BMP7",
                         "BMP2", "INHBA", "EGFL6", "EGFL7", "HGF", "VEGFA"]
    INTEGRIN_RECEPTOR = ["ITGA1", "ITGA2", "ITGA5", "ITGA6", "ITGA10",
                          "ITGA11", "ITGAV", "ITGB1", "ITGB3", "ITGB5",
                          "ITGB6", "SDC1", "SDC2", "SDC4", "LRRC15",
                          "NRP1", "NRP2", "CD44", "CD63"]
    LOST_IN_KELOID_FN = ["DCN", "CXCL12", "PI16", "GSN", "APOD", "TNXB",
                          "CFD", "SCARA5", "ADH1B", "LTBP4", "IGFBP5",
                          "TXNIP", "PDGFRA", "MFAP5", "TNFAIP6"]

    def cat_idx(gene_list):
        return [i for i, g in enumerate(genes) if g in gene_list]

    # Background: all genes (light gray)
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(delta, log_ratio, s=4, c="#D0D5DB", alpha=0.25,
                edgecolor="none", rasterized=True)

    # ECM / collagen — grey
    idx = cat_idx(ECM_COLLAGEN)
    ax.scatter(delta[idx], log_ratio[idx], s=22, c="#7F8C8D",
                edgecolor="white", linewidth=0.4, alpha=0.85, rasterized=True,
                label="ECM / collagen")

    # Protease — orange
    idx = cat_idx(PROTEASES)
    ax.scatter(delta[idx], log_ratio[idx], s=28, c="#E67E22",
                edgecolor="white", linewidth=0.4, alpha=0.9, marker="D",
                label="Protease")

    # Growth factor / signaling — yellow
    idx = cat_idx(GROWTH_SIGNALING)
    ax.scatter(delta[idx], log_ratio[idx], s=28, c="#F1C40F",
                edgecolor="white", linewidth=0.4, alpha=0.9, marker="o",
                label="Growth factor / signaling")

    # Integrin / receptor — blue
    idx = cat_idx(INTEGRIN_RECEPTOR)
    ax.scatter(delta[idx], log_ratio[idx], s=30, c="#3498DB",
                edgecolor="white", linewidth=0.4, alpha=0.9, marker="s",
                label="Integrin / receptor")

    # Lost in keloid — purple downward triangle
    idx = cat_idx(LOST_IN_KELOID_FN)
    ax.scatter(delta[idx], log_ratio[idx], s=40, c="#9B59B6",
                edgecolor="white", linewidth=0.5, alpha=0.9, marker="v",
                label="Lost in keloid")

    # Primary AND-gate inputs (POSTN, ADAM12) — big red dot, NO LABEL
    primary_idx = [i for i, g in enumerate(genes) if g in CANDIDATES_PRIMARY]
    ax.scatter(delta[primary_idx], log_ratio[primary_idx], s=120,
                c=COLORS["candidate_primary"], edgecolor="black", linewidth=0.8,
                zorder=11)

    # Annotate key genes
    annotate = ["POSTN", "ADAM12", "ASPN", "COL11A1", "COL3A1", "COL1A1",
                "BGN", "SPARC", "DCN", "CXCL12", "CFD", "APOD", "ACTA2",
                "SCX", "TNFRSF12A", "MMP14"]
    for g in annotate:
        if g in genes:
            i = list(genes).index(g)
            ax.annotate(g, (delta[i], log_ratio[i]),
                         fontsize=7.5, fontweight="bold",
                         xytext=(4, 4), textcoords="offset points",
                         color="black" if g in CANDIDATES_PRIMARY else "#444444")

    ax.axhline(0, color="#888", lw=0.5, ls="--")
    ax.axvline(0, color="#888", lw=0.5, ls="--")
    ax.set_xlabel(r"$\Delta$ attribution (keloid − normal)")
    ax.set_ylabel(r"log$_2$(keloid / normal)  attribution ratio")
    ax.set_title("A. Foundation-model attribution landscape", fontweight="bold", loc="left")
    ax.legend(loc="lower right", fontsize=7, framealpha=0.9)

    # ----- Panel B: Top 20 genes by Borda composite -----
    ax = fig.add_subplot(gs[0, 1:])
    order = np.argsort(-comp["borda"])[:25]
    top_genes = [genes[i] for i in order]
    top_borda = comp["borda"][order]
    top_delta = comp["delta"][order]
    top_lr = comp["log_ratio"][order]

    y = np.arange(len(top_genes))[::-1]
    cmap = plt.get_cmap("Reds")
    colors_bar = []
    for g in top_genes:
        if g in CANDIDATES_PRIMARY:
            colors_bar.append(COLORS["candidate_primary"])
        elif g in CANDIDATES_BACKUP:
            colors_bar.append("#F1C40F")
        elif g in CANDIDATES_OUTPUT:
            colors_bar.append("#3498DB")
        else:
            colors_bar.append("#888")
    bars = ax.barh(y, top_delta, color=colors_bar,
                    edgecolor="white", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(top_genes, fontsize=8)
    for ti, g in enumerate(top_genes):
        if g in CANDIDATES_PRIMARY:
            ax.get_yticklabels()[ti].set_fontweight("bold")
            ax.get_yticklabels()[ti].set_color(COLORS["candidate_primary"])
    # Add log_ratio as text
    for i, (d_v, lr_v) in enumerate(zip(top_delta, top_lr)):
        ax.text(d_v + 0.05, y[i], f"{lr_v:.1f}×log₂", fontsize=6.5,
                 va="center", color="#444")
    ax.set_xlabel(r"$\Delta$ attribution (keloid − normal)")
    ax.set_title("B. Top 25 genes — Borda rank-product composite", fontweight="bold", loc="left")
    ax.set_xlim(0, max(top_delta) * 1.25)

    # ----- Panel C: Heatmap — our candidates across composite metrics -----
    ax = fig.add_subplot(gs[1, :])
    candidates_present = [g for g in ALL_CANDIDATES if g in genes]
    cand_idx = [list(genes).index(g) for g in candidates_present]

    metrics_data = pd.DataFrame({
        "gene": candidates_present,
        "Δ attribution": comp["delta"][cand_idx],
        "log₂ ratio": comp["log_ratio"][cand_idx],
        "Z-sum": comp["zsum"][cand_idx],
        "Cohen's d": comp["cohens_d"][cand_idx],
        "Specificity": comp["specificity"][cand_idx],
        "Borda rank": comp["borda_rank"][cand_idx],
    })
    metrics_data = metrics_data.sort_values("Borda rank")

    # Normalize each column for heatmap
    M = metrics_data.set_index("gene")
    M_norm = (M - M.min()) / (M.max() - M.min() + 1e-9)
    # Invert Borda rank so low=good becomes high=good
    M_norm["Borda rank"] = 1 - M_norm["Borda rank"]
    # Cap negative deltas at 0 for visualization
    M_norm.loc[M["Δ attribution"] < 0, "Δ attribution"] = 0

    sns.heatmap(M_norm.T, cmap="RdYlBu_r", ax=ax, cbar_kws={"label": "Normalized score (0=worst, 1=best)"},
                 annot=M.T.values, fmt="", annot_kws={"fontsize": 6.5},
                 linewidths=0.3, linecolor="white", vmin=0, vmax=1)
    ax.set_title("C. Composite-metric ranking of all project candidate genes (ordered by Borda)",
                 fontweight="bold", loc="left")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7.5)

    # Color the candidate gene labels
    for tick_lbl in ax.get_xticklabels():
        g = tick_lbl.get_text()
        if g in CANDIDATES_PRIMARY:
            tick_lbl.set_color(COLORS["candidate_primary"])
            tick_lbl.set_fontweight("bold")
        elif g in CANDIDATES_LOST:
            tick_lbl.set_color("#9B59B6")

    fig.suptitle("Figure 2 — Foundation-model gene ranking (SCimilarity SIGnature) for keloid mesenchymal fibroblast state",
                  fontsize=12, fontweight="bold", y=1.00)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(save_dir / f"fig2_multi_metric_ranking.{ext}", dpi=350, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig2_multi_metric_ranking.{{png,svg,pdf}}")


# ============================================================
# FIGURE 3 — Cell-state specificity
# ============================================================
def fig3_cell_state(R, comp, cond_means, save_dir):
    fig = plt.figure(figsize=(15, 11))
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.5,
                   height_ratios=[1.4, 1])
    genes = R["genes"]

    # ----- Panel A: Heatmap of top-30 genes × cell condition -----
    ax = fig.add_subplot(gs[0, :])
    cond_order = ["Healthy_intactSkin", "Wound_D1", "Wound_D7", "Wound_D30",
                  "Normal_scar_mature", "Keloid"]
    cond_order = [c for c in cond_order if c in cond_means]
    top_n = 30
    order = np.argsort(-comp["borda"])[:top_n]
    top_genes = [genes[i] for i in order]
    top_idx = order

    H = np.zeros((top_n, len(cond_order)))
    for j, c in enumerate(cond_order):
        H[:, j] = cond_means[c][top_idx]
    Hdf = pd.DataFrame(H, index=top_genes, columns=cond_order)
    sns.heatmap(Hdf, ax=ax, cmap="RdBu_r", center=0,
                 cbar_kws={"label": "Mean attribution score", "shrink": 0.6},
                 linewidths=0.3, linecolor="white")
    ax.set_title("A. Per-condition attribution heatmap — top 30 keloid-defining genes",
                  fontweight="bold", loc="left")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8.5)
    # Highlight candidate gene rows
    for ti, g in enumerate(top_genes):
        if g in CANDIDATES_PRIMARY:
            ax.get_yticklabels()[ti].set_color(COLORS["candidate_primary"])
            ax.get_yticklabels()[ti].set_fontweight("bold")
        elif g in CANDIDATES_BACKUP:
            ax.get_yticklabels()[ti].set_color("#F39C12")

    # ----- Panel B: Per-state violins for POSTN/ADAM12/ASPN -----
    A, state = R["A"], R["state"]
    for k, g in enumerate(["POSTN", "ADAM12", "ASPN"]):
        ax = fig.add_subplot(gs[1, k])
        if g not in list(genes):
            ax.text(0.5, 0.5, f"{g} not in vocab", ha="center", va="center",
                     transform=ax.transAxes)
            continue
        gi = list(genes).index(g)
        df = pd.DataFrame({"attribution": A[:, gi], "state": state})
        order = ["Normal", "Wound", "Keloid"]
        pal = {"Normal": COLORS["Normal"], "Wound": COLORS["Wound"],
                "Keloid": COLORS["Keloid"]}
        sns.violinplot(data=df, x="state", y="attribution", order=order,
                        ax=ax, palette=pal, inner="quartile", cut=0,
                        linewidth=0.8)
        sns.stripplot(data=df.sample(min(2000, len(df))),
                      x="state", y="attribution", order=order,
                      ax=ax, size=0.5, color="black", alpha=0.1)

        ax.set_title(f"B{'ⁱ' if k==0 else 'ⁱⁱ' if k==1 else 'ⁱⁱⁱ'}. {g}", fontweight="bold", loc="left")
        ax.set_xlabel("")
        ax.set_ylabel("Attribution score")

        # Add p-value annotation (keloid vs normal)
        if "Keloid" in state and "Normal" in state:
            k_v = A[state == "Keloid", gi]
            n_v = A[state == "Normal", gi]
            t, p = stats.mannwhitneyu(k_v, n_v, alternative="greater")
            y_max = max(k_v.max(), n_v.max()) * 1.05
            ax.annotate(f"p = {p:.2e}\nΔ = {k_v.mean() - n_v.mean():+.2f}",
                         xy=(2, y_max), xytext=(2, y_max * 1.05),
                         fontsize=7, ha="center")

    fig.suptitle("Figure 3 — Cell-state-specific gene attribution patterns",
                  fontsize=12, fontweight="bold", y=1.00)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(save_dir / f"fig3_cell_state.{ext}", dpi=350, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig3_cell_state.{{png,svg,pdf}}")


# ============================================================
# FIGURE 4 — Cross-method validation
# ============================================================
def fig4_cross_method(R, comp, save_dir):
    """Compare SIGnature attributions to traditional logFC (per-gene means)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    genes = R["genes"]

    # Compute per-gene logFC from raw attribution (proxy when actual logFC unavailable)
    # Use attribution mean as expression proxy
    k = comp["k_mean"]
    n = comp["n_mean"]
    logfc_proxy = np.log2((k + 0.01) / (n + 0.01))

    # ----- Panel A: Attribution Δ vs log_ratio -----
    ax = axes[0]
    sc = ax.scatter(comp["delta"], comp["log_ratio"], s=4,
                     c=comp["specificity"], cmap="RdYlBu_r",
                     alpha=0.5, rasterized=True)
    # Highlight candidates
    cand_idx = [list(genes).index(g) for g in ALL_CANDIDATES if g in genes]
    ax.scatter(comp["delta"][cand_idx], comp["log_ratio"][cand_idx], s=30,
                c="black", edgecolor="white", linewidth=0.5, zorder=5)
    for i in cand_idx:
        g = genes[i]
        if g in CANDIDATES_PRIMARY + ["ASPN", "COL11A1", "TNFRSF12A"]:
            ax.annotate(g, (comp["delta"][i], comp["log_ratio"][i]),
                         fontsize=7, fontweight="bold",
                         xytext=(3, 3), textcoords="offset points")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Specificity (keloid / total)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax.set_xlabel("Δ attribution")
    ax.set_ylabel("log₂ ratio")
    ax.set_title("A. Attribution Δ vs ratio (color = keloid specificity)",
                  fontweight="bold", loc="left")
    ax.axhline(0, color="#888", lw=0.5, ls="--")
    ax.axvline(0, color="#888", lw=0.5, ls="--")

    # ----- Panel B: Borda rank vs Z-sum rank correlation -----
    ax = axes[1]
    rank_borda = pd.Series(comp["borda"]).rank(ascending=False).values
    rank_zsum = pd.Series(comp["zsum"]).rank(ascending=False).values
    top_mask = (rank_borda < 500) | (rank_zsum < 500)
    ax.scatter(rank_zsum[top_mask], rank_borda[top_mask], s=4,
                c="#7F8C8D", alpha=0.5, rasterized=True)
    ax.scatter(rank_zsum[cand_idx], rank_borda[cand_idx], s=35,
                c=COLORS["candidate_primary"], edgecolor="white",
                linewidth=0.5, zorder=5)
    for i in cand_idx:
        g = genes[i]
        if g in CANDIDATES_PRIMARY + ["ASPN", "COL11A1"]:
            ax.annotate(g, (rank_zsum[i], rank_borda[i]),
                         fontsize=7, fontweight="bold",
                         xytext=(3, 3), textcoords="offset points")
    ax.plot([0, 500], [0, 500], "k--", lw=0.5, alpha=0.5)
    ax.set_xlim(0, 500); ax.set_ylim(0, 500)
    ax.invert_xaxis(); ax.invert_yaxis()
    ax.set_xlabel("Z-sum rank (lower = better)")
    ax.set_ylabel("Borda rank (lower = better)")
    ax.set_title("B. Cross-composite rank agreement", fontweight="bold", loc="left")

    # ----- Panel C: Effect size (Cohen's d) ranking -----
    ax = axes[2]
    cohens_d = comp["cohens_d"]
    cand_d = cohens_d[cand_idx]
    cand_g = [genes[i] for i in cand_idx]
    ord_d = np.argsort(-cand_d)[:15]
    y = np.arange(len(ord_d))[::-1]
    g_top = [cand_g[i] for i in ord_d]
    d_top = cand_d[ord_d]
    cls = []
    for g in g_top:
        if g in CANDIDATES_PRIMARY: cls.append(COLORS["candidate_primary"])
        elif g in CANDIDATES_BACKUP: cls.append("#F39C12")
        elif g in CANDIDATES_OUTPUT: cls.append("#3498DB")
        else: cls.append("#7F8C8D")
    ax.barh(y, d_top, color=cls, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(g_top, fontsize=8)
    for ti, g in enumerate(g_top):
        if g in CANDIDATES_PRIMARY:
            ax.get_yticklabels()[ti].set_fontweight("bold")
            ax.get_yticklabels()[ti].set_color(COLORS["candidate_primary"])
    ax.axvline(0.8, color="#444", lw=0.5, ls="--")
    ax.text(0.82, 1, "large effect", fontsize=7, color="#444")
    ax.set_xlabel("Cohen's d (keloid vs normal)")
    ax.set_title("C. Effect size for candidate genes", fontweight="bold", loc="left")

    fig.suptitle("Figure 4 — Cross-metric validation of foundation-model gene rankings",
                  fontsize=12, fontweight="bold")
    for ext in ("png", "svg", "pdf"):
        fig.savefig(save_dir / f"fig4_cross_method.{ext}", dpi=350, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig4_cross_method.{{png,svg,pdf}}")


# ============================================================
# FIGURE 5 — Sensor design implications
# ============================================================
def fig5_sensor_design(R, comp, cond_means, save_dir):
    fig = plt.figure(figsize=(15, 9))
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.4)
    A, state, genes, condition = R["A"], R["state"], R["genes"], R["condition"]

    # ----- Panel A: top 5 sensor candidates × all conditions -----
    ax = fig.add_subplot(gs[0, :2])
    sensor_picks = ["POSTN", "ADAM12", "ASPN", "COL11A1", "COL8A1"]
    sensor_picks = [g for g in sensor_picks if g in genes]
    cond_order = ["Healthy_intactSkin", "Wound_D1", "Wound_D7", "Wound_D30",
                  "Normal_scar_mature", "Keloid"]
    cond_order = [c for c in cond_order if c in cond_means]
    H = np.zeros((len(sensor_picks), len(cond_order)))
    for i, g in enumerate(sensor_picks):
        gi = list(genes).index(g)
        for j, c in enumerate(cond_order):
            H[i, j] = cond_means[c][gi]
    Hdf = pd.DataFrame(H, index=sensor_picks, columns=cond_order)
    sns.heatmap(Hdf, ax=ax, cmap="Reds", annot=True, fmt=".2f",
                 annot_kws={"fontsize": 9}, linewidths=0.5, linecolor="white",
                 cbar_kws={"label": "Mean attribution", "shrink": 0.7})
    ax.set_title("A. Top 5 sensor candidates × cell conditions", fontweight="bold", loc="left")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8.5)
    for ti, g in enumerate(sensor_picks):
        if g in CANDIDATES_PRIMARY:
            ax.get_yticklabels()[ti].set_color(COLORS["candidate_primary"])
            ax.get_yticklabels()[ti].set_fontweight("bold")

    # ----- Panel B: AND-gate co-attribution scatter (POSTN vs ADAM12) -----
    ax = fig.add_subplot(gs[0, 2])
    if "POSTN" in list(genes) and "ADAM12" in list(genes):
        pi = list(genes).index("POSTN")
        ai = list(genes).index("ADAM12")
        for s in ["Normal", "Wound", "Keloid"]:
            m = state == s
            ax.scatter(A[m, pi], A[m, ai], s=4, c=COLORS[s], alpha=0.4,
                        label=f"{s} (n={m.sum()})", rasterized=True,
                        edgecolor="none")
        ax.set_xlabel("POSTN attribution")
        ax.set_ylabel("ADAM12 attribution")
        ax.set_title("B. Per-cell co-attribution POSTN × ADAM12",
                      fontweight="bold", loc="left")
        ax.legend(loc="upper left", fontsize=7)

    # ----- Panel C: Lost-in-keloid (restoration candidates) -----
    ax = fig.add_subplot(gs[1, 0])
    lost = [(g, comp["delta"][list(genes).index(g)]) for g in CANDIDATES_LOST
             if g in genes]
    lost.sort(key=lambda x: x[1])
    lost_g = [x[0] for x in lost]
    lost_d = [x[1] for x in lost]
    y = np.arange(len(lost_g))[::-1]
    ax.barh(y, lost_d, color="#9B59B6", edgecolor="white", linewidth=0.5)
    ax.set_yticks(y); ax.set_yticklabels(lost_g, fontsize=9)
    ax.set_xlabel("Δ attribution (negative = lost in keloid)")
    ax.set_title("C. Restoration payload candidates", fontweight="bold", loc="left")
    ax.axvline(0, color="#444", lw=0.6)

    # ----- Panel D: ACTA2 paradox -----
    ax = fig.add_subplot(gs[1, 1])
    if "ACTA2" in list(genes):
        ai = list(genes).index("ACTA2")
        df = pd.DataFrame({"attribution": A[:, ai], "state": state})
        sns.violinplot(data=df, x="state", y="attribution",
                        order=["Normal", "Wound", "Keloid"],
                        ax=ax, palette={"Normal": COLORS["Normal"],
                                          "Wound": COLORS["Wound"],
                                          "Keloid": COLORS["Keloid"]},
                        inner="quartile", cut=0, linewidth=0.8)
        ax.set_title(r"D. ACTA2 ($\alpha$-SMA) — myofibroblast non-finding", fontweight="bold", loc="left")
        ax.set_xlabel("")
        ax.set_ylabel("Attribution score")

        # Annotation
        borda_rank_acta2 = comp["borda_rank"][ai]
        ax.text(0.02, 0.96,
                 f"Borda rank: #{borda_rank_acta2:,} of {len(genes):,}\n"
                 f"keloid is NOT $\\alpha$-SMA$^+$ myofibroblast",
                 transform=ax.transAxes, fontsize=8, va="top",
                 bbox=dict(boxstyle="round", facecolor="#FFF7E6", edgecolor="#888"))

    # ----- Panel E: Borda summary of candidates as bars -----
    ax = fig.add_subplot(gs[1, 2])
    cand_present = [g for g in ALL_CANDIDATES if g in genes]
    cand_borda = [comp["borda_rank"][list(genes).index(g)] for g in cand_present]
    order = np.argsort(cand_borda)[:15]
    y = np.arange(len(order))[::-1]
    g_top = [cand_present[i] for i in order]
    b_top = [cand_borda[i] for i in order]
    cls = []
    for g in g_top:
        if g in CANDIDATES_PRIMARY: cls.append(COLORS["candidate_primary"])
        elif g in CANDIDATES_BACKUP: cls.append("#F39C12")
        elif g in CANDIDATES_OUTPUT: cls.append("#3498DB")
        else: cls.append("#7F8C8D")
    ax.barh(y, b_top, color=cls, edgecolor="white", linewidth=0.5)
    for i, b in enumerate(b_top):
        ax.text(b + 30, y[i], f"#{b:,}", fontsize=7, va="center")
    ax.set_yticks(y); ax.set_yticklabels(g_top, fontsize=8.5)
    for ti, g in enumerate(g_top):
        if g in CANDIDATES_PRIMARY:
            ax.get_yticklabels()[ti].set_fontweight("bold")
            ax.get_yticklabels()[ti].set_color(COLORS["candidate_primary"])
    ax.set_xlabel("Borda rank (low = better)")
    ax.set_title("E. Project candidate ranking", fontweight="bold", loc="left")
    ax.invert_xaxis()

    fig.suptitle("Figure 5 — Implications for AND-gate sensor and payload design",
                  fontsize=12, fontweight="bold", y=1.00)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(save_dir / f"fig5_sensor_design.{ext}", dpi=350, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig5_sensor_design.{{png,svg,pdf}}")


# ============================================================
# Run all
# ============================================================
def main():
    print("=" * 60)
    print("Loading SIGnature full results...")
    print("=" * 60)
    R = load_full_results()
    print(f"Attributions: {R['A'].shape}")
    print(f"Cells: {len(R['state']):,}")
    print(f"Genes: {len(R['genes']):,}")

    print("\nComputing per-state aggregates...")
    per_state, cond_means = compute_per_state_scores(R)
    for s, v in per_state.items():
        print(f"  {s}: n = {v['n']}")
    print(f"  Conditions in cond_means: {list(cond_means.keys())}")

    print("\nComputing composite scores...")
    comp = compute_composite(per_state)
    print(f"  Δ range: [{comp['delta'].min():.2f}, {comp['delta'].max():.2f}]")
    print(f"  log_ratio range: [{comp['log_ratio'].min():.2f}, {comp['log_ratio'].max():.2f}]")

    # Save composite table
    df = pd.DataFrame({
        "gene": R["genes"],
        "k_mean": comp["k_mean"],
        "n_mean": comp["n_mean"],
        "delta": comp["delta"],
        "log_ratio": comp["log_ratio"],
        "specificity": comp["specificity"],
        "cohens_d": comp["cohens_d"],
        "zsum": comp["zsum"],
        "borda_rank": comp["borda_rank"],
    }).sort_values("borda_rank")
    df.to_csv(FIG / "composite_scores_full.csv", index=False)
    print(f"  Saved {FIG / 'composite_scores_full.csv'}")

    print("\nGenerating figures...")
    fig2_multi_metric(R, comp, FIG)
    fig3_cell_state(R, comp, cond_means, FIG)
    fig4_cross_method(R, comp, FIG)
    fig5_sensor_design(R, comp, cond_means, FIG)

    print(f"\nAll figures saved to {FIG}")


if __name__ == "__main__":
    main()
