# Keloid scRNA-seq Analysis

Single-cell RNA-seq analysis of human keloid dermal fibroblasts across three published datasets, used to identify cell-state-defining transcriptional inputs for a programmable RNA-sensor therapeutic strategy.

This repository contains the analysis pipeline (loaders, differential expression, cross-dataset integration, foundation-model attribution scoring, cell-cell communication, pathway/TF activity inference, hypothesis tests) and the resulting figures and tables.

---

## Datasets

All three datasets are publicly available on GEO and were downloaded directly into `data/raw/{GSE accession}/`.

| GEO Accession | Study | Cells (fibroblast subset) | Conditions | Platform |
|---|---|---:|---|---|
| **GSE163973** | Deng et al, *Cell Discovery* 2021 — *Single-cell RNA-seq reveals fibroblast heterogeneity and increased mesenchymal fibroblasts in human fibrotic skin diseases* | 12,177 | 3 keloid + 3 normal scar biopsies | 10x Genomics Chromium |
| **GSE181316** | Direder et al, *Matrix Biology* 2022 — keloid scRNA-seq atlas | 37,713 | 4 keloid + 3 normal scar + 1 healthy skin | 10x Genomics Chromium |
| **GSE241132** | Liu and Landén 2024 — human wound healing time course | 12,259 | Intact skin / Wound D1 / Wound D7 / Wound D30 (n=3 donors) | 10x Genomics Chromium |
| **GSE188952** | Onoufriadis et al — bulk RNA-seq (used for sc-vs-bulk concordance check) | (bulk) | 4 keloid + 5 hypertrophic + 3 normotrophic scar | Bulk RNA-seq |

### Download

Raw matrices are not committed to this repository (size). To reproduce the analysis end-to-end:

```bash
mkdir -p data/raw
cd data/raw

# Deng et al GSE163973
wget -O GSE163973_RAW.tar 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE163973&format=file'
mkdir GSE163973 && tar -xf GSE163973_RAW.tar -C GSE163973

# Direder et al GSE181316
wget -O GSE181316_RAW.tar 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE181316&format=file'
mkdir GSE181316 && tar -xf GSE181316_RAW.tar -C GSE181316

# Liu/Landén GSE241132
wget -O GSE241132_RAW.tar 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE241132&format=file'
mkdir GSE241132 && tar -xf GSE241132_RAW.tar -C GSE241132

# Onoufriadis bulk GSE188952
wget -O GSE188952_Processed_FPKM.tsv.gz 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE188952&format=file&file=GSE188952_Processed_FPKM.tsv.gz'
```

Total raw download: approximately 3.3 GB.

---

## Repository layout

```
.
├── scripts/                  # analysis pipeline
│   ├── load_deng_scrna.py            # GSE163973 loader  → deng_fibroblasts.h5ad
│   ├── load_direder_scrna.py         # GSE181316 loader  → direder_*.h5ad
│   ├── load_wound_scrna.py           # GSE241132 loader  → wound_*.h5ad
│   ├── analyze_deng_scrna.py         # Deng DE + figures
│   ├── analyze_gwas.py               # keloid GWAS × scRNA intersection
│   ├── cross_dataset_analysis.py     # cross-study integration
│   ├── build_pooled_fibroblasts.py   # build unified pooled fibroblast AnnData
│   ├── make_scrna_figures.py         # primary scRNA figures (s1–s7)
│   ├── make_cross_dataset_figures.py # cross-study figures (c1–c6)
│   ├── make_extensive_figures.py     # extended volcano / forest / time-course figures (v1–v7)
│   ├── test_h5_mesenchymal_persistence.py  # hypothesis test: does MFB program resolve by D30?
│   ├── sc_and_gate_coexpression.py   # per-cell POSTN ∩ ADAM12 hit rate
│   ├── sc_hallmark_scoring.py        # MSigDB Hallmark signature scoring
│   ├── sc_decoupler_pathway.py       # PROGENy + CollecTRI per-cell activity
│   ├── sc_paga_trajectory.py         # integrated UMAP + PAGA trajectory
│   ├── sc_liana_cellcell.py          # LIANA cell-cell communication
│   ├── signature_full_analysis.py    # SIGnature attribution composite scoring + figures
│   └── modal_signature_full.py       # Modal cloud-GPU SIGnature attribution computation
│
├── data/
│   ├── raw/                  # not committed — download from GEO (see above)
│   └── processed/scRNA/      # small CSVs committed; .h5ad files regenerable from raw
│
└── figures/
    ├── scrna/                # primary scRNA figures (Deng analysis)
    ├── cross/                # cross-dataset comparison figures
    ├── stats/                # extended statistical figures (volcano, forest, etc.)
    ├── hypothesis/           # H5 hypothesis test outputs
    ├── sc_extended/          # AND-gate, hallmark, PROGENy, LIANA, PAGA outputs
    ├── signature/            # SIGnature foundation-model attribution outputs
    └── gwas/                 # GWAS × scRNA intersection
```

---

## Reproducing the analysis

### 1. Environment

Python 3.11+ required. Install dependencies:

```bash
pip install -r requirements.txt
```

Additional packages installed during analysis:

```bash
pip install scanpy==1.10.4 anndata==0.10.9 decoupler==2.1.6 liana==1.7.1 \
            omnipath==1.0.12 gseapy==1.2.1 bbknn==1.6.0 statannotations \
            sc-signature==1.0.0
```

### 2. Load datasets

After downloading the raw data above:

```bash
python scripts/load_deng_scrna.py
python scripts/load_direder_scrna.py
python scripts/load_wound_scrna.py
```

This produces processed AnnData objects in `data/processed/scRNA/`.

### 3. Run analyses

```bash
# Deng-only differential expression and initial figures
python scripts/analyze_deng_scrna.py
python scripts/make_scrna_figures.py

# Cross-dataset integration and figures
python scripts/cross_dataset_analysis.py
python scripts/make_cross_dataset_figures.py
python scripts/make_extensive_figures.py

# Build pooled three-dataset fibroblast object
python scripts/build_pooled_fibroblasts.py

# Hypothesis tests and extended analyses
python scripts/test_h5_mesenchymal_persistence.py
python scripts/sc_and_gate_coexpression.py
python scripts/sc_hallmark_scoring.py
python scripts/sc_decoupler_pathway.py
python scripts/sc_paga_trajectory.py
python scripts/sc_liana_cellcell.py

# GWAS × scRNA
python scripts/analyze_gwas.py
```

### 4. Foundation-model attribution scoring (SIGnature)

The SIGnature analysis uses the SCimilarity foundation model (Heimberg et al, *Nat Methods* 2024) interpreted via the SIGnature framework (Gold et al, *Nat Biotechnology* 2026). Compute is run on Modal cloud (A10G GPU); the SCimilarity model (~30 GB) is cached to a Modal Volume on first run.

```bash
# Run full attribution computation on Modal cloud
modal run scripts/modal_signature_full.py

# Local downstream analysis and figure generation
python scripts/signature_full_analysis.py
```

---

## Key outputs

| Output | Path | Description |
|---|---|---|
| AND-gate per-cell hit rate | `figures/sc_extended/fig_andgate_per_cell.{png,svg}` | Fraction of cells co-expressing POSTN and ADAM12 across all conditions |
| Hallmark signature scores | `figures/sc_extended/fig_hallmark_heatmap.{png,svg}` | MSigDB Hallmark scores per condition |
| PROGENy pathway activity | `figures/sc_extended/fig_progeny_heatmap.{png,svg}` | Per-cell pathway activity inference |
| CollecTRI TF activity | `figures/sc_extended/fig_dorothea_heatmap.{png,svg}` | Per-cell transcription factor activity |
| LIANA ligand-receptor pairs | `figures/sc_extended/fig_liana_*.{png,svg}` | Cell-cell communication, ranked LR pairs |
| PAGA trajectory | `figures/sc_extended/fig_paga_*.{png,svg}` | Integrated cross-dataset trajectory |
| H5 hypothesis test | `figures/hypothesis/fig_h5_*.{png,svg}` | Mesenchymal-fibroblast persistence across timepoints |
| SIGnature composite ranking | `figures/signature/composite_scores_full.csv` | Foundation-model gene importance scores (28,231 genes) |
| SIGnature figures | `figures/signature/fig{2,3,4,5}_*.{png,svg,pdf}` | Attribution landscape, cell-state heatmaps, cross-method validation, sensor design |

---

## Software dependencies

Core: scanpy, anndata, numpy, pandas, scipy, scikit-learn, matplotlib, seaborn

Extended analyses: bbknn (batch correction), decoupler-py (PROGENy + CollecTRI), liana (cell-cell communication), gseapy (Hallmark signatures), statannotations (statistical annotation)

Foundation models: sc-signature, scimilarity (run via Modal cloud A10G)

---

## Citation

If you use this analysis pipeline, please cite the underlying datasets:

- Deng C-C, Hu Y-F, Zhu D-H, et al. *Nat Commun* 2021; 12:3709.
- Direder M, Wielscher M, Weiss S, et al. *Front Immunol* 2022; 13:940645.
- Liu Y, Landén NX, et al. GSE241132.
- Onoufriadis A, et al. GSE188952.

And the methods used:

- Gold MJ et al. SIGnature. *Nat Biotechnology* 2026.
- Heimberg G et al. SCimilarity. *Nat Methods* 2024.
- Wolf FA et al. PAGA. *Genome Biol* 2019.
- Badia-i-Mompel P et al. decoupler-py. *Bioinformatics* 2022.
- Türei D et al. LIANA. *Nat Cell Biol* 2024.

---

## License

MIT
