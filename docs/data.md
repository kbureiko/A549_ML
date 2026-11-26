# Data: what the code reads and where it has to be

Nothing listed here is part of the repository. Put the files anywhere on your computer and name the folders
in `paths.toml`. Names in `code font` are file or folder names that the scripts look for.

## 1. Inputs of the classifiers — `[inputs] ml_inputs`

| File | Format | Content |
|---|---|---|
| `202208_prematrix.csv` | tab-separated | one row per enhancer: `chr`, `start`, `end`; the PRO-seq dex-vs-control statistics (`ctrl vs. dex Log2 Fold Change`, `ctrl vs. dex p-value`, `ctrl vs. dex adj. p-value`); one column per ENCODE signal track, named `<target>_<treatment>_<dose>_<duration>_<ENCODE file accession>` |
| `atac_matrix.csv` | tab-separated, row index | `ATAC_ctrl`, `ATAC_dex_1h` |
| `matrix_motifs.csv` | tab-separated, row index | one column per HOMER motif (score; > 0 is counted as motif present); used when `MOTIF_SET = "original"` |
| `matrix_motif_groups.tsv` | tab-separated, row index, `enhancer_id` | written by `scripts/21_motif_groups.py` from the HOMER tables of `scripts/18`: `motif::<group>` (representative motif of each group of correlated motifs, present / absent) and `motifscore::GRE(NR),IR3` (best GRE score); matched on `enhancer_id` |
| `matrix_neighbourhood.tsv` | tab-separated, row index, `enhancer_id` | written by `scripts/19_neighbourhood_features.py`: other enhancers and GR peaks around each enhancer; matched on `enhancer_id` |
| `20220908_annotation.csv` | tab-separated, row index | genomic annotation of the enhancer (first column is used) |
| `20220919_annotation.csv` | tab-separated, row index | `Distance to TSS` |
| `20220919_gc_content.csv` | `;`-separated, row index | `5_pct_gc` |
| `ENCFF395FIP.bed.gz` | narrowPeak | GR (NR3C1) ChIP-seq peaks from ENCODE, used for the GR-bound annotation |

**The six tables must list the enhancers in the same row order.** They are combined by row position, as in
the published analysis; the enhancer ID `chr_start_end` is built from the first table.

The enhancer set and its statistics come from public A549 PRO-seq data (dREG enhancer calls, HOMER
quantification and differential analysis); the signal tracks are ENCODE A549 experiments. Accessions are
given in the article.

## 2. Downstream tables — `[results] analysis`

Default sub-folders (rename any of them in `[layout]` of `paths.toml`):

| Location | Content | Made by |
|---|---|---|
| `deeptools/chip/` | `matrix_sorted_<track>.tab.gz` for GR, POL2RA, ATAC, H3K27ac, EP300, FOXA1, cJUN, BCL3, CREB1, H2AFZ, TCF12, GATA3, TEAD4 (deepTools `computeMatrix`, enhancer centre ± 2 kb, regions = the three sorted class BED files) | deepTools |
| `deeptools/proseq/` | `matrix_enh_{ctrl,dex}_{plus,minus}.tab.gz`; `NR_ctrlPlusDex_sorted.bed`, `UP_sorted_by_dex.bed`, `DN_sorted_by_ctrl.bed`; optional `A549_MA_plot_data.tsv` | deepTools |
| `enhancer_gene/` | `AllPredictions.AvgHiC.ABC0.015.minus150.ForABCPaperV3.txt.gz` (Nasser et al. 2021, Nature 593:238; from the Engreitz lab resources page), `GSE104714_DEX_EtOH_expression.tsv.gz` (GEO GSE104714), PRO-seq bigWig files for the locus tracks | download |
| `enhancer_gene/ABC_Nasser2021_A549_hg38.tsv.gz` | A549 links lifted to hg38 | `scripts/12_lift_abc_to_hg38.py` |
| `enhancer_gene/tx2gene_GSE104714.tsv`, `A5_locus_PROseq_tracks.tsv.gz` | transcript table; binned locus signal | `R/08_…`, `scripts/13_…` |
| `feature_map/` | `umap_pcs.tsv`, `umap_grid_embeddings_supervised.npz`, check tables | `scripts/09_…`, `scripts/10_…` |
| `nr_topics/` | `MS_nr_topics_loadings.tsv`, `MS_nr_topics_usage.tsv` | `scripts/11_nr_topics.py` |
| `knockdown/` | output folders of the R scripts; unzipped LISA result folders (any depth); `A6_exports/` | `R/01_…` to `R/07_…`, `scripts/14_…` |
| `figures/` | figures and their source tables (or `[results] figures`) | `figures/*.py` |

`ml_outputs` holds one folder per cross-validation run, plus `label_permutation_test/`,
`cv_scheme_comparison/`, `feature_sets/`, the LIME output folder,
`full_feature_matrix/` and `feature_matrix_all_classes_GRannotated.csv`.

## 3. Knockdown count tables — `[knockdown]`

| Entry | File | Content |
|---|---|---|
| `proseq_counts` | `raw.genes.A549_PROseq_siEP300.txt` | HOMER gene-level PRO-seq counts, siNON / siEP300 × EtOH / dex, 2 replicates |
| `proseq_counts` | `raw.eRNA_combined_A549_PROseq_siEP300.txt` | HOMER counts over eRNA regions, same samples (columns 20–27) |
| `brbseq_counts` | `raw.counts.A549.BRBseq.exons.txt` | HOMER exon counts, siNON / siFOXA1 / siTEAD4 × Veh / Dex / TNF / DexTNF, 3 replicates (columns 9–44) |

Sequencing data of the knockdown experiments: see the Availability section of the article.
