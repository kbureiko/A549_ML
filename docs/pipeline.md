# Run order

Run from the repository root, after `uv sync` and after filling in `paths.toml`. Python steps:
`uv run python <script>`; R steps: `Rscript <script>`.

## Models

| Step | Script | Reads | Writes |
|---|---|---|---|
| 1 | `scripts/01_run_cv.py --task <task> --models catboost --tune-threshold --threshold-method max_f1 --run-prefix <date>` (three tasks) | input tables | `<ml_outputs>/<run>/` |
| 2 | `scripts/02_run_final_model.py --task <task> --models catboost` | run folder | `<run>/final_model/` |
| 3 | `scripts/03_lime_rules_final_models.py --n-jobs 6` | final models, input tables | `<ml_outputs>/<date>__lime_final_models/` |

After step 1, write the three run-folder names into `[runs]` of `paths.toml`.

## Validation

| Step | Script | Writes |
|---|---|---|
| 4 | `scripts/04_label_permutation_test.py --n-permutations 99 --cv-repeats 1` | `<ml_outputs>/label_permutation_test/` |
| 5 | `scripts/05_cv_scheme_comparison.py` | `<ml_outputs>/cv_scheme_comparison/` |
| 6 | `scripts/06_feature_set_models.py` | `<ml_outputs>/feature_sets/` |

Steps 5 and 6 refit CatBoost on the saved task matrices with the preprocessing, model settings and
threshold rule of step 1; only the folds (step 5) or the feature set (step 6) differ. Step 4 reruns the
pipeline of step 1 unchanged.

## Feature matrix, feature map, topics

| Step | Script | Writes |
|---|---|---|
| 7 | `scripts/07_save_full_feature_matrix.py` | `<ml_outputs>/full_feature_matrix__…/feature_matrix_all_classes.csv` |
| 8 | `scripts/08_annotate_gr_binding.py` | `<ml_outputs>/feature_matrix_all_classes_GRannotated.csv` |
| 9 | `scripts/09_feature_map_prep.py` | `feature_map/umap_pcs.tsv` (top-30 features of each model, rank-normal transform, redundancy filter, PCA) |
| 10 | `scripts/10_feature_map_layouts.py` | `feature_map/umap_grid_embeddings_supervised.npz`; the figures use layout `nn30_md0.1_cosine_sup0.5` |
| 11 | `scripts/11_nr_topics.py` | `nr_topics/MS_nr_topics_*.tsv` (consensus NMF, K = 5) |

## Enhancer–gene links

| Step | Script | Writes |
|---|---|---|
| 12 | `scripts/12_lift_abc_to_hg38.py` | `enhancer_gene/ABC_Nasser2021_A549_hg38.tsv.gz` |
| 13 | `scripts/13_export_locus_tracks.py` | `enhancer_gene/A5_locus_PROseq_tracks.tsv.gz` |
| — | `R/08_export_tx2gene_GSE104714.R` | `enhancer_gene/tx2gene_GSE104714.tsv` |

## Knockdown experiments (R)

| Script | Content |
|---|---|
| `R/01_proseq_siEP300_enhancers.R` | siEP300 PRO-seq, eRNA regions: DESeq2, sets, Venn, heatmap, fold-change box plots, BED files |
| `R/02_proseq_siEP300_genes.R` | siEP300 PRO-seq, genes: DESeq2, sets, Venn, heatmap, box plots, GO, gene lists for LISA |
| `R/03_brbseq_siFOXA1_siTEAD4.R` | BRB-seq: the same for siFOXA1 and siTEAD4 |
| `R/04_export_proseq_siEP300_tables.R`, `R/05_export_brbseq_tables.R` | full DESeq2 tables for the Python figures (Figure 4, Supplementary Figure S14) |
| `R/06_export_gene_sets_go.R` | shared / unique gene sets and GO with the tested genes as background (Supplementary Figure S14C) |
| `R/07_export_erna_siEP300_tables.R` | eRNA tables for `figures/text_numbers_siEP300_enhancers.py` |
| `R/09_lisa_scatter_plots.R` | LISA scatter plots (Supplementary Figures S9, S11, S13) |
| `scripts/14_collect_lisa.py` | one table from the LISA result files (Supplementary Figure S14D) |

Scripts 04–07 refit the models from the count tables and do not depend on 01–03. Their gene sets can differ
from those of 01–03 by a few genes: 05 removes genes with a row sum of counts ≤ 10 before fitting, 03 does
not. GO uses the genes tested in the experiment as background in all scripts.

## Figures

See [figures.md](figures.md).
