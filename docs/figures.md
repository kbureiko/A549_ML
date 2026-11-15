# Figure → code

| Figure | Code | Needs |
|---|---|---|
| 1 | `figures/fig1_classification_overview.py` | steps 1, 4, 8–11 |
| 2 | `figures/fig2_lime_rules_signal.py` | step 3, 8; deepTools matrices |
| 3B | `R/01_proseq_siEP300_enhancers.R` | eRNA count table |
| 3A, 3C | western blot; HOMER motif enrichment on the BED files of `R/01_…` | not in this repository |
| 4 | `figures/fig4_enhancer_gene_links.py` | steps 8, 12, 13; `R/04_…`, `R/08_…` |
| 5 | `R/02_proseq_siEP300_genes.R` | gene count table |
| 6B–C, 7 | `R/03_brbseq_siFOXA1_siTEAD4.R` | BRB-seq count table |
| S1 | `figures/figS1_model_performance.py` | steps 1, 4, 5, 6, 8 |
| S2 | `figures/figS2_feature_correlation.py` | steps 1, 8 |
| S3 | `figures/figS3_nr_topics.py` | steps 8, 11 |
| S4, S5 | `figures/figS4_umap_features.py`, `figures/figS5_umap_features.py` | steps 8–10 |
| S6 | `figures/figS6_factors.py` | step 8; deepTools matrices |
| S7A–C | `R/01_proseq_siEP300_enhancers.R` | |
| S8 | `R/02_proseq_siEP300_genes.R` | |
| S10B–D, S12B–D | `R/03_brbseq_siFOXA1_siTEAD4.R` | |
| S14 | `figures/figS14_knockdown_genes.py` | `R/04_…`–`R/06_…`, `scripts/14_…` |
| S9, S11, S13 | `R/09_lisa_scatter_plots.R` | LISA result files |
| S7D, RT-qPCR panels, schematic | made outside this code | — |

Numbers quoted in the text:

| Script | Content |
|---|---|
| `figures/text_numbers_feature_map.py` | class summaries, neighbourhoods on the feature map, NR subsets |
| `figures/text_numbers_matched_comparison.py` | UP and DN enhancers against matched NR enhancers |
| `figures/text_numbers_siEP300_enhancers.py` | dex response of enhancers after EP300 depletion |

The R scripts write single panels (PNG); the published multi-panel figures were assembled from them, and
block labels in the heatmaps were added there.
