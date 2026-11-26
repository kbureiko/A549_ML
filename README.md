# Glucocorticoid-responsive enhancers in A549 cells: models, validation and figures

Code for the manuscript on dexamethasone-responsive enhancers in A549 cells.
Enhancers detected from public PRO-seq data are split into three classes (induced **UP**, repressed **DN**,
non-responsive **NR**), described by ENCODE chromatin and transcription-factor tracks, motif calls and genomic
annotation, and separated by three pairwise CatBoost classifiers that are explained with LIME. The repository
also holds the validation analyses, the feature map, the enhancer-gene analysis, the R analyses of the
knockdown experiments (siEP300 PRO-seq, siFOXA1 / siTEAD4 BRB-seq) and the code of the figures.

**No data and no folder names of any computer are stored here.** Every script reads its locations from a
local file, `paths.toml`, which git ignores.

## Layout

```
src/a549_enhancers/   importable pipeline: settings, input assembly, labels, features, fold preprocessing,
                      models, threshold tuning, evaluation, permutation importance, paths
scripts/              numbered analysis steps (Python), run in this order
figures/              one script per figure made in Python, and the tables behind numbers quoted in the text
R/                    knockdown analyses (DESeq2, Venn diagrams, heatmaps, GO) and exports for the Python figures
tests/                tests on synthetic data (no real data needed)
docs/                 data.md (inputs and where to get them), pipeline.md (run order), figures.md (panel -> script)
paths.example.toml    template of the local path file
```

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync                               # creates .venv with the pinned versions of pyproject.toml / uv.lock
cp paths.example.toml paths.toml      # then edit paths.toml: your folders
uv run pytest                         # optional: checks the installation on synthetic data
```

`paths.toml` names four things: the folder with the input tables, the folder for model runs, the folder for
downstream tables and figures, and the count tables of the knockdown experiments. It can also live outside
the repository: set the environment variable `A549_PATHS` to its full path. The sub-folders that the scripts
expect inside the `analysis` folder are listed in [docs/data.md](docs/data.md); each can be renamed in the
`[layout]` section of `paths.toml`.

R (4.3 or later) with DESeq2, dplyr, tidyr, data.table, eulerr, ComplexHeatmap, circlize, ggplot2, Hmisc,
clusterProfiler, org.Hs.eg.db and, for one export, EnsDb.Hsapiens.v75. The R scripts read the same
`paths.toml` (`R/paths.R`).

## Running

All commands are run from the repository root. The full order with inputs and outputs is in
[docs/pipeline.md](docs/pipeline.md); the three models are:

One command runs the cross-validation and the final model of the chosen task(s) into dated run folders, after
checking the settings in `config.py` and the class counts (add `--lime` for the LIME step):

```
uv run python scripts/00_rerun_models.py --tasks up_vs_dn nr_vs_up nr_vs_dn
```

The same command runs two further cross-validations on GR-bound enhancers only (GR-bound UP against GR-bound NR,
with all features and without the features measured after dex):

```
uv run python scripts/00_rerun_models.py --tasks grbound_nr_vs_up grbound_nr_vs_up_pretreatment
```

Every cross-validation run also writes a grouped permutation importance (predefined families of targets and
clusters of correlated tracks, permuted together; `config.FEATURE_FAMILIES`). Descriptive analyses of the classes
(scripts 22 to 25: detection power, graded response, neighbouring enhancers, dex time course) are listed in
[docs/pipeline.md](docs/pipeline.md).

The single steps:

```bash
uv run python scripts/01_run_cv.py --task nr_vs_up \
    --models catboost --tune-threshold --threshold-method max_f1 --run-prefix <date>
uv run python scripts/01_run_cv.py --task up_vs_dn \
    --models catboost --tune-threshold --threshold-method max_f1 --run-prefix <date>
uv run python scripts/01_run_cv.py --task nr_vs_dn \
    --models catboost --tune-threshold --threshold-method max_f1 --run-prefix <date>
```

Figure scripts are plain Python files with one `# %%` cell marker, so they run from the terminal
(`uv run python figures/figS1_model_performance.py`; set `MPLBACKEND=Agg` to skip the preview window) or as
a cell in PyCharm, VS Code or Spyder.

## What the models are

| | |
|---|---|
| Classes | UP: p < 0.05 and log2 fold change >= 0.584; DN: p < 0.05 and log2 fold change <= -0.584; NR: all other enhancers. The p-value is the adjusted p-value of the dex-vs-control comparison (`config.PVAL_COL`). |
| Tasks | NR vs UP (positive UP), UP vs DN (positive UP), NR vs DN (positive DN) |
| Features | ChIP-seq and ATAC-seq signal tracks as measured: one control and one dex (100 nM, 1 h) track per target, both from the same series (`config.EXCLUDE_COLUMNS` lists the tracks left out; the CREB1 control is the mean of two single-replicate experiments, `config.AVERAGED_TRACKS`). Motifs as present / absent, one representative motif per group of correlated HOMER motifs (`config.MOTIF_SET`); best GRE motif score; annotation, distance to TSS, length, GC content; other enhancers and GR peaks within 50 kb (`config.NEIGHBOURHOOD_FEATURES`). Tables added to the prematrix are matched on the enhancer ID. |
| Validation | five-fold stratified cross-validation repeated ten times, 50 held-out evaluations (seed 42); fold models are seeded with 42 + split index |
| Inside each training fold | constant numeric features removed (no variance filter; `config.VARIANCE_THRESHOLD = 0`); annotation and motif indicators passed to CatBoost as categorical features; decision threshold = F1-maximising point of the precision-recall curve of inner 3-fold out-of-fold predictions |
| Model | CatBoost, 800 iterations, depth 6, learning rate 0.03, Bayesian bootstrap; balanced class weights (`auto_class_weights = "Balanced"`) |
| Importance | permutation importance on the held-out fold (average precision, 10 repeats); stable features: positive in at least 80 % of the held-out evaluations and positive median |
| Final models | refitted on all enhancers of the two classes with the stable features, used for LIME only |

All settings are in `src/a549_enhancers/config.py`.

## Not in this repository

- Raw-data processing (read trimming, alignment, dREG enhancer calls, HOMER quantification and motif
  calls, deepTools matrices) was done with command-line tools; see docs/data.md for what each step has to
  produce.
- LISA was run outside this code; `scripts/14_collect_lisa.py` collects its output tables.
- Western blots, RT-qPCR panels, the motif-enrichment panel of Figure 3 and the ChIP-seq signal panel of
  Supplementary Figure S7.
- Multi-panel figures that combine R panels were assembled in a vector-graphics program.

## Licence and citation

MIT licence (see `LICENSE`). Please cite the article (see `CITATION.cff`).

[![DOI](https://zenodo.org/badge/1458427863.svg)](https://doi.org/10.5281/zenodo.24238693)
