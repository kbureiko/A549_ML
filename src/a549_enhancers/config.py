"""Settings of the modelling pipeline.
"""

GLOBAL_RANDOM_STATE = 42

# =========================
# Input files
# =========================
# File names only. The folder that holds them is set per computer in paths.toml
# ([inputs] ml_inputs) and resolved by a549_enhancers.paths.ml_input().
PREMATRIX_FILE = "202208_prematrix.csv"
ATAC_FILE = "atac_matrix.csv"
MOTIFS_FILE = "matrix_motifs.csv"
ANNOTATION_FILE = "20220908_annotation.csv"
ANNOTATION_ADDITIONAL_FILE = "20220919_annotation.csv"
GC_FILE = "20220919_gc_content.csv"
MOTIF_GROUPS_FILE = "matrix_motif_groups.tsv"        # scripts/18 + 21: motif groups and GRE score
NEIGHBOURHOOD_FILE = "matrix_neighbourhood.tsv"       # scripts/19

# Output root: [results] ml_outputs in paths.toml (a549_enhancers.paths.ML).

# =========================
# Label thresholds
# =========================

# p-value column of the dex-vs-control comparison used for the class labels.
# "ctrl vs. dex p-value" gives the class labels of the earlier runs (478 UP, 232 DN, 7,560 NR);
PVAL_COL = "ctrl vs. dex adj. p-value"
LOG2FC_COL = "ctrl vs. dex Log2 Fold Change"

PVAL_THRESHOLD = 0.05
LOG2FC_THRESHOLD = 0.584

# =========================
# CV settings
# =========================

N_SPLITS = 5
N_REPEATS = 10          # earlier runs: 5

# =========================
# Feature engineering
# =========================

MOTIF_KEYWORDS = [
    "motif",
    "homer"
]

# ChIP-seq tracks: one control and one dex track per target, both from the same series (selection of 20 Nov 2026,
# from the ENCODE metadata: project, experiment, release date, replicates, audit flags).
#   - targets of the GGR dex time course: its 0 h (untreated) and 1 h tracks
#   - other targets with a dex track: the EtOH and dex 1 h tracks of the 2011-2012 ENCODE series
#   - targets with control tracks only: one track (the more recent experiment)
#   - NR3C1: dex track from GGR, control = EtOH track ENCFF914PNO (kept as in the earlier runs)
#   - CREB1 control: mean of the two single-replicate EtOH experiments (AVERAGED_TRACKS)
# Everything listed here is left out of the feature matrix. "earlier runs: used" marks tracks that were features
# in the September 2026 models and were replaced on 20 Nov.
EXCLUDE_COLUMNS = [
    "ctrl vs. dex Log2 Fold Change",
    "ctrl vs. dex p-value",
    "ctrl vs. dex adj. p-value",

    "BCL3_eth_0.02_1 hour_ENCFF022EEO",
    "CEBPB_ENCFF135TNI",                          # earlier runs: used; now ENCFF863XVE (GGR 0 h)
    "CEBPB_ENCFF552ZCJ",
    "CTCF_dex_100_1 hour_ENCFF353CCY",
    "CTCF_dex_100_1 hour_ENCFF886LCD",
    "CTCF_ENCFF254DAA",
    "CTCF_ENCFF409RVF",
    "CTCF_ENCFF972XNW",
    "CTCF_eth_0.02_1 hour_ENCFF163NCL",
    "CTCF_eth_0.02_1 hour_ENCFF599HRA",
    "EP300_ENCFF842XYY",                          # earlier runs: used; now ENCFF376VNJ (GGR 0 h, series of the dex track)
    "EP300_eth_0.02_1 hour_ENCFF788DYT",
    "FOSL2_eth_0.02_1 hour_ENCFF082UOK",          # earlier runs: used; now ENCFF129REO (GGR 0 h)
    "H3K27ac_eth_0.02_1 hour_ENCFF730VLE",        # earlier runs: used; now ENCFF054OOT (GGR 0 h)
    "H3K27ac_ENCFF831UJW",
    "H3K27ac_dex_100_1 hour_ENCFF496GJM",
    "H3K4me1_eth_0.02_1 hour_ENCFF905WYC",        # earlier runs: used; now the GGR pair ENCFF580SCK / ENCFF718UBQ
    "H3K4me1_dex_100_1 hour_ENCFF633KDT",         # earlier runs: used
    "H3K4me2_eth_0.02_1 hour_ENCFF230LCP",        # earlier runs: used; now the GGR pair ENCFF271CZT / ENCFF133QQY
    "H3K4me2_dex_100_1 hour_ENCFF167HQG",         # earlier runs: used
    "H3K4me3_ENCFF199LRZ",
    "H3K4me3_eth_0.02_1 hour_ENCFF764XQX",        # earlier runs: used; now the GGR pair ENCFF529CES / ENCFF252BFR
    "H3K4me3_dex_100_1 hour_ENCFF021UDY",         # earlier runs: used
    "H3K9me3_eth_0.02_1 hour_ENCFF795UKW",        # earlier runs: used; now ENCFF606BMK (GGR 0 h)
    "JUN_ENCFF131JRH",
    "JUN_ENCFF522ITF",
    "MAX_ENCFF858VXY",
    "NR3C1_dex_100_1 hour_ENCFF113KIG",
    "NR3C1_ENCFF331JNU",
    "NR3C1_eth_0.02_1 hour_ENCFF951EGL",
    "POLR2A_ENCFF155EYW",
    "RAD21_ENCFF244KAP",
    "RAD21_ENCFF250KYI",
    "REST_eth_0.02_1 hour_ENCFF421JPS",           # earlier runs: used next to ENCFF044DWW
    "SIN3A_eth_0.02_1 hour_ENCFF142UMZ",          # earlier runs: used next to ENCFF230NYQ
    "SMC3_ENCFF564INI",
    "USF1_eth_0.02_1 hour_ENCFF577GXR",
]

# Tracks that are averaged into one feature: {new column: [columns of the prematrix]}. The mean is taken per
# enhancer; the source columns are then dropped. Earlier runs used ENCFF545ERO alone.
AVERAGED_TRACKS = {
    "CREB1_eth_0.02_1 hour_ENCFF545ERO_ENCFF609MCU": [
        "CREB1_eth_0.02_1 hour_ENCFF545ERO",
        "CREB1_eth_0.02_1 hour_ENCFF609MCU",
    ],
}

# =========================
# Sequence and neighbourhood features (added 20 Nov 2026)
# =========================

# Motif indicators (present / absent):
#   "groups"    one representative motif per group of correlated HOMER motifs (MOTIF_GROUPS_FILE, columns motif::...)
#   "original"  the 238 motifs of the earlier runs (MOTIFS_FILE)
MOTIF_SET = "groups"

# Best log-odds score of the A549 GRE motif in the enhancer (continuous), read from MOTIF_GROUPS_FILE.
# None = not used.
GRE_SCORE_SOURCE_COLUMN = "motifscore::GRE(NR),IR3"
GRE_SCORE_FEATURE = "GRE_score"

# Neighbourhood table of scripts/19_neighbourhood_features.py. Empty list = not used.
#   nbr_enhancer*   other enhancers around the enhancer (enhancer set only; no treatment information)
#   nbr_GRpeak*, nbr_strongGRpeak*   GR peaks after dex around the enhancer (dex information)
NEIGHBOURHOOD_FEATURES = [
    "nbr_enhancers_50kb",
    "nbr_enhancer_distance",
    "nbr_GRpeaks_50kb",
    "nbr_strongGRpeaks_50kb",
    "nbr_GRpeak_distance",
    "nbr_strongGRpeak_distance",
]

# =========================
# Always-keep feature columns
# =========================

# Some features are biologically valid but do not match the standard
# dex + 100 nM + 1 hour naming pattern.
# Example: ATAC_dex_1h does not include "100", but should be retained.
ALWAYS_KEEP_FEATURE_COLUMNS = [
    "ATAC_dex_1h",
]

# =========================
# Column filtering
# =========================

# Keep only dex-treated feature columns matching the intended ENCODE condition.
FILTER_DEX_COLUMNS_TO_100NM_1H = True

# A dex column must match at least one term from each group.
DEX_REQUIRED_TERM_GROUPS = [
    ["dex"],
    ["100"],
    ["1 hour", "1h", "1_hour"],
]

# =========================
# Preprocessing
# =========================

# Continuous features with a training-fold variance <= this value are removed. 0.0 removes constant features only.
VARIANCE_THRESHOLD = 0.0          # earlier runs: 5.0

# Treat binary motif indicators as categorical in CatBoost.
MOTIFS_AS_CATEGORICAL_FOR_CATBOOST = True

# Include annotation as categorical feature if present.
ANNOTATION_COL = "annotation"

# =========================
# Model settings
# =========================

RF_PARAMS = {
    "n_estimators": 800,
    "max_depth": None,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "class_weight": "balanced_subsample",
    "n_jobs": -1,
}

CATBOOST_PARAMS = {
    "iterations": 800,
    "depth": 6,
    "learning_rate": 0.03,
    "loss_function": "Logloss",
    "eval_metric": "PRAUC",
    "random_seed": GLOBAL_RANDOM_STATE,
    "verbose": False,
    "allow_writing_files": False,
    # Class imbalance: weight each class inversely to its size in the training data (earlier runs: no class weights).
    "auto_class_weights": "Balanced",

    # Good defaults for tabular genomic signal matrix
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 1.0,
    "random_strength": 1.0,
    "l2_leaf_reg": 3.0,
    "rsm": 1.0,
    "border_count": 128,
}

# =========================
# Permutation importance
# =========================

PERMUTATION_N_REPEATS = 10

# For rare positive class, average precision is more informative than accuracy.
PERMUTATION_SCORING = "average_precision"

# =========================
# Stable feature definition
# =========================

STABLE_MIN_FRACTION_POSITIVE_IMPORTANCE = 0.80          # earlier runs: 0.60
STABLE_MIN_MEDIAN_IMPORTANCE = 0.0
STABLE_TOP_N_FALLBACK = 50

# =========================
# Decision-threshold tuning
# =========================

# Default is False so normal runs still use 0.5 threshold unless requested.
TUNE_THRESHOLD_DEFAULT = False

# Threshold is tuned inside the outer training fold using inner CV.
THRESHOLD_INNER_CV_SPLITS = 3

# Options:
#   max_f1
#   max_f2
#   recall_at_min_precision
THRESHOLD_METHOD_DEFAULT = "max_f1"

# Used only for recall_at_min_precision.
MIN_PRECISION_DEFAULT = 0.30

# =========================
# Grouped permutation importance (added 20 Nov 2026)
# =========================

# All features of a group are permuted together (same row shuffle), so the score drop shows what the group carries
# as a whole. Groups are fixed here, before any model is fitted, and never use the class labels.
# Families: ENCODE targets by name; control and dex tracks of a family form separate groups. Targets that are not
# in the feature matrix are ignored.
FEATURE_FAMILIES = {
    "AP-1": ["JUN", "JUNB", "JUND", "FOSL2", "FOSB", "ATF3"],
    "CTCF and cohesin": ["CTCF", "RAD21", "SMC3"],
    "active histone marks": ["H3K27ac", "H3K4me1", "H3K4me2", "H3K4me3", "H3K9ac"],
    "transcription": ["POLR2A", "POLR2AphosphoS2", "H3K79me2", "H3K36me3"],
    "repressive marks and corepressors": ["H3K27me3", "H3K9me3", "H4K20me1", "CBX2", "CBX8", "RNF2", "EHMT2",
                                          "REST", "RCOR1", "SIN3A", "HDAC2", "KDM1A"],
    "FOX factors": ["FOXA1", "FOXA2", "FOXF2", "FOXS1"],
}

# Correlation clusters of the signal features: average linkage on 1 - Spearman r over all enhancers, cut so that
# the mean correlation between merged clusters is at least this value.
GROUP_CORRELATION_THRESHOLD = 0.7

# Shuffles per group and held-out fold.
GROUPED_PERMUTATION_N_REPEATS = 10

# =========================
# Subsets (added 20 Nov 2026)
# =========================

# GR peak file that defines GR-bound enhancers (overlap of at least 1 bp), in [inputs] ml_inputs.
GR_PEAK_FILE = "ENCFF395FIP.bed.gz"

# =========================
# Dex time course (descriptive, scripts/25_timecourse_by_class.py)
# =========================

# Untreated (0 min) tracks of the minute series (5 to 25 min; GGR, released 2017-10-19). They are not model
# features. The hour series (0.5 to 12 h) starts from the control track of the model (EXCLUDE_COLUMNS above).
TIMECOURSE_MINUTE_ZERO = {
    "NR3C1": "NR3C1_ENCFF331JNU",
    "EP300": "EP300_ENCFF842XYY",
    "H3K27ac": "H3K27ac_ENCFF831UJW",
    "JUN": "JUN_ENCFF522ITF",
}

# Transcribed subset (added 21 Nov 2026; scripts/01_run_cv.py --subset transcribed).
# A decrease in enhancer transcription can only be detected where the enhancer is transcribed, so NR enhancers
# with little PRO-seq signal are not a fair comparison for DN enhancers. The subset keeps the enhancers of both
# classes whose PRO-seq signal is at least the given quantile of the positive class. The signal is the mean of
# both conditions (it does not contain the direction of the response).
PROSEQ_SIGNAL_COLUMN = "Mean_signal"            # column of the PRO-seq table ([layout] proseq_ma_table)
TRANSCRIBED_MIN_QUANTILE_OF_POSITIVE = 0.05

# Matched subset (scripts/01_run_cv.py --subset matched): every enhancer of the positive class with its nearest
# enhancers of the negative class in (log10 length, PRO-seq signal), both standardised. Length and transcription
# level are then similar in the two classes, so the model cannot separate them by detection power alone.
MATCHED_NEGATIVES_PER_POSITIVE = 5
