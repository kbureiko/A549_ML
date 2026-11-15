"""Settings of the modelling pipeline. Values are those of the published runs."""

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

# Output root: [results] ml_outputs in paths.toml (a549_enhancers.paths.ML).

# =========================
# Label thresholds
# =========================

# Benjamini-Hochberg adjusted p-value of the dex-vs-control comparison
PVAL_COL = "ctrl vs. dex adj. p-value"
LOG2FC_COL = "ctrl vs. dex Log2 Fold Change"

PVAL_THRESHOLD = 0.05
LOG2FC_THRESHOLD = 0.584

# =========================
# CV settings
# =========================

N_SPLITS = 5
N_REPEATS = 5

# =========================
# Feature engineering
# =========================

MOTIF_KEYWORDS = [
    "motif",
    "homer"
]

# Features to exclude from the original pipeline
EXCLUDE_COLUMNS = [
    "ctrl vs. dex Log2 Fold Change",
    "ctrl vs. dex p-value",
    "ctrl vs. dex adj. p-value"
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

VARIANCE_THRESHOLD = 5.0

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
    # Class imbalance: weight each class inversely to its size in the training data.
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

STABLE_MIN_FRACTION_POSITIVE_IMPORTANCE = 0.60
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
