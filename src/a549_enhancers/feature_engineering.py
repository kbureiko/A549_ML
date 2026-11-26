"""Feature matrix before cross-validation: motif scores become present / absent indicators, everything else
is used as measured. Fixed rules on column names only; no labels and no train/test information are used."""
import pandas as pd

from .config import (
    MOTIF_KEYWORDS,
    MOTIFS_AS_CATEGORICAL_FOR_CATBOOST,
    ANNOTATION_COL,
    GRE_SCORE_FEATURE,
    NEIGHBOURHOOD_FEATURES,
)


def normalize_colname(name: str) -> str:
    return str(name).lower().replace("-", "_").replace(" ", "_")


def identify_motif_columns(
    X: pd.DataFrame,
    original_motif_cols: list[str] | None = None,
) -> list[str]:
    """
    Identify motif-score columns.

    Uses both:
    1. columns from matrix_motifs.csv, if provided
    2. name keywords: motif, homer, jaspar, pwm
    """
    motif_cols = set()

    if original_motif_cols is not None:
        motif_cols.update([c for c in original_motif_cols if c in X.columns])

    for col in X.columns:
        col_l = normalize_colname(col)
        if any(k in col_l for k in MOTIF_KEYWORDS):
            motif_cols.add(col)

    # continuous sequence and neighbourhood features are never turned into indicators
    motif_cols -= {GRE_SCORE_FEATURE, *NEIGHBOURHOOD_FEATURES}

    return sorted(motif_cols)


def binarize_motif_features(
    X: pd.DataFrame,
    motif_cols: list[str],
) -> tuple[pd.DataFrame, dict]:
    """
    Convert raw motif scores to binary present/absent features.

    Raw motif-score columns are dropped.
    New columns are named:
        motif_binary::<original_col>
    """
    mapping = {}
    binary = {}

    for col in motif_cols:
        if col not in X.columns:
            continue

        new_col = f"motif_binary::{col}"
        raw = pd.to_numeric(X[col], errors="coerce").fillna(0)
        binary[new_col] = (raw > 0).astype(int)
        mapping[col] = new_col

    X = pd.concat([X.drop(columns=list(mapping)), pd.DataFrame(binary, index=X.index)], axis=1)

    return X, mapping


def build_feature_matrix(
    X_raw: pd.DataFrame,
    original_motif_cols: list[str] | None,
) -> tuple[pd.DataFrame, dict]:
    """
    Build final feature matrix before CV.

    Important:
    This step uses fixed rules and column names only.
    It does not use labels or train/test information.
    """
    X_raw = X_raw.copy()

    # 1. Detect and binarize motif features.
    motif_cols = identify_motif_columns(X_raw, original_motif_cols)
    X_final, motif_mapping = binarize_motif_features(X_raw, motif_cols)

    # 2. Define categorical columns.
    categorical_cols = []

    if ANNOTATION_COL in X_final.columns:
        categorical_cols.append(ANNOTATION_COL)

    motif_binary_cols = [
        c for c in X_final.columns
        if c.startswith("motif_binary::")
    ]

    if MOTIFS_AS_CATEGORICAL_FOR_CATBOOST:
        categorical_cols.extend(motif_binary_cols)

    metadata = {
        "n_final_features": X_final.shape[1],
        "motif_mapping": motif_mapping,
        "categorical_cols": categorical_cols,
    }

    return X_final, metadata
