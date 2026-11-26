from pathlib import Path
import pandas as pd

from .config import (
    PREMATRIX_FILE,
    ATAC_FILE,
    MOTIFS_FILE,
    ANNOTATION_FILE,
    ANNOTATION_ADDITIONAL_FILE,
    GC_FILE,
    EXCLUDE_COLUMNS,
    FILTER_DEX_COLUMNS_TO_100NM_1H,
    DEX_REQUIRED_TERM_GROUPS,
    ALWAYS_KEEP_FEATURE_COLUMNS,
    AVERAGED_TRACKS,
    MOTIF_SET,
    MOTIF_GROUPS_FILE,
    GRE_SCORE_SOURCE_COLUMN,
    GRE_SCORE_FEATURE,
    NEIGHBOURHOOD_FILE,
    NEIGHBOURHOOD_FEATURES,
)
from .paths import ml_input
from .utils import read_table


def load_raw_inputs():
    """
    Load all original input files.

    Returns
    -------
    dict of DataFrames
    """
    prematrix = read_table(ml_input(PREMATRIX_FILE), sep="\t", index_col=None)
    atac_scores = read_table(ml_input(ATAC_FILE), sep="\t", index_col=0)
    motifs_scores = read_table(ml_input(MOTIFS_FILE), sep="\t", index_col=0)
    annotation = read_table(ml_input(ANNOTATION_FILE), sep="\t", index_col=0)
    annotation_additional = read_table(ml_input(ANNOTATION_ADDITIONAL_FILE), sep="\t", index_col=0)
    gc = read_table(ml_input(GC_FILE), sep=";", index_col=0)

    raw_extra = {}
    if MOTIF_SET == "groups" or GRE_SCORE_SOURCE_COLUMN:
        raw_extra["motif_groups"] = read_table(ml_input(MOTIF_GROUPS_FILE), sep="\t", index_col=0)
    if NEIGHBOURHOOD_FEATURES:
        raw_extra["neighbourhood"] = read_table(ml_input(NEIGHBOURHOOD_FILE), sep="\t", index_col=0)

    return {
        **raw_extra,
        "prematrix": prematrix,
        "atac_scores": atac_scores,
        "motifs_scores": motifs_scores,
        "annotation": annotation,
        "annotation_additional": annotation_additional,
        "gc": gc,
    }


def add_combined_coordinate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create enhancer ID: chr_start_end.
    """
    required = ["chr", "start", "end"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"prematrix is missing required columns: {missing}")

    df = df.copy()
    df["combined"] = (
        df["chr"].astype(str)
        + "_"
        + df["start"].astype(str)
        + "_"
        + df["end"].astype(str)
    )
    return df


def align_by_enhancer_id(table: pd.DataFrame, enhancer_ids: pd.Series, name: str) -> pd.DataFrame:
    """
    Rows of `table` in the order of `enhancer_ids`, matched on its 'enhancer_id' column (chr_start_end), never
    on row position. Stops if the table has no ID column, repeats an ID, or does not cover every enhancer.
    """
    if "enhancer_id" not in table.columns:
        raise ValueError(f"{name}: no 'enhancer_id' column")
    if table["enhancer_id"].duplicated().any():
        raise ValueError(f"{name}: {int(table['enhancer_id'].duplicated().sum())} repeated enhancer IDs")
    if enhancer_ids.duplicated().any():
        raise ValueError(f"prematrix: {int(enhancer_ids.duplicated().sum())} repeated enhancer IDs")
    missing = ~enhancer_ids.isin(table["enhancer_id"])
    if missing.any():
        raise ValueError(f"{name}: {int(missing.sum())} of {len(enhancer_ids)} enhancers are not in the table, "
                         f"for example {enhancer_ids[missing].iloc[0]}. Rebuild it from the current enhancer set.")
    out = table.set_index("enhancer_id").loc[enhancer_ids.to_numpy()]
    out.index = enhancer_ids.index
    return out


def build_base_feature_matrix(raw: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Rebuild the original feature matrix in a controlled way.

    Returns
    -------
    X_raw : pd.DataFrame
        Feature matrix before motif binarization.
    prematrix_with_ids : pd.DataFrame
        prematrix with combined enhancer ID.
    original_motif_cols : list[str]
        Columns originating from motifs_scores.
    """

    prematrix = add_combined_coordinate(raw["prematrix"])
    atac_scores = raw["atac_scores"]
    motifs_scores = raw["motifs_scores"]
    annotation = raw["annotation"]
    annotation_additional = raw["annotation_additional"]
    gc = raw["gc"]

    extra = []
    if "motif_groups" in raw:
        mg = align_by_enhancer_id(raw["motif_groups"], prematrix["combined"], MOTIF_GROUPS_FILE)
        if MOTIF_SET == "groups":
            group_cols = [c for c in mg.columns if c.startswith("motif::")]
            if not group_cols:
                raise ValueError(f"{MOTIF_GROUPS_FILE}: no 'motif::' columns")
            motifs_scores = mg[group_cols].rename(columns=lambda c: c[len("motif::"):])
        if GRE_SCORE_SOURCE_COLUMN:
            if GRE_SCORE_SOURCE_COLUMN not in mg.columns:
                raise ValueError(f"{MOTIF_GROUPS_FILE}: no column {GRE_SCORE_SOURCE_COLUMN!r}")
            extra.append(mg[[GRE_SCORE_SOURCE_COLUMN]].rename(columns={GRE_SCORE_SOURCE_COLUMN: GRE_SCORE_FEATURE}))
    elif MOTIF_SET != "original":
        raise ValueError(f"MOTIF_SET must be 'groups' or 'original', not {MOTIF_SET!r}")
    if "neighbourhood" in raw:
        nb = align_by_enhancer_id(raw["neighbourhood"], prematrix["combined"], NEIGHBOURHOOD_FILE)
        absent = [c for c in NEIGHBOURHOOD_FEATURES if c not in nb.columns]
        if absent:
            raise ValueError(f"{NEIGHBOURHOOD_FILE}: missing columns {absent}")
        extra.append(nb[NEIGHBOURHOOD_FEATURES])

    original_motif_cols = motifs_scores.columns.tolist()

    # Tracks averaged into one feature (replicate experiments of one series).
    prematrix = prematrix.copy()
    for new_col, sources in AVERAGED_TRACKS.items():
        absent = [c for c in sources if c not in prematrix.columns]
        if absent:
            raise ValueError(f"AVERAGED_TRACKS: columns not in the prematrix: {absent}")
        at = prematrix.columns.get_loc(sources[0])
        prematrix.insert(at, new_col, prematrix[sources].mean(axis=1))
        prematrix = prematrix.drop(columns=sources)

    not_found = [c for c in EXCLUDE_COLUMNS if c not in prematrix.columns]
    if not_found:
        print(f"EXCLUDE_COLUMNS: {len(not_found)} names are not in the prematrix: {not_found}")

    matrix = (
        prematrix
        .join(atac_scores, how="left")
        .join(motifs_scores, how="left")
    )
    for table in extra:
        matrix = matrix.join(table, how="left")

    # Annotation handling.
    if isinstance(annotation, pd.DataFrame):
        if annotation.shape[1] == 1:
            matrix["annotation"] = annotation.iloc[:, 0].values
        else:
            matrix["annotation"] = annotation.iloc[:, 0].values
    else:
        matrix["annotation"] = annotation

    matrix["Distance to TSS"] = annotation_additional["Distance to TSS"].abs().values
    matrix["Enh_len"] = prematrix["end"] - prematrix["start"]
    matrix["gc"] = gc["5_pct_gc"].values

    # Drop genomic coordinates after combined ID is created.
    matrix = matrix.drop(columns=["chr", "start", "end"], errors="ignore")

    # Remove label columns and manually excluded columns.
    matrix = matrix.loc[:, ~matrix.columns.isin(EXCLUDE_COLUMNS)]

    # Set enhancer ID as index.
    matrix = matrix.set_index("combined")

    # Remove CRISPR-related columns.
    crispr_cols = [
        c for c in matrix.columns
        if "crispr" in str(c).lower()
    ]
    matrix = matrix.drop(columns=crispr_cols, errors="ignore")

    # Keep only biologically intended dex condition columns:
    # dex + 100 nM + 1 hour.
    # Non-dex columns are retained.
    if FILTER_DEX_COLUMNS_TO_100NM_1H:
        (
            matrix,
            kept_dex_cols,
            dropped_dex_cols,
            forced_kept_cols,
        ) = filter_dex_columns_by_condition(
            matrix=matrix,
            required_term_groups=DEX_REQUIRED_TERM_GROUPS,
            always_keep_columns=ALWAYS_KEEP_FEATURE_COLUMNS,
        )

        print("Dex condition filtering applied.")
        print(f"Kept dex columns: {len(kept_dex_cols)}")
        print(f"Dropped dex columns not matching 100 nM 1 hour: {len(dropped_dex_cols)}")
        print(f"Forced kept columns: {forced_kept_cols}")

        matrix.attrs["kept_dex_columns_100nm_1h"] = kept_dex_cols
        matrix.attrs["dropped_dex_columns_not_100nm_1h"] = dropped_dex_cols
        matrix.attrs["forced_kept_feature_columns"] = forced_kept_cols

    return matrix, prematrix, original_motif_cols


def save_base_matrix(X_raw: pd.DataFrame, out_path: Path) -> None:
    X_raw.to_csv(out_path)

def _normalize_for_column_filtering(text: str) -> str:
    """
    Normalize column names for robust condition matching.
    """
    text = str(text).lower()
    text = text.replace("-", "_")
    text = text.replace(" ", "_")
    return text

def is_always_keep_column(
    col: str,
    always_keep_columns: list[str],
) -> bool:
    """
    Check whether a column should be kept regardless of dex-condition filtering.

    Uses exact matching and normalized matching.
    """
    col_raw = str(col)
    col_norm = _normalize_for_column_filtering(col_raw)

    keep_raw = [str(c) for c in always_keep_columns]
    keep_norm = [_normalize_for_column_filtering(c) for c in always_keep_columns]

    return (col_raw in keep_raw) or (col_norm in keep_norm)

def column_matches_required_term_groups(
    col: str,
    required_term_groups: list[list[str]],
) -> bool:
    """
    Match treatment terms using only the descriptive part of the
    column name, excluding the ENCODE file accession.
    """
    # Example:
    # EP300_dex_100_2 hour_ENCFF041HHZ
    # becomes:
    # ep300_dex_100_2 hour
    col_raw = str(col).lower().split("_encff", maxsplit=1)[0]
    col_norm = _normalize_for_column_filtering(col_raw)

    for group in required_term_groups:
        group_match = any(
            str(term).lower() in col_raw
            or _normalize_for_column_filtering(term) in col_norm
            for term in group
        )

        if not group_match:
            return False

    return True


def filter_dex_columns_by_condition(
    matrix: pd.DataFrame,
    required_term_groups: list[list[str]],
    always_keep_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """
    Keep only dex columns matching the intended condition.

    Standard rule:
        keep dex columns only if they match:
            dex + 100 + 1 hour

    Exception:
        always_keep_columns are retained even if they do not match the rule.
        Example:
            ATAC_dex_1h

    Non-dex columns are retained.

    Returns
    -------
    filtered_matrix
    kept_dex_columns
    dropped_dex_columns
    forced_kept_columns
    """
    if always_keep_columns is None:
        always_keep_columns = []

    keep_cols = []
    kept_dex_cols = []
    dropped_dex_cols = []
    forced_kept_cols = []

    for col in matrix.columns:
        col_l = str(col).lower()

        # Explicit exception list.
        if is_always_keep_column(col, always_keep_columns):
            keep_cols.append(col)
            forced_kept_cols.append(col)

            if "dex" in col_l:
                kept_dex_cols.append(col)

            continue

        # Standard dex condition filtering.
        if "dex" in col_l:
            if column_matches_required_term_groups(
                col,
                required_term_groups=required_term_groups,
            ):
                keep_cols.append(col)
                kept_dex_cols.append(col)
            else:
                dropped_dex_cols.append(col)

        # Non-dex columns are retained.
        else:
            keep_cols.append(col)

    filtered = matrix.loc[:, keep_cols].copy()

    return filtered, kept_dex_cols, dropped_dex_cols, forced_kept_cols
