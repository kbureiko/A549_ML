"""One table of all enhancers for the descriptive analyses: coordinates, PRO-seq statistics, class, GR binding.

Everything is matched on the enhancer ID (chr_start_end). The class labels are those of the models
(labels.generate_multiclass_labels), GR binding is the overlap with a peak of config.GR_PEAK_FILE.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import GR_PEAK_FILE, LOG2FC_COL, PREMATRIX_FILE, PVAL_COL
from .data_loading import add_combined_coordinate
from .intervals import overlaps_any, read_peaks
from .labels import generate_multiclass_labels
from .paths import ml_input
from .utils import read_table


def load_prematrix() -> pd.DataFrame:
    """The prematrix with the enhancer ID as index (all signal columns kept)."""
    pre = add_combined_coordinate(read_table(ml_input(PREMATRIX_FILE), sep="\t", index_col=None))
    if pre["combined"].duplicated().any():
        raise ValueError("prematrix: repeated enhancer IDs")
    return pre.set_index("combined", drop=False)


def enhancer_table(prematrix: pd.DataFrame | None = None, peaks: str | Path | None = None,
                   need_gr: bool = True) -> pd.DataFrame:
    """Columns: enhancer_id, chr, start, end, log2FC, pvalue, class, GR_bound (index = enhancer_id)."""
    pre = load_prematrix() if prematrix is None else prematrix
    out = pd.DataFrame({
        "enhancer_id": pre.index.to_numpy(),
        "chr": pre["chr"].astype(str).to_numpy(),
        "start": pre["start"].astype(int).to_numpy(),
        "end": pre["end"].astype(int).to_numpy(),
        "log2FC": pd.to_numeric(pre[LOG2FC_COL], errors="coerce").to_numpy(),
        "pvalue": pd.to_numeric(pre[PVAL_COL], errors="coerce").to_numpy(),
    }, index=pre.index.to_numpy())
    out["class"] = generate_multiclass_labels(pre).reindex(out.index).to_numpy()
    peak_file = Path(peaks) if peaks else ml_input(GR_PEAK_FILE)
    if peak_file.exists():
        out["GR_bound"] = overlaps_any(out[["chr", "start", "end"]], read_peaks(peak_file))
    elif need_gr:
        raise FileNotFoundError(f"GR peak file not found: {peak_file}")
    return out


def mean_heldout_score(run_dir: Path, model: str = "catboost") -> pd.DataFrame:
    """Held-out probability of the positive class per enhancer, averaged over the repeats of a run.
    Columns: score (mean), n_repeats, true_label, frac_called_positive; plus the attribute .attrs['positive']."""
    pred = pd.read_csv(Path(run_dir) / "summaries" / "all_cv_predictions.csv", low_memory=False)
    pred = pred[pred["model"] == model]
    if pred.empty:
        raise ValueError(f"{run_dir}: no predictions of model {model!r}")
    positive = str(pred["positive_label"].iloc[0])
    col = f"proba_{positive}"
    out = pred.groupby("enhancer_id").agg(score=(col, "mean"), n_repeats=(col, "size"),
                                          true_label=("true_label", "first"),
                                          frac_called_positive=("pred_label", lambda v: (v == positive).mean()))
    out.attrs["positive"] = positive
    out.attrs["negative"] = str(pred["negative_label"].iloc[0])
    return out


def proseq_signal(ids, path: Path, column: str) -> pd.Series:
    """One column of the PRO-seq table (Chr, Start, End, ...) per enhancer ID, matched on the coordinates.
    HOMER tables are 1-based, so both conventions are tried; stops if the table does not cover the enhancers."""
    M = pd.read_csv(path, sep="\t")
    missing = [c for c in ("Chr", "Start", "End", column) if c not in M.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}; columns are {list(M.columns)[1:]}")
    ids = pd.Index(ids)
    for shift in (1, 0):
        key = M["Chr"].astype(str) + "_" + (M["Start"] - shift).astype(str) + "_" + M["End"].astype(str)
        if ids.isin(key).all():
            s = pd.Series(pd.to_numeric(M[column], errors="coerce").to_numpy(), index=key.to_numpy())
            return s[~s.index.duplicated()].reindex(ids)
    raise ValueError(f"{path}: the coordinates do not cover all {len(ids)} enhancers")
