"""Overlap of enhancers with a peak file, without external tools.

Coordinates are BED-like (0-based start, end not included). Two intervals overlap when they share at least one
base pair, which is what ``bedtools intersect -u`` reports.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ENHANCER_ID_PATTERN = r"^(chr[^_]+)_(\d+)_(\d+)$"


def parse_enhancer_ids(enhancer_ids) -> pd.DataFrame:
    """``chr1_1000_1500`` -> columns chr, start, end (one row per ID, same order)."""
    ids = pd.Series(list(enhancer_ids), dtype="object")
    coords = ids.str.extract(ENHANCER_ID_PATTERN)
    coords.columns = ["chr", "start", "end"]
    bad = coords["chr"].isna()
    if bad.any():
        raise ValueError(f"enhancer IDs that are not chr_start_end: {ids[bad].head(5).tolist()}")
    coords["start"] = coords["start"].astype(int)
    coords["end"] = coords["end"].astype(int)
    return coords


def read_peaks(path) -> pd.DataFrame:
    """First three columns (chr, start, end) of a BED / narrowPeak file, plain or gzipped."""
    peaks = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2], comment="#",
                        names=["chr", "start", "end"], dtype={0: str})
    return peaks.astype({"start": int, "end": int})


def overlaps_any(queries: pd.DataFrame, peaks: pd.DataFrame, min_overlap: int = 1) -> np.ndarray:
    """True for every query interval that shares at least ``min_overlap`` bp with one of the peaks.

    Both tables need the columns chr, start, end. Intervals shorter than ``min_overlap`` cannot qualify and are
    left out. Among the rest, a query [s, e) is hit if some peak starts at or before e - min_overlap and ends at
    or after s + min_overlap. With the peaks of a chromosome sorted by start, the running maximum of their ends
    answers that with one binary search per query.
    """
    if min_overlap < 1:
        raise ValueError("min_overlap must be at least 1")
    hit = np.zeros(len(queries), dtype=bool)
    q_chr = queries["chr"].to_numpy()
    q_start = queries["start"].to_numpy()
    q_end = queries["end"].to_numpy()
    long_enough = (q_end - q_start) >= min_overlap
    peaks = peaks[(peaks["end"] - peaks["start"]) >= min_overlap]
    for chrom, p in peaks.groupby("chr", sort=False):
        rows = np.flatnonzero((q_chr == chrom) & long_enough)
        if rows.size == 0:
            continue
        p = p.sort_values("start")
        starts = p["start"].to_numpy()
        max_end = np.maximum.accumulate(p["end"].to_numpy())
        n_before = np.searchsorted(starts, q_end[rows] - min_overlap, side="right")
        has = n_before > 0
        furthest_end = np.where(has, max_end[np.maximum(n_before - 1, 0)], -1)
        hit[rows] = has & (furthest_end >= q_start[rows] + min_overlap)
    return hit
