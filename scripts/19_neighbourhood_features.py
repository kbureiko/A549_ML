"""Genomic neighbourhood of each enhancer: other enhancers and GR peaks around it.

Features (one row per enhancer; distances in bp between the nearest edges, 0 = overlap)

  before treatment (enhancer set only)
    nbr_enhancers_<W>kb          other enhancers within W kb of the enhancer
    nbr_enhancer_distance        distance to the nearest other enhancer

  after dex (GR ChIP-seq peak file ENCFF395FIP, the file that defines GR-bound enhancers)
    nbr_GRpeaks_<W>kb            GR peaks within W kb, not counting peaks that overlap the enhancer itself
    nbr_GRpeak_distance          distance to the nearest GR peak that does not overlap the enhancer itself
    nbr_strongGRpeaks_<W>kb      the same two for strong GR peaks only (top quarter of the peaks by signal value)
    nbr_strongGRpeak_distance

Peaks that overlap the enhancer are left out on purpose: whether the enhancer itself is GR-bound is already in the
model (GR signal), and the question here is what lies around it. Overlapping peaks of the file are merged first
(a narrowPeak file can list a region once per summit). Distances are capped at --cap (default 10 Mb), which is
also the value when the chromosome has no other enhancer or peak.

    uv run python scripts/19_neighbourhood_features.py                       # prematrix and peak file from paths.toml
    uv run python scripts/19_neighbourhood_features.py --windows 50 500      # two window sizes (kb)

Inputs
    <ml_inputs>/202208_prematrix.csv        enhancer coordinates (--coords: any table with chr, start, end or enhancer_id)
    <ml_inputs>/ENCFF395FIP.bed.gz          GR peaks, narrowPeak (--peaks)
    <ml_outputs>/feature_matrix_all_classes_GRannotated.csv   optional: classes and GR-binding status for the summary
Outputs
    <ml_inputs>/matrix_neighbourhood.tsv            row order and row index of the coordinate table, as the other
                                                    input tables; 'enhancer_id' is there for checking, not a feature
    <analysis>/neighbourhood/neighbourhood_by_class.tsv      medians per class and GR-binding status, with a
                                                    Mann-Whitney test against the NR enhancers of the same status
Nothing in the modelling pipeline reads the new table yet.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from a549_enhancers import paths
from a549_enhancers.intervals import parse_enhancer_ids

PEAK_FILE = "ENCFF395FIP.bed.gz"
STRONG_QUANTILE = 0.75            # strong GR peaks: signal value in the top quarter of the (merged) peaks


def read_coords(path):
    d = pd.read_csv(path, sep=None, engine="python", usecols=lambda c: c in ("chr", "start", "end", "enhancer_id"))
    if {"chr", "start", "end"} <= set(d.columns):
        d = d[["chr", "start", "end"]].astype({"start": int, "end": int})
    elif "enhancer_id" in d.columns:
        d = parse_enhancer_ids(d["enhancer_id"])
    else:
        raise ValueError(f"{path}: needs the columns chr, start, end or enhancer_id")
    d["enhancer_id"] = d["chr"].astype(str) + "_" + d["start"].astype(str) + "_" + d["end"].astype(str)
    if d.enhancer_id.duplicated().any():
        raise ValueError(f"{path}: duplicated enhancers")
    return d.reset_index(drop=True)


def read_narrowpeak(path):
    """chr, start, end, signal value; overlapping entries merged (signal = the highest of the merged entries)."""
    p = pd.read_csv(path, sep="\t", header=None, comment="#", dtype={0: str})
    p = p.rename(columns={0: "chr", 1: "start", 2: "end"})
    p["signal"] = pd.to_numeric(p[6], errors="coerce") if p.shape[1] > 6 else np.nan
    p = p[["chr", "start", "end", "signal"]].astype({"start": int, "end": int})
    p = p.sort_values(["chr", "start", "end"]).reset_index(drop=True)
    new = (p.chr != p.chr.shift()) | (p.start >= p.groupby("chr").end.cummax().shift().fillna(-1))
    p["block"] = new.cumsum()
    m = p.groupby("block").agg(chr=("chr", "first"), start=("start", "min"), end=("end", "max"),
                               signal=("signal", "max")).reset_index(drop=True)
    return m, len(p)


def around(q, t, window, cap, same_set=False):
    """For each query interval: number of target intervals within `window` bp (edge to edge) and distance to the
    nearest one. Targets that overlap the query are not counted; with same_set the query itself is a target and
    is left out the same way (the intervals of one set do not overlap each other after merging)."""
    n = np.zeros(len(q), dtype=int)
    dist = np.full(len(q), float(cap))
    q_chr = q["chr"].to_numpy(); qs = q["start"].to_numpy(); qe = q["end"].to_numpy()
    for chrom, tt in t.groupby("chr", sort=False):
        rows = np.flatnonzero(q_chr == chrom)
        if rows.size == 0:
            continue
        tt = tt.sort_values("start")
        ts = tt["start"].to_numpy(); te = np.sort(tt["end"].to_numpy())      # merged / disjoint: ends sorted too
        s, e = qs[rows], qe[rows]
        # targets entirely left of the query: end <= start ; entirely right: start >= end
        k_left = np.searchsorted(te, s, side="right")            # number of targets with end <= s
        k_right = np.searchsorted(ts, e, side="left")            # index of the first target with start >= e
        d_left = np.where(k_left > 0, s - te[np.maximum(k_left - 1, 0)], np.inf)
        d_right = np.where(k_right < len(ts), ts[np.minimum(k_right, len(ts) - 1)] - e, np.inf)
        dist[rows] = np.minimum(np.minimum(d_left, d_right), cap)
        n_left = k_left - np.searchsorted(te, s - window, side="left")          # ends in [s - window, s]
        n_right = np.searchsorted(ts, e + window, side="right") - k_right       # starts in [e, e + window]
        n[rows] = n_left + n_right
    return n, dist


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coords", type=Path, default=None, help="table with chr, start, end (default: the prematrix)")
    ap.add_argument("--peaks", type=Path, default=None, help="GR narrowPeak file (default: ENCFF395FIP in ml_inputs)")
    ap.add_argument("--classes", type=Path, default=None, help="table with enhancer_id, class, GR_bound for the summary")
    ap.add_argument("--windows", type=int, nargs="+", default=[50], help="window sizes in kb (default: 50)")
    ap.add_argument("--cap", type=int, default=10_000_000, help="largest distance reported, bp")
    ap.add_argument("--out", type=Path, default=None, help="feature table (default: ml_inputs/matrix_neighbourhood.tsv)")
    a = ap.parse_args()

    if a.coords is None:
        from a549_enhancers.config import PREMATRIX_FILE
        a.coords = paths.ml_input(PREMATRIX_FILE)
    peaks_file = a.peaks or paths.ML_INPUTS / PEAK_FILE
    out = a.out or paths.ML_INPUTS / "matrix_neighbourhood.tsv"

    enh = read_coords(a.coords)
    peaks, n_raw = read_narrowpeak(peaks_file)
    strong = peaks[peaks.signal >= peaks.signal.quantile(STRONG_QUANTILE)] if peaks.signal.notna().all() else None
    print(f"{len(enh):,} enhancers ({Path(a.coords).name}) | {n_raw:,} GR peak entries, {len(peaks):,} after merging"
          + (f", {len(strong):,} strong (signal >= {peaks.signal.quantile(STRONG_QUANTILE):.1f})" if strong is not None
             else " | no signal column: strong-peak features are left out"))

    F = pd.DataFrame({"enhancer_id": enh.enhancer_id})
    for w in a.windows:
        F[f"nbr_enhancers_{w}kb"], d_enh = around(enh, enh, w * 1000, a.cap, same_set=True)
        F[f"nbr_GRpeaks_{w}kb"], d_gr = around(enh, peaks, w * 1000, a.cap)
        if strong is not None:
            F[f"nbr_strongGRpeaks_{w}kb"], d_strong = around(enh, strong, w * 1000, a.cap)
    F["nbr_enhancer_distance"] = d_enh
    F["nbr_GRpeak_distance"] = d_gr
    if strong is not None:
        F["nbr_strongGRpeak_distance"] = d_strong
    feats = [c for c in F.columns if c != "enhancer_id"]
    out.parent.mkdir(parents=True, exist_ok=True)
    F.to_csv(out, sep="\t")
    print(f"wrote {out}  ({len(feats)} features)\n")
    print(F[feats].describe().loc[["min", "25%", "50%", "75%", "max"]].round(0).T.to_string())

    # ---- summary by class and GR-binding status (not needed for the features)
    cls_file = a.classes or paths.ML / "feature_matrix_all_classes_GRannotated.csv"
    if not Path(cls_file).exists():
        print(f"\nno class table ({cls_file}); summary by class skipped")
        raise SystemExit(0)
    C = pd.read_csv(cls_file, usecols=["enhancer_id", "class", "GR_bound"], low_memory=False)
    M = F.merge(C, on="enhancer_id", how="inner", validate="one_to_one")
    if len(M) != len(F):
        print(f"WARNING: {len(F) - len(M)} enhancers are not in the class table")
    M["GR_bound"] = M.GR_bound.astype(str).str.lower().isin(["true", "1"])
    rows = []
    for status, label in ((True, "GR-bound"), (False, "not GR-bound"), (None, "all")):
        sub = M if status is None else M[M.GR_bound == status]
        nr = sub[sub["class"] == "NR"]
        for f in feats:
            r = {"GR_status": label, "feature": f}
            for c in ("UP", "NR", "DN"):
                x = sub.loc[sub["class"] == c, f]
                r[f"n_{c}"] = len(x); r[f"median_{c}"] = x.median()
                if c != "NR" and len(x) and len(nr):
                    u = stats.mannwhitneyu(x, nr[f], alternative="two-sided")
                    r[f"P_{c}_vs_NR"] = u.pvalue
                    r[f"AUC_{c}_vs_NR"] = u.statistic / (len(x) * len(nr))       # 0.5 = no difference
            rows.append(r)
    S = pd.DataFrame(rows)
    pcols = [c for c in S.columns if c.startswith("P_")]
    p = S[pcols].to_numpy().ravel(); ok = ~np.isnan(p)
    adj = np.full(p.shape, np.nan); adj[ok] = stats.false_discovery_control(p[ok], method="bh")
    for i, c in enumerate(pcols):
        S["Padj_" + c[2:]] = adj.reshape(len(S), len(pcols))[:, i]
    out_dir = paths.PROJECT / "neighbourhood"              # [results] analysis in paths.toml
    out_dir.mkdir(parents=True, exist_ok=True)
    S.to_csv(out_dir / "neighbourhood_by_class.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    show = ["GR_status", "feature", "median_UP", "median_NR", "median_DN", "AUC_UP_vs_NR", "Padj_UP_vs_NR",
            "AUC_DN_vs_NR", "Padj_DN_vs_NR"]
    print("\nmedians by class; AUC = probability that an UP (DN) enhancer has a higher value than an NR enhancer")
    print(S[show].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"\nwrote {out_dir / 'neighbourhood_by_class.tsv'}")
