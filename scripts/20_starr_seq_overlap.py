"""Overlap of the enhancers with dex-responsive STARR-seq regions (Johnson et al. 2018, Nat Commun 9:5317,
Supplementary Data 1; A549 cells, whole-genome reporter library, 100 nM dex for 1, 4, 8 and 12 h against 0 h, hg38).

The supplementary table lists only the regions whose reporter activity changed with dex (FDR <= 5 %), one sheet per
time point and direction. An enhancer "overlaps" a region when they share at least 1 bp.

Standalone: needs pandas, numpy, scipy and openpyxl.

    python scripts/20_starr_seq_overlap.py --starr 41467_2018_7607_MOESM3_ESM.xlsx \
        --classes feature_matrix_all_classes_GRannotated.csv [--out-dir FOLDER]

Inputs
    --starr     Supplementary Data 1 of the paper (xlsx; sheets 1hr_induced, 1hr_repressed, ... 12hr_repressed)
    --classes   table with enhancer_id (chr_start_end), class and GR_bound
Outputs (in --out-dir, default: next to --classes)
    matrix_starr.tsv        one row per enhancer, in the order of --classes: overlap with an induced / repressed
                            region at each time point and at any time point, and the reporter log2 fold change of the
                            overlapping region with the largest change (empty when there is no overlap)
    starr_by_class.tsv      number and share of enhancers with an overlap per class and GR-binding status, with
                            Fisher's exact test against the NR enhancers of the same status
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

TIMES = (1, 4, 8, 12)


def read_starr(path):
    book = pd.ExcelFile(path)
    rows = []
    for sheet in book.sheet_names:
        time, direction = sheet.split("_")
        d = book.parse(sheet)
        c = d["coordinate"].astype(str).str.extract(r"^(chr[^:]+):(\d+)-(\d+)$")
        if c[0].isna().any():
            raise ValueError(f"{sheet}: coordinates that are not chr:start-end")
        rows.append(pd.DataFrame({"chr": c[0], "start": c[1].astype(int), "end": c[2].astype(int),
                                  "logFC": d["logFC"].astype(float), "time": int(time.replace("hr", "")),
                                  "direction": direction}))
    return pd.concat(rows, ignore_index=True)


def overlap(enh, regions):
    """Per enhancer: overlaps a region (bool), and the logFC of the overlapping region with the largest |logFC|."""
    hit = np.zeros(len(enh), dtype=bool)
    lfc = np.full(len(enh), np.nan)
    e_chr, e_s, e_e = enh["chr"].to_numpy(), enh["start"].to_numpy(), enh["end"].to_numpy()
    for chrom, t in regions.groupby("chr", sort=False):
        ts, te, tl = t["start"].to_numpy(), t["end"].to_numpy(), t["logFC"].to_numpy()
        for i in np.flatnonzero(e_chr == chrom):
            m = (ts < e_e[i]) & (te > e_s[i])
            if m.any():
                hit[i] = True
                lfc[i] = tl[m][np.argmax(np.abs(tl[m]))]
    return hit, lfc


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--starr", type=Path, required=True)
    ap.add_argument("--classes", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    a = ap.parse_args()
    out_dir = a.out_dir or a.classes.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    R = read_starr(a.starr)
    n_regions = R[["chr", "start", "end"]].drop_duplicates().shape[0]
    print(f"STARR-seq: {n_regions:,} dex-responsive regions")
    print(R.groupby(["time", "direction"]).size().unstack().to_string())

    C = pd.read_csv(a.classes, usecols=["enhancer_id", "class", "GR_bound"], low_memory=False)
    if C.enhancer_id.duplicated().any():
        raise ValueError("duplicated enhancer IDs")
    c = C.enhancer_id.str.extract(r"^(chr[^_]+)_(\d+)_(\d+)$")
    if c[0].isna().any():
        raise ValueError("enhancer IDs that are not chr_start_end")
    C["chr"], C["start"], C["end"] = c[0], c[1].astype(int), c[2].astype(int)
    C["GR_bound"] = C.GR_bound.astype(str).str.lower().isin(["true", "1"])

    M = C[["enhancer_id"]].copy()
    for direction in ("induced", "repressed"):
        for t in TIMES:
            M[f"starr_{direction}_{t}h"], _ = overlap(C, R[(R.time == t) & (R.direction == direction)])
        M[f"starr_{direction}_any"] = M[[f"starr_{direction}_{t}h" for t in TIMES]].any(axis=1)
        _, M[f"starr_{direction}_logFC"] = overlap(C, R[R.direction == direction])
    flags = [c for c in M.columns if c.startswith("starr_") and not c.endswith("logFC")]
    M[flags] = M[flags].astype(int)
    M.to_csv(out_dir / "matrix_starr.tsv", sep="\t")
    either = (M.starr_induced_any | M.starr_repressed_any).sum()
    print(f"\n{int(either):,} of {len(M):,} enhancers ({100 * either / len(M):.1f} %) overlap a dex-responsive region")

    D = M.merge(C[["enhancer_id", "class", "GR_bound"]], on="enhancer_id", validate="one_to_one")
    rows = []
    for status, label in ((True, "GR-bound"), (False, "not GR-bound"), (None, "all")):
        S = D if status is None else D[D.GR_bound == status]
        nr = S[S["class"] == "NR"]
        for f in flags:
            r = {"GR_status": label, "feature": f}
            for cl in ("UP", "NR", "DN"):
                x = S.loc[S["class"] == cl, f]
                r[f"n_{cl}"] = len(x); r[f"overlap_{cl}"] = int(x.sum()); r[f"pct_{cl}"] = 100 * x.mean()
                if cl != "NR":
                    table = [[int(x.sum()), int(len(x) - x.sum())], [int(nr[f].sum()), int(len(nr) - nr[f].sum())]]
                    r[f"OR_{cl}_vs_NR"], r[f"P_{cl}_vs_NR"] = stats.fisher_exact(table)
            rows.append(r)
    S = pd.DataFrame(rows)
    pcols = [c for c in S.columns if c.startswith("P_")]
    p = S[pcols].to_numpy().ravel()
    adj = stats.false_discovery_control(p, method="bh").reshape(len(S), len(pcols))
    for i, c in enumerate(pcols):
        S["Padj_" + c[2:]] = adj[:, i]
    S.to_csv(out_dir / "starr_by_class.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    show = S[S.feature.isin(["starr_induced_1h", "starr_induced_any", "starr_repressed_any"])]
    print("\n" + show[["GR_status", "feature", "n_UP", "overlap_UP", "pct_UP", "n_NR", "overlap_NR", "pct_NR", "n_DN",
                       "overlap_DN", "pct_DN", "Padj_UP_vs_NR", "Padj_DN_vs_NR"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"\nwrote {out_dir / 'matrix_starr.tsv'} and {out_dir / 'starr_by_class.tsv'}")
