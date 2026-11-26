"""Do neighbouring enhancers respond together?

Four descriptive tables on the linear genome; no model and no feature matrix are needed.

  1  nearest neighbour     class of the nearest other enhancer, per class, against the expectation when the
                           class labels are shuffled within each chromosome
  2  local composition     UP / DN / NR enhancers within --window kb of an enhancer, per class, against the same
                           expectation
  3  fold-change coupling  Spearman correlation of the PRO-seq log2 fold change between consecutive enhancers,
                           by the distance between them
  4  GR peak of unbound    for enhancers without a GR peak: the nearest GR peak, whether it lies in another
     enhancers             enhancer, and the class of that enhancer

    uv run python scripts/24_neighbour_coresponse.py
    uv run python scripts/24_neighbour_coresponse.py --min-gap 0 2 5 10 --window 50 --n-perm 5000

Input ([inputs] ml_inputs)
    prematrix (coordinates, fold change, p-value -> classes) and the GR peak file (config.GR_PEAK_FILE)
Output (<analysis>/class_analyses/, or --out)
    neighbour_nearest_class.tsv, neighbour_local_composition.tsv, neighbour_log2fc_coupling.tsv,
    neighbour_gr_peak_of_unbound.tsv

Two points for reading the tables.
- Closely spaced dREG calls can come from one transcribed region and then share reads, which makes neighbours
  agree for a technical reason. Tables 1 and 2 are therefore repeated for several minimum distances (--min-gap,
  in kb; neighbours closer than that are ignored); a result should hold beyond the first few kb. Table 3 shows
  the distance dependence directly.
- The expectation keeps every enhancer position and the class counts per chromosome and shuffles which enhancer
  carries which class. It does not model larger domains, so an enrichment says "more than by chance given the
  chromosome", not why.

Distances are between the nearest edges (0 = touching), in base pairs unless stated.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr

from a549_enhancers import paths
from a549_enhancers.config import GLOBAL_RANDOM_STATE, GR_PEAK_FILE
from a549_enhancers.enhancers import enhancer_table
from a549_enhancers.intervals import read_peaks

CLASSES = ["UP", "DN", "NR"]
GAP_BINS_KB = [0, 1, 2, 5, 10, 25, 50, 100, 250, 1000]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--min-gap", nargs="+", type=float, default=[0, 2, 5, 10], help="minimum distances in kb")
    p.add_argument("--window", type=float, default=50, help="window of table 2 in kb")
    p.add_argument("--n-perm", type=int, default=2000)
    p.add_argument("--peaks", default=None, help="GR peak file (default: config.GR_PEAK_FILE in ml_inputs)")
    p.add_argument("--out", default=None)
    return p.parse_args()


def pair_table(E: pd.DataFrame, max_gap: float) -> pd.DataFrame:
    """All pairs (i < j) of enhancers on one chromosome whose edge distance is at most max_gap.
    E must be sorted by chr, start; i and j are row positions."""
    I, J, G = [], [], []
    for _, idx in E.groupby("chr", sort=False).indices.items():
        idx = np.sort(idx)
        s, e = E["start"].to_numpy()[idx], E["end"].to_numpy()[idx]
        for a in range(len(idx)):
            b = a + 1
            while b < len(idx) and s[b] - e[a] <= max_gap:
                I.append(idx[a]); J.append(idx[b]); G.append(max(0, s[b] - e[a]))
                b += 1
    return pd.DataFrame({"i": I, "j": J, "gap": G})


def nearest_neighbour(E: pd.DataFrame, min_gap: float) -> np.ndarray:
    """Row position of the nearest other enhancer at least min_gap away (-1 if there is none)."""
    nn = np.full(len(E), -1)
    start, end = E["start"].to_numpy(), E["end"].to_numpy()
    for _, idx in E.groupby("chr", sort=False).indices.items():
        idx = np.sort(idx)
        s, e = start[idx], end[idx]
        for a in range(len(idx)):
            best, best_gap = -1, np.inf
            b = a - 1
            while b >= 0:
                gap = max(0, s[a] - e[b])
                if gap >= min_gap:
                    best, best_gap = b, gap
                    break
                b -= 1
            b = a + 1
            while b < len(idx):
                gap = max(0, s[b] - e[a])
                if gap >= min_gap:
                    if gap < best_gap:
                        best = b
                    break
                b += 1
            if best >= 0:
                nn[idx[a]] = idx[best]
    return nn


def shuffled_within_chromosome(codes: np.ndarray, chrom_rows: list, rng) -> np.ndarray:
    out = codes.copy()
    for rows in chrom_rows:
        out[rows] = codes[rows][rng.permutation(len(rows))]
    return out


def compare_with_shuffles(observed: np.ndarray, shuffled: np.ndarray):
    """observed: (3, 3); shuffled: (n_perm, 3, 3) -> expected mean, ratio, one-sided empirical P (enrichment)."""
    expected = shuffled.mean(axis=0)
    p = (1 + (shuffled >= observed[None] - 1e-12).sum(axis=0)) / (len(shuffled) + 1)
    return expected, observed / np.where(expected > 0, expected, np.nan), p


def merged_peaks(peaks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for chrom, p in peaks.sort_values(["chr", "start"]).groupby("chr", sort=False):
        cs, ce = None, None
        for s, e in zip(p["start"].to_numpy(), p["end"].to_numpy()):
            if cs is None or s > ce:
                if cs is not None:
                    rows.append((chrom, cs, ce))
                cs, ce = s, e
            else:
                ce = max(ce, e)
        rows.append((chrom, cs, ce))
    return pd.DataFrame(rows, columns=["chr", "start", "end"])


def main():
    a = parse_args()
    out = Path(a.out) if a.out else paths.PROJECT / "class_analyses"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(GLOBAL_RANDOM_STATE)

    E = enhancer_table(peaks=a.peaks).sort_values(["chr", "start"]).reset_index(drop=True)
    n = len(E)
    codes = E["class"].map({c: k for k, c in enumerate(CLASSES)}).to_numpy()
    onehot = np.eye(3)[codes]
    chrom_rows = [np.sort(idx) for idx in E.groupby("chr", sort=False).indices.values()]
    class_n = np.bincount(codes, minlength=3)
    print(f"{n:,} enhancers: " + ", ".join(f"{c} {class_n[k]}" for k, c in enumerate(CLASSES)))
    shuffles = [shuffled_within_chromosome(codes, chrom_rows, rng) for _ in range(a.n_perm)]

    # ------------------------------------------------------------------ 1 nearest neighbour
    rows = []
    for d_kb in a.min_gap:
        nn = nearest_neighbour(E, d_kb * 1000)
        has = nn >= 0
        gap = np.maximum(0, np.maximum(E["start"].to_numpy()[nn[has]] - E["end"].to_numpy()[has],
                                       E["start"].to_numpy()[has] - E["end"].to_numpy()[nn[has]]))

        def shares(c):                                      # row: class of the enhancer, column: class of its neighbour
            m = np.zeros((3, 3))
            np.add.at(m, (c[has], c[nn[has]]), 1)
            return m / np.maximum(m.sum(axis=1, keepdims=True), 1)

        obs = shares(codes)
        exp, ratio, p = compare_with_shuffles(obs, np.stack([shares(c) for c in shuffles]))
        for i, ci in enumerate(CLASSES):
            sel = codes[has] == i
            for j, cj in enumerate(CLASSES):
                rows.append({"min_gap_kb": d_kb, "class": ci, "n_enhancers": int(sel.sum()),
                             "median_distance_to_neighbour_bp": float(np.median(gap[sel])) if sel.any() else np.nan,
                             "neighbour_class": cj, "share_observed": obs[i, j], "share_expected": exp[i, j],
                             "observed_over_expected": ratio[i, j], "empirical_p_enrichment": p[i, j]})
    T1 = pd.DataFrame(rows)
    T1.round(5).to_csv(out / "neighbour_nearest_class.tsv", sep="\t", index=False)

    # ------------------------------------------------------------------ 2 local composition
    P = pair_table(E, a.window * 1000)
    rows = []
    for d_kb in a.min_gap:
        q = P[P["gap"] >= d_kb * 1000]
        A = sparse.coo_matrix((np.ones(2 * len(q)), (np.r_[q["i"], q["j"]], np.r_[q["j"], q["i"]])), shape=(n, n)).tocsr()

        def mean_counts(c):                                 # mean number of class-j enhancers around class-i enhancers
            around = A @ np.eye(3)[c]
            return np.stack([around[c == i].mean(axis=0) if (c == i).any() else np.zeros(3) for i in range(3)])

        obs = mean_counts(codes)
        exp, ratio, p = compare_with_shuffles(obs, np.stack([mean_counts(c) for c in shuffles]))
        total = np.asarray(A.sum(axis=1)).ravel()
        for i, ci in enumerate(CLASSES):
            for j, cj in enumerate(CLASSES):
                rows.append({"min_gap_kb": d_kb, "window_kb": a.window, "class": ci, "n_enhancers": int(class_n[i]),
                             "mean_enhancers_in_window": total[codes == i].mean(), "neighbour_class": cj,
                             "mean_count_observed": obs[i, j], "mean_count_expected": exp[i, j],
                             "observed_over_expected": ratio[i, j], "empirical_p_enrichment": p[i, j]})
    T2 = pd.DataFrame(rows)
    T2.round(5).to_csv(out / "neighbour_local_composition.tsv", sep="\t", index=False)

    # ------------------------------------------------------------------ 3 fold-change coupling
    lfc = E["log2FC"].to_numpy()
    consecutive = []
    for idx in chrom_rows:
        order = idx[np.argsort(E["start"].to_numpy()[idx])]
        consecutive += [(i, j, max(0, E["start"].iat[j] - E["end"].iat[i])) for i, j in zip(order[:-1], order[1:])]
    C = pd.DataFrame(consecutive, columns=["i", "j", "gap"])
    C = C[np.isfinite(lfc[C["i"]]) & np.isfinite(lfc[C["j"]])]
    rows = []
    for lo, hi in zip(GAP_BINS_KB[:-1], GAP_BINS_KB[1:]):
        g = C[(C["gap"] >= lo * 1000) & (C["gap"] < hi * 1000)]
        if len(g) < 10:
            rows.append({"distance_kb": f"{lo}-{hi}", "n_pairs": len(g)})
            continue
        x, y = lfc[g["i"]], lfc[g["j"]]
        rho, p = spearmanr(x, y)
        responsive = (codes[g["i"]] != 2) & (codes[g["j"]] != 2)
        rows.append({"distance_kb": f"{lo}-{hi}", "n_pairs": len(g), "spearman_log2FC": rho, "spearman_p": p,
                     "share_same_sign": float(np.mean(np.sign(x) == np.sign(y))),
                     "n_pairs_both_responsive": int(responsive.sum()),
                     "share_same_class_among_both_responsive":
                         float(np.mean(codes[g["i"]][responsive] == codes[g["j"]][responsive])) if responsive.any() else np.nan})
    T3 = pd.DataFrame(rows)
    T3.round(5).to_csv(out / "neighbour_log2fc_coupling.tsv", sep="\t", index=False)

    # ------------------------------------------------------------------ 4 GR peak of unbound enhancers
    peak_file = Path(a.peaks) if a.peaks else paths.ml_input(GR_PEAK_FILE)
    PK = merged_peaks(read_peaks(peak_file))
    # class(es) of the enhancers that each merged peak overlaps
    peak_classes = [set() for _ in range(len(PK))]
    nearest_peak = np.full(n, -1); nearest_dist = np.full(n, np.nan)
    for chrom, rows_e in E.groupby("chr", sort=False).indices.items():
        pk = PK.index[PK["chr"] == chrom].to_numpy()
        if len(pk) == 0:
            continue
        ps, pe = PK["start"].to_numpy()[pk], PK["end"].to_numpy()[pk]          # merged: sorted and disjoint
        for r in rows_e:
            s, e = E["start"].iat[r], E["end"].iat[r]
            k = np.searchsorted(ps, e, side="left")                             # first peak starting at or after e
            cand = [c for c in (k - 1, k) if 0 <= c < len(pk)]
            dist = [max(0, ps[c] - e, s - pe[c]) for c in cand]
            best = int(np.argmin(dist))
            nearest_peak[r], nearest_dist[r] = pk[cand[best]], dist[best]
            if dist[best] == 0:
                peak_classes[pk[cand[best]]].add(E["class"].iat[r])
            lo = np.searchsorted(pe, s, side="right")                           # all peaks overlapping this enhancer
            for c in range(lo, k):
                if ps[c] < e and pe[c] > s:
                    peak_classes[pk[c]].add(E["class"].iat[r])
    unbound = ~E["GR_bound"].to_numpy()
    bound_share = {c: float(np.mean(E.loc[~unbound, "class"] == c)) for c in CLASSES}
    rows = []
    for c in CLASSES:
        sel = unbound & (E["class"].to_numpy() == c) & (nearest_peak >= 0)
        d = nearest_dist[sel]
        in_enh = np.array([len(peak_classes[k]) > 0 for k in nearest_peak[sel]])
        row = {"class": c, "n_not_GR_bound": int(sel.sum()), "median_distance_to_GR_peak_bp": float(np.median(d)),
               "share_peak_within_5kb": float(np.mean(d <= 5000)), "share_peak_within_50kb": float(np.mean(d <= 50000)),
               "share_nearest_peak_in_another_enhancer": float(in_enh.mean())}
        for cj in CLASSES:
            hit = np.array([cj in peak_classes[k] for k in nearest_peak[sel]])
            row[f"share_nearest_peak_in_{cj}_enhancer"] = float(hit.mean())
            row[f"of_those_in_an_enhancer_share_{cj}"] = float(hit[in_enh].mean()) if in_enh.any() else np.nan
            row[f"reference_share_{cj}_among_GR_bound_enhancers"] = bound_share[cj]
        near = d <= 5000
        row["within_5kb_share_nearest_peak_in_UP_enhancer"] = float(
            np.mean([("UP" in peak_classes[k]) for k in nearest_peak[sel][near]])) if near.any() else np.nan
        rows.append(row)
    T4 = pd.DataFrame(rows)
    T4.round(5).to_csv(out / "neighbour_gr_peak_of_unbound.tsv", sep="\t", index=False)

    pd.set_option("display.width", 240)
    same = T1[T1["class"] == T1["neighbour_class"]]
    print("\n1 nearest neighbour of the same class\n" + same.drop(columns=["neighbour_class"]).round(3).to_string(index=False))
    same = T2[T2["class"] == T2["neighbour_class"]]
    print(f"\n2 enhancers of the same class within {a.window:g} kb\n"
          + same.drop(columns=["neighbour_class", "window_kb"]).round(3).to_string(index=False))
    print("\n3 fold-change coupling of consecutive enhancers\n" + T3.round(3).to_string(index=False))
    print("\n4 nearest GR peak of enhancers without a GR peak\n" + T4.iloc[:, :6].round(3).to_string(index=False))
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
