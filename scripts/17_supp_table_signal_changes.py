"""Supplementary Table S3 - ChIP-seq / ATAC-seq signal per enhancer class and GR-binding status.

The values behind the feature-map section of the Results (Supplementary Figures S4 and S5), in four sheets:

  Dex change        factors measured in control and dex-treated cells: median percentile rank before and after dex
                    and the level-adjusted rank change (delta) per class and per class x GR binding
  Matched to NR     UP and DN enhancers compared with the 20 nearest NR enhancers of the same GR-binding status
  Single condition  GR after dex, factors profiled in control cells only, and motif frequencies
  NR topics         NR enhancers per main chromatin topic: size, GR binding and median delta per factor
  Matching balance  how close the matched NR enhancers are to their UP / DN enhancer in the matching variables,
                    and how often one NR enhancer is reused (the numbers quoted in the Methods)

Definitions are the same as in figures/text_numbers_feature_map.py and figures/text_numbers_matched_comparison.py
(percentile rank within each track over all enhancers; delta = rank after dex minus the median rank after dex of
NR enhancers in the same of 20 bins of control rank; matching with 20 nearest NR enhancers, bootstrap seed 0).

Input:  <ml_outputs>/feature_matrix_all_classes_GRannotated.csv   (scripts/08_annotate_gr_binding.py)
        <nr_topics>/MS_nr_topics_usage.tsv                        (scripts/11_nr_topics.py; optional)
Output: <figures>/Supp_Table_S3_signal_changes.xlsx  and one TSV per sheet next to it

Run:  uv run python scripts/17_supp_table_signal_changes.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon
from sklearn.neighbors import NearestNeighbors

from a549_enhancers.paths import FIGURES, ML, NR_TOPICS   # local folders, set in paths.toml

N_BINS = 20                 # bins of control rank for delta
K = 20                      # matched NR enhancers per UP / DN enhancer
N_BOOT, SEED = 2000, 0      # bootstrap of the median matched difference
GR_DEX = "NR3C1|_dex_"
TRACKS = {   # factor -> (control column, dex column); "prefix|tag" = column starting with prefix_ and containing tag
    "ATAC": ("ATAC_ctrl", "ATAC_dex_1h"), "Pol II": ("POLR2A|_eth_", "POLR2A|_dex_"),
    "H3K27ac": ("H3K27ac_ENCFF054OOT", "H3K27ac|_dex_"), "H3K4me1": ("H3K4me1_ENCFF580SCK", "H3K4me1|_dex_"),
    "H3K4me3": ("H3K4me3_ENCFF529CES", "H3K4me3|_dex_"), "H3K79me2": ("H3K79me2|_eth_", "H3K79me2|_dex_"),
    "H2A.Z": ("H2AFZ|_eth_", "H2AFZ|_dex_"), "EP300": ("EP300_ENCFF376VNJ", "EP300|_dex_"),
    "FOSL2": ("FOSL2_ENCFF129REO", "FOSL2|_dex_"), "cJUN": ("JUN_ENCFF957TUK", "JUN|_dex_"),
    "BCL3": ("BCL3_ENCFF856OZJ", "BCL3|_dex_"), "FOXA1": ("FOXA1|_eth_", "FOXA1|_dex_"),
    "SMC3": ("SMC3_ENCFF029NYA", "SMC3|_dex_"), "RAD21": ("RAD21_ENCFF778YHU", "RAD21|_dex_"),
    "CTCF": ("CTCF_ENCFF083AEY", "CTCF|_dex_"),
}
MATCHED = ["ATAC", "Pol II", "H3K27ac", "H2A.Z", "EP300", "FOSL2", "cJUN", "BCL3", "FOXA1", "SMC3", "RAD21", "CTCF"]
SINGLE = {"GR (after dex)": GR_DEX, "PHF8 (control)": "PHF8_ENCFF690AJW", "TCF12 (control)": "TCF12|_eth_",
          "GATA3 (control)": "GATA3_ENCFF485KDU", "TEAD4 (control)": "TEAD4_ENCFF186WSI"}
# Motif indicators, matched by the start of the column name (first prefix found in the matrix is used). With the
# motif groups of the current matrix: "GRE(NR),IR3 group (5)" = GRE motif of the A549 GR ChIP-seq (representative of
# GRE / ARE / PGR / PR motifs); "Jun-AP1(bZIP) group (10)" = Jun-AP1 motif (representative of ten AP-1 family motifs).
# The earlier matrix (single motifs) is matched by the second prefix.
MOTIF_PREFIXES = {"GRE motif": ["motif_binary::GRE(NR),IR3 group", "motif_binary::GRE(NR),IR3/"],
                  "AP-1 motif": ["motif_binary::Jun-AP1(bZIP) group", "motif_binary::AP-1(bZIP)/"]}


def motif_columns(columns):
    """name -> columns of the feature matrix, from MOTIF_PREFIXES; stops if a motif is not in the matrix"""
    found = {}
    for name, prefixes in MOTIF_PREFIXES.items():
        for p in prefixes:
            hit = [c for c in columns if c.startswith(p)]
            if hit:
                found[name] = hit
                break
        else:
            raise KeyError(f"{name}: no column starting with {prefixes} in the feature matrix")
    return found


MOTIFS = {}       # filled from the feature matrix after it is read: MOTIFS.update(motif_columns(D.columns))
CLASSES = ["UP", "DN", "NR"]
GROUPS = ["UP", "DN", "NR", "UP GR+", "UP GR-", "DN GR+", "DN GR-", "NR GR+", "NR GR-"]
PAIRS = [("UP GR+", "NR GR+"), ("DN GR+", "NR GR+"), ("UP GR-", "NR GR-"), ("DN GR-", "NR GR-")]


def bh(p: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjusted P values; missing values stay missing."""
    out = pd.Series(np.nan, index=p.index, dtype=float)
    v = p.dropna()
    if v.empty:
        return out
    order = v.sort_values()
    adj = (order * len(order) / np.arange(1, len(order) + 1))[::-1].cummin()[::-1].clip(upper=1)
    out[adj.index] = adj
    return out


def main() -> None:
    fm = ML / "feature_matrix_all_classes_GRannotated.csv"
    if not fm.exists():
        raise FileNotFoundError(f"{fm}\nRun scripts/07_save_full_feature_matrix.py and 08_annotate_gr_binding.py first.")
    D = pd.read_csv(fm, low_memory=False).set_index("enhancer_id")
    MOTIFS.update(motif_columns(D.columns))
    cls = D["class"].astype(str)
    grb = D.GR_bound.astype(str).str.lower().eq("true")
    grp = pd.Series(np.where(grb, cls + " GR+", cls + " GR-"), index=D.index)
    nr = cls.eq("NR").to_numpy()
    member = {g: (cls.eq(g) if " " not in g else grp.eq(g)).to_numpy() for g in GROUPS}
    reference = {g: "NR" if " " not in g else "NR " + g.split()[1] for g in GROUPS}

    def colname(spec: str) -> str:
        if spec in D.columns:
            return spec
        pre, tag = spec.split("|")
        hit = [c for c in D.columns if c.startswith(pre + "_") and tag in c]
        if not hit:
            raise KeyError(f"no column for {spec}")
        return hit[0]

    def rank(col: str) -> np.ndarray:
        return 100 * D[col].rank(pct=True).to_numpy()

    gr_dex = rank(colname(GR_DEX))

    # ------------------------------------------------------------------ sheet 1: dex change
    R0, R1, DELTA, rows = {}, {}, {}, []
    for f, (a, b) in TRACKS.items():
        ca, cb = colname(a), colname(b)
        r0, r1 = rank(ca), rank(cb)
        bins = np.clip((r0 / 100 * N_BINS).astype(int), 0, N_BINS - 1)
        ref = pd.Series(r1[nr]).groupby(bins[nr]).median()
        delta = r1 - pd.Series(bins).map(ref).to_numpy()
        R0[f], R1[f], DELTA[f] = r0, r1, delta
        for g in GROUPS:
            m = member[g]
            row = {"factor": f, "group": g, "n_enhancers": int(m.sum()),
                   "median_percentile_control": np.median(r0[m]), "median_percentile_dex": np.median(r1[m]),
                   "median_delta": np.median(delta[m]), "pct_delta_above_0": 100 * (delta[m] > 0).mean(),
                   "compared_with": "" if g.startswith("NR") else reference[g], "p_mann_whitney": np.nan,
                   "control_track": ca, "dex_track": cb}
            if not g.startswith("NR"):
                row["p_mann_whitney"] = mannwhitneyu(delta[m], delta[member[reference[g]]]).pvalue
            rows.append(row)
    S1 = pd.DataFrame(rows)
    S1.insert(S1.columns.get_loc("p_mann_whitney") + 1, "p_adjusted_BH", bh(S1["p_mann_whitney"]))

    # ------------------------------------------------------------------ sheet 2: matched to NR
    rng = np.random.default_rng(SEED)
    grp_arr, grb_arr = grp.to_numpy(), grb.to_numpy()

    def matched(X, r_dex, group, ref_group, check=None):
        ia, ib = np.where(grp_arr == group)[0], np.where(grp_arr == ref_group)[0]
        idx = NearestNeighbors(n_neighbors=K).fit(X[ib]).kneighbors(X[ia], return_distance=False)
        diff = r_dex[ia] - np.median(r_dex[ib][idx], axis=1)
        boot = np.median(rng.choice(diff, (N_BOOT, len(diff))), axis=1)
        if check is not None:        # balance: enhancer against the median of its matched NR enhancers, per variable
            used = np.bincount(idx.ravel(), minlength=len(ib))
            bal = {"n_NR_available": len(ib), "n_NR_used": int((used > 0).sum()), "max_times_one_NR_used": int(used.max())}
            for var, v in check["variables"].items():
                d = np.abs(v[ia] - np.median(v[ib][idx], axis=1))
                bal.update({f"{var}__median_abs_difference": np.median(d), f"{var}__p95_abs_difference": np.percentile(d, 95),
                            f"{var}__max_abs_difference": d.max()})
            balance.append({**check["id"], **bal})
        return {"median_difference": np.median(diff), "ci95_low": np.percentile(boot, 2.5),
                "ci95_high": np.percentile(boot, 97.5), "p_signed_rank": wilcoxon(diff).pvalue}

    rows, balance = [], []
    for f in MATCHED:
        r0, r1 = R0[f], R1[f]
        for group, ref_group in PAIRS:
            row = {"factor": f, "group": group, "matched_NR_group": ref_group, "n_enhancers": int((grp_arr == group).sum()),
                   "spearman_dex_vs_GR_dex_all": spearmanr(r1, gr_dex)[0],
                   "spearman_dex_vs_GR_dex_GR_bound": spearmanr(r1[grb_arr], gr_dex[grb_arr])[0]}
            for name, X in (("matched_on_control", r0[:, None]),
                            ("matched_on_control_and_GR", np.column_stack([r0, gr_dex]))):
                check = {"id": {"factor": f, "group": group, "matched_NR_group": ref_group, "matching": name,
                                "n_enhancers": int((grp_arr == group).sum())},
                         "variables": {"control_rank": r0, "GR_rank_after_dex": gr_dex}}
                row.update({f"{name}__{k}": v for k, v in matched(X, r1, group, ref_group, check).items()})
            rows.append(row)
    S2 = pd.DataFrame(rows)
    S5 = pd.DataFrame(balance)
    for name in ("matched_on_control", "matched_on_control_and_GR"):
        S2.insert(S2.columns.get_loc(f"{name}__p_signed_rank") + 1, f"{name}__p_adjusted_BH",
                  bh(S2[f"{name}__p_signed_rank"]))

    # ------------------------------------------------------------------ sheet 3: single-condition tracks and motifs
    rows = [{"feature": "GR-bound (GR peak after dex)", "value": "% of enhancers",
             **{g: 100 * grb_arr[member[g]].mean() for g in GROUPS}}]
    for f, spec in SINGLE.items():
        r = rank(colname(spec))
        rows.append({"feature": f, "value": "median percentile rank", **{g: np.median(r[member[g]]) for g in GROUPS},
                     "track_or_motif": colname(spec)})
    for f, cols in MOTIFS.items():
        cols = [c for c in cols if c in D.columns]
        v = (D[cols].fillna(0) > 0).any(axis=1).to_numpy()
        rows.append({"feature": f, "value": "% of enhancers", **{g: 100 * v[member[g]].mean() for g in GROUPS},
                     "track_or_motif": "; ".join(c.replace("motif_binary::", "") for c in cols)})
    S3 = pd.DataFrame(rows)
    S3 = pd.concat([pd.DataFrame([{"feature": "Number of enhancers", "value": "n",
                                   **{g: int(member[g].sum()) for g in GROUPS}}]), S3], ignore_index=True)

    # ------------------------------------------------------------------ sheet 4: NR topics (optional)
    S4 = None
    topics = NR_TOPICS / "MS_nr_topics_usage.tsv"
    if topics.exists():
        U = pd.read_csv(topics, sep="\t").set_index("enhancer_id").reindex(D.index[nr])
        tc = [c for c in U.columns if c[:1] == "T" and c[1:2].isdigit()]
        if U[tc].isna().any().any():
            raise ValueError(f"{topics.name} does not contain every NR enhancer of the feature matrix")
        main_topic = U[tc].idxmax(axis=1)
        share = U[tc].max(axis=1)
        idx_nr = D.index[nr]
        rows = []
        for t in sorted(main_topic.unique()):
            m = (main_topic == t).to_numpy()
            row = {"main_topic": t, "n_enhancers": int(m.sum()), "pct_of_NR": 100 * m.mean(),
                   "pct_GR_bound": 100 * grb.loc[idx_nr].to_numpy()[m].mean(),
                   "median_share_of_main_topic": float(share[m].median()),
                   "median_percentile_GR_dex": np.median(gr_dex[nr][m])}
            row.update({f"median_delta_{f}": np.median(DELTA[f][nr][m]) for f in TRACKS})
            rows.append(row)
        S4 = pd.DataFrame(rows)
    else:
        print(f"({topics} not found - sheet 'NR topics' left out)")

    # ------------------------------------------------------------------ write
    readme = [
        ("Supplementary Table S3", "ChIP-seq and ATAC-seq signal per enhancer response class and GR-binding status "
                                   "(values behind Supplementary Figures S4 and S5)."),
        ("Enhancer groups", "UP, DN, NR: response classes. GR+ / GR-: enhancer overlaps / does not overlap a GR peak after dex."),
        ("Percentile rank", "Rank of the mean signal of an enhancer within one track over all enhancers, from 0 to 100."),
        ("delta", f"Level-adjusted rank change: percentile rank after dex minus the median percentile rank after dex of NR "
                  f"enhancers in the same of {N_BINS} equal-width bins of control rank. Per enhancer, not smoothed."),
        ("Sheet 'Dex change'", "Per factor and group: medians of the control rank, the rank after dex and delta; share of "
                               "enhancers with delta above zero; two-sided Mann-Whitney test of delta against NR enhancers "
                               "(for GR+ / GR- groups: NR enhancers of the same GR-binding status); Benjamini-Hochberg "
                               "adjustment over all tests of the sheet."),
        ("Sheet 'Matched to NR'", f"Each UP or DN enhancer is compared with the {K} NR enhancers of the same GR-binding status "
                                  f"that are nearest in (a) the control rank of the factor or (b) the control rank of the "
                                  f"factor and the GR rank after dex. Difference = rank after dex of the enhancer minus the "
                                  f"median rank after dex of its matched NR enhancers. Given: median difference, bootstrap 95% "
                                  f"confidence interval of the median ({N_BOOT} resamples), two-sided Wilcoxon signed-rank test "
                                  f"against zero with Benjamini-Hochberg adjustment per matching, and the Spearman correlation "
                                  f"of the factor after dex with GR after dex."),
        ("Sheet 'Single condition'", "GR after dex and factors profiled in control cells only: median percentile rank per group. "
                                     "Motifs and GR peaks: % of enhancers per group."),
        ("Sheet 'NR topics'", "NR enhancers by main chromatin topic (topic with the largest weight): size, % GR-bound, median "
                              "weight of the main topic, median GR rank after dex and median delta per factor."),
        ("Sheet 'Matching balance'", f"Quality of the matching, per factor, group and matching. For each UP or DN enhancer, the "
                                     f"absolute difference (percentile points) between its value and the median of its {K} "
                                     f"matched NR enhancers, for the control rank of the factor and for the GR rank after dex: "
                                     f"median, 95th percentile and maximum over the enhancers of the group. For "
                                     f"'matched_on_control' the GR rank was not a matching variable and is shown for "
                                     f"comparison. Also: NR enhancers available and used, and the largest number of times one "
                                     f"NR enhancer was matched."),
        ("Tracks", "Column names of the feature matrix; ENCODE file accessions are listed in Supplementary Table S1."),
    ]
    sheets = {"Dex change": S1, "Matched to NR": S2, "Single condition": S3}
    if S4 is not None:
        sheets["NR topics"] = S4
    sheets["Matching balance"] = S5
    FIGURES.mkdir(parents=True, exist_ok=True)
    for name, tab in sheets.items():
        tab.to_csv(FIGURES / f"Supp_Table_S3_{name.replace(' ', '_')}.tsv", sep="\t", index=False, float_format="%.4g")
    xlsx = FIGURES / "Supp_Table_S3_signal_changes.xlsx"
    try:
        with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
            pd.DataFrame(readme, columns=["Item", "Description"]).to_excel(writer, sheet_name="README", index=False)
            for name, tab in sheets.items():
                num = tab.select_dtypes("float").columns
                tab.assign(**{c: tab[c].map(lambda v: float(f"{v:.4g}") if pd.notna(v) else v) for c in num}).to_excel(
                    writer, sheet_name=name, index=False, freeze_panes=(1, 2))
        print("written:", xlsx.name, "and", len(sheets), "TSV files in the figures folder")
    except ImportError:
        print("openpyxl is not installed - only the TSV files were written (uv add openpyxl, then rerun).")

    # ------------------------------------------------------------------ numbers for the Methods (matching balance)
    b = S5[S5.matching == "matched_on_control_and_GR"]
    cols = lambda stat: [f"{v}__{stat}_abs_difference" for v in ("control_rank", "GR_rank_after_dex")]
    worst = b.loc[b[cols("max")].max(axis=1).idxmax()]
    print(f"\nmatching balance (control rank + GR after dex; {b.factor.nunique()} factors, percentile points):")
    print(f"  median absolute difference, largest over factors and groups: {b[cols('median')].max().max():.2f}")
    print(f"  95th percentile, largest: {b[cols('p95')].max().max():.1f}")
    print(f"  maximum: {b[cols('max')].max().max():.1f}  ({worst.factor}, {worst.group}; "
          f"control {worst['control_rank__max_abs_difference']:.1f}, GR {worst['GR_rank_after_dex__max_abs_difference']:.1f})")
    print(f"  one NR enhancer matched at most {int(b.max_times_one_NR_used.max())} times")
    print(b.groupby('group')[cols('median') + cols('p95') + cols('max') + ['max_times_one_NR_used']].max().round(2).to_string())

    # ------------------------------------------------------------------ short report
    print(f"{len(D)} enhancers:", {g: int(member[g].sum()) for g in GROUPS})
    wide = S1.pivot(index="factor", columns="group", values="median_delta").reindex(index=list(TRACKS), columns=GROUPS)
    print("\nmedian delta per factor and group:")
    print(wide.round(1).to_string())
    m2 = S2.pivot(index="factor", columns="group", values="matched_on_control_and_GR__median_difference")
    print("\nmedian difference from matched NR enhancers (control rank + GR after dex):")
    print(m2.reindex(index=MATCHED, columns=[p[0] for p in PAIRS]).round(1).to_string())
    if S4 is not None:
        print("\nNR topics:")
        print(S4.set_index("main_topic").round(1).T.to_string())


if __name__ == "__main__":
    main()
