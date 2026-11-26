"""Supplementary Table S4 - ChIP-seq signal at UP, DN and NR enhancers during dex treatment (ENCODE time courses).

Formats the output of scripts/25_timecourse_by_class.py as one workbook; nothing is recalculated here.

Sheets
    README                     what the numbers are
    By class                   median signal rank and quartiles per track for UP, DN and NR enhancers
    By class and GR binding    the same for each class split by GR binding (GR peak after dex)
    Tracks                     ENCODE file and feature-matrix column of every time point

Input  (<analysis>/class_analyses/, or --input-dir)
    timecourse_by_class.tsv, timecourse_tracks.tsv        from scripts/25_timecourse_by_class.py
Output (<figures>/, or --out-dir)
    Supp_Table_S4_timecourse_by_class.xlsx  and one TSV per data sheet next to it

Run:  uv run python scripts/26_supp_table_timecourse.py
      uv run python scripts/26_supp_table_timecourse.py --table-number S5        # another table number
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from a549_enhancers import paths

RENAME = {"NR3C1": "GR", "JUN": "cJUN", "POLR2A": "Pol II", "H2AFZ": "H2A.Z"}
CLASSES = ["UP", "DN", "NR"]
DEX_CONCENTRATION = "100 nM"      # read from the track names (dex_100); confirm in the ENCODE metadata
COLUMNS = ["series", "target", "time_h", "time_min", "group", "n_enhancers", "median_rank", "rank_Q1", "rank_Q3",
           "ENCODE_file", "time_zero_ENCODE_file", "time_zero_from_same_series"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", default=None, help="folder with the two tables of scripts/25 (default: <analysis>/class_analyses)")
    p.add_argument("--out-dir", default=None, help="folder for the workbook (default: <figures>)")
    p.add_argument("--table-number", default="S4", help="number in the file name and the README (default: S4)")
    return p.parse_args()


def main():
    a = parse_args()
    src = Path(a.input_dir) if a.input_dir else paths.PROJECT / "class_analyses"
    out = Path(a.out_dir) if a.out_dir else paths.FIGURES
    out.mkdir(parents=True, exist_ok=True)
    f_class, f_tracks = src / "timecourse_by_class.tsv", src / "timecourse_tracks.tsv"
    for f in (f_class, f_tracks):
        if not f.exists():
            raise SystemExit(f"{f} not found - run scripts/25_timecourse_by_class.py first")

    d = pd.read_csv(f_class, sep="\t")
    t = pd.read_csv(f_tracks, sep="\t")
    for x in (d, t):
        x["target"] = x["target"].replace(RENAME)
        x["time_min"] = (x["time_h"] * 60).round().astype(int)
    d = d.rename(columns={"n": "n_enhancers", "rank_median": "median_rank", "rank_q25": "rank_Q1", "rank_q75": "rank_Q3",
                          "zero_track": "time_zero_ENCODE_file", "zero_same_series": "time_zero_from_same_series"})
    for c in ("median_rank", "rank_Q1", "rank_Q3"):
        d[c] = d[c].round(1)
    d = d[COLUMNS]
    by_class = d[d["group"].isin(CLASSES)].reset_index(drop=True)
    by_gr = d[~d["group"].isin(CLASSES)].reset_index(drop=True)
    tracks = (t.rename(columns={"column": "feature_matrix_column"})
              [["series", "target", "time_h", "time_min", "ENCODE_file", "feature_matrix_column"]])

    # checks: every track has the three classes, and every group of a track has the same time points
    n_tracks = by_class.groupby(["series", "target", "time_h"]).ngroups
    if len(by_class) != 3 * n_tracks:
        raise SystemExit("timecourse_by_class.tsv: not every track has UP, DN and NR rows")
    if n_tracks != len(tracks):
        raise SystemExit(f"{n_tracks} tracks in timecourse_by_class.tsv but {len(tracks)} in timecourse_tracks.tsv")
    sizes = by_class.groupby("group")["n_enhancers"].first()
    n_all = int(sizes.sum())
    per_series = tracks.groupby("series").agg(targets=("target", "nunique"),
                                              first_min=("time_min", lambda s: int(s[s > 0].min())),
                                              last_h=("time_h", "max"))
    not_same = sorted(d.loc[~d["time_zero_from_same_series"].astype(bool), ["series", "target"]]
                      .drop_duplicates().apply(lambda r: f"{r.target} ({r.series} series)", axis=1))
    gr_groups = "; ".join(f"{g} ({int(n):,})" for g, n in by_gr.groupby("group", sort=False)["n_enhancers"].first().items())

    series_text = []
    for s, r in per_series.iterrows():
        last = f"{r.last_h:g} h" if r.last_h >= 1 else f"{int(round(r.last_h * 60))} min"
        first = f"{r.first_min / 60:g} h" if r.first_min >= 30 and r.last_h >= 1 else f"{int(r.first_min)} min"
        series_text.append(f"{s}: {int(r.targets)} targets from {first} to {last} of {DEX_CONCENTRATION} dex")
    readme = pd.DataFrame([
        (f"Supplementary Table {a.table_number}",
         "ChIP-seq signal at UP, DN and NR enhancers during dex treatment of A549 cells (ENCODE time courses)."),
        ("Signal", f"Each track was converted to percentile ranks over all {n_all:,} enhancers (0 = lowest, 100 = highest "
                   "signal of that track). The sheets give the median rank and the quartiles (Q1, Q3) per group."),
        ("Series", "; ".join(series_text) + ". The two series are separate experiments."),
        ("Time zero", "Hour series: the control track of the feature matrix for that target. Minute series: the "
                      "untreated track of that series. 'time_zero_from_same_series' is False where the control comes "
                      "from another experiment" + (f": {', '.join(not_same)}." if not_same else " (none)."),),
        ("Groups", "Sheet 'By class': " + ", ".join(f"{c} ({int(sizes[c]):,})" for c in CLASSES) + ". Sheet 'By class "
                   "and GR binding': the classes split by overlap with a GR peak after dex (ENCFF395FIP): " + gr_groups + "."),
        ("Limits", "Class medians from one sample per time point; ranks are relative within a track; the ChIP-seq "
                   "series are separate experiments from the PRO-seq that defines the classes."),
        ("Tracks", "Sheet 'Tracks' lists the ENCODE file and the feature-matrix column of every time point."),
        ("Source", "scripts/25_timecourse_by_class.py (values), scripts/26_supp_table_timecourse.py (this workbook)."),
    ], columns=["Item", "Description"])

    stem = f"Supp_Table_{a.table_number}_timecourse"
    sheets = {"By class": by_class, "By class and GR binding": by_gr, "Tracks": tracks}
    for name, tab in sheets.items():
        tab.to_csv(out / f"{stem}_{name.lower().replace(' ', '_')}.tsv", sep="\t", index=False)
    xlsx = out / f"{stem}_by_class.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        readme.to_excel(w, sheet_name="README", index=False)
        for name, tab in sheets.items():
            tab.to_excel(w, sheet_name=name, index=False, freeze_panes=(1, 0 if name == "Tracks" else 5))
        for name, tab in {"README": readme, **sheets}.items():
            ws = w.sheets[name]
            for i, c in enumerate(tab.columns, 1):
                width = min(110, max(len(str(c)), int(tab[c].astype(str).str.len().max())) + 2)
                ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = width

    print(f"{len(tracks)} tracks | By class: {len(by_class)} rows | By class and GR binding: {len(by_gr)} rows")
    print("written:", xlsx)


if __name__ == "__main__":
    main()
