"""Dex time course of the ChIP-seq signal at UP, DN and NR enhancers (descriptive).

ENCODE has ChIP-seq of sixteen targets from 0.5 to 12 h of dex (GGR time course) and of four targets from 5 to
25 min (a second series). The script summarises every track per enhancer class and time point, so the order of
events can be read: for example when H3K27ac and EP300 change at UP enhancers relative to GR, and when AP-1 and
EP300 signal falls at DN enhancers. Nothing here enters the models.

    uv run python scripts/25_timecourse_by_class.py
    uv run python scripts/25_timecourse_by_class.py --targets NR3C1 EP300 H3K27ac JUN FOSL2

How the signal is summarised. Tracks differ in scale, so each track is first converted to percentile ranks over
all enhancers (0 = lowest, 100 = highest signal of that track); the table gives the median rank and the
quartiles per group. This is the rank scale used in the text. A flat line for NR enhancers near 50 is expected
by construction; the comparison is between the classes within a time point, and of one class across time.

Time zero
    hour series     the control track of the model for that target (config.EXCLUDE_COLUMNS decides which);
                    'zero_same_series' in the table says whether it is the 0 h sample of the time course
    minute series   the untreated tracks of that series (config.TIMECOURSE_MINUTE_ZERO)

Groups: UP, DN, NR, and the same split by GR binding (GR-bound UP, GR-bound NR, ...).

Limits to keep in mind: the classes come from PRO-seq at one time point, the ChIP-seq series are other
experiments, and the table shows class medians, not the course of single enhancers.

Input ([inputs] ml_inputs): prematrix (all signal columns), GR peak file
Output (<analysis>/class_analyses/, or --out)
    timecourse_by_class.tsv       long table: series, target, time, group, n, median rank, quartiles
    timecourse_tracks.tsv         the tracks used (column, target, series, time, ENCODE file)
    timecourse_hours.pdf, timecourse_minutes.pdf     one panel per target; lines = UP, DN, NR (median rank)
    timecourse_hours_GRbound.pdf  the same for GR-bound UP and GR-bound NR enhancers
"""
import argparse
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from a549_enhancers import paths
from a549_enhancers.config import AVERAGED_TRACKS, EXCLUDE_COLUMNS, TIMECOURSE_MINUTE_ZERO
from a549_enhancers.enhancers import enhancer_table, load_prematrix

CLS = {"UP": "#D1495B", "DN": "#3B6FB6", "NR": "#5DAA82"}
DEX = re.compile(r"^(?P<target>[^_]+)_dex_(?P<dose>[0-9.]+)_(?P<time>[0-9.]+) (?P<unit>hour|minute)_(?P<acc>ENCFF[0-9A-Z]{6})$")
CONTROL = re.compile(r"^(?P<target>[^_]+)(?:_eth_[0-9.]+_[^_]+)?_(?P<acc>ENCFF[0-9A-Z]{6})$")
RENAME = {"NR3C1": "GR", "JUN": "cJUN", "POLR2A": "Pol II", "H2AFZ": "H2A.Z"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--targets", nargs="+", default=None, help="targets to include (default: all with a series)")
    p.add_argument("--out", default=None)
    return p.parse_args()


def track_table(columns) -> pd.DataFrame:
    """Dex tracks at 100 nM with their time, and the time-zero track of each series and target."""
    rows = []
    for c in columns:
        m = DEX.match(str(c))
        if not m or float(m["dose"]) != 100 or "CRISPR" in str(c).upper():
            continue
        series = "minutes" if m["unit"] == "minute" else "hours"
        hours = float(m["time"]) / 60 if series == "minutes" else float(m["time"])
        rows.append({"column": c, "target": m["target"], "series": series, "time_h": hours, "ENCODE_file": m["acc"]})
    T = pd.DataFrame(rows)
    # the 1 h track that is not a model feature is left out (two tracks exist for some targets)
    T = T[~T["column"].isin(EXCLUDE_COLUMNS)]
    hour_targets = T.loc[(T["series"] == "hours") & (T["time_h"] != 1), "target"].unique()
    T = T[(T["series"] == "minutes") | T["target"].isin(hour_targets)]

    zero = []
    controls = {}                                           # target -> control column(s) of the model
    for c in columns:
        m = CONTROL.match(str(c))
        if m and "_dex_" not in str(c) and c not in EXCLUDE_COLUMNS:
            controls.setdefault(m["target"], []).append(c)
    averaged = {src: new for new, srcs in AVERAGED_TRACKS.items() for src in srcs}
    for target in hour_targets:
        cols = controls.get(target, [])
        if not cols:
            continue
        zero.append({"column": cols[0] if len(cols) == 1 else averaged.get(cols[0], cols[0]), "sources": cols,
                     "target": target, "series": "hours", "time_h": 0.0,
                     "ENCODE_file": "; ".join(CONTROL.match(c)["acc"] for c in cols)})
    for target, col in TIMECOURSE_MINUTE_ZERO.items():
        if col in columns and (T["series"].eq("minutes") & T["target"].eq(target)).any():
            zero.append({"column": col, "sources": [col], "target": target, "series": "minutes", "time_h": 0.0,
                         "ENCODE_file": CONTROL.match(col)["acc"]})
    T["sources"] = T["column"].map(lambda c: [c])
    return pd.concat([pd.DataFrame(zero), T], ignore_index=True).sort_values(["series", "target", "time_h"])


def draw(S: pd.DataFrame, series: str, groups: dict, path: Path):
    D = S[(S["series"] == series) & S["group"].isin(groups)]
    targets = list(dict.fromkeys(D["target"]))
    if not targets:
        return
    ncol = min(4, len(targets)); nrow = math.ceil(len(targets) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.3 * ncol, 2.0 * nrow), squeeze=False, sharey=True)
    unit = 60.0 if series == "minutes" else 1.0
    for ax, target in zip(axes.ravel(), targets):
        for group, (colour, style) in groups.items():
            g = D[(D["target"] == target) & (D["group"] == group)].sort_values("time_h")
            ax.fill_between(g["time_h"] * unit, g["rank_q25"], g["rank_q75"], color=colour, alpha=0.12, lw=0)
            ax.plot(g["time_h"] * unit, g["rank_median"], color=colour, ls=style, marker="o", ms=2.5, lw=1.1, label=group)
        ax.set_title(RENAME.get(target, target), fontsize=8)
        ax.set_ylim(0, 100)
        ax.tick_params(labelsize=6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes.ravel()[len(targets):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("dex (min)" if series == "minutes" else "dex (h)", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("signal rank among\nall enhancers (median)", fontsize=7)
    axes[0, 0].legend(fontsize=6, frameon=False)
    plt.rcParams["pdf.fonttype"] = 42
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    a = parse_args()
    out = Path(a.out) if a.out else paths.PROJECT / "class_analyses"
    out.mkdir(parents=True, exist_ok=True)
    pre = load_prematrix()
    E = enhancer_table(prematrix=pre)
    T = track_table(list(pre.columns))
    if a.targets:
        T = T[T["target"].isin(a.targets)]
    if T.empty:
        raise SystemExit("no time-course tracks found in the prematrix")
    zero_series = T[T["time_h"] == 0].set_index(["series", "target"])["ENCODE_file"]

    cls, bound = E["class"], E["GR_bound"]
    groups = {c: cls.eq(c) for c in CLS}
    groups.update({f"{c}, GR-bound": cls.eq(c) & bound for c in CLS})
    groups.update({f"{c}, not GR-bound": cls.eq(c) & ~bound for c in CLS})

    rows = []
    for _, t in T.iterrows():
        signal = pre[t["sources"]].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        rank = signal.rank(pct=True) * 100
        for name, mask in groups.items():
            v = rank[mask.to_numpy()]
            rows.append({"series": t["series"], "target": t["target"], "time_h": t["time_h"], "group": name,
                         "n": int(mask.sum()), "rank_median": v.median(), "rank_q25": v.quantile(0.25),
                         "rank_q75": v.quantile(0.75), "ENCODE_file": t["ENCODE_file"]})
    S = pd.DataFrame(rows)
    # an EtOH control is a track of the older ENCODE series, not the 0 h sample of the time course
    eth_zero = set(T.loc[(T["time_h"] == 0) & T["column"].str.contains("_eth_"), "target"])
    S["zero_same_series"] = (S["series"] == "minutes") | ~S["target"].isin(eth_zero)
    S["zero_track"] = [zero_series.get((s, t), "") for s, t in zip(S["series"], S["target"])]
    S.round(3).to_csv(out / "timecourse_by_class.tsv", sep="\t", index=False)
    T.drop(columns=["sources"]).to_csv(out / "timecourse_tracks.tsv", sep="\t", index=False)

    classes = {c: (col, "-") for c, col in CLS.items()}
    draw(S, "hours", classes, out / "timecourse_hours.pdf")
    draw(S, "minutes", classes, out / "timecourse_minutes.pdf")
    draw(S, "hours", {"UP, GR-bound": (CLS["UP"], "-"), "NR, GR-bound": (CLS["NR"], "-"),
                      "DN, GR-bound": (CLS["DN"], "-")}, out / "timecourse_hours_GRbound.pdf")

    n = T.groupby("series").agg(targets=("target", "nunique"), tracks=("column", "size"))
    print(n.to_string())
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
