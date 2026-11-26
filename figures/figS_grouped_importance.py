"""Grouped permutation importance of one or more cross-validation runs.

One panel per model. Each row is a predefined feature group (config.FEATURE_FAMILIES, or a cluster of correlated
tracks); all features of a group were permuted together on every held-out fold (scripts/01_run_cv.py).

    dot        mean decrease in average precision over the evaluations
               filled: positive in at least config.STABLE_MIN_FRACTION_POSITIVE_IMPORTANCE of the evaluations
               open:   below that share
    line       10th to 90th percentile of the evaluations
    tick       the largest single-feature importance among the members of the group (same evaluations), to show
               what permuting the group together adds
    colour     how the group was defined (family of targets / correlation cluster)
    right      share of evaluations with a positive decrease

    uv run python figures/figS_grouped_importance.py                       # the three models of paths.toml
    uv run python figures/figS_grouped_importance.py --tasks nr_vs_dn
    uv run python figures/figS_grouped_importance.py --tasks nr_vs_dn --top 20
    uv run python figures/figS_grouped_importance.py --run-dir nr_vs_dn=<folder>

Input (per run folder)
    summaries/all_cv_grouped_permutation_importance.csv, summaries/feature_importance_stability_summary.csv,
    data/feature_groups.csv
Output ([results] figures)
    FigS_grouped_importance.pdf / .png and FigS_grouped_importance_source.tsv
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from a549_enhancers import paths
from a549_enhancers.config import STABLE_MIN_FRACTION_POSITIVE_IMPORTANCE
from a549_enhancers.feature_groups import feature_layer, target_of

TYPE_COLOUR = {"family": "#2a78d6", "correlation cluster": "#eb6834"}      # validated pair (colour-blind safe)
TYPE_LABEL = {"family": "family of targets", "correlation cluster": "cluster of correlated tracks"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
RENAME = {"NR3C1": "GR", "JUN": "cJUN", "POLR2A": "Pol II", "POLR2AphosphoS2": "Pol II S2P", "H2AFZ": "H2A.Z"}
TITLE = {"nr_vs_up": "NR versus UP", "nr_vs_dn": "NR versus DN", "up_vs_dn": "UP versus DN"}
FAMILY_LABEL = {
    "family: all motif indicators": "all motif indicators",
    "family: GRE (motif group and score)": "GRE: motif and best score",
    "family: neighbourhood, other enhancers": "neighbourhood: other enhancers",
    "family: neighbourhood, GR peaks after dex": "neighbourhood: GR peaks after dex",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="+", default=["nr_vs_up", "up_vs_dn", "nr_vs_dn"])
    p.add_argument("--model", default="catboost")
    p.add_argument("--top", type=int, default=None, help="only the N groups with the largest mean importance")
    p.add_argument("--run-dir", nargs="*", default=[], metavar="TASK=FOLDER")
    p.add_argument("--out", default=None, help="output folder (default: [results] figures)")
    return p.parse_args()


def cluster_label(members) -> str:
    """'GATA3, JUND, SP1, TCF12, TEAD4 (control)' or 'H3K27ac (control, dex), H3K9ac (control)'."""
    conditions = {}
    for f in members:
        conditions.setdefault(RENAME.get(target_of(f), target_of(f)), set()).add(
            "dex" if feature_layer(f) == "dex" else "control")
    shared = set.union(*conditions.values()) if len({frozenset(v) for v in conditions.values()}) == 1 else None
    names = sorted(conditions, key=str.lower)
    if shared is not None:
        return f"{', '.join(names)} ({', '.join(sorted(shared))})"
    return ", ".join(f"{t} ({', '.join(sorted(conditions[t]))})" for t in names)


def group_label(group: str, kind: str, members) -> str:
    if kind == "correlation cluster":
        return cluster_label(members)
    return FAMILY_LABEL.get(group, group.replace("family: ", ""))


def load(run: Path, model: str) -> pd.DataFrame:
    """One row per group: label, type, n_features, mean, p10, p90, share positive, largest single member."""
    G = pd.read_csv(run / "summaries" / "all_cv_grouped_permutation_importance.csv")
    G = G[G["model"] == model]
    members = pd.read_csv(run / "data" / "feature_groups.csv")
    single = pd.read_csv(run / "summaries" / "feature_importance_stability_summary.csv")
    single = single[single["model"] == model].set_index("feature")["mean_importance"]
    kind = members.drop_duplicates("group").set_index("group")["group_type"]
    rows = []
    for group, g in G.groupby("group", sort=False):
        feats = members.loc[members["group"] == group, "feature"].tolist()
        v = g["importance_mean"].to_numpy()
        rows.append({"group": group, "group_type": kind[group], "label": group_label(group, kind[group], feats),
                     "n_features": int(g["n_features"].max()), "n_evaluations": len(v), "mean": v.mean(),
                     "p10": np.percentile(v, 10), "p90": np.percentile(v, 90), "share_positive": (v > 0).mean(),
                     "largest_single_member": single.reindex(feats).max(),
                     "baseline_average_precision": g["baseline_average_precision"].mean()})
    return pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)


def draw_panel(ax, D: pd.DataFrame, title: str):
    y = np.arange(len(D))[::-1]
    for yi, (_, r) in zip(y, D.iterrows()):
        colour = TYPE_COLOUR[r["group_type"]]
        stable = r["share_positive"] >= STABLE_MIN_FRACTION_POSITIVE_IMPORTANCE and r["mean"] > 0
        ax.plot([r["p10"], r["p90"]], [yi, yi], color=colour, lw=1.2, solid_capstyle="round", zorder=2)
        ax.plot(r["mean"], yi, "o", ms=5.5, mfc=colour if stable else "white", mec=colour, mew=1.3, zorder=3)
        if np.isfinite(r["largest_single_member"]):
            ax.plot(r["largest_single_member"], yi, marker="|", ms=7, mew=1.3, color=INK, zorder=4)
    ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.label}  [{r.n_features}]" for r in D.itertuples()], fontsize=7, color=INK)
    ax.set_ylim(-0.7, len(D) - 0.3)
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=7, colors=MUTED, length=0)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    base = D["baseline_average_precision"].iloc[0]
    ax.set_xlabel("decrease in average precision when the group is permuted\n"
                  f"(held-out average precision {base:.2f}; {int(D['n_evaluations'].iloc[0])} evaluations)",
                  fontsize=7.5, color=INK)
    ax.set_title(title, fontsize=9, color=INK, loc="left", fontweight="bold")
    # share of evaluations with a positive decrease, as a text column right of the plot
    for yi, r in zip(y, D.itertuples()):
        ax.annotate(f"{100 * r.share_positive:.0f}%", xy=(1.02, yi), xycoords=("axes fraction", "data"),
                    fontsize=6.5, color=MUTED, va="center", ha="left")
    ax.annotate("positive in", xy=(1.02, len(D) - 0.25), xycoords=("axes fraction", "data"), fontsize=6.5,
                color=MUTED, va="bottom", ha="left")


def main():
    a = parse_args()
    given = dict(x.split("=", 1) for x in a.run_dir)
    out = Path(a.out) if a.out else paths.FIGURES
    out.mkdir(parents=True, exist_ok=True)

    tables = {}
    for task in a.tasks:
        run = Path(given[task]) if task in given else paths.run_dir(task)
        if not (run / "summaries" / "all_cv_grouped_permutation_importance.csv").exists():
            print(f"{task}: no grouped importance in {run}; skipped")
            continue
        D = load(run, a.model)
        tables[task] = D.head(a.top) if a.top else D
    if not tables:
        raise SystemExit("no run with a grouped permutation importance found")

    plt.rcParams.update({"pdf.fonttype": 42, "font.family": "sans-serif"})
    heights = [0.19 * len(D) + 0.9 for D in tables.values()]
    fig, axes = plt.subplots(len(tables), 1, figsize=(7.0, sum(heights) + 0.5), squeeze=False,
                             gridspec_kw={"height_ratios": heights})
    for ax, (task, D) in zip(axes[:, 0], tables.items()):
        draw_panel(ax, D, TITLE.get(task, task))
    share = f"{100 * STABLE_MIN_FRACTION_POSITIVE_IMPORTANCE:.0f}%"
    handles = [Line2D([], [], marker="o", ls="-", color=c, mfc=c, ms=5, lw=1.2, label=TYPE_LABEL[k])
               for k, c in TYPE_COLOUR.items()]
    handles += [Line2D([], [], marker="o", ls="", color=MUTED, mfc=MUTED, ms=5, label=f"mean; positive in at least {share}"),
                Line2D([], [], marker="o", ls="", color=MUTED, mfc="white", ms=5, label=f"mean; positive in fewer than {share}"),
                Line2D([], [], marker="", ls="-", color=MUTED, lw=1.2, label="10th to 90th percentile"),
                Line2D([], [], marker="|", ls="", color=INK, ms=7, mew=1.3, label="largest single member")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6.5, frameon=False, labelcolor=INK,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.45 / (sum(heights) + 0.5), 0.96, 1))
    fig.savefig(out / "FigS_grouped_importance.pdf")
    fig.savefig(out / "FigS_grouped_importance.png", dpi=300)
    source = pd.concat([D.assign(task=task) for task, D in tables.items()], ignore_index=True)
    source.round(5).to_csv(out / "FigS_grouped_importance_source.tsv", sep="\t", index=False)
    print(f"saved FigS_grouped_importance.pdf / .png and the source table to {out}")


if __name__ == "__main__":
    main()
