"""Is the response graded? Held-out model score against the PRO-seq fold change inside the NR class.

The classes are cut from a continuous measurement (p-value and fold-change thresholds). If NR enhancers that the
NR-versus-UP model scores as UP-like have small positive fold changes, they are sub-threshold responders rather
than a separate kind of enhancer; the same for DN. The script uses the held-out probabilities of the
cross-validation runs (mean over the repeats, each from a model that did not see the enhancer).

    uv run python scripts/23_graded_response.py
    uv run python scripts/23_graded_response.py --tasks nr_vs_up --bins 20

Input
    <run folder>/summaries/all_cv_predictions.csv   of nr_vs_up and nr_vs_dn ([runs] in paths.toml)
    prematrix (fold change and p-value) and the GR peak file, from [inputs] ml_inputs

Output (<analysis>/class_analyses/, or --out)
    graded_response_bins.tsv      NR enhancers in equal-sized bins of the held-out score: median fold change,
                                  share with a fold change in the direction of the positive class, share with
                                  nominal p < 0.05 in that direction, share GR-bound
    graded_response_summary.tsv   per model: Spearman correlation of score and fold change among NR enhancers
                                  (with a bootstrap interval), and NR enhancers called positive in most repeats
                                  against the other NR enhancers
    graded_response.pdf           median fold change (with quartiles) per score bin

The fold change is signed so that positive values go in the direction of the positive class (for nr_vs_dn the
sign is flipped).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

from a549_enhancers import paths
from a549_enhancers.config import GLOBAL_RANDOM_STATE, PVAL_THRESHOLD
from a549_enhancers.enhancers import enhancer_table, mean_heldout_score

CLS = {"UP": "#D1495B", "DN": "#3B6FB6", "NR": "#5DAA82"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="+", default=["nr_vs_up", "nr_vs_dn"], choices=["nr_vs_up", "nr_vs_dn"])
    p.add_argument("--model", default="catboost")
    p.add_argument("--bins", type=int, default=10, help="equal-sized bins of the score among NR enhancers")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--run-dir", nargs="*", default=[], metavar="TASK=FOLDER", help="run folders, if not those of paths.toml")
    p.add_argument("--out", default=None)
    return p.parse_args()


def main():
    a = parse_args()
    out = Path(a.out) if a.out else paths.PROJECT / "class_analyses"
    out.mkdir(parents=True, exist_ok=True)
    given = dict(x.split("=", 1) for x in a.run_dir)
    E = enhancer_table(need_gr=False)
    rng = np.random.default_rng(GLOBAL_RANDOM_STATE)
    bins, summary = [], []
    fig, axes = plt.subplots(1, len(a.tasks), figsize=(3.8 * len(a.tasks), 3.2), squeeze=False)

    for ax, task in zip(axes[0], a.tasks):
        run = Path(given[task]) if task in given else paths.run_dir(task)
        S = mean_heldout_score(run, a.model)
        positive = S.attrs["positive"]
        sign = 1.0 if positive == "UP" else -1.0
        missing = S.index.difference(E.index)
        if len(missing):
            raise ValueError(f"{task}: {len(missing)} enhancers of the run are not in the prematrix, e.g. {missing[0]}")
        T = S.join(E[["log2FC", "pvalue", "class"] + (["GR_bound"] if "GR_bound" in E else [])])
        T["signed_log2FC"] = sign * T["log2FC"]
        NR = T[T["class"] == "NR"].copy()
        NR["bin"] = pd.qcut(NR["score"].rank(method="first"), a.bins, labels=False) + 1

        for b, g in NR.groupby("bin"):
            bins.append({
                "task": task, "positive_class": positive, "score_bin": int(b), "n_NR": len(g),
                "score_min": g["score"].min(), "score_max": g["score"].max(),
                "signed_log2FC_median": g["signed_log2FC"].median(),
                "signed_log2FC_q25": g["signed_log2FC"].quantile(0.25),
                "signed_log2FC_q75": g["signed_log2FC"].quantile(0.75),
                "share_log2FC_in_direction": (g["signed_log2FC"] > 0).mean(),
                "share_nominal_p_in_direction": ((g["pvalue"] < PVAL_THRESHOLD) & (g["signed_log2FC"] > 0)).mean(),
                "share_GR_bound": g["GR_bound"].mean() if "GR_bound" in g else np.nan,
            })

        ok = NR[["score", "signed_log2FC"]].dropna()
        rho, p = spearmanr(ok["score"], ok["signed_log2FC"])
        boot = []
        v = ok.to_numpy()
        for _ in range(a.n_boot):
            s = v[rng.integers(0, len(v), len(v))]
            boot.append(spearmanr(s[:, 0], s[:, 1])[0])
        called = NR["frac_called_positive"] > 0.5
        mw = (mannwhitneyu(NR.loc[called, "signed_log2FC"].dropna(), NR.loc[~called, "signed_log2FC"].dropna(),
                           alternative="two-sided").pvalue if called.sum() >= 5 else np.nan)
        pos = T[T["class"] == positive]
        summary.append({
            "task": task, "positive_class": positive, "n_NR": len(NR),
            "spearman_score_vs_signed_log2FC_NR": rho, "ci95_low": np.percentile(boot, 2.5),
            "ci95_high": np.percentile(boot, 97.5), "spearman_p": p,
            "n_NR_called_positive_in_most_repeats": int(called.sum()),
            "signed_log2FC_median_called": NR.loc[called, "signed_log2FC"].median(),
            "signed_log2FC_median_other_NR": NR.loc[~called, "signed_log2FC"].median(),
            "share_log2FC_in_direction_called": (NR.loc[called, "signed_log2FC"] > 0).mean(),
            "share_log2FC_in_direction_other_NR": (NR.loc[~called, "signed_log2FC"] > 0).mean(),
            "mann_whitney_p_called_vs_other": mw,
            "spearman_score_vs_signed_log2FC_within_positive_class": spearmanr(pos["score"], pos["signed_log2FC"])[0]
            if len(pos) > 2 else np.nan,
        })

        B = pd.DataFrame([r for r in bins if r["task"] == task])
        ax.fill_between(B["score_bin"], B["signed_log2FC_q25"], B["signed_log2FC_q75"], color=CLS["NR"], alpha=0.25, lw=0)
        ax.plot(B["score_bin"], B["signed_log2FC_median"], color=CLS[positive], marker="o", ms=3.5, lw=1.2)
        ax.axhline(0, color="0.5", lw=0.6)
        ax.set_xlabel(f"NR enhancers, bin of held-out {positive} score\n(low to high, {len(NR) // a.bins} per bin)", fontsize=8)
        ax.set_ylabel(("log2 fold change (dex / control)" if sign > 0 else "- log2 fold change (dex / control)")
                      + "\nmedian and quartiles")
        ax.set_title(f"{task.replace('_', ' ')}: Spearman r = {rho:.2f}", fontsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    pd.DataFrame(bins).round(4).to_csv(out / "graded_response_bins.tsv", sep="\t", index=False)
    pd.DataFrame(summary).round(5).to_csv(out / "graded_response_summary.tsv", sep="\t", index=False)
    plt.rcParams["pdf.fonttype"] = 42
    fig.tight_layout()
    fig.savefig(out / "graded_response.pdf")
    pd.set_option("display.width", 220)
    print(pd.DataFrame(summary).round(3).T.to_string())
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
