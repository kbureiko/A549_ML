"""Supplementary Table S2 - held-out permutation importance of every feature in the three models.

Two sheets per model (NR vs UP, UP vs DN, NR vs DN) plus a README sheet: the single features and the feature groups
(families of targets and clusters of correlated tracks, permuted together). For each feature:
number and fraction of the held-out evaluations with positive importance, mean / median / SD of the
importance (decrease in average precision when the feature is permuted in the held-out fold), whether the
feature is stable (positive in at least 80 % of the evaluations with a positive median; the fraction is read from
config.py) and its rank among the stable features (by fraction of positive evaluations, then mean importance).
Figure 1C-E shows the groups and the single features that belong to no group, ordered by mean importance.

Input:  <run folder>/summaries/feature_importance_stability_summary.csv  (written by scripts/01_run_cv.py)
        <run folder>/summaries/all_cv_grouped_permutation_importance.csv and <run folder>/data/feature_groups.csv
        (groups; optional - without them only the single-feature sheets are written)
Output: <figures>/Supp_Table_S2_feature_importance.xlsx  and one TSV per model next to it

Run:  uv run python scripts/15_supp_table_feature_importance.py
"""
from __future__ import annotations

import re

import pandas as pd

from a549_enhancers.config import (N_REPEATS, N_SPLITS,
                                   STABLE_MIN_FRACTION_POSITIVE_IMPORTANCE as STABLE_FRACTION)
from a549_enhancers.paths import FIGURES, RUNS   # local folders, set in paths.toml

TASK_LABEL = {"nr_vs_up": "NR vs UP", "up_vs_dn": "UP vs DN", "nr_vs_dn": "NR vs DN"}
GENOMIC = {"Enh_len", "Distance to TSS", "annotation", "gc", "GRE_score"}


def feature_group(feature: str) -> str:
    """Same three groups as the colours of Figure 1C-E."""
    if feature.startswith("motif_binary::"):
        return "sequence motif"
    if feature in GENOMIC:
        return "genomic"
    if feature.startswith("nbr_enhancer"):
        return "neighbourhood, other enhancers"
    if feature.startswith("nbr_"):
        return "neighbourhood, GR peaks after dex"
    if "_dex_" in feature or feature == "ATAC_dex_1h":
        return "signal, dex-treated cells"
    return "signal, control cells"   # EtOH-treated or untreated


def accession(feature: str) -> str:
    return "; ".join(re.findall(r"ENCFF[0-9A-Z]{6}", feature))   # two accessions = mean of two tracks


def build(task: str) -> pd.DataFrame:
    path = RUNS[task] / "summaries" / "feature_importance_stability_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path}\nRun scripts/01_run_cv.py for task {task} first.")
    d = pd.read_csv(path)
    n_eval = int(d["n_folds"].max())
    out = pd.DataFrame({
        "feature": d["feature"],
        "feature_group": d["feature"].map(feature_group),
        "ENCODE_accession": d["feature"].map(accession),
        "n_evaluations": d["n_folds"].astype(int),
        "n_positive": (d["fraction_positive_importance"] * d["n_folds"]).round().astype(int),
        "fraction_positive": d["fraction_positive_importance"].round(3),
        "mean_importance": d["mean_importance"],
        "median_importance": d["median_importance"],
        "sd_importance": d["sd_importance"],
    })
    out["stable"] = (out["fraction_positive"] >= STABLE_FRACTION) & (out["median_importance"] > 0)
    out = out.sort_values(["fraction_positive", "mean_importance"], ascending=False, kind="mergesort")
    out = out.reset_index(drop=True)
    out["rank_among_stable"] = pd.array([pd.NA] * len(out), dtype="Int64")
    out.loc[out["stable"], "rank_among_stable"] = range(1, int(out["stable"].sum()) + 1)
    out["rank_by_mean_importance"] = out["mean_importance"].rank(ascending=False, method="min").astype(int)
    print(f"{TASK_LABEL[task]}: {len(out)} features, {int(out['stable'].sum())} stable "
          f"({int((out['stable'] & out['feature_group'].eq('sequence motif')).sum())} motifs), "
          f"{n_eval} evaluations; top by mean importance: "
          f"{out.loc[out['mean_importance'].idxmax(), 'feature']} ({out['mean_importance'].max():.3f})")
    return out


def build_groups(task: str) -> pd.DataFrame | None:
    """Grouped permutation importance per group over the held-out evaluations; None if the run has no group files."""
    run = RUNS[task]
    per_fold, members = run / "summaries" / "all_cv_grouped_permutation_importance.csv", run / "data" / "feature_groups.csv"
    if not (per_fold.exists() and members.exists()):
        print(f"{TASK_LABEL[task]}: no grouped importance in {run.name} - group sheet left out")
        return None
    g, m = pd.read_csv(per_fold), pd.read_csv(members)
    col = "importance_mean"
    agg = g.groupby("group")[col].agg(n_evaluations="size", n_positive=lambda v: int((v > 0).sum()), mean_importance="mean",
                                      median_importance="median", sd_importance="std").reset_index()
    agg["fraction_positive"] = (agg["n_positive"] / agg["n_evaluations"]).round(3)
    info = m.groupby("group").agg(group_type=("group_type", "first"), n_features=("feature", "size"),
                                  members=("feature", lambda v: "; ".join(v))).reset_index()
    out = info.merge(agg, on="group", how="right", validate="1:1")
    if out["group_type"].isna().any():
        raise ValueError(f"{members}: groups of the importance table are missing: {out.loc[out.group_type.isna(), 'group'].tolist()}")
    out["stable"] = (out["fraction_positive"] >= STABLE_FRACTION) & (out["median_importance"] > 0)
    out = out.sort_values("mean_importance", ascending=False, kind="mergesort").reset_index(drop=True)
    out["rank_by_mean_importance"] = range(1, len(out) + 1)
    out = out[["group", "group_type", "n_features", "n_evaluations", "n_positive", "fraction_positive", "mean_importance",
               "median_importance", "sd_importance", "stable", "rank_by_mean_importance", "members"]]
    print(f"{TASK_LABEL[task]}: {len(out)} groups ({int((out.group_type == 'family').sum())} families, "
          f"{int((out.group_type != 'family').sum())} clusters), {int(out['stable'].sum())} stable; "
          f"top: {out.loc[0, 'group']} ({out.loc[0, 'mean_importance']:.3f})")
    return out


README = [
    ("Supplementary Table S2", "Held-out permutation importance of all features and feature groups in the three pairwise CatBoost models."),
    ("Sheets", "Per model (NR vs UP, UP vs DN, NR vs DN): one sheet with the single features and one with the feature groups ('... groups')."),
    ("feature", "Feature name as in the feature matrix (ENCODE target, treatment and file accession; motif name; or genomic feature)."),
    ("feature_group", "signal, dex-treated cells / signal, control cells (EtOH-treated or untreated) / sequence motif / genomic / neighbourhood (other enhancers, or GR peaks after dex)."),
    ("ENCODE_accession", "ENCODE file accession of the signal track (see Supplementary Table S1)."),
    ("n_evaluations", f"Number of held-out evaluations ({N_SPLITS}-fold cross-validation repeated {N_REPEATS} times)."),
    ("n_positive, fraction_positive", "Evaluations in which the permutation importance of the feature was positive."),
    ("mean / median / sd_importance", "Permutation importance = decrease in average precision when the feature is permuted "
                                      "in the held-out fold (10 permutations per feature and fold), summarised over the evaluations."),
    ("stable", f"TRUE when fraction_positive >= {STABLE_FRACTION:.2f} and the median importance is positive."),
    ("rank_among_stable", "Rank among stable features: by fraction_positive, then mean_importance."),
    ("rank_by_mean_importance", "Rank of the feature by mean_importance among all features of the model."),
    ("Group sheets", "Grouped permutation importance: all features of a group are permuted together (same reordering of the "
                     "held-out enhancers; 10 permutations per group and fold). group_type: family (targets defined by name; "
                     "control and dex tracks separately; all motif features; GRE motif and score; neighbourhood features) or "
                     "correlation cluster (signal tracks with Spearman correlation of at least 0.7, average linkage). "
                     "members: the features of the group. A feature can belong to a family and to a cluster."),
    ("Figure 1C-E", "Shows the 15 rows with the largest mean importance among the groups and the single features that belong to no group."),
]


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    tables = {TASK_LABEL[t]: build(t) for t in TASK_LABEL if t in RUNS}
    if not tables:
        raise SystemExit("No run folders configured under [runs] in paths.toml.")
    groups = {f"{TASK_LABEL[t]} groups": build_groups(t) for t in TASK_LABEL if t in RUNS}
    tables = {k: v for pair in zip(tables.items(), groups.items()) for k, v in pair if v is not None}   # feature sheet, then its groups
    for label, tab in tables.items():
        tab.to_csv(FIGURES / f"Supp_Table_S2_feature_importance_{label.replace(' ', '_')}.tsv", sep="\t", index=False)
    xlsx = FIGURES / "Supp_Table_S2_feature_importance.xlsx"
    try:
        with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
            pd.DataFrame(README, columns=["Item", "Description"]).to_excel(writer, sheet_name="README", index=False)
            for label, tab in tables.items():
                tab.to_excel(writer, sheet_name=label, index=False, freeze_panes=(1, 1))
        print("written:", xlsx.name, "and", len(tables), "TSV files in the figures folder")
    except ImportError:
        print("openpyxl is not installed - only the TSV files were written (uv add openpyxl, then rerun).")


if __name__ == "__main__":
    main()
