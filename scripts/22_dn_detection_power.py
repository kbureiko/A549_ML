"""Does the NR-versus-DN model separate the classes among enhancers of similar length and control transcription?

Background. The classes are defined by a p-value threshold on PRO-seq counts. A decrease is easier to detect at
an enhancer that is long and well transcribed before treatment, so part of what separates DN from NR enhancers
can be detection power. This script asks how much of the held-out discrimination remains once length and control
PRO-seq signal are held (approximately) constant. It reads the cross-validation run; no model is fitted again
except a small logistic regression on the two covariates.

    uv run python scripts/22_dn_detection_power.py
    uv run python scripts/22_dn_detection_power.py --task nr_vs_up          # the same check for UP
    uv run python scripts/22_dn_detection_power.py --control-column <name>  # column of the PRO-seq table to use

Input
    <run folder of the task>/summaries/all_cv_predictions.csv     held-out probabilities (scripts/01_run_cv.py)
    <run folder of the task>/data/final_assembled_matrix.csv      enhancer length (Enh_len)
    PRO-seq table ([layout] proseq_ma_table in paths.toml, or --ma-table): Chr, Start, End and the PRO-seq signal.
        The control signal is taken, in this order, from
          - the column named with --control-column;
          - a control-condition column (name with ctrl / control / EtOH);
          - a mean column (Mean_signal, baseMean, A ...) together with a log2 fold-change column: the control
            level is then derived as  mean - log2FC / 2  (exact when the mean is the average of the log2 signal
            of the two conditions, as in an MA plot; --no-derive uses the mean of both conditions as it is);
          - a mean column alone.
        Identifier columns (PeakID ..., integer counters) are never used. Without the table only length is used.

Output (<analysis>/class_analyses/, or --out)
    <task>_detection_strata.tsv      held-out ROC AUC and average precision of the model inside tertiles of length,
                                     of control signal, and inside the nine length x signal cells (mean and SD
                                     over the repeats)
    <task>_detection_matched.tsv     the same on a matched set: every positive enhancer with its --k nearest
                                     negative enhancers in (log length, log signal); with the balance achieved
    <task>_detection_covariates.tsv  how well length and signal alone separate the classes (logistic regression,
                                     same repeated cross-validation), next to the full model

Reading the tables: if the model keeps most of its ROC AUC inside the cells and on the matched set, and the two
covariates alone separate the classes much less well than the model, detection power does not explain the model.
A cell needs at least --min-positives enhancers of the positive class to be reported.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from a549_enhancers import paths
from a549_enhancers.config import GLOBAL_RANDOM_STATE, N_REPEATS, N_SPLITS

CONTROL_PATTERN = re.compile(r"ctrl|control|etoh|eth|veh", re.I)
MEAN_PATTERN = re.compile(r"^(basemean|mean|mean_signal|meansignal|a|avg|average|logcpm|aveexpr)$", re.I)
LFC_PATTERN = re.compile(r"^(log2fc|log2foldchange|lfc|m|logfc)$", re.I)
ID_PATTERN = re.compile(r"peakid|^id$|cmd=|^index$|^unnamed", re.I)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="nr_vs_dn", choices=["nr_vs_dn", "nr_vs_up", "up_vs_dn"])
    p.add_argument("--run-dir", default=None, help="run folder (default: [runs] of paths.toml)")
    p.add_argument("--model", default="catboost")
    p.add_argument("--ma-table", default=None, help="PRO-seq table (default: [layout] proseq_ma_table)")
    p.add_argument("--control-column", default=None, help="column of the PRO-seq table with the control signal")
    p.add_argument("--no-derive", action="store_true",
                   help="do not derive the control level from mean and log2FC; use the mean column as it is")
    p.add_argument("--k", type=int, default=5, help="matched negative enhancers per positive enhancer")
    p.add_argument("--min-positives", type=int, default=15)
    p.add_argument("--out", default=None)
    return p.parse_args()


def to_model_scale(values: pd.Series) -> pd.Series:
    """Counts and lengths are log10-transformed; values that are already on a log scale are left as they are."""
    v = pd.to_numeric(values, errors="coerce")
    if v.min() < 0 or v.max() < 50:
        return v
    return np.log10(v.clip(lower=0) + 1.0)


def read_control_signal(path: Path, column: str | None, ids: pd.Index, derive: bool = True):
    """Signal per enhancer ID from the PRO-seq table; returns (Series or None, description)."""
    if not path.exists():
        return None, f"{path.name} not found: length only"
    M = pd.read_csv(path, sep="\t")
    need = {"Chr", "Start", "End"}
    if not need <= set(M.columns):
        return None, f"{path.name} has no Chr / Start / End columns: length only"
    numeric = [c for c in M.columns if c not in need and pd.api.types.is_numeric_dtype(M[c])
               and not pd.api.types.is_bool_dtype(M[c]) and not ID_PATTERN.search(str(c))]
    if column is None:
        hits = [c for c in numeric if CONTROL_PATTERN.search(c) and not re.search(r"log2?fc|pval|padj|fdr", c, re.I)]
        means = [c for c in numeric if MEAN_PATTERN.match(c)]
        lfcs = [c for c in numeric if LFC_PATTERN.match(c)]
        if hits:
            if len(hits) > 1:
                print(f"several candidate columns {hits}; their mean is used (choose one with --control-column)")
            signal, what = M[hits].mean(axis=1), f"control signal: {', '.join(hits)}"
        elif means and lfcs and derive:
            signal = M[means[0]] - M[lfcs[0]] / 2.0
            what = f"control level derived as {means[0]} - {lfcs[0]} / 2"
        elif means:
            signal, what = M[means[0]], f"mean signal of both conditions: {means[0]}"
        else:
            raise SystemExit(f"{path}: no control or mean signal column recognised among {numeric}. "
                             "Name it with --control-column.")
    else:
        if column not in M.columns:
            raise SystemExit(f"{path}: no column {column!r}. Columns: {list(M.columns)}")
        signal, what = M[column], f"column {column}"
    for shift in (1, 0):                                  # HOMER positions are 1-based, the enhancer IDs 0-based
        key = M["Chr"].astype(str) + "_" + (M["Start"] - shift).astype(str) + "_" + M["End"].astype(str)
        if pd.Index(ids).isin(key).mean() > 0.99:
            s = pd.Series(signal.to_numpy(dtype=float), index=key.to_numpy())
            return s[~s.index.duplicated()].reindex(ids), what
    return None, f"{path.name}: coordinates do not match the enhancer IDs: length only"


def per_repeat_metrics(pred: pd.DataFrame, keep_ids, positive: str, proba: str) -> dict:
    """ROC AUC and average precision of the held-out probabilities on a subset, per repeat -> mean and SD."""
    sub = pred[pred["enhancer_id"].isin(keep_ids)]
    auc, ap = [], []
    for _, g in sub.groupby("repeat"):
        y = (g["true_label"] == positive).to_numpy()
        if y.sum() == 0 or y.sum() == len(y):
            continue
        auc.append(roc_auc_score(y, g[proba])); ap.append(average_precision_score(y, g[proba]))
    one = sub.drop_duplicates("enhancer_id")
    n_pos = int((one["true_label"] == positive).sum())
    return {"n_positive": n_pos, "n_negative": int(len(one) - n_pos),
            "prevalence": n_pos / max(len(one), 1),
            "roc_auc_mean": np.mean(auc) if auc else np.nan, "roc_auc_sd": np.std(auc) if auc else np.nan,
            "average_precision_mean": np.mean(ap) if ap else np.nan, "average_precision_sd": np.std(ap) if ap else np.nan}


def main():
    a = parse_args()
    run = Path(a.run_dir) if a.run_dir else paths.run_dir(a.task)
    out = Path(a.out) if a.out else paths.PROJECT / "class_analyses"
    out.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(run / "summaries" / "all_cv_predictions.csv", low_memory=False)
    pred = pred[pred["model"] == a.model]
    positive, negative = str(pred["positive_label"].iloc[0]), str(pred["negative_label"].iloc[0])
    proba = f"proba_{positive}"
    D = pd.read_csv(run / "data" / "final_assembled_matrix.csv", usecols=["enhancer_id", "binary_target", "Enh_len"],
                    low_memory=False).set_index("enhancer_id")
    D["length"] = pd.to_numeric(D["Enh_len"], errors="coerce")
    ma = Path(a.ma_table) if a.ma_table else paths.PROSEQ_MA_TABLE
    signal, what = read_control_signal(ma, a.control_column, D.index, derive=not a.no_derive)
    print(f"run: {run.name} | positive class {positive}, negative class {negative} | PRO-seq signal: {what}")
    covariates = ["length"]
    if signal is not None:
        D["signal"] = signal.to_numpy()
        n_missing = int(D["signal"].isna().sum())
        if n_missing:
            print(f"{n_missing} enhancers without PRO-seq signal are left out")
            D = D[D["signal"].notna()]
        covariates.append("signal")
    pred = pred[pred["enhancer_id"].isin(D.index)]
    is_pos = D["binary_target"].astype(str).eq(positive)

    # ------------------------------------------------------------------ 1 strata
    for c in covariates:                      # tertiles over all enhancers of the two classes
        D[f"{c}_tertile"] = pd.qcut(D[c].rank(method="first"), 3, labels=["low", "middle", "high"]).astype(str)
    rows = [{"stratum": "all enhancers", **per_repeat_metrics(pred, D.index, positive, proba)}]
    for c in covariates:
        for level in ("low", "middle", "high"):
            ids = D.index[D[f"{c}_tertile"] == level]
            rows.append({"stratum": f"{c}: {level}", f"{c}_median": D.loc[ids, c].median(),
                         **per_repeat_metrics(pred, ids, positive, proba)})
    if len(covariates) == 2:
        for l1 in ("low", "middle", "high"):
            for l2 in ("low", "middle", "high"):
                ids = D.index[(D["length_tertile"] == l1) & (D["signal_tertile"] == l2)]
                rows.append({"stratum": f"length: {l1} / signal: {l2}", "length_median": D.loc[ids, "length"].median(),
                             "signal_median": D.loc[ids, "signal"].median(),
                             **per_repeat_metrics(pred, ids, positive, proba)})
    S = pd.DataFrame(rows)
    few = S["n_positive"] < a.min_positives
    S.loc[few, [c for c in S.columns if c.startswith(("roc_auc", "average_precision"))]] = np.nan
    S["note"] = np.where(few, f"fewer than {a.min_positives} {positive} enhancers: not reported", "")
    S.round(4).to_csv(out / f"{a.task}_detection_strata.tsv", sep="\t", index=False)

    # ------------------------------------------------------------------ 2 matched set
    Z = D[covariates].apply(to_model_scale)
    Z = (Z - Z.mean()) / Z.std(ddof=0)
    neg_ids = D.index[~is_pos]
    nn = NearestNeighbors(n_neighbors=min(a.k, len(neg_ids))).fit(Z.loc[neg_ids].to_numpy())
    _, idx = nn.kneighbors(Z.loc[is_pos].to_numpy())
    matched_neg = pd.Index(neg_ids[np.unique(idx)])
    matched = D.index[is_pos].union(matched_neg)
    row = {"set": f"matched: every {positive} enhancer with its {a.k} nearest {negative} enhancers",
           "n_matched_negative_unique": len(matched_neg), **per_repeat_metrics(pred, matched, positive, proba)}
    for c in covariates:
        row[f"{c}_median_{positive}"] = D.loc[is_pos, c].median()
        row[f"{c}_median_matched_{negative}"] = D.loc[matched_neg, c].median()
        row[f"{c}_median_all_{negative}"] = D.loc[neg_ids, c].median()
    if len(matched_neg) >= 0.9 * len(neg_ids):
        row["note"] = (f"the matched set holds {len(matched_neg)} of {len(neg_ids)} {negative} enhancers, so it is close "
                       "to the full data set: not informative for this comparison")
    Mt = pd.DataFrame([{"set": "all enhancers", **per_repeat_metrics(pred, D.index, positive, proba)}, row])
    Mt.round(4).to_csv(out / f"{a.task}_detection_matched.tsv", sep="\t", index=False)

    # ------------------------------------------------------------------ 3 covariates alone
    y = is_pos.to_numpy().astype(int)
    Xc = D[covariates].apply(to_model_scale).to_numpy()
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=GLOBAL_RANDOM_STATE)
    res = {name: {"auc": [], "ap": []} for name in ["length"] + (["signal", "length + signal"] if len(covariates) == 2 else [])}
    cols_of = {"length": [0], "signal": [1], "length + signal": [0, 1]}
    for tr, te in cv.split(Xc, y):
        for name in res:
            m = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
            m.fit(Xc[tr][:, cols_of[name]], y[tr])
            s = m.predict_proba(Xc[te][:, cols_of[name]])[:, 1]
            res[name]["auc"].append(roc_auc_score(y[te], s)); res[name]["ap"].append(average_precision_score(y[te], s))
    full = per_repeat_metrics(pred, D.index, positive, proba)
    C = pd.DataFrame([{"features": f"logistic regression: {name}", "roc_auc_mean": np.mean(v["auc"]),
                       "roc_auc_sd": np.std(v["auc"]), "average_precision_mean": np.mean(v["ap"]),
                       "average_precision_sd": np.std(v["ap"]), "sd_over": "held-out folds"} for name, v in res.items()] +
                     [{"features": f"full model ({a.model}, held-out)", "roc_auc_mean": full["roc_auc_mean"],
                       "roc_auc_sd": full["roc_auc_sd"], "average_precision_mean": full["average_precision_mean"],
                       "average_precision_sd": full["average_precision_sd"], "sd_over": "repeats"}])
    C["prevalence"] = full["prevalence"]
    C.round(4).to_csv(out / f"{a.task}_detection_covariates.tsv", sep="\t", index=False)

    pd.set_option("display.width", 220)
    print("\nstrata\n" + S[["stratum", "n_positive", "n_negative", "roc_auc_mean", "average_precision_mean", "prevalence"]]
          .round(3).to_string(index=False))
    print("\nmatched set\n" + Mt[["set", "n_positive", "n_negative", "roc_auc_mean", "average_precision_mean"]]
          .round(3).to_string(index=False))
    print("\ncovariates alone\n" + C.round(3).to_string(index=False))
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
