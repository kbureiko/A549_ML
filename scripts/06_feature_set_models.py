"""Baseline-only vs dex-associated-only vs combined models for the three enhancer tasks.

Question (reviewer): do the dominant post-treatment predictors only describe the chromatin state that accompanies
activation, or does the pre-existing regulatory context already predict the response? The answer needs models that
see only one of the two layers, evaluated on the SAME folds so that the comparison is paired.

Feature sets (columns of the published final_assembled_matrix.csv, assigned by name - see feature_layer()):
    genomic   motifs (motif_binary::), annotation, Distance to TSS, Enh_len, gc              - sequence/position floor
    baseline  genomic + EtOH (*_eth_*) + untreated ENCODE (<TF>_ENCFF...) + ATAC_ctrl         - pre-existing context
    dex       genomic + dex-treated (*_dex*, incl. ATAC_dex_1h)                              - activation-associated state
    combined  genomic + baseline + dex                                                      - = the published model
    change    genomic + dex + log2(dex/EtOH) of every mark measured in both    (optional, --sets ... change)
              (the per-enhancer ratio is a fixed transform, no fitted parameters, so it does not leak across folds)

Everything else is the published pipeline:
    CatBoost with the settings stored in the published cv_models; repeated stratified 5-fold CV (same splits for all
    sets: StratifiedKFold(random_state=repeat)); preprocessing fitted on the training fold only - numeric features
    with low variance dropped; 'annotation' and the motif indicators passed to CatBoost as categorical; decision
    threshold = max F1 on inner 3-fold OOF predictions of the training fold (the rule of scripts/01_run_cv.py).

Usage
    uv run python scripts/06_feature_set_models.py                          # 4 sets x 3 tasks x 5 repeats (~1,200 fits)
    uv run python scripts/06_feature_set_models.py --repeats 2              # quick look
    uv run python scripts/06_feature_set_models.py --tasks "NR vs UP" --sets baseline combined
    uv run python scripts/06_feature_set_models.py --importance             # + held-out permutation importance per fold
    uv run python scripts/06_feature_set_models.py --jobs 5                 # 5 folds in parallel, cores/5 threads each
Runs can be split (by --tasks / --sets) and merged: every run appends to the same per-fold table and skips
task x set x repeat combinations that are already in it.

Outputs (in --out, default: <ml_outputs>/feature_sets; feature_sets_folds.tsv is read by
figures/figS1_model_performance.py, panel G)
    feature_sets_folds.tsv        one row per task x set x repeat x fold (metrics, threshold, n features)
    feature_sets_summary.tsv      mean / SD per task x set
    feature_sets_paired.tsv       paired differences between sets on identical folds, with the Nadeau-Bengio
                                  corrected resampled t-test (accounts for overlapping training sets)
    feature_sets_importance.tsv   (--importance) mean held-out permutation importance per task x set x feature
"""
import argparse
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy import stats
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold

from a549_enhancers import paths
from a549_enhancers.thresholding import choose_threshold_from_scores

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------- configuration
ML = paths.ML                                               # [results] ml_outputs in paths.toml
RUNS = {"NR vs UP": paths.RUN_NAMES["nr_vs_up"],            # folder names of the finished runs ([runs] in paths.toml)
        "NR vs DN": paths.RUN_NAMES["nr_vs_dn"],
        "UP vs DN": paths.RUN_NAMES["up_vs_dn"]}
N_SPLITS, N_REPEATS, INNER_SPLITS = 5, 5, 3
VAR_THR = 5.0                       # numeric features; fitted on the training fold
CATEGORICAL = ["annotation"]
CATBOOST = dict(iterations=800, learning_rate=0.03, depth=6, l2_leaf_reg=3, rsm=1, border_count=128,
                bootstrap_type="Bayesian", bagging_temperature=1, random_strength=1, loss_function="Logloss",
                eval_metric="PRAUC", random_seed=42, allow_writing_files=False, verbose=0,
                auto_class_weights="Balanced")          # as in config.CATBOOST_PARAMS
GENOMIC = {"annotation", "Distance to TSS", "Enh_len", "gc"}
PAIRS_TESTED = [("baseline", "genomic"), ("dex", "genomic"), ("dex", "baseline"), ("combined", "baseline"),
                ("combined", "dex"), ("change", "dex")]
METRICS = ["balanced_accuracy", "roc_auc", "average_precision", "positive_f1"]


def hms(sec):
    sec = int(round(sec)); return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


class Clock:
    """Elapsed time and a running ETA from the average duration of the finished units."""
    def __init__(self, n_units):
        self.t0, self.n, self.done = time.time(), n_units, 0

    def tick(self):
        self.done += 1
        el = time.time() - self.t0
        eta = el / self.done * (self.n - self.done) if self.done else float("nan")
        return f"[{self.done}/{self.n} | elapsed {hms(el)} | ETA {hms(eta)} | ends ~{time.strftime('%H:%M', time.localtime(time.time() + eta))}]"


def feature_layer(col):
    """'genomic', 'baseline' or 'dex' from the column name (the naming of the assembled matrix)."""
    if col.startswith("motif_binary::") or col in GENOMIC:
        return "genomic"
    if "_dex" in col:                                   # *_dex_100_1 hour_ENCFF..., ATAC_dex_1h
        return "dex"
    if "_eth_" in col or col == "ATAC_ctrl" or re.match(r"^[\w.]+_ENCFF\w+$", col):
        return "baseline"                               # EtOH series + untreated ENCODE experiments
    raise ValueError(f"unassigned feature column: {col}")


def mark_of(col):
    return "ATAC" if col.startswith("ATAC") else col.split("_")[0]


# ----------------------------------------------------------------------------- data
def load(task):
    df = pd.read_csv(ML / RUNS[task] / "data" / "final_assembled_matrix.csv", low_memory=False)
    pos = df.binary_target.value_counts().idxmin()
    y = (df.binary_target == pos).values.astype(int)
    X = df.drop(columns=["enhancer_id", "multiclass_label", "binary_target"])
    for c in X.columns:
        if c not in CATEGORICAL:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    layers = {c: feature_layer(c) for c in X.columns}
    # change features: log2((dex + 0.25) / (EtOH + 0.25)) for marks with both an EtOH and a dex track
    eth = {mark_of(c): c for c, l in layers.items() if l == "baseline" and ("_eth_" in c or c == "ATAC_ctrl")}
    dex = {mark_of(c): c for c, l in layers.items() if l == "dex"}
    change = {f"log2FC::{m}": np.log2(X[dex[m]] + 0.25) - np.log2(X[eth[m]] + 0.25) for m in sorted(set(eth) & set(dex))}
    X = pd.concat([X, pd.DataFrame(change)], axis=1)
    layers.update({c: "change" for c in change})
    sets = {"genomic": [c for c, l in layers.items() if l == "genomic"]}
    sets["baseline"] = sets["genomic"] + [c for c, l in layers.items() if l == "baseline"]
    sets["dex"] = sets["genomic"] + [c for c, l in layers.items() if l == "dex"]
    sets["combined"] = sets["genomic"] + [c for c, l in layers.items() if l in ("baseline", "dex")]
    sets["change"] = sets["dex"] + [c for c, l in layers.items() if l == "change"]
    return X, y, pos, sets


# ----------------------------------------------------------------------------- preprocessing (training fold only)
class Prep:
    """As in scripts/01_run_cv.py: variance filter on the numeric features; 'annotation' and the motif indicators
    are CatBoost categorical features."""

    def fit(self, X):
        self.cat = [c for c in X.columns if c in CATEGORICAL or c.startswith("motif_binary::")]
        num = [c for c in X.columns if c not in self.cat]
        if X[num].isna().any().any():
            bad = X[num].columns[X[num].isna().any()].tolist()
            raise ValueError(f"NaN in numeric features (no imputation in the pipeline): {bad[:10]}")
        var = X[num].var(ddof=0)
        self.keep = [c for c in num if var[c] > VAR_THR]
        return self

    def transform(self, X):
        out = X[self.keep].copy()
        for c in self.cat:
            out[c] = X[c].astype("object").fillna("__MISSING__").astype(str)
        return out


def tuned_threshold(X, y, cat, seed):
    oof = np.zeros(len(y))
    for tr, va in StratifiedKFold(INNER_SPLITS, shuffle=True, random_state=seed).split(X, y):
        m = CatBoostClassifier(**CATBOOST).fit(X.iloc[tr], y[tr], cat_features=cat)
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    return choose_threshold_from_scores(np.where(y == 1, "pos", "neg"), oof, positive_label="pos", method="max_f1")


def run_fold(X, y, tr, te, seed, importance):
    pre = Prep().fit(X.iloc[tr])
    Xtr, Xte = pre.transform(X.iloc[tr]), pre.transform(X.iloc[te])
    thr = tuned_threshold(Xtr, y[tr], pre.cat, seed)
    model = CatBoostClassifier(**CATBOOST).fit(Xtr, y[tr], cat_features=pre.cat)
    p = model.predict_proba(Xte)[:, 1]; yh = p >= thr
    res = dict(n_features=Xtr.shape[1], threshold=thr, balanced_accuracy=balanced_accuracy_score(y[te], yh),
               roc_auc=roc_auc_score(y[te], p), average_precision=average_precision_score(y[te], p),
               positive_f1=f1_score(y[te], yh, zero_division=0),
               positive_precision=precision_score(y[te], yh, zero_division=0),
               positive_recall=recall_score(y[te], yh, zero_division=0))
    imp = None
    if importance:
        imp = permutation_ap_drop(model, Xte, y[te], res["average_precision"], seed)
    return res, imp


def permutation_ap_drop(model, X, y, base_ap, seed, n_repeats=5):
    """Held-out permutation importance: mean drop in average precision when one column is shuffled
    (written out instead of sklearn.inspection.permutation_importance, which does not recognise every wrapper
    as a classifier in recent scikit-learn versions)."""
    rng = np.random.default_rng(seed)
    out = {}
    for c in X.columns:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy(); Xp[c] = rng.permutation(Xp[c].values)
            drops.append(base_ap - average_precision_score(y, model.predict_proba(Xp)[:, 1]))
        out[c] = float(np.mean(drops))
    return pd.Series(out)


# ----------------------------------------------------------------------------- parallel jobs (one outer fold each)
TASK_ORDER = ["NR vs UP", "NR vs DN", "UP vs DN"]          # largest first: better load balancing


@lru_cache(maxsize=None)
def load_cached(task):
    return load(task)


def init_worker(threads):
    CATBOOST["thread_count"] = threads                     # CatBoost threads per worker process


def fold_job(task, s, rep, k, importance):
    """One outer fold of one task x feature set x repeat; the split is recomputed deterministically, so it is the
    same fold for every feature set wherever the job runs."""
    X, y, pos, sets = load_cached(task)
    tr, te = list(StratifiedKFold(N_SPLITS, shuffle=True, random_state=rep).split(X, y))[k - 1]
    t = time.time()
    res, imp = run_fold(X[sets[s]], y, tr, te, rep, importance)
    row = dict(task=task, feature_set=s, repeat=rep, fold=k, n_train=len(tr), n_test=len(te), **res,
               seconds=round(time.time() - t, 1))
    imp_df = None
    if imp is not None:
        imp_df = (imp.rename("importance").reset_index().rename(columns={"index": "feature"})
                  .assign(task=task, feature_set=s, repeat=rep, fold=k))
    return row, imp_df


# ----------------------------------------------------------------------------- paired comparison
def corrected_ttest(d, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test for repeated k-fold CV differences."""
    d = np.asarray(d); k = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return np.nan, np.nan
    t = d.mean() / np.sqrt((1 / k + n_test / n_train) * var)
    return t, 2 * stats.t.sf(abs(t), df=k - 1)


def paired_table(F):
    rows = []
    for task, g in F.groupby("task"):
        for a, b in PAIRS_TESTED:
            A = g[g.feature_set == a].set_index(["repeat", "fold"]); B = g[g.feature_set == b].set_index(["repeat", "fold"])
            idx = A.index.intersection(B.index)
            if len(idx) < 3:
                continue
            n_test = A.loc[idx, "n_test"].mean(); n_train = A.loc[idx, "n_train"].mean()
            for m in METRICS:
                d = A.loc[idx, m] - B.loc[idx, m]
                t, p = corrected_ttest(d, n_train, n_test)
                rows.append(dict(task=task, comparison=f"{a} − {b}", metric=m, n_folds=len(idx),
                                 mean_diff=d.mean(), sd_diff=d.std(ddof=1), folds_a_better=int((d > 0).sum()),
                                 t_corrected=t, p_corrected=p))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=list(RUNS), choices=list(RUNS))
    ap.add_argument("--sets", nargs="+", default=["genomic", "baseline", "dex", "combined"],
                    choices=["genomic", "baseline", "dex", "combined", "change"])
    ap.add_argument("--repeats", type=int, default=N_REPEATS)
    ap.add_argument("--importance", action="store_true")
    ap.add_argument("--out", type=Path, default=ML / "feature_sets")
    ap.add_argument("--jobs", type=int, default=1, help="parallel worker processes (one outer fold each)")
    ap.add_argument("--threads", type=int, default=0, help="CatBoost threads per worker (default: cores / jobs)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    f_folds = a.out / "feature_sets_folds.tsv"; f_imp = a.out / "feature_sets_importance_folds.tsv"
    F = pd.read_csv(f_folds, sep="\t") if f_folds.exists() else pd.DataFrame()
    IMP = pd.read_csv(f_imp, sep="\t") if f_imp.exists() else pd.DataFrame()

    def already(task, s, rep):
        return len(F) and ((F.task == task) & (F.feature_set == s) & (F.repeat == rep)).sum() == N_SPLITS
    todo = sum(not already(t, s, r) for t in a.tasks for s in a.sets for r in range(1, a.repeats + 1))
    clock = Clock(todo * N_SPLITS)                       # one unit = one outer fold (4 CatBoost fits)
    print(f"started {time.strftime('%Y-%m-%d %H:%M')} | {todo} task x set x repeat blocks to run "
          f"({todo * N_SPLITS} outer folds)", flush=True)
    for task in a.tasks:
        X, y, pos, sets = load_cached(task)
        print(f"{task}: {len(y)} enhancers, {y.sum()} positive ({pos}) | features per set: "
              + ", ".join(f"{s} {len(sets[s])}" for s in a.sets), flush=True)
    jobs = [(t, s, r, k) for t in sorted(a.tasks, key=TASK_ORDER.index) for r in range(1, a.repeats + 1)
            for s in a.sets if not already(t, s, r) for k in range(1, N_SPLITS + 1)]
    threads = a.threads or max(1, (os.cpu_count() or 1) // a.jobs)
    print(f"{a.jobs} worker process(es) x {threads} CatBoost thread(s)\n", flush=True)
    pending = {}                                          # (task, set, repeat) -> finished folds

    def collect(row, imp_df):
        global F, IMP
        key = (row["task"], row["feature_set"], row["repeat"])
        pending.setdefault(key, []).append((row, imp_df))
        print(f"    {key[0]} | {key[1]} | repeat {key[2]} | fold {row['fold']}/{N_SPLITS} ({row['seconds']:.0f} s) "
              f"{clock.tick()}", flush=True)
        if len(pending[key]) == N_SPLITS:                 # block complete: save it
            done = sorted(pending.pop(key), key=lambda x: x[0]["fold"])
            rows = [r for r, _ in done]; imps = [i for _, i in done if i is not None]
            F = pd.concat([F, pd.DataFrame(rows)], ignore_index=True); F.to_csv(f_folds, sep="\t", index=False)
            if imps:
                IMP = pd.concat([IMP, *imps], ignore_index=True); IMP.to_csv(f_imp, sep="\t", index=False)
            r = pd.DataFrame(rows)
            print(f"  ✓ {key[0]} | {key[1]:9s} | repeat {key[2]}: BA {r.balanced_accuracy.mean():.3f}  "
                  f"AUC {r.roc_auc.mean():.3f}  AP {r.average_precision.mean():.3f}  "
                  f"({r.n_features.iloc[0]} features after filtering)", flush=True)

    if a.jobs == 1:
        init_worker(threads)
        for j in jobs:
            collect(*fold_job(*j, a.importance))
    else:
        with ProcessPoolExecutor(max_workers=a.jobs, initializer=init_worker, initargs=(threads,)) as ex:
            futs = [ex.submit(fold_job, *j, a.importance) for j in jobs]
            for fu in as_completed(futs):
                collect(*fu.result())

    S = F.groupby(["task", "feature_set"])[METRICS].agg(["mean", "std"]).round(4)
    S.to_csv(a.out / "feature_sets_summary.tsv", sep="\t")
    P = paired_table(F); P.round(5).to_csv(a.out / "feature_sets_paired.tsv", sep="\t", index=False)
    if len(IMP):
        (IMP.groupby(["task", "feature_set", "feature"]).importance
            .agg(mean="mean", sd="std", frac_positive=lambda v: (v > 0).mean()).reset_index()
            .sort_values(["task", "feature_set", "mean"], ascending=[True, True, False])
            .to_csv(a.out / "feature_sets_importance.tsv", sep="\t", index=False))
    print(f"\nfinished {time.strftime('%Y-%m-%d %H:%M')} | total {hms(time.time() - clock.t0)}")
    print("\n" + S.to_string())
    print("\n" + P[P.metric == "balanced_accuracy"].round(4).to_string(index=False))
