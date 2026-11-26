"""Spatial-leakage audit of the enhancer classifiers, with CatBoost and the pipeline's own preprocessing.

Answers the reviewer question "are nearby enhancers independent, and do random CV splits
inflate the performance estimate?" in three parts:

  (1) CV scheme comparison - random stratified 5-fold (as published) vs whole chromosomes
      held out vs 5 Mb genomic blocks held out. Same features, same CatBoost settings, the same
      fold-fitted preprocessing and the same max-F1 threshold tuning inside each outer training
      fold, so only the fold assignment differs between the three schemes.
  (2) Spatial dependence - correlation of the feature profiles of neighbouring enhancers and
      how often they share a class, as a function of the distance between them.
  (3) Fold co-assignment - how often neighbouring enhancers fell in the SAME fold in the
      published random splits (chance = 1/n_splits).

Reads the assembled matrices of the finished runs, so no feature re-assembly is needed.

Leakage control as in the published pipeline: the preprocessing (FoldPrep below) is fitted on the outer training fold
only (variance filter and categorical levels), the outer held-out fold is transformed with the
train-derived objects, and the decision threshold is chosen from inner-CV out-of-fold
predictions of that training fold. The outer fold is used for evaluation only.

Usage
-----
    uv run python scripts/05_cv_scheme_comparison.py                       # everything, all three tasks
    uv run python scripts/05_cv_scheme_comparison.py --repeats 2           # quick check
    uv run python scripts/05_cv_scheme_comparison.py --tasks "NR vs UP"    # one task
    uv run python scripts/05_cv_scheme_comparison.py --skip-cv             # only parts (2) and (3)
    uv run python scripts/05_cv_scheme_comparison.py --jobs 5              # 5 folds in parallel, cores/5 threads each

Runtime: part (1) is 3 tasks x 3 schemes x repeats x 5 folds x 4 CatBoost fits (3 inner folds
for the threshold + 1 outer model). With the default 5 repeats that is 900 fits of an
800-iteration model; budget a few hours, or start with --repeats 2. Partial results are written
after every task, so the run can be stopped and resumed task by task.

Outputs (in <ml_outputs>/cv_scheme_comparison, or in --out):
    cv_scheme_comparison_folds.tsv     one row per task x scheme x repeat x fold
    cv_scheme_comparison_summary.tsv   mean +- SD over folds
    cv_neighbour_dependence.tsv        part (2)
    cv_fold_coassignment.tsv           part (3)
cv_scheme_comparison_folds.tsv is read by figures/figS1_model_performance.py (panel F).

Preprocessing, model settings and threshold rule are those of scripts/01_run_cv.py. The positive class is the
smaller class of each task, so for UP vs DN it is DN.
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, balanced_accuracy_score,
                             f1_score, precision_score, recall_score)

from a549_enhancers import paths
from a549_enhancers.config import N_REPEATS, N_SPLITS, VARIANCE_THRESHOLD   # same settings as scripts/01_run_cv.py
from a549_enhancers.thresholding import choose_threshold_from_scores

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------- configuration
ML = paths.ML                                               # [results] ml_outputs in paths.toml
RUNS = {"NR vs UP": paths.RUN_NAMES["nr_vs_up"],            # folder names of the finished runs ([runs] in paths.toml)
        "NR vs DN": paths.RUN_NAMES["nr_vs_dn"],
        "UP vs DN": paths.RUN_NAMES["up_vs_dn"]}

# folds, repeats and the variance filter are read from config.py, as in the runs themselves
INNER_SPLITS = 3                       # inner CV used to pick the decision threshold
CATEGORICAL_COLS = ["annotation"]      # categorical for CatBoost, together with the motif_binary:: columns

BLOCK = 5_000_000                      # genomic block size for the block-held-out scheme
DIST_BINS = ((1e4, "< 10 kb"), (2.5e4, "< 25 kb"), (5e4, "< 50 kb"), (1e5, "< 100 kb"),
             (np.inf, "any adjacent"))

# exactly the settings stored in the published cv_models/*_catboost.joblib
CATBOOST = dict(iterations=800, learning_rate=0.03, depth=6, l2_leaf_reg=3, rsm=1, border_count=128,
                bootstrap_type="Bayesian", bagging_temperature=1, random_strength=1,
                loss_function="Logloss", eval_metric="PRAUC", random_seed=42,
                allow_writing_files=False, verbose=0,
                auto_class_weights="Balanced")          # as in config.CATBOOST_PARAMS



class FoldPrep:
    """Fold-fitted preprocessing, as in scripts/01_run_cv.py: numeric coercion without imputation, variance filter
    on the numeric features, 'annotation' and the motif indicators passed to CatBoost as categorical features."""

    def fit(self, X):
        self.cat = [c for c in X.columns if c in CATEGORICAL_COLS or c.startswith("motif_binary::")]
        num = [c for c in X.columns if c not in self.cat]
        Xn = X[num].apply(pd.to_numeric, errors="coerce")
        if Xn.isna().any().any():
            raise ValueError(f"NaN in numeric features: {Xn.columns[Xn.isna().any()].tolist()[:10]}")
        var = Xn.var(ddof=0)
        self.keep = [c for c in num if var[c] > VARIANCE_THRESHOLD]
        return self

    def transform_for_catboost(self, X):
        out = X[self.keep].apply(pd.to_numeric, errors="coerce")
        for c in self.cat:
            out[c] = X[c].astype("object").fillna("__MISSING__").astype(str)
        return out, self.cat


def hms(sec):
    sec = int(round(sec)); return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


class Clock:
    """Elapsed time and a running ETA from the average duration of the finished units."""
    def __init__(self, n_units):
        self.t0, self.n, self.done = time.time(), n_units, 0

    def tick(self):
        self.done += 1
        el = time.time() - self.t0
        eta = el / self.done * (self.n - self.done)
        return f"[{self.done}/{self.n} | elapsed {hms(el)} | ETA {hms(eta)} | ends ~{time.strftime('%H:%M', time.localtime(time.time() + eta))}]"


# ----------------------------------------------------------------------------- data
def load(run):
    """Raw feature frame, binary target and the genomic position of every enhancer of one run.

    No preprocessing here: FoldPrep is fitted inside each outer training fold.
    """
    df = pd.read_csv(ML / run / "data" / "final_assembled_matrix.csv", low_memory=False)
    pos_label = df.binary_target.value_counts().idxmin()       # the minority class is the positive one
    y = (df.binary_target == pos_label).values.astype(int)
    m = df.enhancer_id.str.extract(r"(chr[\w]+)_(\d+)_(\d+)")
    chrom = m[0].values
    mid = ((m[1].astype(int) + m[2].astype(int)) // 2).values
    X = df.drop(columns=[c for c in ("enhancer_id", "multiclass_label", "binary_target") if c in df])
    return df.enhancer_id.values, X.reset_index(drop=True), y, chrom, mid, pos_label


# ----------------------------------------------------------------------------- threshold tuning
def tuned_threshold(X, y, cat_features, seed):
    """Threshold maximising F1 on inner-CV out-of-fold predictions of the training fold.

    Same rule as scripts/01_run_cv.py (F1-maximising point of the precision-recall curve). The outer held-out
    fold is never involved.
    X is the already-transformed training fold, so the inner models see exactly the features the
    outer model will see.
    """
    oof = np.zeros(len(y))
    for tr, va in StratifiedKFold(INNER_SPLITS, shuffle=True, random_state=seed).split(X, y):
        clf = CatBoostClassifier(**CATBOOST).fit(X.iloc[tr], y[tr], cat_features=cat_features)
        oof[va] = clf.predict_proba(X.iloc[va])[:, 1]
    return choose_threshold_from_scores(np.where(y == 1, "pos", "neg"), oof, positive_label="pos", method="max_f1")


def evaluate_fold(X, y, tr, te, seed):
    """Fit the preprocessing, the threshold and the model on the training fold; score the held-out fold."""
    pre = FoldPrep().fit(X.iloc[tr])
    X_tr, cat_features = pre.transform_for_catboost(X.iloc[tr])
    X_te, _ = pre.transform_for_catboost(X.iloc[te])
    thr = tuned_threshold(X_tr, y[tr], cat_features, seed)
    clf = CatBoostClassifier(**CATBOOST).fit(X_tr, y[tr], cat_features=cat_features)
    p = clf.predict_proba(X_te)[:, 1]
    yhat = p >= thr
    return dict(n_test=len(te), n_pos=int(y[te].sum()), n_features=X_tr.shape[1], threshold=thr,
                balanced_accuracy=balanced_accuracy_score(y[te], yhat),
                roc_auc=roc_auc_score(y[te], p),
                average_precision=average_precision_score(y[te], p),
                positive_f1=f1_score(y[te], yhat, zero_division=0),
                positive_precision=precision_score(y[te], yhat, zero_division=0),
                positive_recall=recall_score(y[te], yhat, zero_division=0))


# ----------------------------------------------------------------------------- (1) CV schemes
SCHEMES = ["random", "chromosome", f"{BLOCK // 10**6} Mb block"]
TASK_ORDER = ["NR vs UP", "NR vs DN", "UP vs DN"]          # largest first: better load balancing


@lru_cache(maxsize=None)
def load_cached(task):
    return load(RUNS[task])


def init_worker(threads):
    CATBOOST["thread_count"] = threads


def group_folds(groups, y, n_splits, seed):
    """Folds that keep every group (chromosome or block) together, with a different assignment for every seed.
    Groups are taken in random order and each goes to the fold that so far has the fewest enhancers of the positive
    class (ties: fewest enhancers), so the folds are balanced in both. StratifiedGroupKFold is not used because its
    assignment is practically the same for every random state, which made the repeats identical."""
    rng = np.random.RandomState(seed)
    groups, y = np.asarray(groups), np.asarray(y)
    ids = rng.permutation(np.unique(groups))
    pos = {k: int(y[groups == k].sum()) for k in ids}
    size = {k: int((groups == k).sum()) for k in ids}
    fold_of, n_pos, n_all = {}, np.zeros(n_splits), np.zeros(n_splits)
    for k in ids:
        f = int(np.lexsort((rng.rand(n_splits), n_all, n_pos))[0])
        fold_of[k] = f
        n_pos[f] += pos[k]; n_all[f] += size[k]
    assigned = np.array([fold_of[k] for k in groups])
    return [(np.flatnonzero(assigned != f), np.flatnonzero(assigned == f)) for f in range(n_splits)]


def split_for(task, scheme, rep):
    """Deterministic outer split, so a fold is identical wherever the job runs."""
    _, X, y, chrom, mid, _ = load_cached(task)
    if scheme == "random":
        return list(StratifiedKFold(N_SPLITS, shuffle=True, random_state=rep).split(X, y))
    g = pd.factorize(chrom)[0] if scheme == "chromosome" else \
        pd.factorize(pd.Series(chrom) + "_" + (mid // BLOCK).astype(str))[0]
    return group_folds(g, y, N_SPLITS, rep)


def fold_job(task, scheme, rep, k):
    _, X, y, _, _, _ = load_cached(task)
    tr, te = split_for(task, scheme, rep)[k - 1]
    t = time.time()
    res = evaluate_fold(X, y, tr, te, rep)
    return dict(task=task, scheme=scheme, repeat=rep, fold=k, **res, seconds=round(time.time() - t, 1))


def cv_schemes(tasks, n_repeats, out, n_jobs=1, threads=0):
    rows, pending = [], {}
    tasks = sorted(tasks, key=TASK_ORDER.index)
    for task in tasks:
        _, X, y, _, _, pos_label = load_cached(task)
        print(f"{task}: {X.shape[0]} enhancers, {X.shape[1]} raw columns, {y.sum()} positive ({pos_label})", flush=True)
    jobs = [(t, sc, r, k) for t in tasks for sc in SCHEMES for r in range(1, n_repeats + 1) for k in range(1, N_SPLITS + 1)]
    threads = threads or max(1, (os.cpu_count() or 1) // n_jobs)
    clock = Clock(len(jobs))                                       # one unit = one outer fold (4 CatBoost fits)
    print(f"started {time.strftime('%Y-%m-%d %H:%M')} | {len(jobs)} outer folds | "
          f"{n_jobs} worker process(es) x {threads} CatBoost thread(s)\n", flush=True)

    def collect(row):
        rows.append(row)
        key = (row["task"], row["scheme"], row["repeat"])
        pending.setdefault(key, []).append(row)
        print(f"    {key[0]} | {key[1]} | repeat {key[2]} | fold {row['fold']}/{N_SPLITS} ({row['seconds']:.0f} s) "
              f"{clock.tick()}", flush=True)
        if len(pending[key]) == N_SPLITS:
            f = pd.DataFrame(pending.pop(key))
            print(f"  ✓ {key[0]} | {key[1]:<12} repeat {key[2]}  BA={f.balanced_accuracy.mean():.3f}"
                  f"  AUC={f.roc_auc.mean():.3f}  AP={f.average_precision.mean():.3f}"
                  f"  ({f.n_features.iloc[0]} features kept)", flush=True)
            write_cv(pd.DataFrame(rows), out)                      # partial save after every finished block

    if n_jobs == 1:
        init_worker(threads)
        for j in jobs:
            collect(fold_job(*j))
    else:
        with ProcessPoolExecutor(max_workers=n_jobs, initializer=init_worker, initargs=(threads,)) as ex:
            for fu in as_completed([ex.submit(fold_job, *j) for j in jobs]):
                collect(fu.result())
    print(f"CV finished {time.strftime('%Y-%m-%d %H:%M')} | total {hms(time.time() - clock.t0)}", flush=True)
    return pd.DataFrame(rows).sort_values(["task", "scheme", "repeat", "fold"])


def write_cv(R, out):
    R.to_csv(out / "cv_scheme_comparison_folds.tsv", sep="\t", index=False)
    (R.groupby(["task", "scheme"])[["balanced_accuracy", "roc_auc", "average_precision", "positive_f1"]]
      .agg(["mean", "std"]).round(4).to_csv(out / "cv_scheme_comparison_summary.tsv", sep="\t"))


# ----------------------------------------------------------------------------- (2) and (3)
def neighbours(task, out, n_sample=1500, seed=0):
    """Dependence between neighbouring enhancers, and their fold co-assignment in the published splits.

    The feature profiles are compared on the full numeric matrix (standardised), without the
    fold-fitted variance filter: this part describes the data, not a model, so no fold exists.
    """
    run = RUNS[task]
    ids, X, _, chrom, mid, _ = load(run)
    lab = pd.read_csv(ML / run / "data" / "final_assembled_matrix.csv",
                      usecols=["multiclass_label"], low_memory=False).multiclass_label.values
    num = X.drop(columns=[c for c in CATEGORICAL_COLS if c in X]).apply(pd.to_numeric, errors="coerce")
    num = num.loc[:, ~(num.isna().all() | (num.nunique(dropna=True) <= 1))].values

    o = np.lexsort((mid, chrom))                   # order the enhancers along the genome
    Xo, co, mo, lo, io = num[o], chrom[o], mid[o], lab[o], ids[o]
    Z = np.nan_to_num(Xo)
    Z = (Z - Z.mean(0)) / np.where(Z.std(0) == 0, 1, Z.std(0))
    adj = np.array([k for k in range(len(Z) - 1) if co[k] == co[k + 1]])      # neighbouring pairs
    dist = mo[adj + 1] - mo[adj]
    rng = np.random.default_rng(seed)
    p_same = sum((lab == c).mean() ** 2 for c in np.unique(lab))              # expected class concordance

    dep = []
    for t, name in DIST_BINS:
        sel = adj[dist < t]
        s = rng.choice(sel, min(n_sample, len(sel)), replace=False)
        dep.append(dict(pair_set=name, n_pairs=len(sel),
                        mean_profile_r=np.mean([np.corrcoef(Z[k], Z[k + 1])[0, 1] for k in s]),
                        same_class_pct=100 * np.mean(lo[sel] == lo[sel + 1]),
                        expected_same_class_pct=100 * p_same))
    i, j = rng.integers(0, len(Z), n_sample), rng.integers(0, len(Z), n_sample)
    dep.append(dict(pair_set="random pairs", n_pairs=len(Z),
                    mean_profile_r=np.mean([np.corrcoef(Z[a], Z[b])[0, 1] for a, b in zip(i, j)]),
                    same_class_pct=100 * np.mean(lo[i] == lo[j]), expected_same_class_pct=100 * p_same))
    pd.DataFrame(dep).to_csv(out / "cv_neighbour_dependence.tsv", sep="\t", index=False, float_format="%.4g")

    P = pd.read_csv(ML / run / "summaries" / "all_cv_predictions.csv",
                    usecols=["enhancer_id", "repeat", "fold"])
    co_rows = []
    for t, name in DIST_BINS[:-1]:
        sel = adj[dist < t]
        vals = []
        for _, g in P.groupby("repeat"):
            f = g.set_index("enhancer_id").fold
            a, b = f.reindex(io[sel]).values, f.reindex(io[sel + 1]).values
            ok = ~(pd.isna(a) | pd.isna(b))
            vals.append(np.mean(a[ok] == b[ok]))
        co_rows.append(dict(pair_set=name, n_pairs=len(sel), same_fold_pct=100 * np.mean(vals),
                            same_fold_sd=100 * np.std(vals), chance_same_fold_pct=100 / N_SPLITS))
    pd.DataFrame(co_rows).to_csv(out / "cv_fold_coassignment.tsv", sep="\t", index=False, float_format="%.4g")


# ----------------------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=list(RUNS), choices=list(RUNS))
    ap.add_argument("--repeats", type=int, default=N_REPEATS)
    ap.add_argument("--out", type=Path, default=ML / "cv_scheme_comparison")
    ap.add_argument("--neighbour-task", default="NR vs UP", choices=list(RUNS),
                    help="which matrix to describe in parts (2) and (3); NR vs UP carries all three classes")
    ap.add_argument("--skip-cv", action="store_true", help="only the dependence and co-assignment tables")
    ap.add_argument("--jobs", type=int, default=1, help="parallel worker processes (one outer fold each)")
    ap.add_argument("--threads", type=int, default=0, help="CatBoost threads per worker (default: cores / jobs)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    if not a.skip_cv:
        R = cv_schemes(a.tasks, a.repeats, a.out, a.jobs, a.threads)
        write_cv(R, a.out)
        print("\n" + R.groupby(["task", "scheme"])[["balanced_accuracy", "roc_auc", "average_precision"]]
              .agg(["mean", "std"]).round(3).to_string())
    neighbours(a.neighbour_task, a.out)
    print(f"\nWritten to {a.out}")
