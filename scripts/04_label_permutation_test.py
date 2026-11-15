"""
Label-permutation test for the repeated-CV CatBoost classifiers
(Ojala & Garriga, JMLR 2010, "Test 1").

Question answered
-----------------
Is the cross-validated performance of each binary task (NR vs UP, UP vs DN,
NR vs DN) higher than expected if the enhancer labels carried no information
about the features?

Procedure
---------
For each run directory produced by run_cv.py:

1. Load the saved task matrix (data/final_assembled_matrix.csv), which is the
   exact input to repeated CV, plus categorical columns and threshold
   settings from the run's metadata JSON files.
2. Observed statistic = mean balanced accuracy over the outer held-out
   evaluations (25 by default; 5 with --cv-repeats 1), read from
   summaries/all_cv_metrics.csv (model == catboost).
3. For b = 1..B: shuffle the binary labels (class counts preserved), then rerun
   the WHOLE CV pipeline exactly as run_cv.py does it:
     - RepeatedStratifiedKFold(N_SPLITS, N_REPEATS, GLOBAL_RANDOM_STATE),
       stratified on the permuted labels
     - FoldPreprocessor fitted on each outer training fold
     - fit_catboost with the same fold seeds
     - inner-CV threshold tuning (if the original run used it)
     - balanced accuracy / ROC-AUC / PR-AUC on each outer held-out fold
   and record the mean over the same outer evaluations.

Faster exact variant: --cv-repeats 1 uses only repeat 1 of the 5x5 design for
BOTH the observed value and every permutation (same splits and fold seeds as
the original repeat 1). The test stays exact because the observed and null
statistics come from the identical procedure; it is ~5x cheaper per pass.
4. p = (1 + #{null >= observed}) / (B + 1).

The fold-level scores are never treated as independent observations: each
permutation contributes ONE value (the 25-fold mean) to the null distribution.

Why this is fast(er) than rerunning run_cv.py B times
-----------------------------------------------------
- Imports the pipeline's own modules (config, preprocessing, models,
  thresholding, evaluation), so model settings stay identical.
- Skips everything that does not affect the test statistic: held-out
  permutation importance (10 x n_features extra predictions per fold),
  model/preprocessor pickling and per-fold CSV writing.
- Runs permutations in parallel worker processes, with CatBoost thread count
  set per worker so the cores are not oversubscribed.
- Checkpoints every finished permutation to its own small CSV. If the run is
  interrupted, re-running the same command resumes where it stopped, and you
  can raise --n-permutations later (e.g. 100 -> 1000) without redoing work.

Usage
-----
The published test (99 permutations, repeat 1 of the 5 x 5 design, the three runs named in [runs] of
paths.toml; results in <ml_outputs>/label_permutation_test):

    uv run python scripts/04_label_permutation_test.py --n-permutations 99 --cv-repeats 1 \
        --n-jobs 4 --threads-per-job 2

Other run folders, labels or output folder:

    uv run python scripts/04_label_permutation_test.py \
        --run-dirs <run folder 1> <run folder 2> <run folder 3> \
        --task-labels "NR vs UP" "UP vs DN" "NR vs DN" \
        --n-permutations 100 --n-jobs 4 --threads-per-job 2 \
        --output-dir <output folder>

Optional sanity check (one unpermuted rerun per task, compared with the saved
CV metrics):  add --check-observed
"""

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold

from a549_enhancers import paths
from a549_enhancers.config import (
    GLOBAL_RANDOM_STATE,
    N_REPEATS,
    N_SPLITS,
    VARIANCE_THRESHOLD,
)

MODEL_NAME = "catboost"
METRICS = ["balanced_accuracy", "roc_auc", "average_precision"]
PRIMARY_METRIC = "balanced_accuracy"
PERMUTATION_SEED_BASE = 20101  # seed for label shuffling; perm b uses base + b
DEFAULT_TASKS = {"nr_vs_up": "NR vs UP", "up_vs_dn": "UP vs DN", "nr_vs_dn": "NR vs DN"}  # used without --run-dirs


# =========================================================================
# Arguments
# =========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Label-permutation test for repeated-CV CatBoost models."
    )
    p.add_argument(
        "--run-dirs", nargs="+", default=None,
        help=(
            "One or more run_cv.py output folders (one per task). "
            "Default: the three runs named in [runs] of paths.toml."
        ),
    )
    p.add_argument(
        "--task-labels", nargs="+", default=None,
        help="Display labels for the run dirs, in the same order.",
    )
    p.add_argument(
        "--n-permutations", type=int, default=100,
        help="Number of label permutations per task (default 100).",
    )
    p.add_argument(
        "--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2),
        help="Parallel worker processes (permutations run concurrently).",
    )
    p.add_argument(
        "--threads-per-job", type=int, default=2,
        help=(
            "CatBoost thread_count inside each worker. Keep "
            "n_jobs x threads_per_job <= number of CPU cores."
        ),
    )
    p.add_argument(
        "--output-dir", default=None,
        help=(
            "Where combined results and figures are written "
            "(default: <ml_outputs>/label_permutation_test)."
        ),
    )
    p.add_argument(
        "--check-observed", action="store_true",
        help=(
            "Also rerun the pipeline once with the TRUE labels and compare "
            "with the saved CV metrics (verifies this script matches run_cv.py)."
        ),
    )
    p.add_argument(
        "--cv-repeats", type=int, default=N_REPEATS,
        help=(
            "Number of CV repeats used for BOTH the observed statistic and the "
            "null (1..N_REPEATS). 1 = repeat 1 of the original run (identical "
            "splits and fold seeds), about 5x faster. Default: all repeats."
        ),
    )
    p.add_argument(
        "--fig-format", default="png", choices=["png", "pdf", "svg"],
    )
    return p.parse_args()


# =========================================================================
# Loading a run
# =========================================================================

def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path) as f:
        return json.load(f)


def load_run(run_dir: Path, cv_repeats: int) -> dict:
    """Load the exact CV input matrix, labels and settings of one run."""
    data_dir = run_dir / "data"
    summ_dir = run_dir / "summaries"

    run_meta = _read_json(summ_dir / "run_metadata.json")
    mat_meta = _read_json(data_dir / "final_assembled_matrix_metadata.json")
    fe_meta = _read_json(data_dir / "feature_engineering_metadata.json")

    if MODEL_NAME not in run_meta.get("models", [MODEL_NAME]):
        raise ValueError(f"{run_dir}: run did not include {MODEL_NAME}.")

    if (run_meta.get("n_splits"), run_meta.get("n_repeats")) != (N_SPLITS, N_REPEATS):
        raise ValueError(
            f"{run_dir}: run used n_splits={run_meta.get('n_splits')}, "
            f"n_repeats={run_meta.get('n_repeats')}, but config.py now has "
            f"{N_SPLITS}/{N_REPEATS}. Restore config.py to the original values."
        )

    mat = pd.read_csv(
        data_dir / "final_assembled_matrix.csv",
        index_col="enhancer_id",
        low_memory=False,
    )
    y = mat["binary_target"].astype(str)
    X = mat.drop(columns=["multiclass_label", "binary_target"])

    if X.shape[1] != mat_meta["n_feature_columns"]:
        raise ValueError(
            f"{run_dir}: expected {mat_meta['n_feature_columns']} feature "
            f"columns, found {X.shape[1]}."
        )

    categorical_cols = [c for c in fe_meta.get("categorical_cols", []) if c in X.columns]

    # Observed per-fold metrics from the original run.
    metrics_path = summ_dir / "all_cv_metrics.csv"
    if metrics_path.exists():
        obs = pd.read_csv(metrics_path)
    else:
        files = sorted((run_dir / "cv_metrics").glob(f"*_{MODEL_NAME}_metrics.csv"))
        if not files:
            raise FileNotFoundError(f"{run_dir}: no CV metrics found.")
        obs = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    obs = obs[obs["model"] == MODEL_NAME].copy()
    obs = obs[pd.to_numeric(obs["repeat"]) <= cv_repeats].copy()
    expected = N_SPLITS * cv_repeats
    if len(obs) != expected:
        raise ValueError(
            f"{run_dir}: expected {expected} {MODEL_NAME} fold rows, found {len(obs)}."
        )

    observed = {m: float(pd.to_numeric(obs[m]).mean()) for m in METRICS if m in obs}

    settings = {
        "task": run_meta["task"],
        "positive_label": str(mat_meta["positive_label"]),
        "negative_label": str(mat_meta["negative_label"]),
        "tune_threshold": bool(run_meta.get("threshold_tuning", False)),
        "threshold_method": run_meta.get("threshold_method", "max_f1"),
        "threshold_inner_splits": int(run_meta.get("threshold_inner_splits") or 3),
        "min_precision": (
            float(run_meta["min_precision"])
            if run_meta.get("min_precision") is not None else 0.30
        ),
    }

    return {
        "X": X,
        "y": y,
        "categorical_cols": categorical_cols,
        "settings": settings,
        "observed": observed,
        "observed_folds": obs,
    }


# =========================================================================
# One full CV pass (mirrors run_cv.py, minus outputs not needed here)
# =========================================================================

def run_cv_once(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_cols: list,
    settings: dict,
    threads_per_job: int,
    cv_repeats: int,
) -> pd.DataFrame:
    # Imported inside the worker so each process sets its own thread count.
    from a549_enhancers import models
    from a549_enhancers.evaluation import evaluate_binary_predictions  # noqa: F401 (parity)
    from a549_enhancers.models import fit_catboost, get_positive_class_proba
    from a549_enhancers.preprocessing import FoldPreprocessor
    from a549_enhancers.thresholding import tune_threshold_on_outer_train

    # fit_catboost copies CATBOOST_PARAMS at call time, so setting the thread
    # count here applies to every model fitted in this worker (outer + inner).
    models.CATBOOST_PARAMS["thread_count"] = int(threads_per_job)

    pos = settings["positive_label"]
    neg = settings["negative_label"]

    # With an int random_state, repeat r of an n-repeat run is identical to
    # repeat r of the original 5-repeat run, so cv_repeats < N_REPEATS
    # reproduces the first cv_repeats repeats exactly.
    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=cv_repeats,
        random_state=GLOBAL_RANDOM_STATE,
    )

    rows = []
    for split_id, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        fold_seed = GLOBAL_RANDOM_STATE + split_id

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pre = FoldPreprocessor(
            categorical_cols=categorical_cols,
            variance_threshold=VARIANCE_THRESHOLD,
        )
        pre.fit(X_train)
        X_train_cb, cat_feats = pre.transform_for_catboost(X_train)
        X_test_cb, _ = pre.transform_for_catboost(X_test)

        model = fit_catboost(
            X_train=X_train_cb,
            y_train=y_train,
            cat_features=cat_feats,
            random_state=fold_seed,
        )
        proba = get_positive_class_proba(model, X_test_cb, positive_label=pos)

        if settings["tune_threshold"]:
            thr, _ = tune_threshold_on_outer_train(
                X_outer_train_raw=X_train,
                y_outer_train=y_train,
                model_name=MODEL_NAME,
                categorical_cols=categorical_cols,
                variance_threshold=VARIANCE_THRESHOLD,
                positive_label=pos,
                random_state=fold_seed,
                n_splits=settings["threshold_inner_splits"],
                threshold_method=settings["threshold_method"],
                min_precision=settings["min_precision"],
            )
        else:
            thr = 0.5

        y_pred = np.where(proba >= thr, pos, neg)
        y_bin = (y_test.values == pos).astype(int)

        rows.append({
            "repeat": split_id // N_SPLITS + 1,
            "fold": split_id % N_SPLITS + 1,
            "decision_threshold": float(thr),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_bin, proba),
            "average_precision": average_precision_score(y_bin, proba),
        })

    return pd.DataFrame(rows)


def permutation_worker(
    perm_id: int,
    X: pd.DataFrame,
    y: pd.Series,
    categorical_cols: list,
    settings: dict,
    threads_per_job: int,
    cv_repeats: int,
    out_path: Path,
) -> tuple:
    """Run one permutation (perm_id >= 1) or the unpermuted check (perm_id 0)."""
    t0 = time.time()
    warnings.filterwarnings("ignore")

    if perm_id == 0:
        y_use = y
    else:
        rng = np.random.default_rng(PERMUTATION_SEED_BASE + perm_id)
        y_use = pd.Series(rng.permutation(y.values), index=y.index, name=y.name)

    folds = run_cv_once(X, y_use, categorical_cols, settings, threads_per_job,
                        cv_repeats)
    folds["pass_seconds"] = round(time.time() - t0, 1)
    folds.insert(0, "permutation", perm_id)
    folds.insert(0, "task", settings["task"])

    # Atomic write: a half-written file is never mistaken for a finished one.
    tmp = out_path.with_suffix(".tmp")
    folds.to_csv(tmp, index=False)
    os.replace(tmp, out_path)

    return perm_id, time.time() - t0


# =========================================================================
# Summaries
# =========================================================================

def p_value(null: np.ndarray, observed: float) -> float:
    """One-sided permutation p-value with the +1 correction (never 0)."""
    null = np.asarray(null, dtype=float)
    return (1.0 + np.sum(null >= observed)) / (len(null) + 1.0)


def summarize_task(label: str, run: dict, perm_dir: Path, n_perm: int) -> tuple:
    files = [perm_dir / f"perm_{b:04d}.csv" for b in range(1, n_perm + 1)]
    folds = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    null = folds.groupby("permutation")[METRICS].mean()
    null.insert(0, "task_label", label)

    rows = []
    for m in METRICS:
        obs = run["observed"].get(m, np.nan)
        v = null[m].to_numpy()
        rows.append({
            "task_label": label,
            "task": run["settings"]["task"],
            "metric": m,
            "cv_repeats": run["cv_repeats"],
            "observed_mean_over_folds": obs,
            "n_permutations": len(v),
            "null_mean": float(np.mean(v)),
            "null_sd": float(np.std(v, ddof=1)),
            "null_95th_percentile": float(np.quantile(v, 0.95)),
            "null_max": float(np.max(v)),
            "n_null_ge_observed": int(np.sum(v >= obs)),
            "p_value": p_value(v, obs),
            "minimum_attainable_p": 1.0 / (len(v) + 1.0),
        })

    return null.reset_index(), pd.DataFrame(rows)


def plot_null_distributions(null_all: pd.DataFrame, results: pd.DataFrame,
                            labels: list, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.0), squeeze=False)

    for ax, label in zip(axes[0], labels):
        v = null_all.loc[null_all["task_label"] == label, PRIMARY_METRIC]
        r = results[(results["task_label"] == label)
                    & (results["metric"] == PRIMARY_METRIC)].iloc[0]
        obs = r["observed_mean_over_folds"]
        p = r["p_value"]

        # Bins span the null AND the observed value, so a tight null near 0.5
        # stays visible next to a distant observed line.
        lo = min(v.min(), obs, 0.5)
        hi = max(v.max(), obs, 0.5)
        pad = 0.05 * (hi - lo) + 1e-3
        bins = np.linspace(lo - pad, hi + pad, 60)

        ax.hist(v, bins=bins, color="#5b6b7c", edgecolor="#5b6b7c", linewidth=0.5,
                label="Permuted labels")
        ax.axvline(obs, color="#c0392b", linewidth=1.8, label="Observed")
        ax.axvline(0.5, color="#555555", linewidth=0.8, linestyle=":", label="Chance")
        ax.set_xlim(lo - pad, hi + pad)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        # With no null value >= observed, p equals its floor 1/(B+1).
        p_txt = f"p ≤ {r['minimum_attainable_p']:.3f}" if r["n_null_ge_observed"] == 0 \
            else f"p = {p:.3f}"
        ax.set_title(f"{label}\nobserved {obs:.3f}, {p_txt}", fontsize=9)
        n_f = N_SPLITS * int(r["cv_repeats"])
        ax.set_xlabel(f"Mean balanced accuracy ({n_f} folds)", fontsize=8)
        ax.tick_params(labelsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    axes[0][0].set_ylabel(f"Permutations (n = {int(results['n_permutations'].iloc[0])})",
                          fontsize=8)
    handles, leg_labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, fontsize=7, frameon=False,
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# =========================================================================
# Main
# =========================================================================

def main():
    args = parse_args()

    if args.run_dirs:
        run_dirs = [Path(d) for d in args.run_dirs]
        labels = args.task_labels or [d.name for d in run_dirs]
    else:
        run_dirs = [paths.run_dir(task) for task in DEFAULT_TASKS]
        labels = args.task_labels or list(DEFAULT_TASKS.values())
    if len(labels) != len(run_dirs):
        raise ValueError("--task-labels must match --run-dirs in length.")

    out_dir = Path(args.output_dir) if args.output_dir else paths.ML / "label_permutation_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    cores = os.cpu_count() or 1
    if args.n_jobs * args.threads_per_job > cores:
        print(f"WARNING: n_jobs x threads_per_job = "
              f"{args.n_jobs * args.threads_per_job} > {cores} cores; "
              f"expect oversubscription.")

    if not 1 <= args.cv_repeats <= N_REPEATS:
        raise ValueError(f"--cv-repeats must be between 1 and {N_REPEATS}.")

    runs = {}
    jobs = []
    for run_dir, label in zip(run_dirs, labels):
        run = load_run(run_dir, args.cv_repeats)
        run["cv_repeats"] = args.cv_repeats
        runs[label] = run

        # Separate checkpoint folders per CV depth, so results from different
        # --cv-repeats settings are never mixed. The full 5-repeat setting
        # keeps the original folder name.
        sub = "per_permutation" if args.cv_repeats == N_REPEATS \
            else f"per_permutation_{args.cv_repeats}rep"
        perm_dir = out_dir / sub / run["settings"]["task"]
        perm_dir.mkdir(parents=True, exist_ok=True)
        run["perm_dir"] = perm_dir

        s = run["settings"]
        print(f"\n{label}: {run['X'].shape[0]} enhancers x {run['X'].shape[1]} features | "
              f"{run['y'].value_counts().to_dict()} | threshold tuning="
              f"{s['tune_threshold']} ({s['threshold_method']}) | observed mean "
              f"balanced accuracy={run['observed'][PRIMARY_METRIC]:.4f}")

        ids = list(range(1, args.n_permutations + 1))
        if args.check_observed:
            ids = [0] + ids

        todo = [b for b in ids if not (perm_dir / f"perm_{b:04d}.csv").exists()]
        print(f"  {len(ids) - len(todo)} already done, {len(todo)} to run")

        for b in todo:
            jobs.append((label, b))

    # Interleave tasks so the progress/ETA is representative early on.
    jobs.sort(key=lambda t: (t[1], labels.index(t[0])))

    if jobs:
        print(f"\nRunning {len(jobs)} CV passes on {args.n_jobs} workers x "
              f"{args.threads_per_job} CatBoost threads ...")
        t_start = time.time()

        tasks = (
            delayed(permutation_worker)(
                b,
                runs[label]["X"],
                runs[label]["y"],
                runs[label]["categorical_cols"],
                runs[label]["settings"],
                args.threads_per_job,
                args.cv_repeats,
                runs[label]["perm_dir"] / f"perm_{b:04d}.csv",
            )
            for label, b in jobs
        )

        done = 0
        for _perm_id, _secs in Parallel(
            n_jobs=args.n_jobs,
            return_as="generator_unordered",
        )(tasks):
            done += 1
            elapsed = time.time() - t_start
            eta = elapsed / done * (len(jobs) - done)
            print(f"  [{done}/{len(jobs)}] elapsed {elapsed / 60:.1f} min | "
                  f"ETA {eta / 60:.1f} min", flush=True)

    # ---------- summaries ----------
    null_tables, result_tables = [], []
    for label in labels:
        run = runs[label]
        null, res = summarize_task(label, run, run["perm_dir"], args.n_permutations)
        null_tables.append(null)
        result_tables.append(res)

        if args.check_observed:
            chk = pd.read_csv(run["perm_dir"] / "perm_0000.csv")
            rerun = chk[PRIMARY_METRIC].mean()
            saved = run["observed"][PRIMARY_METRIC]
            flag = "OK" if abs(rerun - saved) < 0.005 else "MISMATCH - check settings"
            print(f"\n{label}: unpermuted rerun {rerun:.4f} vs saved {saved:.4f} -> {flag}")

    null_all = pd.concat(null_tables, ignore_index=True)
    results = pd.concat(result_tables, ignore_index=True)

    tag = "" if args.cv_repeats == N_REPEATS else f"_{args.cv_repeats}rep"
    null_all.to_csv(out_dir / f"null_distribution_mean_over_folds{tag}.csv", index=False)
    results.to_csv(out_dir / f"label_permutation_test_results{tag}.csv", index=False)

    plot_null_distributions(
        null_all, results, labels,
        out_dir / f"label_permutation_null_balanced_accuracy{tag}.{args.fig_format}",
    )

    pd.set_option("display.width", 160)
    print("\n" + "=" * 80)
    print(results[results["metric"] == PRIMARY_METRIC][[
        "task_label", "cv_repeats", "observed_mean_over_folds", "null_mean", "null_sd",
        "null_95th_percentile", "n_null_ge_observed", "n_permutations", "p_value",
    ]].to_string(index=False))
    print("=" * 80)
    print(f"Results written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
