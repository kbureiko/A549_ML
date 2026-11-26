# =============================================================================
# LIME explanations of the three final CatBoost models (nr_vs_up, up_vs_dn, nr_vs_dn) for EVERY enhancer
# of each model's two classes, plus per-class summaries. Plotting is done by a separate script.
#
# Design:
#   * One explanation per enhancer, for the model's positive class, with SIGNED weights. For a binary model
#     the explanation of class 0 is the same local model with the sign flipped, so weight > 0 pushes
#     towards the positive class and weight < 0 towards the negative class. Both sides of each model are
#     then summarised over the enhancers of that side's true class.
#   * All enhancers of both classes by default (--max-per-class caps a class if wanted; the cap is recorded).
#   * Model files are taken from the given model folders only; the script stops if the folder holds more
#     than one candidate (use --model-file task=path) and never trains a model itself.
#   * Feature names must match the model's feature names exactly; label order is checked (AUC of the
#     model on its own task must be > 0.5).
#   * Rules are mapped back to the feature (LIME as_map) and to the quartile bin of the rule
#     (bin 0 = lowest quartile ... n_bins-1 = highest), so the summaries can say "high X" / "low X".
#   * Feature groups (motif / genomic / control / dex) follow the layer rules of the pipeline
#     (a549_enhancers.feature_groups), not substrings of the rule text; see lime_feature_groups.csv.
#   * Reproducible per enhancer (seed + row position), parallel (--n-jobs), checkpointed per chunk
#     (rerunning resumes; delete <task>/chunks to start again).
#   * Optional stability check: a random subset re-explained with another seed (--stability-n).
#
# Run as a script (not inside Jupyter: the worker processes need the __main__ guard on Windows):
#   uv run python scripts/03_lime_rules_final_models.py --n-jobs 6          # the published run (all defaults)
#   uv run python scripts/03_lime_rules_final_models.py --tasks up_vs_dn --n-jobs 6        # one model only
# The model folders are the three runs named in [runs] of paths.toml; each needs final_model/ from
# scripts/02_run_final_model.py. The default output folder is <ml_outputs>/<date>__lime_final_models.
#
# Outputs (output_dir/):
#   <task>/lime_local_weights.csv.gz     one row per enhancer x rule (rank, feature, group, rule, bin, weight)
#   <task>/lime_enhancers.csv            one row per explained enhancer (true class, p(positive), predicted,
#                                        correct, local R2 of the LIME fit)
#   <task>/lime_summary_by_feature.csv   per side (class) x feature  - main table for the figure
#   <task>/lime_summary_by_rule.csv      per side (class) x exact bin rule - supplementary
#   <task>/lime_stability.csv            (if --stability-n) per side: Spearman of feature summaries, 2 seeds
#   lime_feature_groups.csv, lime_run_metadata.json
#
# Summary columns (for side = class c, over the N enhancers whose true class is c; w_c = weight towards c):
#   frac_present          fraction of the N enhancers where the feature/rule is in the local top-k
#   frac_support          fraction where it is in the top-k AND pushes towards c (w_c > 0)
#   expected_contribution sum(w_c) / N  (absent = 0)  -> ranking used for the figure
#   mean_weight_support   mean w_c where w_c > 0
#   *_correct             the same over enhancers of c that the model classifies correctly
#   frac_high / frac_low  (by feature) share of supporting rules in the top / bottom quartile bin
# =============================================================================
from __future__ import annotations

import argparse
import gc
import json
import multiprocessing
import pickle
import re
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)   # fragmented-frame warnings from feature building

from a549_enhancers import paths
from a549_enhancers.feature_groups import feature_layer
from a549_enhancers.compat import register_legacy_modules

register_legacy_modules()   # run folders written before the code was a package store preprocessing.FoldPreprocessor

ML_ROOT = paths.ML          # [results] ml_outputs in paths.toml
TASKS = {   # task: (model folder, negative label = 0, positive label = 1)
    "nr_vs_up": (paths.run_dir("nr_vs_up"), "NR", "UP"),
    "up_vs_dn": (paths.run_dir("up_vs_dn"), "DN", "UP"),
    "nr_vs_dn": (paths.run_dir("nr_vs_dn"), "NR", "DN"),
}
# up_vs_dn keeps DN = 0 / UP = 1 as in the original script; the AUC check below stops the run if a
# model was trained with the opposite order (then swap the two labels here).

MODEL_SUFFIXES = (".cbm", ".joblib", ".pkl", ".pickle")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    p.add_argument("--model-file", nargs="*", default=[], help="override, e.g. nr_vs_up=path/to/model.joblib")
    p.add_argument("--preprocessor-file", nargs="*", default=[],
                   help="override, e.g. nr_vs_up=path/to/preprocessor.joblib (default: preprocessor next to the model)")
    p.add_argument("--output-dir", default=str(ML_ROOT / f"{date.today():%Y%m%d}__lime_final_models"))
    p.add_argument("--max-per-class", type=int, default=0, help="0 = all enhancers of the class")
    p.add_argument("--num-features", type=int, default=20, help="rules kept per enhancer (local top-k)")
    p.add_argument("--num-samples", type=int, default=5000, help="LIME perturbations per enhancer")
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--chunk-size", type=int, default=100)
    p.add_argument("--tasks-per-worker", type=int, default=0,
                   help="chunks a worker process handles before it is replaced; 0 = never (default). Replacing "
                        "workers can stall the pool on Windows, and the final models are small, so it is off.")
    p.add_argument("--stability-n", type=int, default=200, help="enhancers per class re-run with another seed; 0 = skip")
    p.add_argument("--random-seed", type=int, default=123)
    return p.parse_args()


# ----------------------------------------------------------------------------- data and model
def build_full_matrix():
    from a549_enhancers.data_loading import load_raw_inputs, build_base_feature_matrix
    from a549_enhancers.feature_engineering import build_feature_matrix
    from a549_enhancers.labels import generate_multiclass_labels

    raw = load_raw_inputs()
    X_raw, prematrix_with_ids, motif_cols = build_base_feature_matrix(raw)
    labels = generate_multiclass_labels(prematrix_with_ids)
    X_raw = X_raw.copy(); X_raw.index = X_raw.index.astype(str); X_raw.columns = [str(c) for c in X_raw.columns]
    labels = labels.copy(); labels.index = labels.index.astype(str)
    X, _meta = build_feature_matrix(X_raw=X_raw, original_motif_cols=motif_cols)
    X = X.copy(); X.index = X.index.astype(str); X.columns = [str(c) for c in X.columns]
    return X, labels, [str(c) for c in motif_cols]


def find_model_file(task, folder, overrides, pre_overrides):
    """Final model (+ its preprocessor) inside the model folder; CV fold models are ignored."""
    if task in overrides:
        m = Path(overrides[task])
        if not m.exists():
            raise FileNotFoundError(m)
    else:
        if not folder.exists():
            raise FileNotFoundError(f"Model folder not found: {folder}")
        cands = [f for f in folder.rglob("*") if f.is_file() and f.suffix.lower() in MODEL_SUFFIXES
                 and "cv_models" not in f.parts and not re.search(r"fold|repeat_\d", f.name, re.I)]
        models = [f for f in cands if "preprocess" not in f.name.lower()]
        final = [f for f in models if any("final" in part.lower() for part in f.parts[len(folder.parts):])]
        pool = final or models
        if len(pool) != 1:
            listing = "\n  ".join(str(f) for f in sorted(cands)) or "(none)"
            raise RuntimeError(f"{task}: expected one final model file in {folder}, found {len(pool)}.\n"
                               f"Candidates:\n  {listing}\nPass --model-file {task}=<path>.")
        m = pool[0]
    if task in pre_overrides:
        pre = Path(pre_overrides[task])
        if not pre.exists():
            raise FileNotFoundError(pre)
    else:
        sib = sorted(f for f in m.parent.iterdir() if f.is_file() and "preprocess" in f.name.lower()
                     and f.suffix.lower() in MODEL_SUFFIXES)
        if len(sib) > 1:
            raise RuntimeError(f"{task}: several preprocessors next to {m}: {sib}. Pass --preprocessor-file.")
        pre = sib[0] if sib else None
    return m, pre


class CatBoostPreprocessorAdapter:
    """Wraps the pipeline's FoldPreprocessor so that .transform() = transform_for_catboost()."""

    def __init__(self, pre):
        self.pre = pre
        self.feature_names_in_ = list(getattr(pre, "numeric_cols_", []) or []) + \
            list(getattr(pre, "categorical_cols_", []) or [])
        self.categorical_cols_ = list(getattr(pre, "categorical_cols_", []) or [])
        self.inner_type = f"{type(pre).__module__}.{type(pre).__name__} (transform_for_catboost)"

    def transform(self, X):
        Z = self.pre.transform_for_catboost(X)
        if isinstance(Z, (tuple, list)):          # e.g. (X_for_catboost, cat_feature_indices): keep the table
            tables = [z for z in Z if isinstance(z, (pd.DataFrame, np.ndarray))]
            if not tables:
                raise ValueError(f"transform_for_catboost returned {type(Z).__name__} of "
                                 f"{[type(z).__name__ for z in Z]} - no table found")
            Z = tables[0]
        return Z


def load_preprocessor(f):
    if f is None:
        return None
    if f.suffix.lower() == ".joblib":
        import joblib
        obj = joblib.load(f)
    else:
        with open(f, "rb") as h:
            obj = pickle.load(h)
    if isinstance(obj, dict):
        for k in ("preprocessor", "transformer", "pipeline", *obj):
            if k in obj and (hasattr(obj[k], "transform") or hasattr(obj[k], "transform_for_catboost")):
                obj = obj[k]; break
    if hasattr(obj, "transform_for_catboost"):          # preprocessing.FoldPreprocessor
        return CatBoostPreprocessorAdapter(obj)
    if not hasattr(obj, "transform"):
        raise ValueError(f"{f} has neither transform() nor transform_for_catboost()")
    return obj


def preprocessor_input_names(pre):
    for a in ("feature_names_in_", "input_features_", "feature_names", "columns_", "columns", "input_columns"):
        v = getattr(pre, a, None)
        if v is not None and not callable(v) and len(v):
            return [str(x) for x in v]
    return None


def load_model(f):
    s = f.suffix.lower()
    if s == ".cbm":
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(); m.load_model(str(f)); return m
    if s == ".joblib":
        import joblib
        obj = joblib.load(f)
    else:
        with open(f, "rb") as h:
            obj = pickle.load(h)
    if hasattr(obj, "predict_proba"):
        return obj
    if isinstance(obj, dict):
        for k in ("model", "final_model", "classifier", "estimator", "catboost_model", "best_model", *obj):
            if k in obj and hasattr(obj[k], "predict_proba"):
                return obj[k]
    raise ValueError(f"No predict_proba model in {f}")


def find_threshold(folder):
    """Decision threshold saved with the model (thr_max_f1), if any json in the folder has one."""
    for f in sorted(folder.rglob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        stack = [d]
        while stack:
            x = stack.pop()
            if isinstance(x, dict):
                for k, v in x.items():
                    if "threshold" in str(k).lower() and isinstance(v, (int, float)) and 0 < v < 1:
                        return float(v), f"{f.name}:{k}"
                    stack.append(v)
            elif isinstance(x, list):
                stack.extend(x)
    return 0.5, "default 0.5 (no threshold found in model folder)"


def model_feature_names(model):
    for a in ("feature_names_", "feature_names_in_"):
        v = getattr(model, a, None)
        if v is not None and len(v):
            return [str(x) for x in v]
    return None


def cat_indices(model, n):
    try:
        return sorted({int(i) for i in model.get_cat_feature_indices() if 0 <= int(i) < n})
    except Exception:
        return []


def canon(x):
    if pd.isna(x): return "__MISSING__"
    if isinstance(x, (int, np.integer)): return str(int(x))
    if isinstance(x, (float, np.floating)):
        return str(int(round(x))) if np.isfinite(x) and abs(x - round(x)) < 1e-9 else f"{float(x):.12g}"
    return str(x)


def encode_for_lime(X, cat_idx, raw_levels=False):
    """Numeric matrix for LIME: categorical columns integer-coded, NaN in numeric columns -> median.
    raw_levels=True decodes codes back to the ORIGINAL values (for the pipeline preprocessor), otherwise to
    canonical strings (for a bare CatBoost model with categorical features)."""
    cols, cat_names, decode = [], {}, {}
    for j, c in enumerate(X.columns):
        v = X.iloc[:, j]
        if j in cat_idx:
            cats = v.map(canon); levels = sorted(cats.unique().tolist())
            code = {k: i for i, k in enumerate(levels)}
            cols.append(cats.map(code).astype(float)); cat_names[j] = levels
            if raw_levels:
                first = {k: v.iloc[i] for i, k in reversed(list(enumerate(cats)))}
                decode[j] = [np.nan if k == "__MISSING__" else first[k] for k in levels]
            else:
                decode[j] = levels
        else:
            num = pd.to_numeric(v, errors="coerce").replace([np.inf, -np.inf], np.nan)
            cols.append(num.fillna(num.median() if num.notna().any() else 0.0).astype(float))
    return np.column_stack([c.values for c in cols]), cat_names, decode


class Predictor:
    """Picklable predict_fn: decodes categorical codes back to CatBoost categories."""

    def __init__(self, model, feature_names, decode, threads=1, pre=None, model_names=None):
        self.model, self.names, self.threads = model, list(feature_names), threads
        self.pre, self.model_names = pre, model_names
        self.is_cb = type(model).__module__.startswith("catboost")
        # level arrays keep the original dtype (int for 0/1 columns, object for strings), as the model saw them
        self.levels = {j: (np.asarray(lv, dtype=object) if any(isinstance(x, str) for x in lv) else np.asarray(lv))
                       for j, lv in decode.items()}

    def __call__(self, arr):
        arr = np.atleast_2d(np.asarray(arr, dtype=float))
        cols = {}
        for j, name in enumerate(self.names):
            if j in self.levels:
                lv = self.levels[j]
                cols[name] = lv[np.clip(np.rint(np.nan_to_num(arr[:, j])), 0, len(lv) - 1).astype(int)]
            else:
                cols[name] = arr[:, j]
        df = pd.DataFrame(cols)                       # built in one go: no fragmented blocks to consolidate
        if self.pre is not None:                      # LIME works on the raw features; the model sees these
            Z = self.pre.transform(df)
            if isinstance(Z, pd.DataFrame):
                if self.model_names and list(Z.columns) != self.model_names and set(Z.columns) == set(self.model_names):
                    Z = Z[self.model_names]            # same columns, model order
            else:
                Z = Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z)
                Z = pd.DataFrame(Z, columns=self.model_names if self.model_names and len(self.model_names) == Z.shape[1]
                                 else None)
            df = Z
        p = (self.model.predict_proba(df, thread_count=self.threads) if self.is_cb
             else self.model.predict_proba(df))
        p = np.asarray(p)
        return np.column_stack([1 - p, p]) if p.ndim == 1 else p


def feature_group(name, motif_cols):
    """motif / genomic / control / dex, with the layer rules of the pipeline (a549_enhancers.feature_groups):
    tracks without dex in the name (EtOH and untreated) and the enhancer density are 'control'; dex tracks and the
    GR peaks around the enhancer are 'dex'. Written to lime_feature_groups.csv."""
    if name in motif_cols or str(name).startswith("motif"):
        return "motif"
    return feature_layer(name)


# ----------------------------------------------------------------------------- workers
_W = {}


def _init_worker(model, X_values, feature_names, cat_idx, cat_names, decode, class_names, num_features,
                 num_samples, pre=None, model_names=None):
    from lime.lime_tabular import LimeTabularExplainer
    _W.update(
        X=X_values, cat=set(cat_idx), k=num_features, ns=num_samples,
        pred=Predictor(model, feature_names, decode, threads=1, pre=pre, model_names=model_names),
        expl=LimeTabularExplainer(training_data=X_values, feature_names=list(feature_names),
                                  class_names=class_names, categorical_features=cat_idx,
                                  categorical_names=cat_names, mode="classification",
                                  discretize_continuous=True, discretizer="quartile", random_state=0))


def _explain(positions, seed):
    expl, rows, info = _W["expl"], [], []
    for pos in positions:
        rs = np.random.RandomState(seed + int(pos))          # reproducible per enhancer, independent of chunking
        expl.random_state = rs
        if expl.discretizer is not None:
            expl.discretizer.random_state = rs
        e = expl.explain_instance(_W["X"][pos], _W["pred"], labels=(1,), num_features=_W["k"],
                                  num_samples=_W["ns"])
        for r, ((fi, w), (rule, _)) in enumerate(zip(e.as_map()[1], e.as_list(label=1)), 1):
            if fi in _W["cat"] or expl.discretizer is None:
                b, nb = -1, 0
            else:
                names = expl.discretizer.names[fi]
                b, nb = (names.index(rule) if rule in names else -1), len(names)
            rows.append((int(pos), r, int(fi), rule, b, nb, float(_W["X"][pos][fi]), float(w)))
        info.append((int(pos), float(e.score), float(np.ravel(e.local_pred)[0])))
        del e
    gc.collect()
    return rows, info


# ----------------------------------------------------------------------------- summaries
def summarise(local, enh, neg, pos, key):
    out = []
    for c in (neg, pos):
        sign = 1.0 if c == pos else -1.0
        e = enh[enh.true_class == c]; N = len(e); Nc = int(e.correct.sum())
        L = local[local.enhancer_id.isin(e.enhancer_id)].copy()
        L["w_c"] = sign * L.weight
        L["correct"] = L.enhancer_id.map(e.set_index("enhancer_id").correct)
        g = L.groupby(key)
        s = pd.DataFrame({
            "n_present": g.enhancer_id.nunique(),
            "n_support": g.w_c.apply(lambda x: int((x > 0).sum())),
            "sum_w": g.w_c.sum(),
            "mean_weight_support": g.w_c.apply(lambda x: x[x > 0].mean() if (x > 0).any() else np.nan),
            "mean_abs_weight_present": g.w_c.apply(lambda x: x.abs().mean()),
            "sum_w_correct": L[L.correct].groupby(key).w_c.sum(),
            "n_support_correct": L[L.correct & (L.w_c > 0)].groupby(key).enhancer_id.nunique(),
        }).fillna({"sum_w_correct": 0, "n_support_correct": 0})
        if key == "feature":
            sup = L[(L.w_c > 0) & (L.bin >= 0)]
            gs = sup.groupby("feature")
            s["frac_high"] = gs.apply(lambda x: (x.bin == x.n_bins - 1).mean())
            s["frac_low"] = gs.apply(lambda x: (x.bin == 0).mean())
            s["median_value_support"] = gs.value.median()
            s["median_value_class"] = pd.Series({f: np.nan for f in s.index})
            s["feature_group"] = L.groupby("feature").feature_group.first()
        else:
            s["feature"] = g.feature.first(); s["feature_group"] = g.feature_group.first()
            s["bin"] = g.bin.first(); s["n_bins"] = g.n_bins.first()
        s["frac_present"] = s.n_present / N
        s["frac_support"] = s.n_support / N
        s["expected_contribution"] = s.sum_w / N
        s["frac_support_correct"] = s.n_support_correct / max(Nc, 1)
        s["expected_contribution_correct"] = s.sum_w_correct / max(Nc, 1)
        s = s.drop(columns=["sum_w", "sum_w_correct"]).sort_values("expected_contribution", ascending=False)
        s.insert(0, "rank", np.arange(1, len(s) + 1))
        s.insert(0, "n_class_correct", Nc); s.insert(0, "n_class", N); s.insert(0, "side", c)
        out.append(s.reset_index().rename(columns={"index": key}))
    return pd.concat(out, ignore_index=True)


def add_class_medians(summ, X_task, enh):
    """Median feature value over all enhancers of the side's class (context for 'high'/'low')."""
    for c in summ.side.unique():
        ids = enh.loc[enh.true_class == c, "enhancer_id"]
        med = X_task.loc[ids].apply(pd.to_numeric, errors="coerce").median()
        m = summ.side == c
        summ.loc[m, "median_value_class"] = summ.loc[m, "feature"].map(med)
    return summ


def run_positions(positions, init_args, seed, n_jobs, chunk_size, chunk_dir=None, label="", tasks_per_worker=0):
    chunks = [positions[i:i + chunk_size] for i in range(0, len(positions), chunk_size)]
    rows, info, todo = [], [], []
    for i, ch in enumerate(chunks):
        f = chunk_dir / f"chunk_{i:05d}.pkl" if chunk_dir else None
        if f is not None and f.exists():
            with open(f, "rb") as h:
                r, inf = pickle.load(h)
            rows += r; info += inf
        else:
            todo.append((i, ch, f))
    if not todo:
        return rows, info
    t0, done = time.time(), 0
    print(f"  {label}: {len(positions)} enhancers, {len(todo)}/{len(chunks)} chunks to run, n_jobs={n_jobs}")

    def keep(i, f, res):
        nonlocal done, rows, info
        r, inf = res; rows += r; info += inf; done += 1
        if f is not None:
            with open(f, "wb") as h:
                pickle.dump(res, h)
        el = time.time() - t0
        print(f"    {done}/{len(todo)} chunks | {el / 60:.1f} min | ETA {el / done * (len(todo) - done) / 60:.1f} min",
              flush=True)

    if n_jobs <= 1:
        _init_worker(*init_args)
        for i, ch, f in todo:
            keep(i, f, _explain(ch, seed))
        return rows, info
    # worker processes are replaced after tasks_per_worker chunks (memory that CatBoost / pandas keep is freed);
    # a chunk whose worker fails (e.g. MemoryError) is retried with half the workers, twice
    pending, workers = todo, n_jobs
    for attempt in range(3):
        failed = []
        kw = {"max_tasks_per_child": tasks_per_worker} if tasks_per_worker else {}
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=init_args,
                                 mp_context=multiprocessing.get_context("spawn"), **kw) as ex:
            futs = {ex.submit(_explain, ch, seed): (i, ch, f) for i, ch, f in pending}
            for fu in as_completed(futs):
                i, ch, f = futs[fu]
                err = fu.exception()
                if err is None:
                    keep(i, f, fu.result())
                else:
                    print(f"    chunk {i} failed: {type(err).__name__}: {str(err)[:100]}", flush=True)
                    failed.append((i, ch, f))
        if not failed:
            break
        pending, workers = failed, max(1, workers // 2)
        print(f"  retrying {len(pending)} failed chunk(s) with {workers} worker(s)", flush=True)
    if failed:
        raise RuntimeError(f"{label}: {len(failed)} chunk(s) failed after retries: {[i for i, _, _ in failed]}. "
                           f"Rerun the same command - finished chunks are kept.")
    return rows, info


LOCAL_COLS = ["pos", "rank", "feature_index", "rule", "bin", "n_bins", "value", "weight"]


# ----------------------------------------------------------------------------- main
def main():
    a = parse_args()
    from sklearn.metrics import roc_auc_score
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    overrides = dict(x.split("=", 1) for x in a.model_file)
    pre_overrides = dict(x.split("=", 1) for x in a.preprocessor_file)
    print("Building feature matrix ...")
    X_all, labels, motif_cols = build_full_matrix()
    groups = pd.DataFrame({"feature": X_all.columns, "feature_group": [feature_group(c, motif_cols) for c in X_all.columns]})
    groups.to_csv(out / "lime_feature_groups.csv", index=False)
    print("feature groups:", groups.feature_group.value_counts().to_dict())
    meta = {"args": vars(a), "tasks": {}}

    for task in a.tasks:
        folder, neg, pos = TASKS[task]
        tdir = out / task; (tdir / "chunks").mkdir(parents=True, exist_ok=True)
        print("\n" + "=" * 80 + f"\n{task}\n" + "=" * 80)
        mfile, pfile = find_model_file(task, folder, overrides, pre_overrides)
        model, pre = load_model(mfile), load_preprocessor(pfile)
        print(f"model file: {mfile.stat().st_size / 2**20:.0f} MB")
        # class order: CatBoost trained on string labels sorts them (e.g. ['DN', 'NR']), so column 1 of
        # predict_proba is the alphabetically later class - take neg/pos from the model, not from TASKS
        cls = [str(c) for c in getattr(model, "classes_", [])]
        if len(cls) == 2 and set(cls) == {neg, pos} and (cls[0], cls[1]) != (neg, pos):
            print(f"  model.classes_ = {cls}: using {cls[0]} = 0, {cls[1]} = 1 (TASKS had {neg} = 0, {pos} = 1)")
            neg, pos = cls[0], cls[1]
        elif len(cls) == 2 and set(cls) != {neg, pos} and set(cls) != {"0", "1"}:
            raise RuntimeError(f"{task}: model.classes_ = {cls} does not match the TASKS labels {neg}/{pos}")
        print("model:", mfile, "\npreprocessor:", pfile or "none", f"({getattr(pre, 'inner_type', type(pre).__name__)})" if pre is not None else "")

        common = X_all.index.intersection(labels.index)
        ids = labels.loc[common][labels.loc[common].isin([neg, pos])].index
        mnames = model_feature_names(model)
        if pre is not None:
            # LIME runs on the preprocessor's INPUT columns (raw, interpretable features)
            names = preprocessor_input_names(pre)
            if names is None:
                names = [c for c in X_all.columns]
                print("  preprocessor has no stored input names - passing all rebuilt columns in matrix order")
        else:
            names = mnames
        if names is None:
            raise ValueError(f"{task}: no feature names on the model or preprocessor; cannot verify the feature order.")
        missing = [f for f in names if f not in X_all.columns]
        if missing:
            raise ValueError(f"{task}: {len(missing)} model input features missing from the rebuilt matrix, e.g. {missing[:10]}")
        extra = [c for c in X_all.columns if c not in set(names)]
        X_task = X_all.loc[ids, names]
        print(f"  LIME features: {len(names)} (rebuilt matrix has {len(extra)} columns the model does not use)"
              + (f" | model input after preprocessing: {len(mnames)}" if pre is not None and mnames else ""))
        gmap = groups.set_index("feature").feature_group
        used = pd.DataFrame({"feature": names, "feature_group": [gmap.get(n, "other") for n in names]})
        used.to_csv(tdir / "lime_features_used.csv", index=False)
        if task == a.tasks[0]:
            for gname, gg in used.groupby("feature_group"):
                print(f"  [{gname}] " + ", ".join(gg.feature))
        y = (labels.loc[ids] == pos).astype(int).values
        if pre is None:
            cidx = cat_indices(model, len(names))
        else:   # categorical = raw columns that are non-numeric, or that the model lists as categorical by name
            mcat = {mnames[i] for i in cat_indices(model, len(mnames))} if mnames else set()
            mcat |= set(getattr(pre, "categorical_cols_", []) or [])
            cidx = [j for j, c in enumerate(names)
                    if c in mcat or not pd.api.types.is_numeric_dtype(X_task[c])]
        Xv, cat_names, decode = encode_for_lime(X_task, cidx, raw_levels=pre is not None)

        if pre is not None:
            z = pre.transform(X_task.head(5))
            zc = list(z.columns) if isinstance(z, pd.DataFrame) else None
            print(f"  preprocessor output: {type(z).__name__} {getattr(z, 'shape', '?')}"
                  + (f", columns == model features: {zc == mnames}" if zc is not None and mnames else ""))
            if zc is not None and mnames and set(zc) != set(mnames):
                raise ValueError(f"{task}: preprocessed columns differ from the model's features; "
                                 f"only in preprocessor: {sorted(set(zc) - set(mnames))[:10]}, "
                                 f"only in model: {sorted(set(mnames) - set(zc))[:10]}")
        p1 = Predictor(model, names, decode, threads=-1, pre=pre, model_names=mnames)(Xv)[:, 1]
        auc = roc_auc_score(y, p1)
        thr, thr_src = find_threshold(mfile.parent)
        if thr_src.startswith("default"):
            thr, thr_src = find_threshold(folder)
        print(f"{neg} = 0, {pos} = 1 | in-sample AUC = {auc:.3f} | threshold {thr:.3f} ({thr_src}) | {len(names)} features, "
              f"{len(cidx)} categorical")
        if auc < 0.5:
            raise RuntimeError(f"{task}: AUC < 0.5 - label order is probably reversed; swap {neg}/{pos} in TASKS.")

        # enhancers to explain: all of both classes (or a recorded random cap)
        rng = np.random.default_rng(a.random_seed)
        sel = []
        for c in (neg, pos):
            p_c = np.flatnonzero(labels.loc[ids].values == c)
            if a.max_per_class and len(p_c) > a.max_per_class:
                p_c = np.sort(rng.choice(p_c, a.max_per_class, replace=False))
            sel.append(p_c)
            print(f"  {c}: {len(p_c)} of {int((labels.loc[ids] == c).sum())} enhancers explained")
        positions = np.sort(np.concatenate(sel)).tolist()

        # chunks from an earlier run are reused only if they were made with the same settings and model
        params = {"model_file": str(mfile), "preprocessor_file": str(pfile), "num_features": a.num_features, "num_samples": a.num_samples,
                  "random_seed": a.random_seed, "chunk_size": a.chunk_size, "max_per_class": a.max_per_class, "n_task": len(ids)}
        pf = tdir / "chunks" / "params.json"
        if pf.exists() and json.loads(pf.read_text()) != params:
            raise RuntimeError(f"{tdir / 'chunks'} was made with other settings - delete it or use another --output-dir.")
        pf.write_text(json.dumps(params, indent=1))
        init_args = (model, Xv, names, cidx, cat_names, decode, [neg, pos], a.num_features, a.num_samples,
                     pre, mnames)
        rows, info = run_positions(positions, init_args, a.random_seed, a.n_jobs, a.chunk_size,
                                   tdir / "chunks", label=task, tasks_per_worker=a.tasks_per_worker)

        pid = pd.Series(np.asarray(ids), index=range(len(ids)))
        local = pd.DataFrame(rows, columns=LOCAL_COLS)
        local.insert(0, "enhancer_id", local.pos.map(pid)); local.insert(0, "task", task)
        local["feature"] = [names[i] for i in local.feature_index]
        local["feature_group"] = local.feature.map(groups.set_index("feature").feature_group)
        local["toward"] = np.where(local.weight > 0, pos, neg)
        local = local.drop(columns="pos").sort_values(["enhancer_id", "rank"])
        local.to_csv(tdir / "lime_local_weights.csv.gz", index=False)

        inf = pd.DataFrame(info, columns=["pos", "lime_local_r2", "lime_local_pred"]).set_index("pos")
        enh = pd.DataFrame({"enhancer_id": pid.loc[positions].values, "true_class": labels.loc[ids].values[positions],
                            f"p_{pos}": p1[positions]}, index=positions)
        enh["predicted"] = np.where(enh[f"p_{pos}"] >= thr, pos, neg)
        enh["correct"] = enh.predicted == enh.true_class
        enh = enh.join(inf)
        enh.insert(0, "task", task)
        enh.to_csv(tdir / "lime_enhancers.csv", index=False)

        by_f = add_class_medians(summarise(local, enh, neg, pos, "feature"), X_task, enh)
        by_f.insert(0, "task", task); by_f.to_csv(tdir / "lime_summary_by_feature.csv", index=False)
        by_r = summarise(local, enh, neg, pos, "rule"); by_r.insert(0, "task", task)
        by_r.to_csv(tdir / "lime_summary_by_rule.csv", index=False)
        for c in (neg, pos):
            top = by_f[by_f.side == c].head(8)
            print(f"  top {c}: " + ", ".join(f"{f} ({fs:.0%})" for f, fs in zip(top.feature, top.frac_support)))

        tm = {"model_file": str(mfile), "preprocessor_file": str(pfile), "negative": neg, "positive": pos, "in_sample_auc": auc,
              "threshold": thr, "threshold_source": thr_src, "n_features": len(names),
              "categorical_features": [names[i] for i in cidx],
              "n_explained": {c: int((enh.true_class == c).sum()) for c in (neg, pos)},
              "n_in_class": {c: int((labels.loc[ids] == c).sum()) for c in (neg, pos)},
              "accuracy_at_threshold": {c: float(enh.loc[enh.true_class == c, "correct"].mean()) for c in (neg, pos)},
              "median_lime_local_r2": float(enh.lime_local_r2.median())}

        # stability: same enhancers, different seed -> Spearman of per-feature expected contribution
        if a.stability_n:
            sub = []
            for c in (neg, pos):
                pc = enh.index[enh.true_class == c].values
                sub.append(rng.choice(pc, min(a.stability_n, len(pc)), replace=False))
            sub = np.sort(np.concatenate(sub)).tolist()
            r2, i2 = run_positions(sub, init_args, a.random_seed + 10 ** 7, a.n_jobs, a.chunk_size,
                                   None, label=f"{task} stability", tasks_per_worker=a.tasks_per_worker)
            loc2 = pd.DataFrame(r2, columns=LOCAL_COLS)
            loc2.insert(0, "enhancer_id", loc2.pos.map(pid))
            loc2["feature"] = [names[i] for i in loc2.feature_index]
            loc2["feature_group"] = loc2.feature.map(groups.set_index("feature").feature_group)
            e_sub = enh.loc[sub]
            s1 = summarise(local[local.enhancer_id.isin(e_sub.enhancer_id)], e_sub, neg, pos, "feature")
            s2 = summarise(loc2, e_sub, neg, pos, "feature")
            st = []
            for c in (neg, pos):
                m = s1[s1.side == c].set_index("feature").expected_contribution.to_frame("seed1").join(
                    s2[s2.side == c].set_index("feature").expected_contribution.rename("seed2"), how="outer").fillna(0)
                top20_1 = set(m.seed1.nlargest(20).index); top20_2 = set(m.seed2.nlargest(20).index)
                st.append({"task": task, "side": c, "n_enhancers": int((e_sub.true_class == c).sum()),
                           "spearman_all_features": m.seed1.corr(m.seed2, method="spearman"),
                           "top20_overlap": len(top20_1 & top20_2)})
            st = pd.DataFrame(st); st.to_csv(tdir / "lime_stability.csv", index=False)
            print("  stability:", st[["side", "spearman_all_features", "top20_overlap"]].to_dict("records"))
            tm["stability"] = st.to_dict("records")
        meta["tasks"][task] = tm
        with open(out / "lime_run_metadata.json", "w", encoding="utf-8") as h:
            json.dump(meta, h, indent=2, default=str)

    print("\nSaved to", out)


if __name__ == "__main__":
    main()
