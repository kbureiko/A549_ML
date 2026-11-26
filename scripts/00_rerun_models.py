"""Rerun of the pairwise enhancer models: cross-validation, final model and (optionally) LIME, for the tasks you choose.

One command per model (or several models in one command). Each run is written to its own dated folder in
[results] ml_outputs, so earlier runs are never overwritten:

    <date>__<task>__thr_max_f1/            cross-validation (scripts/01_run_cv.py) and final_model/ (scripts/02)
    <date>__lime_final_models/<task>/      LIME (scripts/03), only with --lime

Settings of the rerun (agreed on 18 Nov 2026). They live in src/a549_enhancers/config.py, where every model script
reads them; this script checks them before it starts anything and stops if one differs:

    variance filter        none (VARIANCE_THRESHOLD = 0.0: only constant features are removed)
    class weights          CatBoost auto_class_weights = "Balanced"
    cross-validation       5 folds x 10 repeats = 50 held-out evaluations
    decision threshold     max F1 on inner 3-fold predictions of each training fold (--tune-threshold)
    stable features        positive held-out permutation importance in >= 80 % of the evaluations, positive median
    class labels           the labels of the earlier runs: unadjusted p-value < 0.05 and |log2FC| >= 0.584
                           (478 UP, 232 DN, 7,560 NR); the label counts are checked before the run starts

Usage (from the repository folder)

    uv run python scripts/00_rerun_models.py --tasks up_vs_dn                      # one model: CV + final model
    uv run python scripts/00_rerun_models.py --tasks nr_vs_dn nr_vs_up             # two models, one after the other
    uv run python scripts/00_rerun_models.py --tasks up_vs_dn nr_vs_up nr_vs_dn --lime      # everything
    uv run python scripts/00_rerun_models.py --tasks nr_vs_dn --steps final lime --date 20261118   # later steps of an
                                                                                   # existing run of that date
    uv run python scripts/00_rerun_models.py --tasks nr_vs_dn --dry-run            # show what would run

Additional runs (cross-validation only; the final-model and LIME steps do not apply to them):

    uv run python scripts/00_rerun_models.py --tasks grbound_nr_vs_up grbound_nr_vs_up_pretreatment

    grbound_nr_vs_up                 GR-bound UP against GR-bound NR enhancers, all features
    grbound_nr_vs_up_pretreatment    the same enhancers, without the features measured after dex
    nr_vs_dn_transcribed             DN against NR enhancers that are transcribed in the range of the DN class
                                     (needs the PRO-seq table, [layout] proseq_ma_table in paths.toml)
    nr_vs_dn_matched                 DN against their nearest NR enhancers in length and PRO-seq signal (same table)

Every cross-validation run also writes the grouped permutation importance (summaries/grouped_importance_summary.csv).

    --date      prefix of the run folders (default: today, YYYYMMDD)
    --steps     any of: cv final lime (default: cv final; --lime adds lime)
    --force     run a step again although its output is already complete
    --lime-jobs worker processes of the LIME step (default 4)

A step whose output is already complete is skipped, so the same command can be repeated after an interruption
(the cross-validation itself starts again from its first fold).

When all three models are done, put the three folder names into [runs] of paths.toml (the script prints the
lines); every downstream script (tables, figures) reads the runs from there.
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
TASKS = ["nr_vs_up", "up_vs_dn", "nr_vs_dn"]
# Additional cross-validation runs (no final model, no LIME): run name -> (task, options of 01_run_cv.py, folder tag)
EXTRA_RUNS = {
    "grbound_nr_vs_up": ("nr_vs_up", ["--subset", "gr_bound"], "__grbound"),
    "nr_vs_dn_transcribed": ("nr_vs_dn", ["--subset", "transcribed"], "__transcribed"),
    "nr_vs_dn_matched": ("nr_vs_dn", ["--subset", "matched"], "__matched"),
    "grbound_nr_vs_up_pretreatment": ("nr_vs_up", ["--subset", "gr_bound", "--feature-set", "pretreatment"],
                                      "__grbound__pretreatment"),
}
MODEL = "catboost"
THRESHOLD_METHOD = "max_f1"

# setting -> (required value, description)
REQUIRED = {
    "VARIANCE_THRESHOLD": (0.0, "no variance filter (only constant features are removed)"),
    "N_SPLITS": (5, "outer folds"),
    "N_REPEATS": (10, "repeats of the outer cross-validation"),
    "THRESHOLD_INNER_CV_SPLITS": (3, "inner folds for the decision threshold"),
    "PERMUTATION_N_REPEATS": (10, "shuffles per feature in the permutation importance"),
    "PERMUTATION_SCORING": ("average_precision", "score of the permutation importance"),
    "STABLE_MIN_FRACTION_POSITIVE_IMPORTANCE": (0.80, "stability criterion"),
    "STABLE_MIN_MEDIAN_IMPORTANCE": (0.0, "stable features need a positive median importance"),
    "PVAL_COL": ("ctrl vs. dex p-value", "class labels of the earlier runs (unadjusted p-value)"),
    "PVAL_THRESHOLD": (0.05, "p-value threshold of the class labels"),
    "LOG2FC_THRESHOLD": (0.584, "log2 fold-change threshold of the class labels"),
    "MOTIFS_AS_CATEGORICAL_FOR_CATBOOST": (True, "motifs are categorical features"),
}
REQUIRED_CATBOOST = {"auto_class_weights": "Balanced", "iterations": 800, "depth": 6, "learning_rate": 0.03,
                     "l2_leaf_reg": 3.0, "bootstrap_type": "Bayesian"}
OLD_LABEL_COUNTS = {"UP": 478, "DN": 232, "NR": 7560}


def hms(sec):
    sec = int(round(sec)); return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def check_settings():
    """Stop unless config.py holds the agreed settings (all model scripts read them from there)."""
    from a549_enhancers import config
    rows = []
    for name, (want, what) in REQUIRED.items():
        have = getattr(config, name, "<missing>")
        rows.append((name, have, want, what, have == want and type(have) is type(want) or
                     (isinstance(want, float) and isinstance(have, (int, float)) and float(have) == want)))
    for name, want in REQUIRED_CATBOOST.items():
        have = config.CATBOOST_PARAMS.get(name, "<missing>")
        ok = have == want or (isinstance(want, float) and isinstance(have, (int, float)) and float(have) == want)
        rows.append((f"CATBOOST_PARAMS['{name}']", have, want, "CatBoost setting", ok))
    print("settings (src/a549_enhancers/config.py):")
    for name, have, want, what, ok in rows:
        print(f"  {'ok ' if ok else 'NO '} {name} = {have!r}" + ("" if ok else f"   <- must be {want!r} ({what})"))
    bad = [r for r in rows if not r[4]]
    if bad:
        sys.exit(f"\n{len(bad)} setting(s) in src/a549_enhancers/config.py differ from the agreed rerun settings. "
                 "Change them there and start again; nothing was run.")
    return {name: have for name, have, *_ in rows}


def check_labels(allow_other):
    """Class counts on the enhancers of the feature matrix must be those of the earlier runs."""
    from a549_enhancers.data_loading import build_base_feature_matrix, load_raw_inputs
    from a549_enhancers.labels import generate_multiclass_labels
    X_raw, prematrix_with_ids, _ = build_base_feature_matrix(load_raw_inputs())
    labels = generate_multiclass_labels(prematrix_with_ids)
    labels = labels.loc[X_raw.index.intersection(labels.index)]
    counts = {k: int(v) for k, v in labels.value_counts().items()}
    print(f"class labels: {counts}  (earlier runs: {OLD_LABEL_COUNTS})")
    if counts != OLD_LABEL_COUNTS:
        msg = ("The class counts differ from the earlier runs. Check PVAL_COL in config.py and the input tables in "
               "[inputs] ml_inputs.")
        if not allow_other:
            sys.exit(msg + " Nothing was run (use --allow-other-labels to run with these labels).")
        print("WARNING: " + msg)
    return counts


def run(cmd, dry):
    print("\n$ " + " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd), flush=True)
    if dry:
        return 0.0
    t0 = time.time()
    subprocess.run([str(c) for c in cmd], cwd=REPO, check=True)
    return time.time() - t0


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() + (" (uncommitted changes)" if dirty.stdout.strip() else "") if out.returncode == 0 else None
    except Exception:
        return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", required=True, choices=TASKS + list(EXTRA_RUNS), help="model(s) to run")
    ap.add_argument("--date", default=f"{date.today():%Y%m%d}", help="prefix of the run folders (default: today)")
    ap.add_argument("--steps", nargs="+", choices=["cv", "final", "lime"], default=["cv", "final"])
    ap.add_argument("--lime", action="store_true", help="also run the LIME step")
    ap.add_argument("--lime-jobs", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="run a step again although its output is complete")
    ap.add_argument("--allow-other-labels", action="store_true", help="run although the class counts are not those of the earlier runs")
    ap.add_argument("--skip-label-check", action="store_true", help="do not load the input tables to check the class counts")
    ap.add_argument("--dry-run", action="store_true", help="check the settings and print the commands only")
    a = ap.parse_args()
    steps = list(dict.fromkeys(a.steps + (["lime"] if a.lime else [])))
    tasks = [t for t in TASKS + list(EXTRA_RUNS) if t in a.tasks]    # fixed order
    base_tasks = [t for t in tasks if t in TASKS]

    from a549_enhancers import paths
    ml = Path(paths.ML)
    print(f"rerun {a.date} | tasks: {', '.join(tasks)} | steps: {', '.join(steps)} | output: {ml}\n")
    settings = check_settings()
    label_counts = None if a.skip_label_check else check_labels(a.allow_other_labels)

    run_dirs = {t: ml / (f"{a.date}__{EXTRA_RUNS[t][0]}__thr_{THRESHOLD_METHOD}{EXTRA_RUNS[t][2]}" if t in EXTRA_RUNS
                         else f"{a.date}__{t}__thr_{THRESHOLD_METHOD}") for t in tasks}
    lime_dir = ml / f"{a.date}__lime_final_models"
    log = {"date": a.date, "started": f"{datetime.now():%Y-%m-%d %H:%M}", "settings": settings,
           "label_counts": label_counts, "git_commit": git_commit(), "python": sys.version.split()[0], "steps": {}}
    t_all = time.time()

    for t in tasks:
        rd = run_dirs[t]
        cv_done = (rd / "summaries" / "run_metadata.json").exists()
        final_dir = rd / "final_model" / MODEL
        final_done = (final_dir / "metadata.json").exists()
        print("\n" + "=" * 100 + f"\n{t}  ->  {rd.name}\n" + "=" * 100)

        if "cv" in steps:
            if cv_done and not a.force:
                print("cross-validation: complete, skipped (--force to run it again)")
            else:
                if rd.exists() and any(rd.iterdir()) and not cv_done and not a.dry_run:
                    print("cross-validation: the run folder exists but is not complete; it is run again from the first fold")
                cv_task, cv_options = (EXTRA_RUNS[t][0], EXTRA_RUNS[t][1]) if t in EXTRA_RUNS else (t, [])
                sec = run([sys.executable, SCRIPTS / "01_run_cv.py", "--task", cv_task, "--models", MODEL,
                           "--tune-threshold", "--threshold-method", THRESHOLD_METHOD, "--run-prefix", a.date,
                           *cv_options], a.dry_run)
                log["steps"][f"{t}:cv"] = hms(sec); cv_done = cv_done or not a.dry_run
                final_done = False if not a.dry_run else final_done          # a new CV needs a new final model

        if "final" in steps and t in EXTRA_RUNS:
            print("final model: not fitted for this run (cross-validation only)")
        elif "final" in steps:
            if not cv_done and not a.dry_run:
                sys.exit(f"{t}: no complete cross-validation in {rd}; run the cv step first.")
            if final_done and not a.force:
                print("final model: complete, skipped (--force to fit it again)")
            else:
                sec = run([sys.executable, SCRIPTS / "02_run_final_model.py", "--task", t, "--models", MODEL,
                           "--cv-output-root", rd], a.dry_run)
                log["steps"][f"{t}:final"] = hms(sec)
                meta = final_dir / "metadata.json"
                if meta.exists():
                    m = json.loads(meta.read_text(encoding="utf-8"))
                    print(f"final model: {m.get('n_stable_features')} stable features, {m.get('n_samples')} enhancers")

        if not a.dry_run and rd.exists():
            (rd / "rerun_settings.json").write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")

    if "lime" in steps and base_tasks:
        missing = [t for t in base_tasks if not (run_dirs[t] / "final_model" / MODEL / "model.joblib").exists()]
        if missing and not a.dry_run:
            sys.exit(f"LIME: no final model for {', '.join(missing)}; run the final step first.")
        cmd = [sys.executable, SCRIPTS / "03_lime_rules_final_models.py", "--tasks", *base_tasks, "--output-dir", lime_dir,
               "--n-jobs", a.lime_jobs, "--model-file",
               *[f"{t}={run_dirs[t] / 'final_model' / MODEL / 'model.joblib'}" for t in base_tasks], "--preprocessor-file",
               *[f"{t}={run_dirs[t] / 'final_model' / MODEL / 'preprocessor.joblib'}" for t in base_tasks]]
        print("\n" + "=" * 100 + f"\nLIME  ->  {lime_dir.name}\n" + "=" * 100)
        log["steps"]["lime"] = hms(run(cmd, a.dry_run))

    print("\n" + "=" * 100)
    print(f"{'dry run finished' if a.dry_run else 'finished'} | total {hms(time.time() - t_all)}")
    for k, v in log["steps"].items():
        print(f"  {k:18s} {v}")
    print("\nrun folders:")
    for t in tasks:
        print(f"  {run_dirs[t]}")
    if "lime" in steps:
        print(f"  {lime_dir}")
    print("\nWhen the three models are done, set these lines in [runs] of paths.toml:")
    for t in TASKS:
        print(f'  {t} = "{a.date}__{t}__thr_{THRESHOLD_METHOD}"' + ("" if t in tasks else "      # not run in this command"))
