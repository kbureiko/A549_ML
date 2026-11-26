"""Repeated stratified cross-validation of one pairwise enhancer classifier.

Tasks: nr_vs_up (positive class UP), up_vs_dn (positive class UP), nr_vs_dn (positive class DN).

Everything that is learned from data is learned inside the outer training fold: the variance filter, the
categorical levels, the model and - with --tune-threshold - the decision threshold (from inner
cross-validation of that training fold). The outer held-out fold is used for evaluation and for the held-out
permutation importance only.

The three models:

    uv run python scripts/01_run_cv.py --task nr_vs_up --models catboost \
        --tune-threshold --threshold-method max_f1 --run-prefix <date>
    uv run python scripts/01_run_cv.py --task up_vs_dn --models catboost \
        --tune-threshold --threshold-method max_f1 --run-prefix <date>
    uv run python scripts/01_run_cv.py --task nr_vs_dn --models catboost \
        --tune-threshold --threshold-method max_f1 --run-prefix <date>

Options added on 20 Nov 2026:

    --subset gr_bound            only GR-bound enhancers (overlap with a peak of config.GR_PEAK_FILE); with
                                 --task nr_vs_up this is GR-bound UP against GR-bound NR
    --subset transcribed         only enhancers of the two classes with enough PRO-seq signal for a response to be
                                 detectable: signal (config.PROSEQ_SIGNAL_COLUMN of the PRO-seq table) at least
                                 the config.TRANSCRIBED_MIN_QUANTILE_OF_POSITIVE quantile of the positive class.
                                 With --task nr_vs_dn this compares DN enhancers with NR enhancers that are
                                 transcribed in the same range
    --subset matched             every enhancer of the positive class with its config.MATCHED_NEGATIVES_PER_POSITIVE
                                 nearest enhancers of the negative class in length and PRO-seq signal, so the two
                                 classes have similar length and transcription level (stricter than 'transcribed',
                                 which only removes weakly transcribed enhancers)
    --feature-set pretreatment   without the features measured after dex (dex ChIP-seq and ATAC-seq tracks, GR
                                 peaks around the enhancer); sequence and control features only
    --no-grouped-importance      skip the grouped permutation importance

The options add a tag to the run folder name (..__grbound, ..__transcribed, ..__matched, ..__pretreatment). The grouped permutation importance
(groups of config.FEATURE_FAMILIES and correlation clusters, all members shuffled together) is computed on every
held-out fold next to the single-feature importance.

Note that the default of --models also fits a random forest, and that the threshold is tuned only with
--tune-threshold.

Input: the six tables in [inputs] ml_inputs (paths.toml). Output: one run folder in [results] ml_outputs
(data/, cv_predictions/, cv_metrics/, cv_importance/, cv_models/, thresholding/, summaries/).
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from a549_enhancers.config import (
    GLOBAL_RANDOM_STATE,
    N_SPLITS,
    N_REPEATS,
    VARIANCE_THRESHOLD,
    PERMUTATION_N_REPEATS,
    PERMUTATION_SCORING,
    GROUPED_PERMUTATION_N_REPEATS,
    GR_PEAK_FILE,
    PROSEQ_SIGNAL_COLUMN,
    TRANSCRIBED_MIN_QUANTILE_OF_POSITIVE,
    MATCHED_NEGATIVES_PER_POSITIVE,
    THRESHOLD_INNER_CV_SPLITS,
    THRESHOLD_METHOD_DEFAULT,
    MIN_PRECISION_DEFAULT,
)

from a549_enhancers import paths
from a549_enhancers.utils import (
    set_global_seed,
    make_output_dirs,
    safe_filename,
    save_json,
)

from a549_enhancers.data_loading import (
    load_raw_inputs,
    build_base_feature_matrix,
    save_base_matrix,
)

from a549_enhancers.labels import (
    generate_multiclass_labels,
    make_binary_dataset,
    summarize_labels,
)

from a549_enhancers.feature_engineering import build_feature_matrix
from a549_enhancers.preprocessing import FoldPreprocessor

from a549_enhancers.models import (
    fit_random_forest,
    fit_catboost,
    get_positive_class_proba,
)

from a549_enhancers.evaluation import (
    evaluate_binary_predictions,
    make_prediction_table,
)

from a549_enhancers.permutation_importance import (
    compute_permutation_importance_heldout,
    compute_grouped_permutation_importance_heldout,
)
from a549_enhancers.feature_groups import build_feature_groups, feature_layer
from a549_enhancers.enhancers import proseq_signal
from a549_enhancers.intervals import overlaps_any, parse_enhancer_ids, read_peaks

from a549_enhancers.stability import (
    summarize_cv_metrics,
    summarize_importance_stability,
)

from a549_enhancers.thresholding import tune_threshold_on_outer_train


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        choices=["nr_vs_up", "up_vs_dn", "nr_vs_dn"],
        required=True,
        help="Binary classification task.",
    )



    parser.add_argument(
        "--subset",
        choices=["all", "gr_bound", "transcribed", "matched"],
        default="all",
        help="Enhancers of the two classes to use: all, only the GR-bound ones, or only those with enough "
             "PRO-seq signal for a response to be detectable.",
    )

    parser.add_argument(
        "--feature-set",
        choices=["all", "pretreatment"],
        default="all",
        help="all features, or without the features measured after dex.",
    )

    parser.add_argument(
        "--no-grouped-importance",
        action="store_true",
        help="Do not compute the grouped permutation importance.",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=["catboost", "random_forest"],
        default=["catboost", "random_forest"],
        help="Models to train.",
    )

    parser.add_argument(
        "--output-root",
        default=None,
        help="Root output directory (default: [results] ml_outputs in paths.toml).",
    )

    parser.add_argument(
        "--run-prefix",
        default="",
        help=(
            "Optional prefix added to the run output folder name. "
            "Example: --run-prefix <date> creates "
            "20260917__nr_vs_dn__thr_max_f1."
        ),
    )

    # =========================
    # Decision-threshold tuning
    # =========================

    parser.add_argument(
        "--tune-threshold",
        action="store_true",
        help=(
            "Tune the positive-class decision threshold inside each outer "
            "training fold using inner CV. Outer held-out folds are not used "
            "to select thresholds."
        ),
    )

    parser.add_argument(
        "--threshold-method",
        choices=["max_f1", "max_f2", "recall_at_min_precision"],
        default=THRESHOLD_METHOD_DEFAULT,
        help="Method used to choose the positive-class decision threshold.",
    )

    parser.add_argument(
        "--threshold-inner-splits",
        type=int,
        default=THRESHOLD_INNER_CV_SPLITS,
        help="Number of inner CV folds for threshold tuning.",
    )

    parser.add_argument(
        "--min-precision",
        type=float,
        default=MIN_PRECISION_DEFAULT,
        help=(
            "Minimum precision used only when "
            "--threshold-method recall_at_min_precision."
        ),
    )

    return parser.parse_args()


def make_threshold_tag(args) -> str:
    """
    Create output-folder tag for threshold strategy.

    This prevents default-threshold and tuned-threshold runs from
    overwriting each other.
    """
    if not args.tune_threshold:
        return "thr_default"

    if args.threshold_method == "recall_at_min_precision":
        min_prec = str(args.min_precision).replace(".", "p")
        return f"thr_recall_at_min_precision_{min_prec}"

    return f"thr_{args.threshold_method}"


def save_dex_filtering_audit(X_raw: pd.DataFrame, data_dir: Path) -> None:
    """
    Save audit files for dex-condition filtering if these attributes exist.

    These attributes are created in data_loading.py if you implemented
    dex-column filtering and always-keep exception logic.
    """
    kept_dex = X_raw.attrs.get("kept_dex_columns_100nm_1h", [])
    dropped_dex = X_raw.attrs.get("dropped_dex_columns_not_100nm_1h", [])
    forced_kept = X_raw.attrs.get("forced_kept_feature_columns", [])

    pd.DataFrame({"kept_dex_columns_100nm_1h": kept_dex}).to_csv(
        data_dir / "kept_dex_columns_100nm_1h.csv",
        index=False,
    )

    pd.DataFrame({"dropped_dex_columns_not_100nm_1h": dropped_dex}).to_csv(
        data_dir / "dropped_dex_columns_not_100nm_1h.csv",
        index=False,
    )

    pd.DataFrame({"forced_kept_feature_columns": forced_kept}).to_csv(
        data_dir / "forced_kept_feature_columns.csv",
        index=False,
    )


def main():
    args = parse_args()
    set_global_seed(GLOBAL_RANDOM_STATE)

    threshold_tag = make_threshold_tag(args)

    base_run_name = (
        f"{args.task}"
        f"__{threshold_tag}"
        + ("__grbound" if args.subset == "gr_bound" else "")
        + ("__transcribed" if args.subset == "transcribed" else "")
        + ("__matched" if args.subset == "matched" else "")
        + ("__pretreatment" if args.feature_set == "pretreatment" else "")
    )

    run_prefix = safe_filename(str(args.run_prefix).strip()) if args.run_prefix else ""
    run_name = (
        f"{run_prefix}__{base_run_name}"
        if run_prefix
        else base_run_name
    )

    out_root = Path(args.output_root or paths.ML) / safe_filename(run_name)
    dirs = make_output_dirs(out_root)

    # Robust fallback in case utils.py has not yet been updated
    # to include "thresholding" in make_output_dirs().
    threshold_dir = out_root / "thresholding"
    threshold_dir.mkdir(parents=True, exist_ok=True)
    dirs["thresholding"] = threshold_dir

    print("=" * 80)
    print(f"Running: {run_name}")
    print(f"Output: {out_root}")
    print(f"Task: {args.task}")
    print(f"Enhancers: {args.subset} | features: {args.feature_set}")
    print(f"Models: {args.models}")
    print(f"Threshold tuning: {args.tune_threshold}")
    print(f"Run prefix: {run_prefix if run_prefix else '(none)'}")

    if args.tune_threshold:
        print(f"Threshold method: {args.threshold_method}")
        print(f"Threshold inner splits: {args.threshold_inner_splits}")
        print(f"Minimum precision: {args.min_precision}")

    print("=" * 80)

    # =========================
    # 1. Load and build base matrix
    # =========================

    raw = load_raw_inputs()

    X_raw, prematrix_with_ids, original_motif_cols = build_base_feature_matrix(raw)

    save_base_matrix(
        X_raw,
        dirs["data"] / "matrix_base_before_feature_engineering.csv",
    )

    save_dex_filtering_audit(X_raw, dirs["data"])

    # =========================
    # 2. Generate UP / DN / NR labels
    # =========================

    labels = generate_multiclass_labels(prematrix_with_ids)
    labels.to_csv(dirs["data"] / "labels_multiclass_UP_DN_NR.csv")

    label_summary = summarize_labels(labels)
    label_summary.to_csv(
        dirs["data"] / "label_summary_multiclass.csv",
        index=False,
    )

    print("\nMulticlass label summary:")
    print(label_summary)

    # Align X and labels.
    common = X_raw.index.intersection(labels.index)

    if len(common) == 0:
        raise ValueError("No overlapping enhancer IDs between X_raw and labels.")

    X_raw = X_raw.loc[common].copy()
    labels = labels.loc[common].copy()

    if not X_raw.index.equals(labels.index):
        raise ValueError("X_raw and labels are not aligned after intersection.")

    # =========================
    # 3. Feature engineering
    # =========================

    X_features, feature_metadata = build_feature_matrix(
        X_raw=X_raw,
        original_motif_cols=original_motif_cols,
    )

    X_features.to_csv(dirs["data"] / "feature_matrix_before_cv.csv")

    save_json(
        feature_metadata,
        dirs["data"] / "feature_engineering_metadata.json",
    )

    categorical_cols = feature_metadata.get("categorical_cols", [])

    print("\nFeature matrix before CV:")
    print(X_features.shape)
    print(f"Categorical columns: {categorical_cols}")

    # =========================
    # 4. Build binary task
    # =========================

    X_task, y_task, negative_label, positive_label = make_binary_dataset(
        X=X_features,
        labels=labels,
        task=args.task,
    )

    # Optional: GR-bound enhancers only (overlap with a GR peak, as in scripts/08_annotate_gr_binding.py).
    if args.subset == "gr_bound":
        gr_bound = overlaps_any(parse_enhancer_ids(X_task.index), read_peaks(paths.ml_input(GR_PEAK_FILE)))
        print(f"\nSubset gr_bound: {int(gr_bound.sum())} of {len(gr_bound)} enhancers overlap a peak of {GR_PEAK_FILE}")
        print(pd.crosstab(y_task.to_numpy(), gr_bound, rownames=["class"], colnames=["GR-bound"]))
        X_task, y_task = X_task.loc[gr_bound].copy(), y_task.loc[gr_bound].copy()

    # Optional: enhancers transcribed in the range of the positive class (same rule for both classes).
    if args.subset == "transcribed":
        signal = proseq_signal(X_task.index, paths.PROSEQ_MA_TABLE, PROSEQ_SIGNAL_COLUMN)
        if signal.isna().any():
            print(f"{int(signal.isna().sum())} enhancers without PRO-seq signal are left out")
        is_pos = (y_task == positive_label).to_numpy()
        cutoff = float(signal[is_pos].quantile(TRANSCRIBED_MIN_QUANTILE_OF_POSITIVE))
        keep = (signal >= cutoff).to_numpy()
        print(f"\nSubset transcribed: {PROSEQ_SIGNAL_COLUMN} >= {cutoff:.3f} "
              f"({TRANSCRIBED_MIN_QUANTILE_OF_POSITIVE:.0%} quantile of {positive_label}); "
              f"{int(keep.sum())} of {len(keep)} enhancers kept")
        print(pd.crosstab(y_task.to_numpy(), keep, rownames=["class"], colnames=["kept"]))
        pd.DataFrame({"enhancer_id": X_task.index, "class": y_task.to_numpy(), PROSEQ_SIGNAL_COLUMN: signal.to_numpy(),
                      "kept": keep, "cutoff": cutoff}).to_csv(dirs["data"] / "transcribed_subset.csv", index=False)
        X_task, y_task = X_task.loc[keep].copy(), y_task.loc[keep].copy()

    # Optional: positives with their nearest negatives in length and PRO-seq signal.
    if args.subset == "matched":
        from sklearn.neighbors import NearestNeighbors
        signal = proseq_signal(X_task.index, paths.PROSEQ_MA_TABLE, PROSEQ_SIGNAL_COLUMN)
        cov = pd.DataFrame({"log10_length": np.log10(pd.to_numeric(X_task["Enh_len"]).to_numpy()),
                            PROSEQ_SIGNAL_COLUMN: signal.to_numpy()}, index=X_task.index)
        usable = cov.notna().all(axis=1).to_numpy()
        Z = ((cov - cov[usable].mean()) / cov[usable].std(ddof=0)).to_numpy()
        is_pos = (y_task == positive_label).to_numpy()
        neg_rows = np.flatnonzero(~is_pos & usable)
        pos_rows = np.flatnonzero(is_pos & usable)
        k = min(MATCHED_NEGATIVES_PER_POSITIVE, len(neg_rows))
        _, idx = NearestNeighbors(n_neighbors=k).fit(Z[neg_rows]).kneighbors(Z[pos_rows])
        keep = np.zeros(len(X_task), dtype=bool)
        keep[pos_rows] = True
        keep[neg_rows[np.unique(idx)]] = True
        print(f"\nSubset matched: {len(pos_rows)} {positive_label} enhancers and their {k} nearest {negative_label} "
              f"enhancers in length and {PROSEQ_SIGNAL_COLUMN}: {int(keep.sum())} enhancers kept")
        balance = cov.assign(group=np.where(is_pos, positive_label, np.where(keep, f"matched {negative_label}",
                                                                              f"other {negative_label}")))
        balance["length"] = 10 ** balance.pop("log10_length")
        print(balance.groupby("group").agg(n=("length", "size"), length_median=("length", "median"),
                                           signal_median=(PROSEQ_SIGNAL_COLUMN, "median")).round(2))
        pd.DataFrame({"enhancer_id": X_task.index, "class": y_task.to_numpy(), "length": balance["length"].to_numpy(),
                      PROSEQ_SIGNAL_COLUMN: signal.to_numpy(), "kept": keep}).to_csv(
            dirs["data"] / "matched_subset.csv", index=False)
        X_task, y_task = X_task.loc[keep].copy(), y_task.loc[keep].copy()

    # Optional: without the features measured after dex.
    if args.feature_set == "pretreatment":
        dex_cols = [c for c in X_task.columns if feature_layer(c) == "dex"]
        X_task = X_task.drop(columns=dex_cols)
        X_features = X_features.drop(columns=dex_cols)
        categorical_cols = [c for c in categorical_cols if c in X_task.columns]
        pd.DataFrame({"dropped_dex_feature": dex_cols}).to_csv(dirs["data"] / "dropped_dex_features.csv", index=False)
        print(f"\nFeature set pretreatment: {len(dex_cols)} dex features dropped, {X_task.shape[1]} features left")

    # Predefined feature groups for the grouped permutation importance: from all enhancers, without labels.
    feature_groups = {}
    if not args.no_grouped_importance:
        groups_table = build_feature_groups(X_features[X_task.columns])
        groups_table.to_csv(dirs["data"] / "feature_groups.csv", index=False)
        feature_groups = {g: t["feature"].tolist() for g, t in groups_table.groupby("group", sort=False)}
        print(f"\nFeature groups: {len(feature_groups)} "
              f"({groups_table.drop_duplicates('group').group_type.value_counts().to_dict()})")

    X_task.to_csv(dirs["data"] / f"X_{args.task}.csv")
    y_task.to_csv(dirs["data"] / f"y_{args.task}.csv")

    # Save the final assembled task matrix used as input to repeated CV.
    #
    # Important: this is intentionally saved BEFORE fold-specific preprocessing.
    # Variance filtering, categorical encoding, and any other learned
    # preprocessing are fitted separately inside each outer training fold to
    # preserve leakage control. Therefore, there is no single globally
    # preprocessed matrix that should be used for CV evaluation.
    final_assembled_matrix = X_task.copy()

    multiclass_for_task = labels.loc[X_task.index].copy()
    final_assembled_matrix.insert(
        0,
        "multiclass_label",
        multiclass_for_task,
    )
    final_assembled_matrix.insert(
        1,
        "binary_target",
        y_task,
    )

    final_matrix_path = dirs["data"] / "final_assembled_matrix.csv"
    final_assembled_matrix.to_csv(
        final_matrix_path,
        index=True,
        index_label="enhancer_id",
    )

    final_matrix_metadata = {
        "task": args.task,
        "negative_label": negative_label,
        "positive_label": positive_label,
        "n_rows": int(final_assembled_matrix.shape[0]),
        "n_feature_columns": int(X_task.shape[1]),
        "n_total_columns_in_saved_matrix": int(final_assembled_matrix.shape[1]),
        "matrix_stage": (
            "Final assembled binary-task matrix after feature engineering and "
            "task filtering, before fold-specific preprocessing."
        ),
        "leakage_control_note": (
            "FoldPreprocessor is fitted separately on each outer training fold. "
            "The saved final_assembled_matrix.csv is therefore the correct "
            "single run-level matrix to save without introducing global "
            "preprocessing leakage."
        ),
    }
    save_json(
        final_matrix_metadata,
        dirs["data"] / "final_assembled_matrix_metadata.json",
    )

    print("\nSaved final assembled matrix:")
    print(f"  {final_matrix_path}")
    print(f"  shape={final_assembled_matrix.shape}")

    task_summary = summarize_labels(y_task)
    task_summary.to_csv(
        dirs["data"] / f"label_summary_{args.task}.csv",
        index=False,
    )

    print("\nBinary task label summary:")
    print(task_summary)
    print(f"Negative label: {negative_label}")
    print(f"Positive label: {positive_label}")

    if y_task.nunique() != 2:
        raise ValueError(
            f"Binary task {args.task} does not contain exactly 2 classes. "
            f"Observed classes: {y_task.unique()}"
        )

    # =========================
    # 5. Repeated stratified CV
    # =========================

    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=GLOBAL_RANDOM_STATE,
    )

    all_predictions = []
    all_metrics = []
    all_importances = []
    all_group_importances = []
    all_threshold_tables = []

    for split_id, (train_idx, test_idx) in enumerate(cv.split(X_task, y_task)):
        repeat = split_id // N_SPLITS + 1
        fold = split_id % N_SPLITS + 1
        fold_seed = GLOBAL_RANDOM_STATE + split_id

        print("\n" + "-" * 80)
        print(f"Repeat {repeat}/{N_REPEATS}, fold {fold}/{N_SPLITS}")
        print("-" * 80)

        X_train = X_task.iloc[train_idx].copy()
        X_test = X_task.iloc[test_idx].copy()

        y_train = y_task.iloc[train_idx].copy()
        y_test = y_task.iloc[test_idx].copy()

        print("Training label counts:")
        print(y_train.value_counts())
        print("Held-out label counts:")
        print(y_test.value_counts())

        # =========================
        # Leakage-controlled preprocessing
        # =========================
        # The preprocessor is fitted only on the outer training fold.
        # The held-out fold is transformed using train-derived preprocessing.

        preprocessor = FoldPreprocessor(
            categorical_cols=categorical_cols,
            variance_threshold=VARIANCE_THRESHOLD,
        )

        X_train_cb, cb_cat_features, X_train_rf = preprocessor.fit_transform(
            X_train
        )

        X_test_cb, _ = preprocessor.transform_for_catboost(X_test)
        X_test_rf = preprocessor.transform_for_rf(X_test)

        fold_info = {
            "task": args.task,
            "repeat": repeat,
            "fold": fold,
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "n_features_before_preprocessing": int(X_task.shape[1]),
            "n_numeric_retained_after_variance": int(
                len(preprocessor.retained_numeric_cols_)
            ),
            "n_catboost_features": int(X_train_cb.shape[1]),
            "n_rf_features": int(X_train_rf.shape[1]),
        }

        joblib.dump(
            preprocessor,
            dirs["cv_models"]
            / f"repeat_{repeat:02d}_fold_{fold:02d}_preprocessor.joblib",
        )

        # =========================
        # 6. Train and evaluate models
        # =========================

        for model_name in args.models:
            print(f"\n  Model: {model_name}")

            if model_name == "catboost":
                X_tr = X_train_cb
                X_te = X_test_cb

                model = fit_catboost(
                    X_train=X_tr,
                    y_train=y_train,
                    cat_features=cb_cat_features,
                    random_state=fold_seed,
                )

            elif model_name == "random_forest":
                X_tr = X_train_rf
                X_te = X_test_rf

                model = fit_random_forest(
                    X_train=X_tr,
                    y_train=y_train,
                    random_state=fold_seed,
                )

            else:
                raise ValueError(f"Unsupported model: {model_name}")

            model_path = (
                dirs["cv_models"]
                / f"repeat_{repeat:02d}_fold_{fold:02d}_{model_name}.joblib"
            )
            joblib.dump(model, model_path)

            # =========================
            # Outer held-out probabilities
            # =========================
            # These probabilities are allowed for evaluation.
            # If threshold tuning is enabled, the threshold is selected
            # only from the outer training fold, not from these held-out data.

            y_pos_proba = get_positive_class_proba(
                model=model,
                X=X_te,
                positive_label=positive_label,
            )

            # =========================
            # Threshold tuning
            # =========================

            if args.tune_threshold:
                tuned_threshold, threshold_oof = tune_threshold_on_outer_train(
                    X_outer_train_raw=X_train,
                    y_outer_train=y_train,
                    model_name=model_name,
                    categorical_cols=categorical_cols,
                    variance_threshold=VARIANCE_THRESHOLD,
                    positive_label=positive_label,
                    random_state=fold_seed,
                    n_splits=args.threshold_inner_splits,
                    threshold_method=args.threshold_method,
                    min_precision=args.min_precision,
                )

                threshold_oof["task"] = args.task
                threshold_oof["model"] = model_name
                threshold_oof["repeat"] = repeat
                threshold_oof["fold"] = fold
                threshold_oof["negative_label"] = negative_label
                threshold_oof["positive_label"] = positive_label

                threshold_path = (
                    dirs["thresholding"]
                    / f"repeat_{repeat:02d}_fold_{fold:02d}_{model_name}_threshold_oof_predictions.csv"
                )

                threshold_oof.to_csv(threshold_path, index=False)
                all_threshold_tables.append(threshold_oof)

            else:
                tuned_threshold = 0.5

            print(f"  Decision threshold: {tuned_threshold:.4f}")

            # Convert probabilities to class labels using the selected threshold.
            y_pred = np.where(
                y_pos_proba >= tuned_threshold,
                positive_label,
                negative_label,
            )

            # =========================
            # Metrics
            # =========================

            metrics = evaluate_binary_predictions(
                y_true=y_test,
                y_pred=y_pred,
                y_pos_proba=y_pos_proba,
                positive_label=positive_label,
                negative_label=negative_label,
            )

            metrics.update(fold_info)

            metrics["model"] = model_name
            metrics["positive_label"] = positive_label
            metrics["negative_label"] = negative_label

            metrics["decision_threshold"] = float(tuned_threshold)
            metrics["threshold_tuning"] = bool(args.tune_threshold)
            metrics["threshold_method"] = (
                args.threshold_method if args.tune_threshold else "default_0.5"
            )
            metrics["threshold_inner_splits"] = (
                int(args.threshold_inner_splits) if args.tune_threshold else 0
            )
            metrics["min_precision"] = (
                float(args.min_precision) if args.tune_threshold else np.nan
            )

            metrics_df = pd.DataFrame([metrics])

            metrics_path = (
                dirs["cv_metrics"]
                / f"repeat_{repeat:02d}_fold_{fold:02d}_{model_name}_metrics.csv"
            )
            metrics_df.to_csv(metrics_path, index=False)

            all_metrics.append(metrics_df)

            print(
                f"  balanced_accuracy={metrics['balanced_accuracy']:.3f} | "
                f"positive_recall={metrics['positive_recall']:.3f} | "
                f"positive_precision={metrics['positive_precision']:.3f} | "
                f"positive_f1={metrics['positive_f1']:.3f} | "
                f"average_precision={metrics['average_precision']:.3f} | "
                f"TP={metrics['tp']} FP={metrics['fp']} "
                f"FN={metrics['fn']} TN={metrics['tn']}"
            )

            # =========================
            # Prediction table
            # =========================

            pred_df = make_prediction_table(
                enhancer_ids=X_test.index,
                y_true=y_test,
                y_pred=y_pred,
                y_pos_proba=y_pos_proba,
                task=args.task,
                model_name=model_name,
                repeat=repeat,
                fold=fold,
                positive_label=positive_label,
            )

            pred_df["negative_label"] = negative_label
            pred_df["positive_label"] = positive_label
            pred_df["decision_threshold"] = float(tuned_threshold)
            pred_df["threshold_tuning"] = bool(args.tune_threshold)
            pred_df["threshold_method"] = (
                args.threshold_method if args.tune_threshold else "default_0.5"
            )

            pred_path = (
                dirs["cv_predictions"]
                / f"repeat_{repeat:02d}_fold_{fold:02d}_{model_name}_predictions.csv"
            )
            pred_df.to_csv(pred_path, index=False)

            all_predictions.append(pred_df)

            # =========================
            # Held-out permutation importance
            # =========================
            # Computed only on outer held-out fold.
            # The scorer uses positive-class probabilities, so threshold tuning
            # does not affect permutation importance.

            imp = compute_permutation_importance_heldout(
                model=model,
                X_test=X_te,
                y_test=y_test,
                positive_label=positive_label,
                scoring=PERMUTATION_SCORING,
                n_repeats=PERMUTATION_N_REPEATS,
                random_state=fold_seed,
            )

            imp["task"] = args.task
            imp["model"] = model_name
            imp["repeat"] = repeat
            imp["fold"] = fold
            imp["positive_label"] = positive_label
            imp["negative_label"] = negative_label
            imp["scoring"] = PERMUTATION_SCORING
            imp["decision_threshold"] = float(tuned_threshold)
            imp["threshold_tuning"] = bool(args.tune_threshold)
            imp["threshold_method"] = (
                args.threshold_method if args.tune_threshold else "default_0.5"
            )

            imp_path = (
                dirs["cv_importance"]
                / f"repeat_{repeat:02d}_fold_{fold:02d}_{model_name}_permutation_importance.csv"
            )
            imp.to_csv(imp_path, index=False)

            all_importances.append(imp)

            # =========================
            # Grouped permutation importance (held-out fold)
            # =========================
            if feature_groups:
                gimp = compute_grouped_permutation_importance_heldout(
                    model=model,
                    X_test=X_te,
                    y_test=y_test,
                    positive_label=positive_label,
                    groups=feature_groups,
                    n_repeats=GROUPED_PERMUTATION_N_REPEATS,
                    random_state=fold_seed,
                )
                gimp["task"] = args.task
                gimp["model"] = model_name
                gimp["repeat"] = repeat
                gimp["fold"] = fold
                gimp.to_csv(
                    dirs["cv_importance"]
                    / f"repeat_{repeat:02d}_fold_{fold:02d}_{model_name}_grouped_permutation_importance.csv",
                    index=False,
                )
                all_group_importances.append(gimp)

    # =========================
    # 7. Aggregate CV outputs
    # =========================

    all_predictions_df = pd.concat(all_predictions, ignore_index=True)
    all_metrics_df = pd.concat(all_metrics, ignore_index=True)
    all_importances_df = pd.concat(all_importances, ignore_index=True)

    all_predictions_df.to_csv(
        dirs["summaries"] / "all_cv_predictions.csv",
        index=False,
    )

    all_metrics_df.to_csv(
        dirs["summaries"] / "all_cv_metrics.csv",
        index=False,
    )

    all_importances_df.to_csv(
        dirs["summaries"] / "all_cv_permutation_importance.csv",
        index=False,
    )

    if args.tune_threshold and len(all_threshold_tables) > 0:
        all_threshold_oof_df = pd.concat(all_threshold_tables, ignore_index=True)
        all_threshold_oof_df.to_csv(
            dirs["summaries"] / "all_threshold_oof_predictions.csv",
            index=False,
        )

    # =========================
    # 8. General summaries
    # =========================

    metric_summary = summarize_cv_metrics(all_metrics_df)
    metric_summary.to_csv(
        dirs["summaries"] / "cv_metric_summary.csv",
        index=False,
    )

    importance_summary = summarize_importance_stability(all_importances_df)
    importance_summary.to_csv(
        dirs["summaries"] / "feature_importance_stability_summary.csv",
        index=False,
    )

    # Grouped importance: per group, next to the single-feature importance of its members.
    if all_group_importances:
        all_group_df = pd.concat(all_group_importances, ignore_index=True)
        all_group_df.to_csv(dirs["summaries"] / "all_cv_grouped_permutation_importance.csv", index=False)
        group_summary = (
            all_group_df.groupby(["task", "model", "group"], sort=False)
            .agg(
                n_features=("n_features", "max"),
                n_folds=("importance_mean", "count"),
                mean_importance=("importance_mean", "mean"),
                median_importance=("importance_mean", "median"),
                sd_importance=("importance_mean", "std"),
                fraction_positive_importance=("importance_mean", lambda x: np.mean(np.asarray(x) > 0)),
            )
            .reset_index()
        )
        single = importance_summary.set_index(["model", "feature"])["mean_importance"]
        members = groups_table.groupby("group")["feature"].apply(list)
        best, total = [], []
        for _, r in group_summary.iterrows():
            vals = single.reindex([(r["model"], f) for f in members[r["group"]]]).dropna()
            best.append(vals.max() if len(vals) else np.nan)
            total.append(vals.clip(lower=0).sum() if len(vals) else np.nan)
        group_summary["largest_single_member_importance"] = best
        group_summary["sum_of_positive_member_importances"] = total
        group_summary["group_type"] = group_summary["group"].map(
            groups_table.drop_duplicates("group").set_index("group")["group_type"])
        group_summary = group_summary.sort_values(["model", "mean_importance"], ascending=[True, False])
        group_summary.to_csv(dirs["summaries"] / "grouped_importance_summary.csv", index=False)

    # =========================
    # 9. Threshold/performance summary
    # =========================
    # Only aggregate numeric columns.
    # Do not include string columns such as "model" or "threshold_method"
    # in the value columns passed to .agg(["mean", ...]).

    threshold_numeric_cols = [
        "decision_threshold",
        "positive_recall",
        "positive_precision",
        "positive_f1",
        "macro_f1",
        "average_precision",
        "roc_auc",
        "balanced_accuracy",
        "tp",
        "fp",
        "fn",
        "tn",
    ]

    existing_threshold_numeric_cols = [
        c for c in threshold_numeric_cols
        if c in all_metrics_df.columns
    ]

    for col in existing_threshold_numeric_cols:
        all_metrics_df[col] = pd.to_numeric(all_metrics_df[col], errors="coerce")

    threshold_summary = (
        all_metrics_df
        .groupby(["task", "model", "threshold_method"])[existing_threshold_numeric_cols]
        .agg(["mean", "std", "median", "min", "max"])
    )

    threshold_summary.to_csv(
        dirs["summaries"] / "threshold_metric_summary.csv"
    )

    # Also save a cleaner long-format mean summary.
    threshold_summary_long = (
        all_metrics_df
        .groupby(["task", "model", "threshold_method"], as_index=False)
        [existing_threshold_numeric_cols]
        .mean()
    )

    threshold_summary_long.to_csv(
        dirs["summaries"] / "threshold_metric_summary_mean_only.csv",
        index=False,
    )

    # =========================
    # 10. Run metadata
    # =========================

    run_metadata = {
        "run_prefix": run_prefix,
        "run_name": run_name,
        "base_run_name": base_run_name,
        "task": args.task,
        "subset": args.subset,
        "feature_set": args.feature_set,
        "n_feature_groups": len(feature_groups),
        "models": args.models,
        "negative_label": negative_label,
        "positive_label": positive_label,
        "n_splits": N_SPLITS,
        "n_repeats": N_REPEATS,
        "variance_threshold": VARIANCE_THRESHOLD,
        "permutation_scoring": PERMUTATION_SCORING,
        "permutation_n_repeats": PERMUTATION_N_REPEATS,
        "threshold_tuning": bool(args.tune_threshold),
        "threshold_method": (
            args.threshold_method if args.tune_threshold else "default_0.5"
        ),
        "threshold_inner_splits": (
            int(args.threshold_inner_splits) if args.tune_threshold else 0
        ),
        "min_precision": (
            float(args.min_precision) if args.tune_threshold else None
        ),
        "final_assembled_matrix": str(
            dirs["data"] / "final_assembled_matrix.csv"
        ),
        "final_assembled_matrix_shape": [
            int(final_assembled_matrix.shape[0]),
            int(final_assembled_matrix.shape[1]),
        ],
        "leakage_control_note": (
            "All preprocessing steps learned from data, including categorical "
            "encoding and variance filtering, are fitted only on training folds. "
            "Outer held-out folds are used only for final evaluation and held-out "
            "permutation importance."
        ),
        "threshold_note": (
            "When threshold tuning is enabled, the decision threshold is selected "
            "inside each outer training fold using inner CV predictions. The outer "
            "held-out fold is not used to select the threshold."
        ),
        "permutation_importance_note": (
            "Permutation importance is computed only on the outer held-out fold "
            "using positive-class probability scoring."
        ),
        "final_model_note": (
            "Final models should be trained only after CV evaluation is complete, "
            "using scripts/02_run_final_model.py. CV metrics in this folder are the performance "
            "estimates."
        ),
    }

    save_json(
        run_metadata,
        dirs["summaries"] / "run_metadata.json",
    )

    print("\n" + "=" * 80)
    print("Done.")
    print(f"Results saved to: {out_root}")
    print("=" * 80)


if __name__ == "__main__":
    main()
