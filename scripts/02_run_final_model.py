"""Final model of one task, refitted on all enhancers of its two classes, for interpretation (LIME).

Uses the features that were stable in the finished cross-validation run (positive held-out permutation
importance in at least 80 % of the held-out evaluations and a positive median; config.py). Performance is never taken from these
models: all reported metrics come from the held-out folds of scripts/01_run_cv.py.

The three models:

    uv run python scripts/02_run_final_model.py --task nr_vs_up --models catboost
    uv run python scripts/02_run_final_model.py --task up_vs_dn --models catboost
    uv run python scripts/02_run_final_model.py --task nr_vs_dn --models catboost

Output: <run folder>/final_model/ (model.joblib, preprocessor.joblib, feature lists, metadata).
"""
import argparse
from pathlib import Path

import pandas as pd

from a549_enhancers.config import (
    GLOBAL_RANDOM_STATE,
    VARIANCE_THRESHOLD,
)
from a549_enhancers import paths
from a549_enhancers.utils import make_output_dirs, safe_filename, save_json, set_global_seed
from a549_enhancers.data_loading import load_raw_inputs, build_base_feature_matrix
from a549_enhancers.labels import generate_multiclass_labels, make_binary_dataset
from a549_enhancers.feature_engineering import build_feature_matrix
from a549_enhancers.stability import (
    summarize_importance_stability,
    select_stable_features,
)
from a549_enhancers.final_model import fit_final_models


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        choices=["nr_vs_up", "up_vs_dn", "nr_vs_dn"],
        required=True,
    )



    parser.add_argument(
        "--cv-output-root",
        default=None,
        help="Path to the completed CV run folder (default: the folder of this task in [runs] of paths.toml).",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=["catboost", "random_forest"],
        default=["catboost", "random_forest"],
    )

    return parser.parse_args()


def main():
    args = parse_args()
    set_global_seed(GLOBAL_RANDOM_STATE)

    cv_root = Path(args.cv_output_root) if args.cv_output_root else paths.run_dir(args.task)
    importance_path = cv_root / "summaries" / "all_cv_permutation_importance.csv"

    if not importance_path.exists():
        raise FileNotFoundError(f"Missing CV importance file: {importance_path}")

    final_root = cv_root / "final_model"
    final_root.mkdir(parents=True, exist_ok=True)

    # Rebuild the same feature matrix deterministically.
    raw = load_raw_inputs()
    X_raw, prematrix_with_ids, original_motif_cols = build_base_feature_matrix(raw)
    labels = generate_multiclass_labels(prematrix_with_ids)

    common = X_raw.index.intersection(labels.index)
    X_raw = X_raw.loc[common].copy()
    labels = labels.loc[common].copy()

    X_features, feature_metadata = build_feature_matrix(
        X_raw=X_raw,
        original_motif_cols=original_motif_cols,
    )

    categorical_cols = feature_metadata["categorical_cols"]

    X_task, y_task, negative_label, positive_label = make_binary_dataset(
        X=X_features,
        labels=labels,
        task=args.task,
    )

    importance_df = pd.read_csv(importance_path)
    importance_summary = summarize_importance_stability(importance_df)

    stable_features_by_model = {}

    for model_name in args.models:
        stable_features = select_stable_features(
            importance_summary=importance_summary,
            task=args.task,
            model_name=model_name,
        )
        stable_features_by_model[model_name] = stable_features

        print(f"{model_name}: {len(stable_features)} stable features")

    fitted = fit_final_models(
        X=X_task,
        y=y_task,
        stable_features_by_model=stable_features_by_model,
        categorical_cols=categorical_cols,
        variance_threshold=VARIANCE_THRESHOLD,
        task=args.task,
        positive_label=positive_label,
        negative_label=negative_label,
        outdir=final_root,
        random_state=GLOBAL_RANDOM_STATE,
    )

    metadata = {
        "task": args.task,
        "cv_output_root": str(cv_root),
        "positive_label": positive_label,
        "negative_label": negative_label,
        "stable_features_by_model": stable_features_by_model,
        "note": (
            "Final models are for interpretation only. "
            "Performance estimates come from held-out repeated CV."
        ),
    }

    save_json(metadata, final_root / "final_training_metadata.json")

    print(f"Final models saved to: {final_root}")


if __name__ == "__main__":
    main()
