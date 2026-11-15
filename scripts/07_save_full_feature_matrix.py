"""Feature matrix of all enhancers (UP, DN and NR together) with the class label.

Same input tables, feature filtering and feature engineering as scripts/01_run_cv.py, but without the
restriction to two classes. The figures and the feature map read this table after
scripts/08_annotate_gr_binding.py has added the GR-binding column.

    uv run python scripts/07_save_full_feature_matrix.py

Output: <ml_outputs>/full_feature_matrix/feature_matrix_all_classes.csv
"""
import argparse
from pathlib import Path

import pandas as pd

from a549_enhancers import paths
from a549_enhancers.utils import safe_filename
from a549_enhancers.data_loading import load_raw_inputs, build_base_feature_matrix
from a549_enhancers.labels import generate_multiclass_labels
from a549_enhancers.feature_engineering import build_feature_matrix


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build the full enhancer feature matrix for all NR/UP/DN classes "
            "without running binary-task filtering or cross-validation."
        )
    )



    parser.add_argument(
        "--output-root",
        default=None,
        help="Root output directory (default: [results] ml_outputs in paths.toml).",
    )

    parser.add_argument(
        "--run-prefix",
        default=None,
        help=(
            "Optional prefix for the output folder, e.g. 20260917. "
            "If omitted, no prefix is added."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    run_name = "full_feature_matrix"

    if args.run_prefix:
        run_name = f"{args.run_prefix}__{run_name}"

    out_dir = Path(args.output_root or paths.ML) / safe_filename(run_name)
    out_dir.mkdir(parents=True, exist_ok=True)


    print("=" * 80)
    print("Building full feature matrix for NR / UP / DN enhancers")
    print(f"Output: {out_dir}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Load raw inputs and assemble the base enhancer-by-feature matrix
    # ------------------------------------------------------------------
    raw = load_raw_inputs()

    X_raw, prematrix_with_ids, original_motif_cols = build_base_feature_matrix(raw)

    # ------------------------------------------------------------------
    # 2. Generate multiclass enhancer labels
    # ------------------------------------------------------------------
    labels = generate_multiclass_labels(prematrix_with_ids)

    # Align raw feature matrix and labels before feature engineering.
    common = X_raw.index.intersection(labels.index)

    if len(common) == 0:
        raise ValueError("No overlapping enhancer IDs between X_raw and labels.")

    X_raw = X_raw.loc[common].copy()
    labels = labels.loc[common].copy()

    if not X_raw.index.equals(labels.index):
        raise ValueError("X_raw and labels are not aligned after intersection.")

    # ------------------------------------------------------------------
    # 3. Apply the same feature engineering used by run_cv.py
    # ------------------------------------------------------------------
    X_features, _ = build_feature_matrix(
        X_raw=X_raw,
        original_motif_cols=original_motif_cols,
    )

    # Re-align labels to the final engineered matrix in case feature
    # engineering changes row order.
    common_final = X_features.index.intersection(labels.index)

    if len(common_final) == 0:
        raise ValueError(
            "No overlapping enhancer IDs between engineered features and labels."
        )

    X_features = X_features.loc[common_final].copy()
    labels = labels.loc[common_final].copy()

    if not X_features.index.equals(labels.index):
        raise ValueError(
            "Final engineered feature matrix and labels are not aligned."
        )

    # ------------------------------------------------------------------
    # 4. Assemble one full matrix containing all enhancer classes
    # ------------------------------------------------------------------
    final_matrix = X_features.copy()

    if isinstance(labels, pd.DataFrame):
        if labels.shape[1] != 1:
            raise ValueError(
                "Expected generate_multiclass_labels() to return a Series "
                "or a one-column DataFrame."
            )
        class_values = labels.iloc[:, 0]
    else:
        class_values = labels

    final_matrix.insert(0, "class", class_values.astype(str))
    final_matrix.index.name = "enhancer_id"

    output_path = out_dir / "feature_matrix_all_classes.csv"
    final_matrix.to_csv(output_path, index=True)

    # ------------------------------------------------------------------
    # 5. Report what was saved
    # ------------------------------------------------------------------
    print("\nFinal matrix:")
    print(f"  rows: {final_matrix.shape[0]}")
    print(f"  feature columns: {X_features.shape[1]}")
    print(f"  total columns including class: {final_matrix.shape[1]}")

    print("\nClass counts:")
    print(class_values.value_counts(dropna=False))

    print("\nSaved:")
    print(output_path)
    print("=" * 80)


if __name__ == "__main__":
    main()
