# thresholding.py

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve

from .preprocessing import FoldPreprocessor
from .models import (
    fit_catboost,
    fit_random_forest,
    get_positive_class_proba,
)


def choose_threshold_from_scores(
    y_true,
    y_score,
    positive_label: str,
    method: str = "max_f1",
    min_precision: float = 0.30,
) -> float:
    """
    Select a decision threshold from validation predictions.

    Important:
    This function must only receive predictions generated from training data,
    for example inner-CV predictions within the outer training fold.

    It must never receive outer held-out fold predictions.
    """

    y_true = pd.Series(y_true)
    y_true_bin = (y_true == positive_label).astype(int).values

    precision, recall, thresholds = precision_recall_curve(
        y_true_bin,
        y_score,
    )

    # precision and recall are length len(thresholds) + 1.
    # Use values corresponding to actual thresholds.
    precision_t = precision[:-1]
    recall_t = recall[:-1]

    if len(thresholds) == 0:
        return 0.5

    if method == "max_f1":
        beta = 1.0
        score = (1 + beta**2) * precision_t * recall_t / (
            beta**2 * precision_t + recall_t + 1e-12
        )

    elif method == "max_f2":
        beta = 2.0
        score = (1 + beta**2) * precision_t * recall_t / (
            beta**2 * precision_t + recall_t + 1e-12
        )

    elif method == "recall_at_min_precision":
        valid = precision_t >= min_precision

        if valid.sum() == 0:
            # Fallback to max F1 if no threshold satisfies minimum precision.
            beta = 1.0
            score = (1 + beta**2) * precision_t * recall_t / (
                beta**2 * precision_t + recall_t + 1e-12
            )
        else:
            score = np.where(valid, recall_t, -np.inf)

    else:
        raise ValueError(
            "method must be one of: max_f1, max_f2, recall_at_min_precision"
        )

    best_idx = int(np.nanargmax(score))
    threshold = float(thresholds[best_idx])

    return threshold


def tune_threshold_on_outer_train(
    X_outer_train_raw: pd.DataFrame,
    y_outer_train: pd.Series,
    model_name: str,
    categorical_cols: list[str],
    variance_threshold: float,
    positive_label: str,
    random_state: int,
    n_splits: int = 3,
    threshold_method: str = "max_f1",
    min_precision: float = 0.30,
) -> tuple[float, pd.DataFrame]:
    """
    Tune classification threshold using only the outer training fold.

    Leakage control:
    - Inner CV is performed only inside the outer training data.
    - Each inner fold fits preprocessing only on the inner-training split.
    - Outer held-out data are not seen here.
    - The returned threshold is later applied once to the outer held-out fold.
    """

    inner_cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    all_rows = []

    for inner_fold, (inner_train_idx, inner_valid_idx) in enumerate(
        inner_cv.split(X_outer_train_raw, y_outer_train),
        start=1,
    ):
        X_inner_train = X_outer_train_raw.iloc[inner_train_idx].copy()
        X_inner_valid = X_outer_train_raw.iloc[inner_valid_idx].copy()

        y_inner_train = y_outer_train.iloc[inner_train_idx].copy()
        y_inner_valid = y_outer_train.iloc[inner_valid_idx].copy()

        preprocessor = FoldPreprocessor(
            categorical_cols=categorical_cols,
            variance_threshold=variance_threshold,
        )

        X_inner_train_cb, cb_cat_features, X_inner_train_rf = (
            preprocessor.fit_transform(X_inner_train)
        )

        X_inner_valid_cb, _ = preprocessor.transform_for_catboost(X_inner_valid)
        X_inner_valid_rf = preprocessor.transform_for_rf(X_inner_valid)

        inner_seed = random_state + inner_fold

        if model_name == "catboost":
            model = fit_catboost(
                X_train=X_inner_train_cb,
                y_train=y_inner_train,
                cat_features=cb_cat_features,
                random_state=inner_seed,
            )
            X_valid_processed = X_inner_valid_cb

        elif model_name == "random_forest":
            model = fit_random_forest(
                X_train=X_inner_train_rf,
                y_train=y_inner_train,
                random_state=inner_seed,
            )
            X_valid_processed = X_inner_valid_rf

        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        y_score = get_positive_class_proba(
            model=model,
            X=X_valid_processed,
            positive_label=positive_label,
        )

        fold_rows = pd.DataFrame({
            "inner_fold": inner_fold,
            "enhancer_id": X_inner_valid.index,
            "true_label": y_inner_valid.values,
            f"proba_{positive_label}": y_score,
        })

        all_rows.append(fold_rows)

    oof = pd.concat(all_rows, ignore_index=True)

    threshold = choose_threshold_from_scores(
        y_true=oof["true_label"],
        y_score=oof[f"proba_{positive_label}"].values,
        positive_label=positive_label,
        method=threshold_method,
        min_precision=min_precision,
    )

    oof["chosen_threshold"] = threshold
    oof["threshold_method"] = threshold_method
    oof["min_precision"] = min_precision

    return threshold, oof
