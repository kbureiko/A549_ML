import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, balanced_accuracy_score

from .models import get_positive_class_proba


def make_average_precision_scorer(positive_label: str):
    """
    Scorer for permutation importance.
    """
    def scorer(estimator, X, y):
        y_bin = (pd.Series(y) == positive_label).astype(int).values
        y_score = get_positive_class_proba(estimator, X, positive_label)
        return average_precision_score(y_bin, y_score)

    return scorer


def make_balanced_accuracy_scorer():
    def scorer(estimator, X, y):
        y_pred = estimator.predict(X)
        y_pred = np.asarray(y_pred).ravel()
        return balanced_accuracy_score(y, y_pred)

    return scorer


def compute_permutation_importance_heldout(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    positive_label: str,
    scoring: str,
    n_repeats: int,
    random_state: int,
) -> pd.DataFrame:
    """
    Compute permutation importance on the held-out fold only.

    Leakage control:
    - never compute permutation importance on training data
    - never compute it on the full dataset during CV
    """
    if scoring == "average_precision":
        scorer = make_average_precision_scorer(positive_label)
    elif scoring == "balanced_accuracy":
        scorer = make_balanced_accuracy_scorer()
    else:
        raise ValueError("scoring must be average_precision or balanced_accuracy")

    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring=scorer,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    imp = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": result.importances_mean,
        "importance_sd": result.importances_std,
    })

    imp["importance_rank"] = (
        imp["importance_mean"]
        .rank(method="average", ascending=False)
    )

    imp = imp.sort_values("importance_mean", ascending=False)

    return imp
