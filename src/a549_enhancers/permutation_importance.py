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


def compute_grouped_permutation_importance_heldout(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    positive_label: str,
    groups: dict,
    n_repeats: int,
    random_state: int,
) -> pd.DataFrame:
    """
    Permutation importance of feature groups on the held-out fold: decrease in average precision when all
    features of a group are shuffled together. One row order is drawn per repeat and applied to every member, so
    the relations inside the group are kept and only the link to the other features and to the class is broken.

    groups: {group name: [feature names]}; members that are not columns of X_test (for example constant
    features removed in this fold) are ignored, and groups left with fewer than two members are skipped.
    """
    y_bin = (pd.Series(list(y_test)) == positive_label).astype(int).to_numpy()

    def score(X):
        return average_precision_score(y_bin, get_positive_class_proba(model, X, positive_label))

    baseline = score(X_test)
    rng = np.random.default_rng(random_state)
    rows = []
    for name, members in groups.items():
        cols = [c for c in members if c in X_test.columns]
        if len(cols) < 2:
            continue
        drops = []
        for _ in range(n_repeats):
            order = rng.permutation(len(X_test))
            X_perm = X_test.copy()
            shuffled = X_test[cols].iloc[order]
            shuffled.index = X_test.index
            X_perm[cols] = shuffled
            drops.append(baseline - score(X_perm))
        rows.append({
            "group": name,
            "n_features": len(cols),
            "importance_mean": float(np.mean(drops)),
            "importance_sd": float(np.std(drops)),
            "baseline_average_precision": float(baseline),
        })
    return pd.DataFrame(rows, columns=["group", "n_features", "importance_mean", "importance_sd",
                                       "baseline_average_precision"])
