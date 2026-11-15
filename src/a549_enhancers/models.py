import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier

from .config import RF_PARAMS, CATBOOST_PARAMS


def fit_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
) -> RandomForestClassifier:
    params = dict(RF_PARAMS)
    params["random_state"] = random_state

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    return model


def fit_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cat_features: list[str],
    random_state: int,
) -> CatBoostClassifier:
    params = dict(CATBOOST_PARAMS)
    params["random_seed"] = random_state

    model = CatBoostClassifier(**params)

    model.fit(
        X_train,
        y_train,
        cat_features=cat_features if cat_features else None,
    )

    return model


def get_positive_class_proba(
    model,
    X: pd.DataFrame,
    positive_label: str,
) -> np.ndarray:
    """
    Extract probability for the biological positive class.

    Works for sklearn RF and CatBoost.
    """
    proba = model.predict_proba(X)

    classes = list(model.classes_)
    if positive_label not in classes:
        raise ValueError(
            f"Positive label {positive_label} not found in model classes {classes}"
        )

    pos_idx = classes.index(positive_label)
    return proba[:, pos_idx]


def predict_binary(
    model,
    X: pd.DataFrame,
    positive_label: str,
) -> tuple[np.ndarray, np.ndarray]:
    y_pred = model.predict(X)

    # CatBoost may return shape (n, 1)
    y_pred = np.asarray(y_pred).ravel()

    y_pos_proba = get_positive_class_proba(model, X, positive_label)

    return y_pred, y_pos_proba
