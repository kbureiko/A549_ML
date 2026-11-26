import numpy as np
import pandas as pd

from .config import (
    STABLE_MIN_FRACTION_POSITIVE_IMPORTANCE,
    STABLE_MIN_MEDIAN_IMPORTANCE,
    STABLE_TOP_N_FALLBACK,
)


def summarize_cv_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        c for c in metrics_df.columns
        if pd.api.types.is_numeric_dtype(metrics_df[c])
        and c not in ["repeat", "fold"]
    ]

    summary = (
        metrics_df[metric_cols]
        .agg(["mean", "std", "median", "min", "max"])
        .T
        .reset_index()
        .rename(columns={"index": "metric"})
    )

    return summary


def summarize_importance_stability(importance_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize permutation importance stability across held-out folds.

    Requires columns:
        task
        model
        repeat
        fold
        feature
        importance_mean
        importance_sd
        importance_rank
    """

    summary = (
        importance_df
        .groupby(["task", "model", "feature"])
        .agg(
            n_folds=("importance_mean", "count"),
            mean_importance=("importance_mean", "mean"),
            median_importance=("importance_mean", "median"),
            sd_importance=("importance_mean", "std"),
            fraction_positive_importance=("importance_mean", lambda x: np.mean(np.asarray(x) > 0)),
            median_rank=("importance_rank", "median"),
            mean_rank=("importance_rank", "mean"),
            top10_fraction=("importance_rank", lambda x: np.mean(np.asarray(x) <= 10)),
            top20_fraction=("importance_rank", lambda x: np.mean(np.asarray(x) <= 20)),
        )
        .reset_index()
    )

    summary = summary.sort_values(
        [
            "task",
            "model",
            "fraction_positive_importance",
            "median_importance",
            "mean_importance",
        ],
        ascending=[True, True, False, False, False],
    )

    return summary


def select_stable_features(
    importance_summary: pd.DataFrame,
    task: str,
    model_name: str,
    min_fraction_positive: float = STABLE_MIN_FRACTION_POSITIVE_IMPORTANCE,
    min_median_importance: float = STABLE_MIN_MEDIAN_IMPORTANCE,
    top_n_fallback: int = STABLE_TOP_N_FALLBACK,
) -> list[str]:
    """
    Define stable feature set after CV is complete.

    This is not used during CV evaluation.
    It is used only for final interpretation models.

    Primary rule:
        fraction_positive_importance >= STABLE_MIN_FRACTION_POSITIVE_IMPORTANCE (config.py; 0.80)
        median_importance > 0

    Fallback:
        if too few features pass, use top N by stability/importance.
    """
    sub = importance_summary[
        (importance_summary["task"] == task)
        & (importance_summary["model"] == model_name)
    ].copy()

    stable = sub[
        (sub["fraction_positive_importance"] >= min_fraction_positive)
        & (sub["median_importance"] > min_median_importance)
    ].copy()

    if stable.shape[0] == 0:
        stable = (
            sub.sort_values(
                [
                    "fraction_positive_importance",
                    "median_importance",
                    "mean_importance",
                ],
                ascending=False,
            )
            .head(top_n_fallback)
            .copy()
        )

    return stable["feature"].tolist()
