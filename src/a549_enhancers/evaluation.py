import numpy as np
import pandas as pd

from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)


def evaluate_binary_predictions(
    y_true: pd.Series,
    y_pred,
    y_pos_proba,
    positive_label: str,
    negative_label: str,
) -> dict:
    """
    Binary evaluation.

    Positive class:
        UP for both tasks in your current design.
    """
    y_true = pd.Series(y_true)
    y_pred = pd.Series(np.asarray(y_pred).ravel(), index=y_true.index)

    y_true_bin = (y_true == positive_label).astype(int)
    y_pred_bin = (y_pred == positive_label).astype(int)

    metrics = {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "positive_f1": f1_score(y_true_bin, y_pred_bin, zero_division=0),
        "positive_precision": precision_score(y_true_bin, y_pred_bin, zero_division=0),
        "positive_recall": recall_score(y_true_bin, y_pred_bin, zero_division=0),
    }

    try:
        metrics["roc_auc"] = roc_auc_score(y_true_bin, y_pos_proba)
    except ValueError:
        metrics["roc_auc"] = np.nan

    try:
        metrics["average_precision"] = average_precision_score(
            y_true_bin,
            y_pos_proba,
        )
    except ValueError:
        metrics["average_precision"] = np.nan

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[negative_label, positive_label],
    )

    metrics["tn"] = int(cm[0, 0])
    metrics["fp"] = int(cm[0, 1])
    metrics["fn"] = int(cm[1, 0])
    metrics["tp"] = int(cm[1, 1])

    return metrics


def make_prediction_table(
    enhancer_ids,
    y_true,
    y_pred,
    y_pos_proba,
    task: str,
    model_name: str,
    repeat: int,
    fold: int,
    positive_label: str,
) -> pd.DataFrame:
    return pd.DataFrame({
        "enhancer_id": enhancer_ids,
        "task": task,
        "model": model_name,
        "repeat": repeat,
        "fold": fold,
        "true_label": list(y_true),
        "pred_label": list(np.asarray(y_pred).ravel()),
        f"proba_{positive_label}": y_pos_proba,
    })
