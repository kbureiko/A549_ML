import pandas as pd

from .config import (
    PVAL_COL,
    LOG2FC_COL,
    PVAL_THRESHOLD,
    LOG2FC_THRESHOLD,
)


def generate_multiclass_labels(prematrix_with_ids: pd.DataFrame) -> pd.Series:
    """
    Generate UP / DN / NR labels from prematrix.

    UP:
        adj. p-value < 0.05 and log2FC >= 0.584
    DN:
        adj. p-value < 0.05 and log2FC <= -0.584
    NR:
        all other enhancers
    """

    required = ["combined", PVAL_COL, LOG2FC_COL]
    missing = [c for c in required if c not in prematrix_with_ids.columns]
    if missing:
        raise ValueError(f"prematrix is missing required label columns: {missing}")

    df = prematrix_with_ids.copy()

    # Set enhancer ID as the actual index so labels, p-values, and log2FC align.
    df = df.set_index("combined", drop=False)

    pval = pd.to_numeric(df[PVAL_COL], errors="coerce")
    log2fc = pd.to_numeric(df[LOG2FC_COL], errors="coerce")

    labels = pd.Series("NR", index=df.index, name="label")

    up_mask = (pval < PVAL_THRESHOLD) & (log2fc >= LOG2FC_THRESHOLD)
    dn_mask = (pval < PVAL_THRESHOLD) & (log2fc <= -LOG2FC_THRESHOLD)

    labels.loc[up_mask] = "UP"
    labels.loc[dn_mask] = "DN"

    # Safety checks
    if labels.index.duplicated().any():
        n_dup = labels.index.duplicated().sum()
        raise ValueError(f"Duplicated enhancer IDs in labels: {n_dup}")

    print("Generated labels:")
    print(labels.value_counts())

    return labels

def make_binary_dataset(
    X: pd.DataFrame,
    labels: pd.Series,
    task: str,
) -> tuple[pd.DataFrame, pd.Series, str, str]:
    """
    Create binary classification dataset.

    Tasks
    -----
    nr_vs_up:
        negative = NR
        positive = UP

    up_vs_dn:
        negative = DN
        positive = UP

    nr_vs_dn:
        negative = NR
        positive = DN
    """

    if task == "nr_vs_up":
        keep_labels = ["NR", "UP"]
        negative_label = "NR"
        positive_label = "UP"

    elif task == "up_vs_dn":
        keep_labels = ["DN", "UP"]
        negative_label = "DN"
        positive_label = "UP"

    elif task == "nr_vs_dn":
        keep_labels = ["NR", "DN"]
        negative_label = "NR"
        positive_label = "DN"

    else:
        raise ValueError(
            "task must be one of: nr_vs_up, up_vs_dn, nr_vs_dn"
        )

    common = X.index.intersection(labels.index)

    X = X.loc[common].copy()
    labels = labels.loc[common].copy()

    mask = labels.isin(keep_labels)

    X_task = X.loc[mask].copy()
    y_task = labels.loc[mask].copy()

    y_task.name = "label"

    if not X_task.index.equals(y_task.index):
        raise ValueError("X_task and y_task are not aligned.")

    print(f"\nBinary task: {task}")
    print(f"Negative label: {negative_label}")
    print(f"Positive label: {positive_label}")
    print(y_task.value_counts())

    return X_task, y_task, negative_label, positive_label

def summarize_labels(labels: pd.Series) -> pd.DataFrame:
    return (
        labels.value_counts()
        .rename_axis("class")
        .reset_index(name="n")
    )
