import numpy as np
import pandas as pd
import pytest

from a549_enhancers import config
from a549_enhancers.data_loading import add_combined_coordinate, build_base_feature_matrix, load_raw_inputs
from a549_enhancers.feature_engineering import build_feature_matrix
from a549_enhancers.labels import generate_multiclass_labels, make_binary_dataset

from conftest import DROPPED_DEX_COLUMN, EXCLUDED_COLUMN, MOTIF_COLUMNS, SIGNAL_COLUMNS


def label_table(pvals, lfcs):
    df = pd.DataFrame({"chr": "chr1", "start": np.arange(len(pvals)) * 1000, "end": np.arange(len(pvals)) * 1000 + 500,
                       config.PVAL_COL: pvals, config.LOG2FC_COL: lfcs})
    return add_combined_coordinate(df)


def test_class_rule_uses_the_adjusted_p_value_and_inclusive_fold_change():
    t = label_table(pvals=[0.049, 0.049, 0.05, 0.01, 0.01, 0.01, 0.9],
                    lfcs=[0.584, -0.584, 2.0, 0.5839, -0.5839, -3.0, 5.0])
    labels = generate_multiclass_labels(t)
    assert labels.tolist() == ["UP", "DN", "NR", "NR", "NR", "DN", "NR"]
    assert labels.index[0] == "chr1_0_500"


def test_unadjusted_p_value_column_is_not_used():
    t = label_table(pvals=[0.99], lfcs=[1.0])
    t["ctrl vs. dex p-value"] = 0.0001
    assert config.PVAL_COL == "ctrl vs. dex adj. p-value"
    assert generate_multiclass_labels(t).tolist() == ["NR"]


@pytest.mark.parametrize("task, negative, positive", [("nr_vs_up", "NR", "UP"), ("up_vs_dn", "DN", "UP"),
                                                       ("nr_vs_dn", "NR", "DN")])
def test_binary_tasks(task, negative, positive):
    labels = pd.Series(["UP", "DN", "NR", "NR", "UP"], index=list("abcde"))
    X = pd.DataFrame({"f": range(5)}, index=list("abcde"))
    X_task, y_task, neg, pos = make_binary_dataset(X, labels, task)
    assert (neg, pos) == (negative, positive)
    assert set(y_task) == {negative, positive} and X_task.index.equals(y_task.index)


def test_feature_matrix_from_the_input_tables(workspace):
    X_raw, prematrix, motif_cols = build_base_feature_matrix(load_raw_inputs())
    assert X_raw.index[0] == f"{prematrix.chr[0]}_{prematrix.start[0]}_{prematrix.end[0]}"
    assert motif_cols == MOTIF_COLUMNS
    for col in SIGNAL_COLUMNS + ["ATAC_ctrl", "ATAC_dex_1h", "annotation", "Distance to TSS", "Enh_len", "gc"]:
        assert col in X_raw.columns
    assert DROPPED_DEX_COLUMN not in X_raw.columns                 # dex track that is not 100 nM, 1 h
    assert EXCLUDED_COLUMN not in X_raw.columns                    # config.EXCLUDE_COLUMNS
    assert not {config.PVAL_COL, config.LOG2FC_COL, "ctrl vs. dex adj. p-value"} & set(X_raw.columns)   # no label leak
    assert (X_raw["Distance to TSS"] >= 0).all()
    assert (X_raw["Enh_len"].to_numpy() == (prematrix["end"] - prematrix["start"]).to_numpy()).all()


def test_features_binarise_motifs_and_mark_them_categorical(workspace):
    X_raw, _, motif_cols = build_base_feature_matrix(load_raw_inputs())
    X, meta = build_feature_matrix(X_raw, motif_cols)
    binary = [f"motif_binary::{c}" for c in MOTIF_COLUMNS]
    assert set(binary) <= set(X.columns) and not set(MOTIF_COLUMNS) & set(X.columns)
    assert set(np.unique(X[binary].to_numpy())) <= {0, 1}
    assert (X[binary[0]] == (X_raw[MOTIF_COLUMNS[0]] > 0).astype(int)).all()
    assert meta["categorical_cols"] == ["annotation"] + [c for c in X.columns if c.startswith("motif_binary::")]
    assert X.shape[1] == X_raw.shape[1]                              # no derived dex/control contrasts
