import numpy as np
import pandas as pd
import pytest

from a549_enhancers import config
from a549_enhancers.data_loading import add_combined_coordinate, build_base_feature_matrix, load_raw_inputs
from a549_enhancers.feature_engineering import build_feature_matrix
from a549_enhancers.labels import generate_multiclass_labels, make_binary_dataset

from conftest import (AVERAGED_COLUMN, AVERAGED_SOURCES, DROPPED_DEX_COLUMN, EXCLUDED_COLUMN, GROUP_COLUMNS,
                      MOTIF_COLUMNS, SIGNAL_COLUMNS)


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


def test_labels_use_only_the_configured_p_value_column():
    """The class labels come from config.PVAL_COL (unadjusted p-value = labels of the earlier runs); the other
    p-value column is ignored."""
    other = ({"ctrl vs. dex p-value", "ctrl vs. dex adj. p-value"} - {config.PVAL_COL}).pop()
    t = label_table(pvals=[0.99], lfcs=[1.0])
    t[other] = 0.0001
    assert config.PVAL_COL == "ctrl vs. dex p-value"
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
    assert motif_cols == GROUP_COLUMNS                              # config.MOTIF_SET = "groups"
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
    binary = [f"motif_binary::{c}" for c in GROUP_COLUMNS]
    assert set(binary) <= set(X.columns) and not set(GROUP_COLUMNS) & set(X.columns)
    assert set(np.unique(X[binary].to_numpy())) <= {0, 1}
    assert (X[binary[0]] == (X_raw[GROUP_COLUMNS[0]] > 0).astype(int)).all()
    assert meta["categorical_cols"] == ["annotation"] + [c for c in X.columns if c.startswith("motif_binary::")]
    assert X.shape[1] == X_raw.shape[1]                              # no derived dex/control contrasts


def test_new_tables_are_matched_on_enhancer_id_and_tracks_are_averaged(workspace, monkeypatch):
    raw = load_raw_inputs()
    X_raw, _, motif_cols = build_base_feature_matrix(raw)
    ids = X_raw.index.to_numpy()
    groups = raw["motif_groups"].set_index("enhancer_id")
    nbr = raw["neighbourhood"].set_index("enhancer_id")
    assert not groups.index.equals(X_raw.index)                     # the files are in another row order
    assert np.allclose(X_raw[config.GRE_SCORE_FEATURE], groups.loc[ids, config.GRE_SCORE_SOURCE_COLUMN])
    for c in config.NEIGHBOURHOOD_FEATURES:
        assert np.allclose(X_raw[c], nbr.loc[ids, c])
    assert (X_raw[GROUP_COLUMNS[0]].to_numpy() == groups.loc[ids, "motif::" + GROUP_COLUMNS[0]].to_numpy()).all()
    # mean of the two CREB1 control tracks; the single tracks are not features
    pre = workspace["prematrix"]
    assert np.allclose(X_raw[AVERAGED_COLUMN], pre[AVERAGED_SOURCES].mean(axis=1))
    assert not set(AVERAGED_SOURCES) & set(X_raw.columns)
    # the score and the neighbourhood counts stay continuous
    X, meta = build_feature_matrix(X_raw, motif_cols)
    assert config.GRE_SCORE_FEATURE in X.columns and X[config.GRE_SCORE_FEATURE].nunique() > 2
    assert not [c for c in meta["categorical_cols"] if c.startswith(("nbr_", "GRE_score"))]
    # a table that does not cover every enhancer stops the run
    short = dict(raw, neighbourhood=raw["neighbourhood"].iloc[1:])
    with pytest.raises(ValueError, match="are not in the table"):
        build_base_feature_matrix(short)
    # the motifs of the earlier runs remain available
    monkeypatch.setattr("a549_enhancers.data_loading.MOTIF_SET", "original")
    assert build_base_feature_matrix(load_raw_inputs())[2] == MOTIF_COLUMNS


def test_feature_layers_and_predefined_groups(workspace):
    from a549_enhancers.feature_groups import build_feature_groups, feature_layer
    assert feature_layer("H3K27ac_dex_100_1 hour_ENCFF530RMD") == "dex" and feature_layer("ATAC_dex_1h") == "dex"
    assert feature_layer("nbr_GRpeaks_50kb") == "dex" and feature_layer("nbr_strongGRpeak_distance") == "dex"
    assert feature_layer("H3K27ac_ENCFF054OOT") == "control" and feature_layer("nbr_enhancers_50kb") == "control"
    assert feature_layer("CREB1_eth_0.02_1 hour_ENCFF545ERO_ENCFF609MCU") == "control"
    assert feature_layer("GRE_score") == "genomic" and feature_layer("motif_binary::x") == "genomic"
    rng = np.random.default_rng(0)
    base = rng.normal(size=300)
    X = pd.DataFrame({
        "JUN_ENCFFAAAAAA": base + rng.normal(0, 0.1, 300), "JUNB_ENCFFBBBBBB": base + rng.normal(0, 0.1, 300),
        "CTCF_ENCFFCCCCCC": rng.normal(size=300), "FOSL2_dex_100_1 hour_ENCFFDDDDDD": rng.normal(size=300),
        "GRE_score": rng.normal(size=300), "motif_binary::a": rng.integers(0, 2, 300), "motif_binary::b": rng.integers(0, 2, 300),
        "nbr_enhancers_50kb": rng.integers(0, 9, 300), "nbr_enhancer_distance": rng.integers(0, 9, 300)})
    G = build_feature_groups(X)
    members = G.groupby("group")["feature"].apply(set)
    assert members["family: AP-1 (control)"] == {"JUN_ENCFFAAAAAA", "JUNB_ENCFFBBBBBB"}     # dex FOSL2 is not in it
    assert "family: AP-1 (dex)" not in members                                              # one member only
    assert members["family: all motif indicators"] == {"motif_binary::a", "motif_binary::b"}
    clusters = G[G.group_type == "correlation cluster"].groupby("group")["feature"].apply(set).tolist()
    assert clusters == [{"JUN_ENCFFAAAAAA", "JUNB_ENCFFBBBBBB"}]                            # the only correlated pair


def test_grouped_permutation_importance_finds_the_informative_group():
    from sklearn.linear_model import LogisticRegression
    from a549_enhancers.permutation_importance import compute_grouped_permutation_importance_heldout
    rng = np.random.default_rng(1)
    y = pd.Series(rng.choice(["NR", "UP"], 600))
    signal = (y == "UP").to_numpy() * 1.5
    X = pd.DataFrame({"a": signal + rng.normal(size=600), "b": signal + rng.normal(size=600),
                      "c": rng.normal(size=600), "d": rng.normal(size=600)})
    model = LogisticRegression().fit(X[:400], y[:400])
    out = compute_grouped_permutation_importance_heldout(
        model, X[400:], y[400:], "UP", {"informative": ["a", "b"], "noise": ["c", "d"], "single": ["a"],
                                        "partly absent": ["c", "not_a_column"]}, n_repeats=5, random_state=0)
    imp = out.set_index("group")["importance_mean"]
    assert list(out["group"]) == ["informative", "noise"]            # groups with fewer than two columns are skipped
    assert imp["informative"] > 0.1 and abs(imp["noise"]) < 0.05
