import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from a549_enhancers.intervals import overlaps_any, parse_enhancer_ids
from a549_enhancers.preprocessing import FoldPreprocessor
from a549_enhancers.stability import select_stable_features, summarize_importance_stability
from a549_enhancers.thresholding import choose_threshold_from_scores


def frame(n=60, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"wide": rng.normal(0, 10, n), "narrow": rng.normal(0, 0.5, n),
                         "motif_binary::GRE": rng.integers(0, 2, n), "annotation": rng.choice(["a", "b"], n)})


def test_variance_filter_is_fitted_on_the_training_fold_only():
    train, test = frame(seed=1), frame(seed=2)
    test["narrow"] *= 100                                    # high variance in the held-out fold must not rescue it
    pre = FoldPreprocessor(categorical_cols=["annotation", "motif_binary::GRE"], variance_threshold=5.0).fit(train)
    assert pre.retained_numeric_cols_ == ["wide"]
    X_test, cat = pre.transform_for_catboost(test)
    assert list(X_test.columns) == ["wide", "annotation", "motif_binary::GRE"]
    assert cat == ["annotation", "motif_binary::GRE"]         # categorical features bypass the variance filter
    assert X_test["motif_binary::GRE"].map(type).eq(str).all()


def test_missing_values_are_an_error_not_imputed():
    train = frame()
    train.loc[3, "wide"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        FoldPreprocessor(categorical_cols=["annotation"], variance_threshold=5.0).fit(train)


def test_unseen_category_in_held_out_fold_does_not_break_the_random_forest_matrix():
    train, test = frame(seed=1), frame(seed=2)
    test.loc[0, "annotation"] = "new"
    pre = FoldPreprocessor(categorical_cols=["annotation"], variance_threshold=5.0).fit(train)
    assert pre.transform_for_rf(test).shape[1] == pre.transform_for_rf(train).shape[1]


def test_max_f1_threshold_is_the_best_f1_over_all_observed_scores():
    rng = np.random.default_rng(0)
    y = np.array(["UP"] * 40 + ["NR"] * 360)
    score = np.r_[rng.beta(4, 3, 40), rng.beta(2, 6, 360)]
    thr = choose_threshold_from_scores(y, score, positive_label="UP", method="max_f1")
    best = max(f1_score(y == "UP", score >= t) for t in np.unique(score))
    assert f1_score(y == "UP", score >= thr) == pytest.approx(best)
    assert thr in score                                      # an observed score, not a grid value


def test_stable_features_need_positive_importance_in_most_folds():
    imp = pd.DataFrame([{"task": "nr_vs_up", "model": "catboost", "repeat": r, "fold": f, "feature": name,
                         "importance_mean": v[(r * 5 + f) % len(v)], "importance_sd": 0.0, "importance_rank": 1}
                        for r in range(5) for f in range(5)
                        for name, v in {"always": [0.02], "half": [0.02, -0.01], "never": [-0.01]}.items()])
    summary = summarize_importance_stability(imp)
    assert select_stable_features(summary, task="nr_vs_up", model_name="catboost") == ["always"]


def test_overlap_needs_one_shared_base():
    q = parse_enhancer_ids(["chr1_100_200", "chr1_200_300", "chr2_100_200", "chr1_50_100"])
    peaks = pd.DataFrame({"chr": ["chr1", "chr1"], "start": [199, 10], "end": [200, 50]})
    assert overlaps_any(q, peaks).tolist() == [True, False, False, False]      # touching intervals do not overlap
    assert overlaps_any(q, peaks, min_overlap=2).tolist() == [False] * 4


def test_bad_enhancer_id_is_reported():
    with pytest.raises(ValueError):
        parse_enhancer_ids(["chr1:100-200"])
