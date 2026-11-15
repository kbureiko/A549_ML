"""Fold-specific preprocessing: everything that is learned from data is learned from the training fold only."""

import pandas as pd
import numpy as np

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OneHotEncoder


class FoldPreprocessor:
    """
    Fold-specific preprocessing.

    Leakage control:
    - categorical levels are fitted only on X_train
    - variance filter is fitted only on X_train
    - X_test is transformed using train-derived objects

    Important:
    - No numeric imputation is performed.
    - Real zeros remain real zeros.
    - If numeric NaNs are present, the pipeline raises an error.
    """

    def __init__(
        self,
        categorical_cols: list[str],
        variance_threshold: float,
        fail_on_numeric_nan: bool = True,
    ):
        self.categorical_cols = categorical_cols
        self.variance_threshold = variance_threshold
        self.fail_on_numeric_nan = fail_on_numeric_nan

        self.numeric_cols_ = None
        self.categorical_cols_ = None

        self.variance_filter_ = None
        self.retained_numeric_cols_ = None

        self.onehot_encoder_ = None
        self.rf_feature_names_ = None

    def _to_numeric_no_imputation(
        self,
        X: pd.DataFrame,
        context: str,
    ) -> pd.DataFrame:
        """
        Convert numeric columns to numeric values without imputation.

        If NaNs are generated or already present, raise an error.
        """
        X_num = X[self.numeric_cols_].apply(pd.to_numeric, errors="coerce")

        # Check infinite values.
        inf_mask = np.isinf(X_num.to_numpy(dtype=float, copy=False))
        if inf_mask.any():
            bad_cols = X_num.columns[np.isinf(X_num).any(axis=0)].tolist()
            raise ValueError(
                f"Infinite values found in numeric features during {context}. "
                f"Problem columns include: {bad_cols[:20]}"
            )

        # Check NaNs.
        nan_counts = X_num.isna().sum()
        bad_nan_cols = nan_counts[nan_counts > 0]

        if self.fail_on_numeric_nan and len(bad_nan_cols) > 0:
            msg = (
                f"NaN values found in numeric features during {context}. "
                f"No numeric imputation is performed in this pipeline. "
                f"Number of affected columns: {len(bad_nan_cols)}. "
                f"Examples: {bad_nan_cols.head(20).to_dict()}\n\n"
                f"Fix upstream alignment/parsing, or explicitly decide that missing "
                f"signal values should be encoded as 0 before CV."
            )
            raise ValueError(msg)

        return X_num

    def fit(self, X_train: pd.DataFrame):
        X_train = X_train.copy()

        self.categorical_cols_ = [
            c for c in self.categorical_cols
            if c in X_train.columns
        ]

        self.numeric_cols_ = [
            c for c in X_train.columns
            if c not in self.categorical_cols_
        ]

        # Numeric coercion without imputation.
        X_num = self._to_numeric_no_imputation(
            X_train,
            context="training-fold fit",
        )

        # Train-fold-only variance filtering.
        self.variance_filter_ = VarianceThreshold(
            threshold=self.variance_threshold
        )
        self.variance_filter_.fit(X_num)

        self.retained_numeric_cols_ = X_num.columns[
            self.variance_filter_.get_support()
        ].tolist()

        if len(self.retained_numeric_cols_) == 0:
            raise ValueError(
                "No numeric features retained after variance filtering. "
                "Check VARIANCE_THRESHOLD or feature scaling."
            )

        # Train-fold-only one-hot encoding for RF categorical features.
        if self.categorical_cols_:
            X_cat = (
                X_train[self.categorical_cols_]
                .astype("object")
                .fillna("__MISSING__")
                .astype(str)
            )

            self.onehot_encoder_ = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            )
            self.onehot_encoder_.fit(X_cat)

        return self

    def transform_for_catboost(self, X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """
        Returns processed matrix for CatBoost.

        Numeric columns:
            numeric conversion + train-fold variance filtering.
            No numeric imputation.

        Categorical columns:
            retained as string categories.
        """
        X = X.copy()

        X_num = self._to_numeric_no_imputation(
            X,
            context="CatBoost transform",
        )

        X_num = X_num[self.retained_numeric_cols_]

        parts = [X_num]
        cat_features = []

        if self.categorical_cols_:
            X_cat = (
                X[self.categorical_cols_]
                .astype("object")
                .fillna("__MISSING__")
                .astype(str)
            )
            parts.append(X_cat)
            cat_features = self.categorical_cols_

        X_out = pd.concat(parts, axis=1)

        return X_out, cat_features

    def transform_for_rf(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Returns processed matrix for Random Forest.

        Numeric:
            numeric conversion + train-fold variance filtering.
            No numeric imputation.

        Categorical:
            one-hot encoded using categories learned only from training fold.
        """
        X = X.copy()

        X_num = self._to_numeric_no_imputation(
            X,
            context="Random Forest transform",
        )

        X_num = X_num[self.retained_numeric_cols_]

        parts = [X_num]

        if self.categorical_cols_:
            X_cat = (
                X[self.categorical_cols_]
                .astype("object")
                .fillna("__MISSING__")
                .astype(str)
            )

            encoded = self.onehot_encoder_.transform(X_cat)

            encoded_names = self.onehot_encoder_.get_feature_names_out(
                self.categorical_cols_
            )

            X_encoded = pd.DataFrame(
                encoded,
                index=X.index,
                columns=encoded_names,
            )

            parts.append(X_encoded)

        X_out = pd.concat(parts, axis=1)
        self.rf_feature_names_ = X_out.columns.tolist()

        return X_out

    def fit_transform(
        self,
        X_train: pd.DataFrame,
    ):
        self.fit(X_train)

        X_train_cb, cb_cat_features = self.transform_for_catboost(X_train)
        X_train_rf = self.transform_for_rf(X_train)

        return X_train_cb, cb_cat_features, X_train_rf
