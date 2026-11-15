from pathlib import Path
import joblib
import pandas as pd

from .preprocessing import FoldPreprocessor
from .models import fit_random_forest, fit_catboost
from .utils import save_json


def fit_final_models(
    X: pd.DataFrame,
    y: pd.Series,
    stable_features_by_model: dict,
    categorical_cols: list[str],
    variance_threshold: float,
    task: str,
    positive_label: str,
    negative_label: str,
    outdir: Path,
    random_state: int,
):
    """
    Train final models after CV evaluation is complete.

    Leakage note:
    - This model is not used to estimate performance.
    - Performance comes from repeated-CV held-out predictions.
    - Final model exists for SHAP/LIME and biological interpretation.
    """

    outdir.mkdir(parents=True, exist_ok=True)

    fitted = {}

    for model_name, stable_features in stable_features_by_model.items():
        model_dir = outdir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        X_sub = X[stable_features].copy()

        cat_sub = [c for c in categorical_cols if c in X_sub.columns]

        preprocessor = FoldPreprocessor(
            categorical_cols=cat_sub,
            variance_threshold=variance_threshold,
        )

        preprocessor.fit(X_sub)

        if model_name == "catboost":
            X_proc, cat_features = preprocessor.transform_for_catboost(X_sub)
            model = fit_catboost(
                X_proc,
                y,
                cat_features=cat_features,
                random_state=random_state,
            )

            processed_feature_names = X_proc.columns.tolist()
            categorical_feature_names = cat_features

        elif model_name == "random_forest":
            X_proc = preprocessor.transform_for_rf(X_sub)
            model = fit_random_forest(
                X_proc,
                y,
                random_state=random_state,
            )

            processed_feature_names = X_proc.columns.tolist()
            categorical_feature_names = []

        else:
            raise ValueError(f"Unknown model_name: {model_name}")

        joblib.dump(model, model_dir / "model.joblib")
        joblib.dump(preprocessor, model_dir / "preprocessor.joblib")

        X_proc.to_csv(model_dir / "processed_final_feature_matrix.csv")
        y.to_csv(model_dir / "final_labels.csv")

        with open(model_dir / "stable_features.txt", "w") as f:
            for feat in stable_features:
                f.write(feat + "\n")

        metadata = {
            "task": task,
            "model": model_name,
            "positive_label": positive_label,
            "negative_label": negative_label,
            "n_samples": int(X.shape[0]),
            "n_original_features": int(X.shape[1]),
            "n_stable_features": int(len(stable_features)),
            "n_processed_features": int(X_proc.shape[1]),
            "stable_features": stable_features,
            "processed_feature_names": processed_feature_names,
            "categorical_feature_names": categorical_feature_names,
            "class_order": list(model.classes_),
            "note": (
                "Final model trained after CV using stable features. "
                "Do not use this model for performance estimation."
            ),
        }

        save_json(metadata, model_dir / "metadata.json")

        fitted[model_name] = {
            "model": model,
            "preprocessor": preprocessor,
            "X_processed": X_proc,
            "metadata": metadata,
        }

    return fitted
