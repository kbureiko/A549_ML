import json
import random
from pathlib import Path

import numpy as np
import pandas as pd


def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def read_table(path: Path, sep: str = "\t", index_col=None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, sep=sep, index_col=index_col)


def make_output_dirs(root: Path):
    subdirs = {
        "root": root,
        "data": root / "data",
        "cv_predictions": root / "cv_predictions",
        "cv_metrics": root / "cv_metrics",
        "cv_importance": root / "cv_importance",
        "cv_models": root / "cv_models",
        "summaries": root / "summaries",
        "final_model": root / "final_model",
        "figures": root / "figures",
        "thresholding": root / "thresholding",
    }

    for p in subdirs.values():
        ensure_dir(p)

    return subdirs


def safe_filename(text: str) -> str:
    out = (
        text.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("|", "_")
        .replace(">", "gt")
        .replace("<", "lt")
    )
    return out
