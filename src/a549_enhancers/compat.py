"""Opening model files that were saved before this code became a package.

The published run folders were written when the pipeline modules were plain files next to each other
(``preprocessing.py``, ``models.py``, ...). A ``preprocessor.joblib`` from that time stores its class as
``preprocessing.FoldPreprocessor``. ``register_legacy_modules()`` makes the old module name point to the same
class in this package, so those files open unchanged. Files written by the current code do not need it.
"""
from __future__ import annotations

import importlib
import sys

LEGACY_MODULES = ("preprocessing",)


def register_legacy_modules() -> None:
    """Let ``pickle`` / ``joblib`` resolve the pre-package module names."""
    for name in LEGACY_MODULES:
        sys.modules.setdefault(name, importlib.import_module(f"{__package__}.{name}"))
