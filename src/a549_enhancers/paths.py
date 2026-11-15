"""Where the data and the results live on this computer.

No script in this repository contains a folder name from anyone's computer. Every script asks this module, and
this module reads the answers from ``paths.toml``: a small file that you create once from ``paths.example.toml``
and that git ignores, so it never reaches the repository.

How ``paths.toml`` is found
---------------------------
1. the file named by the environment variable ``A549_PATHS``, if it is set;
2. otherwise ``paths.toml`` in the current folder or one of its parents;
3. otherwise ``paths.toml`` in the repository root.

Typical use::

    from a549_enhancers.paths import ML, PROJECT, RUNS      # folders as pathlib.Path

    cv_metrics = RUNS["nr_vs_up"] / "summaries" / "all_cv_metrics.csv"

The names are resolved the first time they are used, so importing the package (for example in the tests) does
not need a ``paths.toml``.

===================  ============================  ==========================================================
name                 ``paths.toml`` entry          content
===================  ============================  ==========================================================
``ML_INPUTS``        ``[inputs] ml_inputs``        the six input tables of the classifiers
``ML``               ``[results] ml_outputs``      run folders written by ``scripts/01_run_cv.py``
``PROJECT``          ``[results] analysis``        downstream tables read and written by the analysis scripts
``FIGURES``          ``[results] figures``         figures and their source tables (default: analysis/figures)
``RUNS``             ``[runs]``                    task -> folder of the published cross-validation run
``RUN_NAMES``        ``[runs]``                    task -> folder name only
``CHIP_MATRICES``    ``[layout] chip_matrices``    deepTools matrices of the ChIP-seq / ATAC-seq tracks
``PROSEQ_MATRICES``  ``[layout] proseq_matrices``  deepTools matrices of the PRO-seq tracks, sorted class BEDs
``PROSEQ_MA_TABLE``  ``[layout] proseq_ma_table``  per-enhancer PRO-seq log2 fold changes (optional table)
``ENHANCER_GENE``    ``[layout] enhancer_gene``    time-course expression, tx2gene, locus tracks, bigWigs
``ABC_LINKS``        ``[layout] abc_links``        ABC enhancer-gene links lifted to hg38
``FEATURE_MAP``      ``[layout] feature_map``      principal components and UMAP layouts
``NR_TOPICS``        ``[layout] nr_topics``        topic loadings and usage of the NR enhancers
``KNOCKDOWN``        ``[layout] knockdown``        tables of the knockdown experiments (written by the R scripts)
===================  ============================  ==========================================================

The ``[layout]`` entries are sub-folders (or files) inside ``analysis``; each has the default listed in
``LAYOUT_DEFAULTS`` and can be renamed in ``paths.toml`` without touching any script.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

ENV_VAR = "A549_PATHS"
REPO_ROOT = Path(__file__).resolve().parents[2]
TASKS = ("nr_vs_up", "up_vs_dn", "nr_vs_dn")

LAYOUT_DEFAULTS = {
    "chip_matrices": "deeptools/chip",
    "proseq_matrices": "deeptools/proseq",
    "proseq_ma_table": "deeptools/proseq/A549_MA_plot_data.tsv",
    "enhancer_gene": "enhancer_gene",
    "abc_links": "enhancer_gene/ABC_Nasser2021_A549_hg38.tsv.gz",
    "feature_map": "feature_map",
    "nr_topics": "nr_topics",
    "knockdown": "knockdown",
}

_HELP = (
    "paths.toml was not found. Copy paths.example.toml to paths.toml in the repository root and fill in the "
    f"folders on your computer, or point the environment variable {ENV_VAR} at such a file. "
    "paths.toml is ignored by git."
)


@dataclass(frozen=True)
class Paths:
    """The folders named in one ``paths.toml``."""

    source: Path
    ml_inputs: Path | None = None
    ml_outputs: Path | None = None
    analysis: Path | None = None
    figures: Path | None = None
    knockdown_inputs: dict[str, Path] = field(default_factory=dict)
    run_names: dict[str, str] = field(default_factory=dict)
    layout: dict[str, str] = field(default_factory=lambda: dict(LAYOUT_DEFAULTS))

    def require(self, name: str) -> Path:
        """Folder ``name``; a clear error if ``paths.toml`` does not define it."""
        value = getattr(self, name)
        if value is None:
            raise KeyError(f"'{name}' is not set in {self.source} (see paths.example.toml)")
        return value

    def run_dir(self, task: str) -> Path:
        """Folder of the cross-validation run of one task."""
        if task not in self.run_names:
            raise KeyError(f"[runs] {task} is not set in {self.source} (see paths.example.toml)")
        return self.require("ml_outputs") / self.run_names[task]

    def sub(self, key: str) -> Path:
        """One of the ``[layout]`` locations inside ``analysis``."""
        return self.require("analysis") / self.layout[key]


def _as_path(value) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return Path(str(value)).expanduser()


def find_config() -> Path:
    """Location of the ``paths.toml`` in use."""
    env = os.environ.get(ENV_VAR, "").strip()
    if env:
        p = Path(env).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"{ENV_VAR} points to {p}, which does not exist")
        return p
    cwd = Path.cwd()
    for folder in (cwd, *cwd.parents, REPO_ROOT):
        p = folder / "paths.toml"
        if p.is_file():
            return p
    raise FileNotFoundError(_HELP)


def read_config(file: Path) -> Paths:
    """Parse one ``paths.toml``."""
    with open(file, "rb") as fh:
        cfg = tomllib.load(fh)
    inputs, results = cfg.get("inputs", {}), cfg.get("results", {})
    unknown = set(cfg.get("layout", {})) - set(LAYOUT_DEFAULTS)
    if unknown:
        raise KeyError(f"unknown [layout] entries in {file}: {sorted(unknown)}; known: {sorted(LAYOUT_DEFAULTS)}")
    analysis = _as_path(results.get("analysis"))
    figures = _as_path(results.get("figures"))
    if figures is None and analysis is not None:
        figures = analysis / "figures"
    return Paths(
        source=Path(file),
        ml_inputs=_as_path(inputs.get("ml_inputs")),
        ml_outputs=_as_path(results.get("ml_outputs")),
        analysis=analysis,
        figures=figures,
        knockdown_inputs={k: _as_path(v) for k, v in cfg.get("knockdown", {}).items() if _as_path(v) is not None},
        run_names={k: str(v) for k, v in cfg.get("runs", {}).items()},
        layout={**LAYOUT_DEFAULTS, **{k: str(v) for k, v in cfg.get("layout", {}).items()}},
    )


@lru_cache(maxsize=1)
def load() -> Paths:
    """The paths of this computer (read once per process)."""
    return read_config(find_config())


def reload() -> Paths:
    """Read ``paths.toml`` again, for example after changing ``A549_PATHS``."""
    load.cache_clear()
    return load()


def ml_input(file_name: str) -> Path:
    """Full path of one of the input tables of the classifiers."""
    return load().require("ml_inputs") / file_name


def run_dir(task: str) -> Path:
    """Folder of the published cross-validation run of ``task``."""
    return load().run_dir(task)


_LAZY = {
    "ML_INPUTS": lambda p: p.require("ml_inputs"),
    "ML": lambda p: p.require("ml_outputs"),
    "PROJECT": lambda p: p.require("analysis"),
    "FIGURES": lambda p: p.require("figures"),
    "RUN_NAMES": lambda p: dict(p.run_names),
    "RUNS": lambda p: {t: p.run_dir(t) for t in p.run_names},
    **{key.upper(): (lambda p, key=key: p.sub(key)) for key in LAYOUT_DEFAULTS},
}


def __getattr__(name: str):
    if name in _LAZY:
        return _LAZY[name](load())
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
