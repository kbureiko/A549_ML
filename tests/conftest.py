"""Synthetic inputs for the tests: small random tables in the format of the six input files, and a paths.toml
that points at a temporary folder. No real data is needed or touched."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from a549_enhancers import config, paths

N_ENHANCERS = 240
SIGNAL_COLUMNS = [
    "NR3C1_dex_100_1 hour_ENCFFAAAAAA",
    "NR3C1_eth_0.02_1 hour_ENCFFBBBBBB",
    "H3K27ac_dex_100_1 hour_ENCFFCCCCCC",
    "H3K27ac_eth_0.02_1 hour_ENCFFDDDDDD",
    "EP300_ENCFFEEEEEE",
]
DROPPED_DEX_COLUMN = "EP300_dex_100_2 hour_ENCFFGGGGGG"      # dex, but not 100 nM for 1 h
EXCLUDED_COLUMN = "CTCF_ENCFF254DAA"                          # listed in config.EXCLUDE_COLUMNS
MOTIF_COLUMNS = ["GRE(NR),IR3/A549-GR-ChIP-Seq/Homer", "AP-1(bZIP)/ThioMac-PU.1-ChIP-Seq/Homer"]


def make_inputs(folder: Path, n: int = N_ENHANCERS, seed: int = 0) -> pd.DataFrame:
    """Write the six input tables; returns the prematrix."""
    rng = np.random.default_rng(seed)
    start = np.sort(rng.integers(10_000, 5_000_000, n))
    pre = pd.DataFrame({"chr": rng.choice(["chr1", "chr2", "chr3"], n), "start": start,
                        "end": start + rng.integers(200, 1500, n)})
    response = rng.choice([1, -1, 0], n, p=[0.25, 0.2, 0.55])
    pre[config.LOG2FC_COL] = response * rng.uniform(0.7, 3.0, n) + (response == 0) * rng.normal(0, 0.2, n)
    pre[config.PVAL_COL] = np.where(response != 0, rng.uniform(1e-6, 0.04, n), rng.uniform(0.06, 1.0, n))
    pre["ctrl vs. dex p-value"] = pre[config.PVAL_COL] / 5
    for i, col in enumerate(SIGNAL_COLUMNS + [DROPPED_DEX_COLUMN, EXCLUDED_COLUMN]):
        pre[col] = rng.gamma(2.0, 8.0, n) + 12.0 * response * (i == 0)       # first track carries class signal
    pre.to_csv(folder / config.PREMATRIX_FILE, sep="\t", index=False)

    atac = pd.DataFrame({"ATAC_ctrl": rng.gamma(2.0, 6.0, n), "ATAC_dex_1h": rng.gamma(2.0, 6.0, n)})
    atac.to_csv(folder / config.ATAC_FILE, sep="\t")
    motifs = pd.DataFrame({c: rng.choice([0.0, 0.0, 7.5], n) for c in MOTIF_COLUMNS})
    motifs.to_csv(folder / config.MOTIFS_FILE, sep="\t")
    pd.DataFrame({"Annotation": rng.choice(["Intergenic", "intron", "promoter-TSS"], n)}).to_csv(
        folder / config.ANNOTATION_FILE, sep="\t")
    pd.DataFrame({"Distance to TSS": rng.integers(-200_000, 200_000, n)}).to_csv(
        folder / config.ANNOTATION_ADDITIONAL_FILE, sep="\t")
    pd.DataFrame({"5_pct_gc": rng.uniform(0.3, 0.7, n)}).to_csv(folder / config.GC_FILE, sep=";")
    return pre


@pytest.fixture()
def workspace(tmp_path, monkeypatch):
    """Temporary inputs + results folders and a paths.toml that names them."""
    for sub in ("ml_inputs", "ml_outputs", "analysis"):
        (tmp_path / sub).mkdir()
    prematrix = make_inputs(tmp_path / "ml_inputs")
    toml = tmp_path / "paths.toml"
    toml.write_text(
        f'[inputs]\nml_inputs = "{(tmp_path / "ml_inputs").as_posix()}"\n'
        f'[results]\nml_outputs = "{(tmp_path / "ml_outputs").as_posix()}"\n'
        f'analysis = "{(tmp_path / "analysis").as_posix()}"\n'
        '[runs]\nnr_vs_up = "test__nr_vs_up__thr_max_f1"\n'
    )
    monkeypatch.setenv(paths.ENV_VAR, str(toml))
    paths.reload()
    yield {"root": tmp_path, "prematrix": prematrix, "toml": toml}
    paths.load.cache_clear()
