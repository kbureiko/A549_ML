"""Feature layers and predefined feature groups.

Layers (from the column name only):
    genomic   sequence and position: motif indicators, GRE score, annotation, distance to TSS, length, GC content
    control   measured or defined without dex: EtOH and untreated ChIP-seq tracks, ATAC_ctrl, enhancer density
    dex       measured after dex: dex ChIP-seq tracks, ATAC_dex_1h, GR peaks around the enhancer

Groups for the grouped permutation importance are fixed before any model is fitted and never use the class
labels: (1) families of related targets listed in config.FEATURE_FAMILIES, separately for control and dex tracks;
(2) clusters of correlated signal features (Spearman correlation over all enhancers, average linkage).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    FEATURE_FAMILIES,
    GRE_SCORE_FEATURE,
    GROUP_CORRELATION_THRESHOLD,
)

GENOMIC_FEATURES = {"annotation", "Distance to TSS", "Enh_len", "gc", GRE_SCORE_FEATURE}
MOTIF_PREFIX = "motif_binary::"
NBR_ENHANCER_PREFIX = "nbr_enhancer"
NBR_GR_PREFIXES = ("nbr_GRpeak", "nbr_strongGRpeak")


def feature_layer(col: str) -> str:
    """'genomic', 'control' or 'dex' for a column of the feature matrix."""
    col = str(col)
    if col.startswith(MOTIF_PREFIX) or col in GENOMIC_FEATURES:
        return "genomic"
    if col.startswith(NBR_GR_PREFIXES):
        return "dex"
    if col.startswith(NBR_ENHANCER_PREFIX):
        return "control"
    if "_dex_" in col or col == "ATAC_dex_1h":
        return "dex"
    return "control"


def target_of(col: str) -> str:
    """ENCODE target of a signal column ('H3K27ac_dex_100_1 hour_ENCFF530RMD' -> 'H3K27ac'; ATAC columns -> 'ATAC')."""
    col = str(col)
    return "ATAC" if col.startswith("ATAC") else col.split("_")[0]


def is_signal_feature(col: str) -> bool:
    """ChIP-seq and ATAC-seq signal columns (not motifs, genomic or neighbourhood features)."""
    col = str(col)
    return (feature_layer(col) != "genomic" and not col.startswith("nbr_")
            and ("ENCFF" in col or col.startswith("ATAC")))


def family_groups(columns) -> dict[str, list[str]]:
    """Groups from config.FEATURE_FAMILIES (per condition), all motif indicators, and the two neighbourhood sets.
    Groups with fewer than two members are left out."""
    columns = [str(c) for c in columns]
    groups: dict[str, list[str]] = {}
    for family, targets in FEATURE_FAMILIES.items():
        for layer in ("control", "dex"):
            members = [c for c in columns if is_signal_feature(c) and target_of(c) in targets
                       and feature_layer(c) == layer]
            groups[f"family: {family} ({layer})"] = members
    groups["family: all motif indicators"] = [c for c in columns if c.startswith(MOTIF_PREFIX)]
    groups["family: GRE (motif group and score)"] = [c for c in columns if c == GRE_SCORE_FEATURE or
                                                     (c.startswith(MOTIF_PREFIX + "GRE(NR),IR3"))]
    groups["family: neighbourhood, other enhancers"] = [c for c in columns if c.startswith(NBR_ENHANCER_PREFIX)]
    groups["family: neighbourhood, GR peaks after dex"] = [c for c in columns if c.startswith(NBR_GR_PREFIXES)]
    return {name: members for name, members in groups.items() if len(members) >= 2}


def correlation_groups(X: pd.DataFrame, threshold: float = GROUP_CORRELATION_THRESHOLD) -> dict[str, list[str]]:
    """Clusters of signal features whose mean pairwise Spearman correlation is at least ``threshold``
    (average linkage on 1 - r). ``X``: all enhancers, no labels. Clusters of one feature are left out."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    cols = [c for c in X.columns if is_signal_feature(c)]
    V = X[cols].apply(pd.to_numeric, errors="coerce")
    V = V.loc[:, V.nunique(dropna=True) > 1]
    if V.shape[1] < 2:
        return {}
    r = V.rank().corr().to_numpy()
    d = np.clip(1.0 - r, 0.0, 2.0)
    d = (d + d.T) / 2.0
    np.fill_diagonal(d, 0.0)
    labels = fcluster(linkage(squareform(d, checks=False), method="average"), t=1.0 - threshold, criterion="distance")
    groups = {}
    for k in np.unique(labels):
        members = [c for c, lab in zip(V.columns, labels) if lab == k]
        if len(members) >= 2:
            short = ", ".join(sorted(f"{target_of(c)} {'dex' if feature_layer(c) == 'dex' else 'ctrl'}" for c in members)[:3])
            groups[f"cluster {len(groups) + 1:02d}: {short}{', ...' if len(members) > 3 else ''} (n={len(members)})"] = members
    return groups


def build_feature_groups(X_all: pd.DataFrame) -> pd.DataFrame:
    """Long table of all predefined groups: group, group_type ('family' or 'correlation cluster'), feature."""
    rows = []
    for name, members in family_groups(X_all.columns).items():
        rows += [{"group": name, "group_type": "family", "feature": c} for c in members]
    for name, members in correlation_groups(X_all).items():
        rows += [{"group": name, "group_type": "correlation cluster", "feature": c} for c in members]
    return pd.DataFrame(rows, columns=["group", "group_type", "feature"])
