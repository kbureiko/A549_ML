"""Add the GR-binding column to the feature matrix of all enhancers.

An enhancer is GR-bound when it overlaps a peak of the GR (NR3C1) ChIP-seq peak file ENCFF395FIP (ENCODE,
A549 cells, hg38) by at least one base pair. With the published enhancer set this marks 2,743 of the 8,270
enhancers.

    uv run python scripts/08_annotate_gr_binding.py

Input
    <ml_outputs>/full_feature_matrix/feature_matrix_all_classes.csv
        (scripts/07_save_full_feature_matrix.py; another file with --matrix)
    <ml_inputs>/ENCFF395FIP.bed.gz    narrowPeak file from ENCODE (another file with --peaks)
Output
    <ml_outputs>/feature_matrix_all_classes_GRannotated.csv   the table every figure script reads (--output)
    <ml_outputs>/a549_enhancers_GRbound.bed                    the GR-bound enhancers (chr, start, end, ID, class)
"""
import argparse
from pathlib import Path

import pandas as pd

from a549_enhancers import paths
from a549_enhancers.intervals import overlaps_any, parse_enhancer_ids, read_peaks

FULL_MATRIX = Path("full_feature_matrix") / "feature_matrix_all_classes.csv"
PEAK_FILE = "ENCFF395FIP.bed.gz"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", default=None, help="feature matrix of all classes (CSV with enhancer_id and class)")
    p.add_argument("--peaks", default=None, help="GR peak file (BED / narrowPeak, may be gzipped)")
    p.add_argument("--output", default=None, help="annotated matrix to write")
    p.add_argument("--min-overlap", type=int, default=1, help="shared base pairs needed to call an overlap")
    return p.parse_args()


def main():
    args = parse_args()
    matrix = Path(args.matrix) if args.matrix else paths.ML / FULL_MATRIX
    peak_file = Path(args.peaks) if args.peaks else paths.ML_INPUTS / PEAK_FILE
    output = Path(args.output) if args.output else paths.ML / "feature_matrix_all_classes_GRannotated.csv"

    df = pd.read_csv(matrix, low_memory=False)
    if not {"enhancer_id", "class"} <= set(df.columns):
        raise ValueError(f"{matrix} needs the columns enhancer_id and class")
    if df["enhancer_id"].duplicated().any():
        raise ValueError("duplicated enhancer IDs in the feature matrix")

    coords = parse_enhancer_ids(df["enhancer_id"])
    peaks = read_peaks(peak_file)
    df["GR_bound"] = overlaps_any(coords, peaks, min_overlap=args.min_overlap)

    print(f"{len(peaks):,} peaks in {peak_file.name}; {int(df['GR_bound'].sum()):,} of {len(df):,} enhancers GR-bound")
    print(pd.crosstab(df["class"], df["GR_bound"], margins=True))

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    bed = pd.concat([coords, df[["enhancer_id", "class"]]], axis=1)[df["GR_bound"].to_numpy()]
    bed.to_csv(output.parent / "a549_enhancers_GRbound.bed", sep="\t", header=False, index=False)
    print("Saved:", output)


if __name__ == "__main__":
    main()
