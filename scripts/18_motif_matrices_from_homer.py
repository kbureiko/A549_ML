"""Motif tables for all HOMER known motifs: number of matches and best match score per enhancer.

Two steps around the HOMER run. The HOMER commands between them (all known vertebrate motifs):

    KNOWN=<homer>/data/knownTFs/vertebrates/known.motifs
    annotatePeaks.pl enhancers.peaks.txt hg38 -size given -m $KNOWN -nmotifs -rmrevopp 5 -noann -nogene > motifs.count.txt
    annotatePeaks.pl enhancers.peaks.txt hg38 -size given -m $KNOWN -mscore -noann -nogene > motifs.bestscore.txt
    annotatePeaks.pl enhancers.peaks.txt hg38 -size given -CpG -noann -nogene > enhancers.cpg.txt


    1. peaks    writes a HOMER peak file from the prematrix, with the enhancer ID of the feature matrix
                (chr_start_end) as peak ID, so that the HOMER output can be joined on the ID.

        uv run python scripts/18_motif_matrices_from_homer.py peaks --out enhancers.peaks.txt

    2. matrix   turns the HOMER outputs (annotatePeaks.pl -nmotifs / -mscore, optionally -CpG) into tables in the
                row order of the prematrix. The rows are matched on the enhancer ID, never on the row position;
                the script stops if an enhancer is missing or duplicated in a HOMER output.

        uv run python scripts/18_motif_matrices_from_homer.py matrix \
            --counts motifs.count.txt --scores motifs.bestscore.txt --cpg enhancers.cpg.txt --out-dir <ml_inputs>

Outputs of step 2 (tab-separated, row index = row number of the prematrix, as the other input tables;
the first data column 'enhancer_id' is there for checking and is not a feature):
    matrix_motifs_counts.tsv        one column per motif: number of matches above the motif's detection threshold
    matrix_motifs_bestscore.tsv     one column per motif: highest log-odds score in the enhancer
    matrix_cpg.tsv                  CpG and GC content (if --cpg is given)
    motif_matrix_summary.tsv        per motif: share of enhancers with a match, mean count, median best score

The prematrix is read from [inputs] ml_inputs in paths.toml (or --prematrix FILE). Nothing in the modelling
pipeline reads the new tables yet; the present / absent table matrix_motifs.csv is left untouched.
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# columns that annotatePeaks.pl writes before the motif columns
HOMER_STANDARD = {"Chr", "Start", "End", "Strand", "Peak Score", "Focus Ratio/Region Size", "Annotation",
                  "Detailed Annotation", "Distance to TSS", "Nearest PromoterID", "Entrez ID", "Nearest Unigene",
                  "Nearest Refseq", "Nearest Ensembl", "Gene Name", "Gene Alias", "Gene Description", "Gene Type",
                  "CpG%", "GC%"}
SUFFIX = re.compile(r"\s+(No\. motifs|Best Motif log-odds Score|Distance From Peak\(sequence,strand,conservation\))\s*$")


def read_prematrix(path):
    if path is None:
        from a549_enhancers.config import PREMATRIX_FILE
        from a549_enhancers.paths import ml_input
        path = ml_input(PREMATRIX_FILE)
    pre = pd.read_csv(path, sep="\t", usecols=["chr", "start", "end"])
    pre["enhancer_id"] = pre["chr"].astype(str) + "_" + pre["start"].astype(str) + "_" + pre["end"].astype(str)
    if pre.enhancer_id.duplicated().any():
        sys.exit(f"{path}: {int(pre.enhancer_id.duplicated().sum())} duplicated enhancer IDs")
    print(f"prematrix: {len(pre)} enhancers ({path})")
    return pre


def read_homer(path, ids, what):
    """HOMER annotatePeaks table -> motif (or CpG) columns, rows in the order of `ids` (matched on the peak ID)."""
    d = pd.read_csv(path, sep="\t", low_memory=False)
    d = d.rename(columns={d.columns[0]: "enhancer_id"})
    d["enhancer_id"] = d["enhancer_id"].astype(str)
    if d.enhancer_id.duplicated().any():
        sys.exit(f"{path}: {int(d.enhancer_id.duplicated().sum())} duplicated peak IDs")
    missing = pd.Index(ids).difference(d.enhancer_id)
    if len(missing):
        sys.exit(f"{path}: {len(missing)} enhancers of the prematrix are not in the HOMER output, e.g. "
                 f"{list(missing[:3])}. Was the peak file written by the 'peaks' step of this script?")
    extra = len(d) - len(ids)
    d = d.set_index("enhancer_id").loc[ids]
    if what == "cpg":
        cols = [c for c in ("CpG%", "GC%") if c in d.columns]
        if not cols:
            sys.exit(f"{path}: no CpG% / GC% columns (run annotatePeaks.pl with -CpG)")
        out = d[cols].rename(columns={"CpG%": "CpG_content", "GC%": "GC_content"})
    else:
        cols = [c for c in d.columns if c not in HOMER_STANDARD]
        if not cols:
            sys.exit(f"{path}: no motif columns found")
        out = d[cols].rename(columns=lambda c: SUFFIX.sub("", c))
        if out.columns.duplicated().any():
            sys.exit(f"{path}: duplicated motif names: {out.columns[out.columns.duplicated()].tolist()[:5]}")
    out = out.apply(pd.to_numeric, errors="coerce")
    if out.isna().any().any():
        bad = out.columns[out.isna().any()].tolist()
        sys.exit(f"{path}: non-numeric or empty values in {len(bad)} column(s), e.g. {bad[:3]}")
    print(f"{path}: {out.shape[1]} columns for {len(out)} enhancers" + (f" ({extra} other rows ignored)" if extra else ""))
    return out


def write(table, ids, path):
    out = table.reset_index(drop=True)                      # row index = row number of the prematrix
    out.insert(0, "enhancer_id", list(ids))
    out.to_csv(path, sep="\t")
    print(f"wrote {path}  ({table.shape[1]} columns)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)
    p = sub.add_parser("peaks", help="HOMER peak file from the prematrix")
    p.add_argument("--prematrix", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path("enhancers.peaks.txt"))
    m = sub.add_parser("matrix", help="HOMER outputs -> motif tables in prematrix row order")
    m.add_argument("--prematrix", type=Path, default=None)
    m.add_argument("--counts", type=Path, help="annotatePeaks.pl ... -nmotifs output")
    m.add_argument("--scores", type=Path, help="annotatePeaks.pl ... -mscore output")
    m.add_argument("--cpg", type=Path, help="annotatePeaks.pl ... -CpG output")
    m.add_argument("--out-dir", type=Path, default=Path("."))
    a = ap.parse_args()
    pre = read_prematrix(a.prematrix)
    ids = pre.enhancer_id.tolist()

    if a.step == "peaks":
        # HOMER peak file: ID, chr, start, end, strand. The prematrix coordinates are treated as BED
        # (0-based start), so 1 is added to the start; the ID keeps the prematrix coordinates.
        peaks = pd.DataFrame({"id": ids, "chr": pre.chr, "start": pre.start.astype(int) + 1,
                              "end": pre.end.astype(int), "strand": "+"})
        peaks.to_csv(a.out, sep="\t", header=False, index=False)
        print(f"wrote {a.out}  ({len(peaks)} enhancers)")
        sys.exit(0)

    if not (a.counts or a.scores or a.cpg):
        sys.exit("give at least one of --counts, --scores, --cpg")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    counts = read_homer(a.counts, ids, "motif") if a.counts else None
    scores = read_homer(a.scores, ids, "motif") if a.scores else None
    if counts is not None:
        if (counts < 0).any().any() or (counts % 1 != 0).any().any():
            sys.exit(f"{a.counts}: values are not counts - is this the -nmotifs output?")
        write(counts.astype(int), ids, a.out_dir / "matrix_motifs_counts.tsv")
    if scores is not None:
        write(scores, ids, a.out_dir / "matrix_motifs_bestscore.tsv")
    if a.cpg:
        write(read_homer(a.cpg, ids, "cpg"), ids, a.out_dir / "matrix_cpg.tsv")
    if counts is not None and scores is not None and set(counts.columns) != set(scores.columns):
        only = sorted(set(counts.columns) ^ set(scores.columns))
        print(f"WARNING: {len(only)} motifs are in only one of the two tables, e.g. {only[:3]}")

    rows = {}
    if counts is not None:
        rows["share_of_enhancers_with_match"] = (counts > 0).mean()
        rows["mean_count"] = counts.mean()
        rows["max_count"] = counts.max()
    if scores is not None:
        rows["median_best_score"] = scores.median()
        rows["sd_best_score"] = scores.std()
    if rows:
        S = pd.DataFrame(rows).rename_axis("motif").sort_values(list(rows)[0], ascending=False)
        S.round(4).to_csv(a.out_dir / "motif_matrix_summary.tsv", sep="\t")
        print(f"wrote {a.out_dir / 'motif_matrix_summary.tsv'}")
        if counts is not None:
            sh = rows["share_of_enhancers_with_match"]
            print(f"motifs: {len(S)} | with a match in < 1 % of enhancers: {int((sh < 0.01).sum())} | "
                  f"in > 90 %: {int((sh > 0.9).sum())}")
