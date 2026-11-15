"""ABC enhancer-gene links of A549 cells (Nasser et al. 2021), lifted from hg19 to hg38.

The published ABC predictions are in hg19 and cover 131 biosamples. This script keeps the A549 rows and lifts
three positions per link to hg38 with pyliftover (UCSC chain hg19 -> hg38): the start and the end of the
element and the TSS of the target gene. A link is kept when all three positions lift to the same chromosome
and the lifted end lies after the lifted start. In the published analysis 59,328 of the 59,634 A549 links
(biosample A549_treated_with_ethanol_0.02_percent_for_1_hour-Roadmap) were kept.

    uv run python scripts/12_lift_abc_to_hg38.py

Input   <enhancer_gene>/AllPredictions.AvgHiC.ABC0.015.minus150.ForABCPaperV3.txt.gz  (or --source)
Output  the file named by [layout] abc_links, default <enhancer_gene>/ABC_Nasser2021_A549_hg38.tsv.gz:
        all columns of the source plus start_hg38, end_hg38 and TSS_hg38. figures/fig4_enhancer_gene_links.py
        reads it.

pyliftover downloads the chain file on first use, so this step needs an internet connection once.
"""
import argparse
import gzip
from pathlib import Path

import pandas as pd
from pyliftover import LiftOver

from a549_enhancers import paths

SOURCE_FILE = "AllPredictions.AvgHiC.ABC0.015.minus150.ForABCPaperV3.txt.gz"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default=None, help="ABC prediction file of all biosamples (hg19)")
    p.add_argument("--output", default=None, help="lifted A549 table to write")
    return p.parse_args()


def main():
    args = parse_args()
    source = Path(args.source) if args.source else paths.ENHANCER_GENE / SOURCE_FILE
    output = Path(args.output) if args.output else paths.ABC_LINKS

    # 1. Stream the file and keep A549 rows only
    with gzip.open(source, "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
        ct = header.index("CellType")
        rows = [l.rstrip("\n").split("\t") for l in f if "a549" in l.split("\t")[ct].lower()]
    abc = pd.DataFrame(rows, columns=header)
    print("A549 biosamples:", abc.CellType.value_counts().to_dict())

    # 2. Lift hg19 -> hg38 (element start/end and target-gene TSS)
    lo = LiftOver("hg19", "hg38")

    def lift(chrom, pos):
        r = lo.convert_coordinate(chrom, int(pos))
        return r[0][1] if r and r[0][0] == chrom else None

    abc["start_hg38"] = [lift(c, s) for c, s in zip(abc.chr, abc.start)]
    abc["end_hg38"] = [lift(c, e) for c, e in zip(abc.chr, abc.end)]
    abc["TSS_hg38"] = [lift(c, t) for c, t in zip(abc.chr, abc.TargetGeneTSS)]
    ok = abc[["start_hg38", "end_hg38", "TSS_hg38"]].notna().all(axis=1) & \
        (abc.end_hg38.astype(float) > abc.start_hg38.astype(float))
    print(f"lifted {ok.sum():,} / {len(abc):,} rows")

    output.parent.mkdir(parents=True, exist_ok=True)
    abc[ok].to_csv(output, sep="\t", index=False)
    print("Saved:", output)


if __name__ == "__main__":
    main()
