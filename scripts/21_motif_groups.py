"""Motif features from the HOMER scan of all known motifs: similar motifs are combined into groups.

Steps (no class labels are used anywhere)
  1. Motifs with a match in fewer than --min-prevalence of the enhancers (default 1 %) are dropped.
  2. Similarity between two motifs = Spearman correlation of their best match scores across the enhancers.
     Motifs with similar matrices score the same sequences high, so this measures how redundant they are here.
  3. Average-linkage hierarchical clustering; the tree is cut at correlation --min-correlation (default 0.7).
  4. One feature per group, taken from ONE representative motif: the most central member (highest mean correlation
     with the other members), or a motif named in --prefer if the group contains it (default: the GRE motif derived
     from GR ChIP-seq in A549 cells). The feature is 1 / 0 for motif present / absent (--encoding presence, default)
     or the number of matches (--encoding count). Combining the members instead (motif of any member present)
     was tested and was worse: the least specific member of a group then decides the feature.
  5. For the motifs named in --score-motifs (default: the A549 GRE motif) the best match score of that motif is
     added as a continuous feature (HOMER log-odds score of the best match in the enhancer, also below threshold).

Standalone: needs pandas, numpy and scipy.

    python scripts/21_motif_groups.py --counts motifs.count.txt --scores motifs.bestscore.txt \
        --order ml_inputs/202208_prematrix.csv [--out-dir FOLDER]

Inputs
    --counts   annotatePeaks.pl ... -nmotifs output          --scores   annotatePeaks.pl ... -mscore output
    --order    table that fixes the row order: the prematrix (chr, start, end) or any table with enhancer_id
Outputs (in --out-dir, default: next to --order)
    matrix_motif_groups.tsv     row order and row index of --order; 'enhancer_id' is for checking, not a feature;
                                columns 'motif::<group>' (present / absent or count of the representative motif)
                                and 'motifscore::<motif>' (best match score)
    motif_groups.tsv            every motif: group, group size, prevalence, whether it is the representative,
                                whether it was dropped
    motif_group_summary.tsv     every group: members, prevalence, lowest correlation within the group
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

HOMER_STANDARD = {"Chr", "Start", "End", "Strand", "Peak Score", "Focus Ratio/Region Size", "Annotation",
                  "Detailed Annotation", "Distance to TSS", "Nearest PromoterID", "Entrez ID", "Nearest Unigene",
                  "Nearest Refseq", "Nearest Ensembl", "Gene Name", "Gene Alias", "Gene Description", "Gene Type",
                  "CpG%", "GC%"}
SUFFIX = re.compile(r"\s+(No\. motifs|Best Motif log-odds Score|Distance From Peak\(sequence,strand,conservation\))\s*$")
GRE_MOTIF = "GRE(NR),IR3/A549-GR-ChIP-Seq(GSE32465)/Homer"


def read_order(path):
    d = pd.read_csv(path, sep=None, engine="python", usecols=lambda c: c in ("chr", "start", "end", "enhancer_id"))
    if {"chr", "start", "end"} <= set(d.columns):
        ids = d["chr"].astype(str) + "_" + d["start"].astype(str) + "_" + d["end"].astype(str)
    elif "enhancer_id" in d.columns:
        ids = d["enhancer_id"].astype(str)
    else:
        raise ValueError(f"{path}: needs the columns chr, start, end or enhancer_id")
    if ids.duplicated().any():
        raise ValueError(f"{path}: duplicated enhancers")
    return ids.tolist()


def read_homer(path, ids):
    d = pd.read_csv(path, sep="\t", low_memory=False)
    d = d.rename(columns={d.columns[0]: "enhancer_id"})
    d["enhancer_id"] = d["enhancer_id"].astype(str)
    if d.enhancer_id.duplicated().any():
        raise ValueError(f"{path}: duplicated peak IDs")
    missing = pd.Index(ids).difference(d.enhancer_id)
    if len(missing):
        raise ValueError(f"{path}: {len(missing)} enhancers are not in the HOMER output, e.g. {list(missing[:3])}")
    d = d.set_index("enhancer_id").loc[ids]
    out = d[[c for c in d.columns if c not in HOMER_STANDARD]].rename(columns=lambda c: SUFFIX.sub("", c))
    out = out.apply(pd.to_numeric, errors="coerce")
    if out.isna().any().any() or out.columns.duplicated().any():
        raise ValueError(f"{path}: non-numeric values or duplicated motif names")
    return out


def short(name):
    return name.split("/")[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--counts", type=Path, required=True)
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--order", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--min-prevalence", type=float, default=0.01)
    ap.add_argument("--min-correlation", type=float, default=0.70)
    ap.add_argument("--encoding", choices=["presence", "count"], default="presence")
    ap.add_argument("--prefer", nargs="*", default=[GRE_MOTIF],
                    help="motifs that represent their group when the group contains them (full HOMER names)")
    ap.add_argument("--score-motifs", nargs="*", default=[GRE_MOTIF],
                    help="motifs whose best match score is added as a continuous feature (full HOMER names)")
    a = ap.parse_args()
    out_dir = a.out_dir or a.order.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = read_order(a.order)
    cnt = read_homer(a.counts, ids)
    sc = read_homer(a.scores, ids)
    if list(cnt.columns) != list(sc.columns):
        raise ValueError("the count and score tables do not hold the same motifs in the same order")
    if (cnt < 0).any().any() or (cnt % 1 != 0).any().any():
        raise ValueError(f"{a.counts}: values are not counts - is this the -nmotifs output?")
    print(f"{len(ids):,} enhancers, {cnt.shape[1]} motifs")

    prevalence = (cnt > 0).mean()
    keep = prevalence[prevalence >= a.min_prevalence].index
    print(f"dropped {cnt.shape[1] - len(keep)} motifs with a match in < {100 * a.min_prevalence:g} % of the enhancers")

    R = sc[keep].corr(method="spearman")
    D = 1 - R.to_numpy()
    np.fill_diagonal(D, 0)
    D = np.clip((D + D.T) / 2, 0, None)
    cluster = pd.Series(fcluster(linkage(squareform(D, checks=False), "average"), 1 - a.min_correlation, "distance"),
                        index=keep)

    features, groups, members_of, rep_of, used = {}, [], {}, {}, set()
    for k, mem in cluster.groupby(cluster).groups.items():
        mem = list(mem)
        sub = R.loc[mem, mem]
        preferred = [m for m in a.prefer if m in mem]
        representative = preferred[0] if preferred else sub.mean().idxmax()
        name = short(representative) + (f" group ({len(mem)})" if len(mem) > 1 else "")
        while name in used:                                   # two HOMER motifs can share a short name
            name += "*"
        used.add(name)
        members_of[name], rep_of[name] = mem, representative
        x = cnt[representative]
        features[f"motif::{name}"] = ((x > 0).astype(int) if a.encoding == "presence" else x).to_numpy()
        low = sub.to_numpy()[np.triu_indices(len(mem), 1)].min() if len(mem) > 1 else np.nan
        groups.append({"group": name, "n_motifs": len(mem), "representative_motif": representative,
                       "prevalence": float((x > 0).mean()), "mean_count": float(x.mean()),
                       "lowest_correlation_within_group": low, "members": "; ".join(short(m) for m in mem)})
    for motif in a.score_motifs:
        if motif not in sc.columns:
            raise ValueError(f"--score-motifs: {motif} is not in the HOMER output")
        features[f"motifscore::{short(motif)}"] = sc[motif].to_numpy()
        print(f"best-score feature: {motif}")

    F = pd.DataFrame(features)
    F.insert(0, "enhancer_id", ids)
    F.to_csv(out_dir / "matrix_motif_groups.tsv", sep="\t")

    G = pd.DataFrame(groups).sort_values(["n_motifs", "prevalence"], ascending=False)
    G.round(4).to_csv(out_dir / "motif_group_summary.tsv", sep="\t", index=False)
    name_of = {m: g for g, mem in members_of.items() for m in mem}
    M = pd.DataFrame({"motif": cnt.columns, "prevalence": prevalence.reindex(cnt.columns).to_numpy()})
    M["dropped_low_prevalence"] = ~M.motif.isin(keep)
    M["group"] = M.motif.map(name_of)
    M["group_size"] = M.group.map(G.set_index("group").n_motifs)
    M["is_representative"] = M.motif.isin(set(rep_of.values()))
    M.round(4).to_csv(out_dir / "motif_groups.tsv", sep="\t", index=False)

    n_multi = int((G.n_motifs > 1).sum())
    print(f"{len(keep)} motifs -> {len(G)} groups ({n_multi} with more than one motif, largest {int(G.n_motifs.max())})")
    print(f"wrote {out_dir / 'matrix_motif_groups.tsv'}  ({F.shape[1] - 1} features, encoding: {a.encoding})")
    pd.set_option("display.width", 220, "display.max_colwidth", 110)
    print("\nlargest groups:")
    print(G.head(15)[["group", "n_motifs", "prevalence", "lowest_correlation_within_group", "members"]]
          .to_string(index=False, float_format=lambda v: f"{v:.2f}"))
