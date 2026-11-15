# =============================================================================
# Figure 4 - Genes linked to UP and DN enhancers respond to dex in the corresponding direction
#   A  ABC enhancer-gene links (Nasser et al. 2021, A549): enhancers with a linked gene per class (rows under the
#      bars: linked enhancers and all enhancers of the class), and the distance from the enhancer to the TSS of
#      its top-linked gene (violin; box = median and quartiles). Numbers above the violins = median number of
#      linked genes per linked enhancer.
#   B  dex response of top-linked genes in siNON cells: % induced (up, solid) and % repressed (down, hatched) at
#      1 h (PRO-seq, nascent transcription); n = top-linked genes tested in the assay
#   C  genes by number of linked UP (left) or DN (right) enhancers -> % induced (repressed). All links of a gene
#      are counted, not only top links; genes tested in the assay and linked to >= 1 enhancer; error bars = 95 %
#      Wilson interval; tested genes per bin under the axis (one row per assay)
#   D  time course of top-linked genes (GSE104714, 1-11 h dex): median log2(dex / time-matched EtOH), band = 95 %
#      bootstrap interval over genes; genes with CPM > 1 in >= 4 samples, hence fewer genes than in B; the class
#      is also written at the end of each line
#   E  example loci (induced DUSP1, FKBP5; repressed CYP24A1, LIF). Tracks: public PRO-seq, EtOH and dex 45 min,
#      mean of 2 replicates per 10 M tags, square-root scale, one scale per locus for both conditions and strands,
#      symmetric around 0 (only the upper limit is labelled); reads map antisense to the RNA, so a minus-strand
#      gene is read on the plus track. Below: enhancers (UP and DN full height, NR half height; triangle =
#      GR-bound), ABC links to the named gene (arc width follows the ABC score), gene model. A dotted "ABC TSS"
#      tick marks the TSS used by ABC when it differs from the annotated 5' end; grey ticks = TSSs of other
#      genes. Titles: log2FC of the gene in siNON cells per assay.
# Links: ABC predictions for A549 (Nasser et al. 2021; the published set is thresholded at ABC score >= 0.015),
#   self-promoter links excluded; an enhancer is "linked" when its dREG interval overlaps an ABC element (>= 1 bp).
#   Top-linked gene = the gene with the highest ABC score for that enhancer. When several genes share the highest
#   score (gene pairs with a shared promoter, mostly TSSs within 1 kb; the cell prints how many enhancers), all of
#   them count as top-linked; the link distance in A uses the nearest of them.
# Gene classes (B, D) are mutually exclusive: a gene is UP-linked (DN-linked) if it is the top-linked gene of
#   >= 1 UP (DN) enhancer, and NR-linked if all its top-linking enhancers are NR (genes top-linked to NR and to
#   UP/DN enhancers count as UP/DN; the cell prints how many, and whether any gene is top-linked to both UP and DN).
# Induced / repressed: DESeq2 padj < PADJ and |log2FC| > LFC, dex vs EtOH in siNON cells.
# ASSAYS: the 1 h PRO-seq only (the BRB-seq experiments come after this figure in the text). Uncomment the BRB-seq
#   line to add the 6 h mRNA data to B, C and the locus titles.
# Inputs: ABC predictions, feature matrix (class, GR_bound), gene-level DESeq2 exports (siNON arms), GSE104714
#   transcript counts + tx2gene, optional locus tracks (scripts/13_export_locus_tracks.py; E is drawn without them)
# Outputs (FIGURES): Fig_4.pdf/png; per-panel PDFs Fig_4_A..E.pdf and Fig_4_E_key.pdf (no letters);
#   Fig_4_ABC_enhancer_gene_links.tsv (all links), Fig_4_panel_numbers.tsv (every number shown in A-D)
# =============================================================================
# %%
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Arc, Rectangle, Patch
from matplotlib.lines import Line2D
from scipy.stats import fisher_exact, norm

from a549_enhancers.paths import PROJECT, ML, FIGURES, ENHANCER_GENE, ABC_LINKS, KNOCKDOWN   # local folders, set in paths.toml
FINAL = FIGURES; FINAL.mkdir(parents=True, exist_ok=True)   # final figures + their tables
A5 = ENHANCER_GENE
ABC = ABC_LINKS
FM = ML / "feature_matrix_all_classes_GRannotated.csv"
TC = A5 / "GSE104714_DEX_EtOH_expression.tsv.gz"             # transcript counts, time course
TX2GENE = A5 / "tx2gene_GSE104714.tsv"
TRACKS = A5 / "A5_locus_PROseq_tracks.tsv.gz"                 # optional; panel E draws without it


def find(d, pattern):
    hits = sorted(Path(d).glob(pattern))
    return hits[-1] if hits else Path(d) / pattern


ASSAYS = [  # (label, short label, DESeq2 table: dex vs EtOH in siNON cells)
    ("1 h nascent\n(PRO-seq)", "1 h", find(KNOCKDOWN / "panel_export_PROseq_siEP300", "*PROseq_DESeq2_siNON_dex_vs_EtOH.tsv")),
    # ("6 h mRNA\n(BRB-seq)", "6 h", find(KNOCKDOWN / "panel_export_BRBseq_siFOXA1_siTEAD4", "*BRBseq_DESeq2_siNON_Dex_vs_Veh.tsv")),
]
PADJ, LFC = 0.05, 0.5          # gene-level dex sets, as for the knockdown gene sets
LOCI = {"DUSP1": ("chr5", 172_700_000, 172_930_000, 172_768_090, 172_771_195, "-"),   # GENCODE gene span (hg38)
        "FKBP5": ("chr6", 35_540_000, 35_760_000, 35_573_585, 35_728_583, "-"),
        "CYP24A1": ("chr20", 54_095_000, 54_190_000, 54_153_446, 54_173_986, "-"),
        "LIF": ("chr22", 30_185_000, 30_268_000, 30_240_447, 30_246_851, "-")}
LOCI_UP, LOCI_DN = ["DUSP1", "FKBP5"], ["CYP24A1", "LIF"]
N_BOOT, SEED = 1000, 0

missing = [str(p) for p in (ABC, FM, TC, TX2GENE, *[a[2] for a in ASSAYS]) if not p.exists()]
if missing:
    raise FileNotFoundError("Not found (edit paths at top):\n  " + "\n  ".join(missing))

# ----------------------------------------------------------------- shared A549 style
plt.style.use("default")       # reset first: a style left active by an earlier cell (grid, grey spines) must not leak in
FONT = "Open Sans" if any("Open Sans" in f.name for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
mpl.rcParams.update({
    "font.family": [FONT, "DejaVu Sans"], "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})
CLS = {"UP": "#D1495B", "DN": "#3B6FB6", "NR": "#5DAA82"}
ORDER = ["UP", "NR", "DN"]
STRAND = {"plus": "#C2185B", "minus": "#0B84A5"}
GREY = "#555555"


def panel_letters(fig, axes):
    fig.canvas.draw(); r = fig.canvas.get_renderer(); out = []
    for letter, ax in axes.items():
        bb = ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
        out.append(fig.text(bb.x0 - 0.004, bb.y1 + 0.003, letter, fontsize=11, fontweight="bold", ha="left", va="bottom"))
    return out


def export_panels(fig, panels, stem, letters, pad=0.04):
    for t in letters: t.set_visible(False)
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    for name, artists in panels.items():
        boxes = [a.get_tightbbox(r) if hasattr(a, "get_tightbbox") else a.get_window_extent(r) for a in artists]
        bb = mpl.transforms.Bbox.union(boxes).transformed(fig.dpi_scale_trans.inverted()).padded(pad)
        fig.savefig(FINAL / f"{stem}_{name}.pdf", bbox_inches=bb)
    for t in letters: t.set_visible(True)


def tracked(fig, fn, *a, **k):
    n_ax, n_tx = len(fig.axes), len(fig.texts)
    ret = fn(*a, **k)
    return ret, list(fig.axes[n_ax:]) + list(fig.texts[n_tx:])


def class_ticks(ax, positions, labels):
    ax.set_xticks(positions); ax.set_xticklabels(labels)
    for t, c in zip(ax.get_xticklabels(), labels):
        t.set_color(CLS[c]); t.set_fontweight("bold")


NUM = []                                   # every number shown in the panels -> Fig_4_panel_numbers.tsv
def keep(panel, group, assay, n, metric, value):
    NUM.append(dict(panel=panel, group=group, assay=assay, n=n, metric=metric, value=value))


# ----------------------------------------------------------------- data: enhancers and ABC links
fm = pd.read_csv(FM, usecols=["enhancer_id", "class", "GR_bound"], low_memory=False)
e = fm.enhancer_id.str.extract(r"(chr\w+)_(\d+)_(\d+)")
fm["chr"], fm["s"], fm["e"] = e[0], e[1].astype(int), e[2].astype(int)
n_cls = fm["class"].value_counts()

abc_all = pd.read_csv(ABC, sep="\t")
abc = abc_all[~abc_all.isSelfPromoter].copy()
el = abc[["chr", "start_hg38", "end_hg38", "name"]].drop_duplicates("name")
hits = []
for c, g in fm.groupby("chr"):
    a = el[el.chr == c].sort_values("start_hg38")
    if a.empty: continue
    st, en, nm = a.start_hg38.values, a.end_hg38.values, a.name.values
    width = int((en - st).max()) + 1
    for eid, s, t in zip(g.enhancer_id, g.s, g.e):
        i = np.searchsorted(st, t); j = np.searchsorted(st, s - width)
        m = (st[j:i] < t) & (en[j:i] > s)
        hits += [(eid, n) for n in nm[j:i][m]]
links = (pd.DataFrame(hits, columns=["enhancer_id", "name"])
         .merge(abc[["name", "TargetGene", "ABC.Score", "distance", "TSS_hg38"]], on="name")
         .merge(fm[["enhancer_id", "class", "GR_bound"]], on="enhancer_id"))
links.to_csv(FINAL / "Fig_4_ABC_enhancer_gene_links.tsv", sep="\t", index=False)
# top-linked gene(s) of each enhancer: all genes that share its highest ABC score (ties are gene pairs with a shared
# promoter, which ABC scores identically), so the result does not depend on the sort order
top = links[links["ABC.Score"] == links.groupby("enhancer_id")["ABC.Score"].transform("max")] \
    .drop_duplicates(["enhancer_id", "TargetGene"])
top1 = top.sort_values(["enhancer_id", "distance", "TargetGene"], kind="mergesort").drop_duplicates("enhancer_id")
_n_tied = int(top.groupby("enhancer_id").TargetGene.nunique().gt(1).sum())
_gcls = top.groupby("TargetGene")["class"].agg(set)
_n_both = int(_gcls.map(lambda s: "UP" in s and "DN" in s).sum())
gene_class = _gcls.map(lambda s: "UP" if "UP" in s else ("DN" if "DN" in s else "NR"))
tg = pd.DataFrame({"TargetGene": gene_class.index, "class": gene_class.values})          # exclusive classes
overlap = {"UP & NR": int(_gcls.map(lambda s: {"UP", "NR"} <= s).sum()),
           "DN & NR": int(_gcls.map(lambda s: {"DN", "NR"} <= s).sum()), "UP & DN": _n_both}
print("top-linked genes per class (exclusive):", tg["class"].value_counts().to_dict(),
      "| genes top-linked to enhancers of two classes:", overlap,
      f"| enhancers whose highest score is shared by several genes: {_n_tied} of {top.enhancer_id.nunique():,}")
if _n_both:
    print(f"NOTE: {_n_both} genes are top-linked to both UP and DN enhancers; they are counted as UP")

# ----------------------------------------------------------------- data: gene responses
DE = {lab: pd.read_csv(f, sep="\t").set_index("gene") for lab, _, f in ASSAYS}


def gene_sets(d):
    ok = d.padj.notna()
    return (set(d.index[ok & (d.padj < PADJ) & (d.log2FoldChange > LFC)]),
            set(d.index[ok & (d.padj < PADJ) & (d.log2FoldChange < -LFC)]), set(d.index[ok]))


SETS = {lab: gene_sets(d) for lab, d in DE.items()}
for lab, (up, dn, tested) in SETS.items():
    print(f"{lab.replace(chr(10), ' ')}: {len(tested):,} genes tested, {len(up):,} induced, {len(dn):,} repressed")

# time course: transcripts -> genes, CPM, log2(dex / time-matched EtOH)
X = pd.read_csv(TC, sep="\t", index_col=0)
X = X.loc[X.index.str.startswith("ENST"), [c for c in X.columns if re.match(r"(DEX|EtOH)_\d+hr_Rep\d", c)]]
X.index = X.index.str.split(".").str[0]
t2g = pd.read_csv(TX2GENE, sep="\t").drop_duplicates("tx_id").set_index("tx_id").gene_name
G = X.groupby(t2g.reindex(X.index).values).sum()
cpm = G / G.sum() * 1e6
cpm = cpm[(cpm > 1).sum(axis=1) >= 4]
HOURS = sorted({int(re.search(r"_(\d+)hr", c).group(1)) for c in cpm.columns})
tc = pd.DataFrame({h: np.log2((cpm.filter(regex=f"^DEX_{h}hr").mean(axis=1) + 1) /
                              (cpm.filter(regex=f"^EtOH_{h}hr").mean(axis=1) + 1)) for h in HOURS})
print(f"font: {FONT} | enhancers with >=1 ABC link: {links.enhancer_id.nunique():,} of {len(fm):,} | "
      f"time-course genes: {len(tc):,}")


# ----------------------------------------------------------------- A: ABC coverage and link distance
def plot_A(fig, spec):
    sub = spec.subgridspec(1, 2, width_ratios=[1, 1.15], wspace=1.0)
    ax = fig.add_subplot(sub[0])
    cov = links.groupby("class").enhancer_id.nunique().reindex(ORDER) / n_cls.reindex(ORDER) * 100
    ax.bar(range(3), cov, color=[CLS[c] for c in ORDER], width=0.62)
    tr = ax.get_xaxis_transform()
    for i, (c, v) in enumerate(cov.items()):
        nl = links.loc[links["class"] == c, "enhancer_id"].nunique()
        ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=6.4)
        ax.text(i, -0.17, f"{nl:,}", ha="center", va="top", fontsize=5.6, color=GREY, transform=tr)
        ax.text(i, -0.255, f"{n_cls[c]:,}", ha="center", va="top", fontsize=5.6, color=GREY, transform=tr)
        keep("A", c, "", int(n_cls[c]), "enhancers with an ABC-linked gene (%)", round(float(v), 1))
        keep("A", c, "", int(nl), "linked enhancers (n)", int(nl))
    for y, lab in ((-0.17, "linked"), (-0.255, "all")):               # row labels of the counts
        ax.text(-0.62, y, lab, ha="right", va="top", fontsize=5.6, color=GREY, transform=tr)
    class_ticks(ax, range(3), ORDER)
    ax.set_ylim(0, 80); ax.set_ylabel("Enhancers with an\nABC-linked gene (%)")
    ax.set_title("Linked enhancers", fontsize=7.6)
    ax2 = fig.add_subplot(sub[1])
    for i, c in enumerate(ORDER):
        dist = top1.loc[top1["class"] == c, "distance"]                              # one value per enhancer
        d = np.log10(dist.clip(lower=100))                                           # distance of the top link
        vp = ax2.violinplot(d, positions=[i], widths=0.75, showextrema=False)
        for b in vp["bodies"]: b.set_facecolor(CLS[c]); b.set_alpha(0.35); b.set_edgecolor("none")
        ax2.boxplot(d, positions=[i], widths=0.18, showfliers=False, medianprops=dict(color="black", lw=1),
                    boxprops=dict(lw=0.6), whiskerprops=dict(lw=0.6), capprops=dict(lw=0))
        g = links.loc[links["class"] == c].groupby("enhancer_id").TargetGene.nunique().median()
        ax2.text(i, 6.42, f"{g:.0f}", ha="center", fontsize=6.8, color=CLS[c], fontweight="bold")
        keep("A", c, "", len(dist), "median distance to the TSS of the top-linked gene (kb)", round(float(dist.median()) / 1000, 1))
        keep("A", c, "", len(dist), "median linked genes per linked enhancer", float(g))
    ax2.text(1, 6.78, "median linked genes / linked enhancer", ha="center", fontsize=5.4, color=GREY)
    class_ticks(ax2, range(3), ORDER)
    ax2.set_yticks([3, 4, 5, 6]); ax2.set_yticklabels(["1 kb", "10 kb", "100 kb", "1 Mb"])
    ax2.set_ylim(2.3, 7.0); ax2.set_ylabel("Enhancer to TSS of its\ntop-linked gene")
    ax2.spines["left"].set_bounds(2.5, 6.2)
    ax2.set_title("Link distance", fontsize=7.6, pad=10)
    return ax


# ----------------------------------------------------------------- B: linked-gene responses
def plot_B(ax):
    x0, xt, xl = 0.0, [], []
    tr = ax.get_xaxis_transform()
    for lab, short, _ in ASSAYS:
        up, dn, tested = SETS[lab]; one = lab.replace("\n", " ")
        for i, c in enumerate(ORDER):
            g = tg.loc[(tg["class"] == c) & tg.TargetGene.isin(tested), "TargetGene"]
            pu, pd_ = g.isin(up).mean() * 100, g.isin(dn).mean() * 100
            x = x0 + i
            ax.bar(x, pu, color=CLS[c], width=0.72)
            ax.bar(x, -pd_, color=CLS[c], width=0.72, alpha=0.45, hatch="////", edgecolor="white", lw=0)
            ax.text(x, pu + 1, f"{pu:.0f}", ha="center", va="bottom", fontsize=6)
            ax.text(x, -pd_ - 1, f"{pd_:.0f}", ha="center", va="top", fontsize=6)
            ax.text(x, -0.15, f"{len(g):,}", ha="center", va="top", fontsize=5.6, color=GREY, transform=tr)
            xt.append(x); xl.append(c)
            keep("B", f"{c}-linked", one, len(g), "induced (%)", round(float(pu), 1))
            keep("B", f"{c}-linked", one, len(g), "repressed (%)", round(float(pd_), 1))
        gN = tg.loc[(tg["class"] == "NR") & tg.TargetGene.isin(tested), "TargetGene"]     # odds ratios vs NR-linked
        for c, s, what in (("UP", up, "induced"), ("DN", dn, "repressed")):
            g = tg.loc[(tg["class"] == c) & tg.TargetGene.isin(tested), "TargetGene"]
            orr, p = fisher_exact([[g.isin(s).sum(), (~g.isin(s)).sum()], [gN.isin(s).sum(), (~gN.isin(s)).sum()]])
            keep("B", f"{c}- vs NR-linked", one, len(g) + len(gN), f"{what}: odds ratio (Fisher)", round(float(orr), 2))
            keep("B", f"{c}- vs NR-linked", one, len(g) + len(gN), f"{what}: Fisher P", float(f"{p:.3g}"))
        ax.text(x0 + 1, -0.27, lab, ha="center", va="top", fontsize=6.6, transform=tr, linespacing=1.1)
        x0 += 3.7
    ax.axhline(0, color="#333333", lw=0.6)
    class_ticks(ax, xt, xl)
    ax.tick_params(axis="x", length=0, pad=2)
    pad = 0.75 if len(ASSAYS) > 1 else 1.9                     # keeps the bar width similar with one assay
    ax.set_xlim(-pad, xt[-1] + pad)
    ax.set_ylim(-40, 62); ax.set_yticks([-40, -20, 0, 20, 40, 60]); ax.set_yticklabels(["40", "20", "0", "20", "40", "60"])
    ax.set_ylabel("Top-linked genes (%)\nrepressed            induced", fontsize=7)
    ax.text(-0.2 - pad, -0.15, "n genes", fontsize=5.6, color=GREY, va="top", ha="right", transform=tr)
    ax.set_title("Dex response of top-linked genes", fontsize=7.6, pad=14)
    h = [Patch(facecolor="#999999", label="induced"),
         Patch(facecolor="#999999", alpha=0.45, hatch="////", edgecolor="white", lw=0, label="repressed")]
    ax.legend(handles=h, frameon=False, fontsize=5.8, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              handlelength=1.1, columnspacing=1.0, handletextpad=0.35, borderaxespad=0.0)
    return ax


# ----------------------------------------------------------------- C: number of linked enhancers
def wilson(k, n, z=norm.ppf(0.975)):
    """Wilson 95% interval (in %) for k successes of n"""
    k, n = np.asarray(k, float), np.asarray(n, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = k / n; den = 1 + z ** 2 / n
        c = (p + z ** 2 / (2 * n)) / den
        h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return (c - h) * 100, (c + h) * 100


def plot_C(fig, spec):
    sub = spec.subgridspec(1, 2, wspace=0.5)
    axes = []
    bins, labs = [-1, 0, 1, 2, 10_000], ["0", "1", "2", "≥3"]
    off = 0.06 if len(ASSAYS) > 1 else 0.0
    for k, (cls, which) in enumerate((("UP", 0), ("DN", 1))):
        ax = fig.add_subplot(sub[k]); axes.append(ax)
        what = "induced" if cls == "UP" else "repressed"
        for j, (lab, short, _) in enumerate(ASSAYS):
            sets = SETS[lab]; target, tested = sets[which], sets[2]
            L = links[links.TargetGene.isin(tested)]
            genes = pd.Index(sorted(set(L.TargetGene)))
            cnt = L[L["class"] == cls].groupby("TargetGene").enhancer_id.nunique().reindex(genes, fill_value=0)
            b = pd.cut(cnt, bins, labels=labs)
            kk = cnt.groupby(b, observed=False).apply(lambda s: int(s.index.isin(list(target)).sum())).reindex(labs)
            n = b.value_counts().reindex(labs)
            frac = kk / n * 100
            lo, hi = wilson(kk.values, n.values)
            xs = np.arange(4) + (-off if j == 0 else off)
            ax.errorbar(xs, frac.values, yerr=[frac.values - lo, hi - frac.values], fmt="none", ecolor=CLS[cls],
                        elinewidth=0.7, capsize=1.5, alpha=0.8, zorder=2)
            ax.plot(xs, frac.values, "-" if j == 0 else "--", marker="o", ms=3.5, color=CLS[cls], lw=1.1,
                    label=lab.replace("\n", " "), mfc=CLS[cls] if j == 0 else "white", zorder=3)
            for i, nn in enumerate(n.values):              # tested genes per bin, one row per assay
                ax.text(i, -0.05 - 0.085 * j, f"{nn:,}", ha="center", va="top", fontsize=5.4, color=GREY,
                        transform=ax.get_xaxis_transform())
                keep("C", f"{labs[i]} linked {cls} enhancers", lab.replace("\n", " "), int(nn), f"genes {what} (%)", round(float(frac.values[i]), 1))
            if k == 0:
                ax.text(-0.62, -0.05 - 0.085 * j, short, ha="right", va="top", fontsize=5.4, color=GREY,
                        transform=ax.get_xaxis_transform())
        ax.set_xticks(range(4)); ax.set_xticklabels(labs)
        ax.tick_params(axis="x", pad=3 + 7.5 * len(ASSAYS))
        ax.set_xlabel(f"Linked {cls} enhancers / gene", fontsize=6.8)
        ax.set_ylabel(f"Genes {what} (%)")
        ax.set_ylim(0, None)
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
        ax.set_title(f"{cls} enhancers", fontsize=7.6, color=CLS[cls], pad=20 if len(ASSAYS) > 1 else 6)
    if len(ASSAYS) > 1:
        h = [Line2D([], [], color=GREY, lw=1.1, ls="-" if j == 0 else "--", marker="o", ms=3.5,
                    mfc=GREY if j == 0 else "white", label=lab.replace("\n", " ")) for j, (lab, _, _) in enumerate(ASSAYS)]
        p0, p1 = axes[0].get_position(), axes[1].get_position()
        axes[0].legend(handles=h, frameon=False, fontsize=5.8, loc="lower center", handlelength=1.8, ncol=2,
                       bbox_to_anchor=((p0.x0 + p1.x1) / 2, p0.y1 + 0.002), bbox_transform=fig.transFigure,
                       borderaxespad=0.0, columnspacing=1.2, handletextpad=0.5)
    axes[0].set_ylim(0, max(52, axes[0].get_ylim()[1]))
    return axes[0]


# ----------------------------------------------------------------- D: time course
def plot_D(ax):
    rng = np.random.default_rng(SEED)
    hrs = [0] + HOURS
    for c in ORDER:
        g = tg.loc[(tg["class"] == c) & tg.TargetGene.isin(tc.index), "TargetGene"].unique()
        M = tc.loc[g].values
        med = np.median(M, axis=0)
        boot = np.array([np.median(M[rng.integers(0, len(M), len(M))], axis=0) for _ in range(N_BOOT)])
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        ax.fill_between(hrs, [0, *lo], [0, *hi], color=CLS[c], alpha=0.18, lw=0)
        ax.plot(hrs, [0, *med], "-o", ms=2.8, lw=1.2, color=CLS[c], label=f"{c}-linked ({len(g):,} genes)")
        ax.text(hrs[-1] + 0.35, med[-1], c, ha="left", va="center", fontsize=6.2, color="#333333")
        for h, m_, l_, u_ in zip(HOURS, med, lo, hi):
            keep("D", f"{c}-linked", f"{h} h", len(g), "median log2(dex / EtOH) [95% CI]", f"{m_:+.2f} [{l_:+.2f}, {u_:+.2f}]")
    ax.plot([-0.5, hrs[-1]], [0, 0], color="#999999", lw=0.5, ls=":", zorder=1)
    ax.set_xticks(hrs); ax.set_xlabel("Dex treatment (h)")
    ax.set_xlim(-0.5, hrs[-1] + 1.5)
    ax.set_ylabel("log2(dex / time-matched EtOH)\nmedian, 95% CI (bootstrap over genes)", fontsize=6.8)
    ax.set_title("Time course of top-linked genes (GSE104714)", fontsize=7.6)
    ax.legend(frameon=False, fontsize=6, loc="upper left", handlelength=1.2, borderaxespad=0.1)
    ax.set_ylim(None, ax.get_ylim()[1] + 0.45)
    return ax


# ----------------------------------------------------------------- E: loci
tracks = pd.read_csv(TRACKS, sep="\t") if TRACKS.exists() else None
fm_pos = fm.set_index("enhancer_id")[["s", "e"]]


def plot_locus(fig, spec, gene):
    chrom, x0, x1, g0, g1, gstr = LOCI[gene]
    n_tr = 2 if tracks is not None else 0
    sub = spec.subgridspec(n_tr + 2, 1, height_ratios=[1] * n_tr + [0.7, 0.75], hspace=0.12)
    axes = []
    if tracks is not None:
        t = tracks[tracks.gene == gene]
        tf = np.sqrt                                                      # square-root scale: no clipping,
        yp = ym = (tf(t.signal.max()) * 1.05) or 1                        # one scale per locus: both strands,
        #                                                                   both conditions (fair comparison)
        for k, cond in enumerate(["EtOH", "Dex"]):
            ax = fig.add_subplot(sub[k]); axes.append(ax)
            ax.axvspan(g0 / 1e6, g1 / 1e6, color="#EFEFEF", zorder=0, lw=0)       # gene body
            for st, sgn in (("plus", 1), ("minus", -1)):
                d = t[(t.condition == cond) & (t.strand == st)]
                ax.fill_between((d.start + d.end) / 2 / 1e6, 0, sgn * tf(d.signal), step="mid",
                                color=STRAND[st], lw=0, zorder=2)
            ax.axhline(0, color="#999999", lw=0.3, zorder=1)
            ax.set_ylim(-ym, yp); ax.set_xlim(x0 / 1e6, x1 / 1e6)
            ax.set_xticks([]); ax.spines["bottom"].set_visible(False)
            vmax = (yp / 1.05) ** 2
            ax.set_yticks([-ym / 1.05, 0, yp / 1.05]); ax.set_yticklabels(["", "0", f"{vmax:.0f}"], fontsize=5.4)   # symmetric scale
            ax.text(0.005, 0.97, cond, transform=ax.transAxes, fontsize=6.4, va="top", fontweight="bold")
            if k == 0:
                ax.set_ylabel("PRO-seq", fontsize=6.4)
    # enhancers + ABC arcs
    axe = fig.add_subplot(sub[n_tr]); axes.append(axe)
    axe.set_xlim(x0 / 1e6, x1 / 1e6); axe.set_ylim(0, 1); axe.axis("off")
    loc = fm[(fm.chr == chrom) & (fm.e > x0) & (fm.s < x1)]
    lk = links[(links.TargetGene == gene) & links.enhancer_id.isin(loc.enhancer_id)]
    tss = abc_all.loc[abc_all.TargetGene == gene, "TSS_hg38"].iloc[0] / 1e6
    for _, r in lk.iterrows():
        mid = fm_pos.loc[r.enhancer_id].mean() / 1e6
        w = abs(mid - tss)
        axe.add_patch(Arc(((mid + tss) / 2, 0.12), w, 1.6, theta1=0, theta2=180,
                          color=CLS[r["class"]], lw=0.4 + 6 * r["ABC.Score"], alpha=0.85))
    for _, r in loc.iterrows():                              # NR at half height: class is not read from colour alone
        axe.add_patch(Rectangle((r.s / 1e6, 0.0), max((r.e - r.s) / 1e6, (x1 - x0) / 1e6 * 0.004),
                                0.11 if r["class"] == "NR" else 0.22, color=CLS[r["class"]], ec="none"))
        if r.GR_bound:
            axe.plot((r.s + r.e) / 2e6, -0.12, "^", ms=3.4, color="#7B3294", clip_on=False)
    # gene model (span + TSS of target and other genes in the window)
    axg = fig.add_subplot(sub[n_tr + 1]); axes.append(axg)
    axg.set_xlim(x0 / 1e6, x1 / 1e6); axg.set_ylim(-1, 1)
    for sp in ("left", "top", "right"): axg.spines[sp].set_visible(False)
    axg.set_yticks([])
    axg.axvspan(g0 / 1e6, g1 / 1e6, ymin=0.5, ymax=1, color="#EFEFEF", zorder=0, lw=0)
    axg.add_patch(Rectangle((g0 / 1e6, 0.22), (g1 - g0) / 1e6, 0.36, color="#333333", zorder=2))
    n_chev = int((g1 - g0) / (x1 - x0) * 14)
    for xc in np.linspace(g0, g1, n_chev + 2)[1:-1] / 1e6:
        axg.text(xc, 0.40, "‹" if gstr == "-" else "›", ha="center", va="center", fontsize=6, color="white",
                 fontweight="bold", zorder=3)
    room_left = (g0 - x0) / (x1 - x0)                            # gene name left of the gene if there is room
    if room_left > 0.22:
        axg.text(g0 / 1e6 - (x1 - x0) / 1e6 * 0.01, 0.40, gene, ha="right", va="center", fontsize=6.6,
                 fontstyle="italic", fontweight="bold")
    else:
        axg.text(g1 / 1e6 + (x1 - x0) / 1e6 * 0.01, 0.40, gene, ha="left", va="center", fontsize=6.6,
                 fontstyle="italic", fontweight="bold")
    g5 = (g1 if gstr == "-" else g0) / 1e6                      # annotated 5' end
    dx = (x1 - x0) / 1e6 * 0.035 * (-1 if gstr == "-" else 1)
    axg.plot([g5, g5], [0.58, 0.9], color="#333333", lw=0.8)
    axg.annotate("", xy=(g5 + dx, 0.9), xytext=(g5, 0.9),
                 arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#333333", mutation_scale=6))
    if abs(tss - g5) > (x1 - x0) / 1e6 * 0.01:                  # ABC uses a different (alternative) TSS
        axg.plot([tss, tss], [0.4, 0.75], color="#333333", lw=0.6, ls=":")
        axg.text(tss, 0.78, "ABC TSS", fontsize=5.6, ha="center", va="bottom", color=GREY)
    other = abc_all[(abc_all.chr == chrom) & abc_all.TSS_hg38.between(x0, x1) & (abc_all.TargetGene != gene)] \
        .drop_duplicates("TargetGene")
    for _, r in other.iterrows():
        axg.plot([r.TSS_hg38 / 1e6] * 2, [0.25, 0.55], color="#AAAAAA", lw=0.6)
        axg.text(r.TSS_hg38 / 1e6, -0.05, r.TargetGene, fontsize=5.4, color="#777777", ha="center", va="top")
    axg.set_xlabel(f"{chrom} (Mb, hg38)", fontsize=6.4, labelpad=1)
    axg.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
    axg.tick_params(axis="x", labelsize=5.8)
    lfc = ", ".join(f"{short} {DE[lab].log2FoldChange.get(gene, np.nan):+.1f}" for lab, short, _ in ASSAYS)
    axes[0].set_title(f"$\\it{{{gene}}}$   (log2FC {lfc})", fontsize=7.0, pad=2)
    return axes[0]


# ----------------------------------------------------------------- assemble
fig = plt.figure(figsize=(7.2, 9.6))
gs = fig.add_gridspec(3, 1, height_ratios=[1, 1.02, 2.0 if tracks is None else 2.8], hspace=0.5, top=0.955, bottom=0.105)
r1 = gs[0].subgridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.45)
r2 = gs[1].subgridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.45)
r3 = gs[2].subgridspec(2, 2, wspace=0.18, hspace=0.45)
parts, lead = {}, {}
lead["A"], parts["A"] = tracked(fig, plot_A, fig, r1[0])
axB = fig.add_subplot(r1[1]); plot_B(axB); lead["B"], parts["B"] = axB, [axB]
lead["C"], parts["C"] = tracked(fig, plot_C, fig, r2[0])
axD = fig.add_subplot(r2[1]); plot_D(axD); lead["D"], parts["D"] = axD, [axD]
n_ax = len(fig.axes)
axE = plot_locus(fig, r3[0, 0], LOCI_UP[0]); plot_locus(fig, r3[0, 1], LOCI_UP[1])
plot_locus(fig, r3[1, 0], LOCI_DN[0]); plot_locus(fig, r3[1, 1], LOCI_DN[1])
lead["E"], parts["E"] = axE, list(fig.axes[n_ax:])
# key for panel E (legend handles instead of symbol glyphs, which some fonts lack)
hE = [Patch(color=CLS[c], label=f"{c} enhancer / link" + (" (half height)" if c == "NR" else "")) for c in ORDER] + \
     [Line2D([], [], marker="^", color="#7B3294", ls="none", ms=4, label="GR-bound enhancer"),
      Line2D([], [], color="#333333", lw=0.5, label="thin arc = low ABC score"),
      Line2D([], [], color="#333333", lw=2.2, label="thick arc = high ABC score")]
if tracks is not None:
    hE += [Patch(color=STRAND["plus"], label="PRO-seq, plus-strand reads (up)"),
           Patch(color=STRAND["minus"], label="PRO-seq, minus-strand reads (down)"),
           Patch(color="#EFEFEF", label="gene body of the named gene")]
fig.canvas.draw()
bbE = mpl.transforms.Bbox.union([a.get_tightbbox(fig.canvas.get_renderer()) for a in parts["E"]]) \
    .transformed(fig.transFigure.inverted())
lgE = fig.legend(handles=hE, loc="upper center", bbox_to_anchor=(0.5, bbE.y0 - 0.006), ncol=3, frameon=False,
                 fontsize=6.2, handlelength=1.6, columnspacing=1.4, handletextpad=0.5)
letters = panel_letters(fig, lead)
for ext in ("pdf", "png"):
    fig.savefig(FINAL / f"Fig_4.{ext}", dpi=300, bbox_inches="tight")
parts["E_key"] = [lgE]
export_panels(fig, parts, "Fig_4", letters)
pd.DataFrame(NUM).to_csv(FINAL / "Fig_4_panel_numbers.tsv", sep="\t", index=False)
plt.show()
print("Saved to", FINAL / "Fig_4.pdf", "+ per-panel PDFs Fig_4_A..E.pdf, Fig_4_E_key.pdf, Fig_4_panel_numbers.tsv")
