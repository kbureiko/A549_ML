# =============================================================================
# Supplementary Figure S14 - dex-responsive genes after EP300, FOXA1 and TEAD4 knockdown: overlaps, pathways, regulators
#   A  dex-induced genes   B  dex-repressed genes: UpSet of the five gene sets (siNON and siEP300 PRO-seq 1 h;
#      siNON, siFOXA1 and siTEAD4 BRB-seq 6 h; DESeq2 padj < PADJ, |log2FC| > LFC). Exclusive intersections, the
#      N_INTERSECT largest shown (plus any intersection tied with the last one); left bars = set sizes.
#   C  pathway summary: GO:BP over-representation (R/06_export_gene_sets_go.R) of the gene sets "shared" (dex-responsive
#      in siNON and the knockdown) and "si<KD>-unique" per knockdown and direction, summarised by topic (keyword rules).
#      Colour = best FDR of the topic's terms in the set (red = induced, blue = repressed
#      sets); size = number of its terms at FDR < 0.05; grey = best FDR >= 0.05.
#   D  LISA motif model: TF families ranked for the same sets (family value = best rank of any member; size =
#      members in the top TOP_N; number = rank <= 5).
# "Unique" = passing the cut-off in one condition and not the other (not a tested difference).
# Tied intersections are ordered deterministically. Panel D needs A6_LISA_all_sets.tsv from scripts/14_collect_lisa.py.
# Inputs (rglob under PROJECT): panel_export_PROseq_siEP300/*_DESeq2_si*_dex_vs_EtOH.tsv,
#   panel_export_BRBseq_siFOXA1_siTEAD4/*_DESeq2_si*_Dex_vs_Veh.tsv, *_A6_GO_BP_all_sets.tsv, *_A6_gene_set_sizes.tsv,
#   A6_LISA_all_sets.tsv
# Outputs (FIGURES): Fig_S14_knockdown_genes.pdf/png (+ _A.._D.pdf), Fig_S14_upset_intersections.tsv,
#   Fig_S14_GO_topics.tsv, Fig_S14_GO_topic_best_terms.tsv, Fig_S14_LISA_families.tsv
# =============================================================================
# %%
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

from a549_enhancers.paths import PROJECT, FIGURES   # local folders, set in paths.toml
FINAL = FIGURES; FINAL.mkdir(parents=True, exist_ok=True)
PADJ, LFC, FDR = 0.05, 0.5, 0.05
N_INTERSECT, TOP_N, MAX_BEST_RANK, MAX_TERM_SIZE = 11, 50, 15, 400


def find(pattern, required=True):
    hits = sorted(p for p in PROJECT.rglob(pattern) if not any(x.startswith("figures") for x in p.parts))
    if not hits and required:
        raise FileNotFoundError(f"{pattern} not found under {PROJECT}")
    return hits[-1] if hits else None


FONT = "Open Sans" if any("Open Sans" in f.name for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
mpl.rcParams.update({
    "font.family": [FONT, "DejaVu Sans"], "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.8, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none",
    "axes.grid": False,
})
DIRCOL = {"up": "#D1495B", "down": "#3B6FB6"}
SET_ORDER = [("siNON", "PRO"), ("siEP300", "PRO"), ("siNON", "BRB"), ("siFOXA1", "BRB"), ("siTEAD4", "BRB")]
SET_LAB = {k: f"{k[0]} ({'PRO-seq' if k[1] == 'PRO' else 'BRB-seq'})" for k in SET_ORDER}
SET_COL = dict(zip(SET_ORDER, ["#E69F00", "#CC79A7", "#D55E00", "#009E73", "#0072B2"]))   # Okabe-Ito
KDS, GROUPS, DIRS = ["EP300", "FOXA1", "TEAD4"], ["shared", "KD_specific"], ["up", "down"]
SETS = [f"{k}_{g}_{d}" for d in DIRS for k in KDS for g in GROUPS]


# ----------------------------------------------------------------- gene sets
def dex_sets(path):
    t = pd.read_csv(path, sep="\t").set_index("gene"); ok = t.padj.notna() & (t.padj < PADJ)
    return {"up": set(t.index[ok & (t.log2FoldChange > LFC)]), "down": set(t.index[ok & (t.log2FoldChange < -LFC)])}


def load_assay(folder, pattern):
    files = sorted(p for p in PROJECT.rglob(f"{folder}/{pattern}") if "interaction" not in p.name)
    if not files:
        raise FileNotFoundError(f"{folder}/{pattern} not found under {PROJECT}")
    out = {}
    for p in files:                                     # newest file per silencing wins (sorted by name/date)
        out[p.name.split("DESeq2_")[1].split("_")[0]] = p
    return {k: dex_sets(p) for k, p in out.items()}


PRO = load_assay("panel_export_PROseq_siEP300", "*_DESeq2_si*_dex_vs_EtOH.tsv")
BRB = load_assay("panel_export_BRBseq_siFOXA1_siTEAD4", "*_DESeq2_si*_Dex_vs_Veh.tsv")
GS = {k: (PRO if k[1] == "PRO" else BRB)[k[0]] for k in SET_ORDER}
sizes = pd.read_csv(find("*_A6_gene_set_sizes.tsv"), sep="\t").set_index("set").n_genes
set_label = lambda s_: ("shared" if "_shared_" in s_ else f"si{s_.split('_')[0]}-unique") + f" ({sizes.get(s_, 0)})"


# ----------------------------------------------------------------- A, B: UpSet
def upset(spec, dirn, rows_out):
    sets = {k: GS[k][dirn] for k in SET_ORDER}
    universe = set().union(*sets.values())
    memb = pd.Series([tuple(g in sets[k] for k in SET_ORDER) for g in universe])
    counts = memb.value_counts()
    # deterministic order: size, then membership pattern (the order of tied intersections otherwise changes between runs)
    counts = counts.loc[sorted(counts.index, key=lambda c: (-counts[c], tuple(not f for f in c)))]
    cut = counts.iloc[min(N_INTERSECT, len(counts)) - 1]
    combos = [c for c in counts.index if counts[c] >= cut]          # ties with the last shown intersection are kept
    g2 = spec.subgridspec(2, 3, height_ratios=[1.7, 1.0], width_ratios=[0.42, 1.0, 2.3], hspace=0.05, wspace=0.02)
    axb = fig.add_subplot(g2[0, 2]); axm = fig.add_subplot(g2[1, 2], sharex=axb); axs = fig.add_subplot(g2[1, 0], sharey=axm)
    axl = fig.add_subplot(g2[1, 1], sharey=axm); axl.axis("off")
    x = np.arange(len(combos)); vals = [counts[c] for c in combos]
    axb.bar(x, vals, color="#555555", width=0.65, lw=0)
    for xi, v in zip(x, vals):
        axb.text(xi, v, f"{v}", ha="center", va="bottom", fontsize=5.6)
    axb.set_ylabel("Intersection size", fontsize=6.6); axb.tick_params(axis="x", length=0, labelbottom=False)
    axb.set_ylim(0, max(vals) * 1.15); axb.spines["bottom"].set_visible(False)
    axb.set_title("Dex-induced genes" if dirn == "up" else "Dex-repressed genes", fontsize=7.6, color=DIRCOL[dirn],
                  fontweight="bold", loc="left")
    n = len(SET_ORDER)
    for i in range(n):
        if i % 2 == 1:
            axm.axhspan(i - 0.5, i + 0.5, color="#F2F2F2", lw=0, zorder=0)
    for xi, c in zip(x, combos):
        on = [i for i, f in enumerate(c) if f]
        if len(on) > 1:
            axm.plot([xi, xi], [min(on), max(on)], c="#555555", lw=1.0, zorder=1)
        axm.scatter([xi] * n, range(n), s=18, lw=0, zorder=2,
                    c=[SET_COL[k] if f else "#DDDDDD" for k, f in zip(SET_ORDER, c)])
    axm.set_ylim(n - 0.5, -0.5); axm.set_xlim(-0.6, len(combos) - 0.4)
    axm.set_yticks(range(n)); axm.tick_params(length=0, labelbottom=False, labelleft=False)
    for i, k in enumerate(SET_ORDER):
        axl.text(0.97, i, SET_LAB[k], ha="right", va="center", fontsize=5.4, transform=axl.get_yaxis_transform(),
                 color="#222222")
    for sp in axm.spines.values():
        sp.set_visible(False)
    tot = [len(sets[k]) for k in SET_ORDER]
    axs.barh(range(n), tot, color=[SET_COL[k] for k in SET_ORDER], height=0.6, lw=0)
    axs.set_xlim(max(tot) * 1.1, 0)
    axs.set_xlabel("Set size", fontsize=6.2); axs.tick_params(axis="y", length=0, labelleft=False)
    axs.tick_params(axis="x", labelsize=5.4); axs.spines["left"].set_visible(False)
    for c, v in zip(counts.index, counts.values):
        rows_out.append(dict(direction=dirn, members=" & ".join(SET_LAB[k] for k, f in zip(SET_ORDER, c) if f),
                             n_genes=int(v)))
    return [axb, axm, axs, axl]


# ----------------------------------------------------------------- C: GO topics (same rules as the A6 GO panel)
TOPICS = [
    ("Inflammation / immune", r"inflammat|leukocyte|cytokine|myeloid|macrophage|immune|NF-kappaB|interleukin|"
                              r"tumor necrosis|neutrophil|lymphocyte|lipopolysaccharide|bacteri"),
    ("Migration / wound healing", r"wound|migration|ameboidal|motility|chemotaxis|locomotion|taxis|coagulation|hemostasis"),
    ("ECM / adhesion / cytoskeleton", r"extracellular matrix|extracellular structure|external encapsulating|adhesion|"
                                      r"junction|integrin|actin|actomyosin|cytoskeleton|membrane raft"),
    ("Growth-factor / Hippo signalling", r"hippo|growth factor|transforming growth|SMAD|BMP|ERK1|MAPK|p38|"
                                         r"protein phosphorylation|Wnt|receptor tyrosine|Rho protein|small GTPase"),
    ("Epithelial differentiation", r"epithelial cell differentiation|epithelium development|epithelial cell proliferation|"
                                   r"lung|respiratory|branching"),
    ("Oxidative stress / glutathione / metals", r"glutathione|oxidative stress|chemical stress|copper|metal ion|"
                                                r"detoxification|reactive oxygen|xenobiotic|zinc|cadmium"),
    ("Hypoxia", r"hypoxia|oxygen levels|decreased oxygen|oxygen-glucose"),
    ("Apoptosis / survival", r"apoptotic|cell death|programmed"),
    ("Glucose / insulin / nutrients", r"glucose|carbohydrate|insulin|nutrient|starvation"),
    ("Lipid / retinoid metabolism", r"lipid|fat cell|fatty acid|retinoid|retinol|terpenoid|isoprenoid|cholesterol|"
                                    r"steroid metabolic|steroid biosynthetic"),
    ("Amino acid / one-carbon metabolism", r"amino acid|one-carbon|folate|sulfur|nonribosomal peptide"),
    ("Nuclear receptor / hormone signalling", r"intracellular receptor signaling|steroid hormone mediated|"
                                              r"hormone-mediated signaling"),
    ("Cell cycle / chromosome segregation", r"cell cycle|chromosome segregation|sister chromatid|nuclear division|"
                                            r"DNA replication|mitotic|organelle fission|spindle|cell division"),
]
EXCLUDE = (r"kidney|nephron|renal|pancrea|osteoblast|bone|ossification|keratinocyte|epiderm|neuron|nervous|axon|"
           r"synap|glial|astrocyte|heart|cardiac|muscle|sperm|blood pressure|circulat|vascul|angiogen|blood vessel|"
           r"endothel|hair|tooth|eye|lens|retina|inner ear|cartilage|chondro|skeletal|limb|placenta|gland|"
           r"mammary|prostate|uter|ovar|testis|germ cell|embryo|gastrulation|hepatocyte")
GENERIC = (r"^(cellular )?response to (steroid hormone|hormone|hormone stimulus|corticosteroid|glucocorticoid|"
           r"oxygen levels|decreased oxygen levels|wounding|lipid|peptide|peptide hormone|organic cyclic compound|"
           r"inorganic substance|chemical|chemical stress|nutrient levels|extracellular stimulus|"
           r"external stimulus|molecule of bacterial origin|bacterium|other organism|biotic stimulus)$")


def topic_of(name):
    if re.search(EXCLUDE, name, re.I) or re.search(GENERIC, name, re.I):
        return None
    return next((t for t, p in TOPICS if re.search(p, name, re.I)), None)


def col_header(ax, m, y_kd, y_dir):
    for b, dirn in enumerate(DIRS):
        for j, k in enumerate(KDS):
            ax.text(b * 6 + j * 2 + 0.5, y_kd, f"si{k}", ha="center", va="bottom", fontsize=6.2, fontweight="bold")
        ax.text(b * 6 + 2.5, y_dir, "Dex-induced genes" if dirn == "up" else "Dex-repressed genes", ha="center",
                va="bottom", fontsize=7.2, color=DIRCOL[dirn], fontweight="bold")
    ax.axvline(len(KDS) * len(GROUPS) - 0.5, color="#999999", lw=0.8)
    for k in range(1, len(KDS) * 2):
        if k != len(KDS):
            ax.axvline(k * len(GROUPS) - 0.5, color="#E6E6E6", lw=0.6, zorder=0)


def plot_C(ax):
    go = pd.read_csv(find("*_A6_GO_BP_all_sets.tsv"), sep="\t")
    go["size"] = go.BgRatio.astype(str).str.split("/").str[0].astype(int)
    go["topic"] = [topic_of(n) if s <= MAX_TERM_SIZE else None for n, s in zip(go.Description, go["size"])]
    t = go[go.topic.notna() & go.set.isin(SETS)]
    best = t.pivot_table(index="topic", columns="set", values="p.adjust", aggfunc="min").reindex(columns=SETS)
    nsig = t[t["p.adjust"] < FDR].pivot_table(index="topic", columns="set", values="ID", aggfunc="nunique") \
        .reindex(columns=SETS)
    topics = [tp for tp, _ in TOPICS if tp in best.index and (best.loc[tp] < FDR).any()]
    best, nsig = best.reindex(topics), nsig.reindex(topics).fillna(0)
    rep = t[t.topic.isin(topics) & (t["p.adjust"] < FDR)].sort_values("pvalue").groupby(["topic", "set"]).Description.first()
    n, m = len(topics), len(SETS)
    B = best.values.astype(float); ok = np.isfinite(B); sig = ok & (B < FDR)
    L = -np.log10(np.where(ok, B, 1))
    yy, xx = np.meshgrid(range(n), range(m), indexing="ij")
    for i in range(n):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#F7F7F7", zorder=0, lw=0)
    ax.scatter(xx[ok & ~sig], yy[ok & ~sig], s=8, c="#D8D8D8", lw=0, zorder=2)
    size = lambda k: 10 + 16 * np.sqrt(k)
    scs = {}
    for dirn, cmap in (("up", "Reds"), ("down", "Blues")):
        cols = np.array([s_.endswith("_" + dirn) for s_ in SETS])
        msk = sig & cols[None, :]
        scs[dirn] = ax.scatter(xx[msk], yy[msk], s=size(nsig.values[msk]), c=np.clip(L[msk], 0, 4),
                               cmap=cmap, norm=mpl.colors.Normalize(0.5, 4), lw=0.3, edgecolor="#444444", zorder=3)
    ax.set_xlim(-0.6, m - 0.4); ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks(range(n)); ax.set_yticklabels(topics, fontsize=6.3)
    ax.set_xticks(range(m)); ax.set_xticklabels([set_label(s_) for s_ in SETS], rotation=60, ha="right", fontsize=6.0)
    ax.tick_params(length=0, pad=2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    col_header(ax, m, -0.7, -1.7)
    ax.text((m - 1) / 2, -3.0, "Pathway summary (GO:BP topics)", ha="center", va="bottom", fontsize=8, clip_on=False)
    out = pd.concat({"best_FDR": best, "n_terms_FDR<0.05": nsig}, axis=1)
    out.to_csv(FINAL / "Fig_S14_GO_topics.tsv", sep="\t")
    rep.rename("best_term").to_csv(FINAL / "Fig_S14_GO_topic_best_terms.tsv", sep="\t")
    return scs, size


# ----------------------------------------------------------------- D: LISA TF families
FAMILIES = [
    ("GR / AR / PR (GRE)", r"^(NR3C1|NR3C2|AR|PGR|PR)$"),
    ("AP-1 (JUN / FOS / BATF)", r"^(JUN|JUNB|JUND|FOS|FOSB|FOSL1|FOSL2|BATF\d?|JDP2|ATF3)$"),
    ("NRF2 / MAF / BACH", r"^(NFE2|NFE2L\d|MAF[A-Z]?|BACH\d)$"),
    ("ATF4 / CEBP", r"^(ATF4|ATF5|CEBP[A-Z]|DDIT3|DBP|TEF|HLF|NFIL3)$"),
    ("TEAD", r"^TEAD\d$"),
    ("NF-κB", r"^(NFKB\d|RELA|RELB|REL)$"),
    ("KLF / SP", r"^(KLF\d+|SP\d|ZNF281|ZFP281|ZFP740|PATZ1|MAZ)$"),
    ("FOX", r"^FOX[A-Z]\d*$"),
    ("POU / homeobox", r"^(POU\d+F\d|BARHL\d|LHX\d|LMX\d[AB]?|HOX[A-D]\d+|NKX\d-\d|DLX\d|PRRX\d|SHOX\d?|VAX\d|VSX\d|"
                       r"UNCX|PROX1|SIX\d|IRX\d|HMBOX1|MEOX\d|VENTX|RHOX\d|TLX\d|BSX|RAXL1)$"),
    ("NFI", r"^NFI[ABCX]$"),
    ("TFAP2", r"^(TFAP2[A-E]|TCFAP2[A-E])$"),
    ("PPAR / RXR / RAR / VDR", r"^(PPAR[ADG]|PPARG::RXRA|RXR[ABG]|RAR[ABG]|VDR|ROR[ABC])$"),
    ("p53 family", r"^(TP53|TRP53|TP63|TP73)$"),
    ("SOX", r"^SOX\d+$"),
    ("ETS", r"^(ETS\d|ELF\d|ELK\d|EHF|ETV\d|GABPA|SPDEF|FLI1|ERG)$"),
    ("E2F", r"^E2F\d$"),
    ("SMAD", r"^SMAD\d"),
]


def family_of(tf):
    return next((lab for lab, p in FAMILIES if re.search(p, str(tf).upper())), None)


def plot_D(ax):
    lisa = pd.read_csv(find("A6_LISA_all_sets.tsv"), sep="\t")
    d = lisa[(lisa.model == "motif") & lisa.set.isin(SETS)].copy()
    d["family"] = d.TF.map(family_of)
    fam = d[d.family.notna()]
    best = fam.pivot_table(index="family", columns="set", values="rank", aggfunc="min")
    ntop = fam[fam["rank"] <= TOP_N].pivot_table(index="family", columns="set", values="TF", aggfunc="nunique")
    fsize = fam.groupby("family").TF.nunique()
    fams = [f for f, _ in FAMILIES if f in best.index and (best.loc[f] <= MAX_BEST_RANK).any()]
    best = best.reindex(index=fams, columns=SETS); ntop = ntop.reindex(index=fams, columns=SETS).fillna(0)
    missing = [s_ for s_ in SETS if s_ not in set(d.set)]
    R = best.values.astype(float); ok = np.isfinite(R); strong = ok & (R <= TOP_N)
    n, m = R.shape
    yy, xx = np.meshgrid(range(n), range(m), indexing="ij")
    size = lambda k: 14 + 26 * np.sqrt(k)
    for j, s_ in enumerate(SETS):
        if s_ in missing:
            ax.scatter([j] * n, range(n), marker="x", s=12, c="#BBBBBB", lw=0.6, zorder=2)
    for i in range(n):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#F7F7F7", zorder=0, lw=0)
        ax.text(m - 0.25, i, f"{fsize.get(fams[i], 0)}", va="center", ha="left", fontsize=5.6, color="#777777")
    ax.scatter(xx[ok & ~strong], yy[ok & ~strong], s=10, c="#D8D8D8", lw=0, zorder=2)
    sc = ax.scatter(xx[strong], yy[strong], s=size(ntop.values[strong]), c=np.clip(R[strong], 1, 400),
                    cmap=mpl.colormaps["magma"], norm=mpl.colors.LogNorm(1, 400), lw=0.3, edgecolor="#444444", zorder=3)
    for i in range(n):
        for j in range(m):
            if ok[i, j] and R[i, j] <= 5:
                ax.text(j, i, f"{int(R[i, j])}", ha="center", va="center", fontsize=5.2, color="white", zorder=4)
    ax.text(m - 0.25, -0.75, "TFs in\nfamily", va="bottom", ha="left", fontsize=5.4, color="#777777", linespacing=1.0)
    ax.set_xlim(-0.6, m - 0.4); ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks(range(n)); ax.set_yticklabels(fams, fontsize=6.3)
    ax.set_xticks(range(m))
    ax.set_xticklabels([set_label(s_) + ("\nno result" if s_ in missing else "") for s_ in SETS], rotation=60,
                       ha="right", fontsize=6.0)
    ax.tick_params(length=0, pad=2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    col_header(ax, m, -0.7, -1.7)
    ax.text((m - 1) / 2, -3.0, "LISA-ranked TF families (motif model)", ha="center", va="bottom", fontsize=8,
            clip_on=False)
    pd.concat({"best_rank": best, f"n_in_top{TOP_N}": ntop}, axis=1).assign(family_size=fsize.reindex(fams)) \
        .to_csv(FINAL / "Fig_S14_LISA_families.tsv", sep="\t")
    return sc, size, bool(missing)


# ----------------------------------------------------------------- assemble
rows = []
fig = plt.figure(figsize=(7.4, 14.3))
gs = fig.add_gridspec(3, 1, height_ratios=[0.85, 1.35, 1.75], hspace=0.62, left=0.05, right=0.97, top=0.975,
                      bottom=0.05)
r1 = gs[0].subgridspec(1, 2, wspace=0.3)
axA = upset(r1[0], "up", rows); axB = upset(r1[1], "down", rows)
r2 = gs[1].subgridspec(1, 3, width_ratios=[0.62, 2.0, 0.42], wspace=0.0)
axC = fig.add_subplot(r2[1]); scsC, sizeC = plot_C(axC)
r3 = gs[2].subgridspec(1, 3, width_ratios=[0.62, 2.0, 0.42], wspace=0.0)
axD = fig.add_subplot(r3[1]); scD, sizeD, has_missing = plot_D(axD)
fig.canvas.draw()
# C legends: two colour bars (induced / repressed) + size key
p = axC.get_position()
for k, (dirn, lab) in enumerate((("up", "induced"), ("down", "repressed"))):
    cax = fig.add_axes([p.x1 + 0.05 + k * 0.07, p.y0 + p.height * 0.55, 0.01, p.height * 0.36])
    cb = fig.colorbar(scsC[dirn], cax=cax); cb.set_ticks([1, 2, 3, 4]); cb.set_ticklabels(["1", "2", "3", "≥4"])
    cb.ax.tick_params(labelsize=5.4, length=1.5); cb.outline.set_linewidth(0.4)
    cb.ax.set_title(lab, fontsize=5.4, pad=3)
fig.text(p.x1 + 0.05, p.y0 + p.height * 0.98, "−log10 FDR", fontsize=6.0)
for k in (1, 5, 15):
    axC.scatter([], [], s=sizeC(k), c="#999999", lw=0, label=str(k))
axC.scatter([], [], s=8, c="#D8D8D8", lw=0, label="n.s.")
axC.legend(title="terms at\nFDR < 0.05", frameon=False, fontsize=5.8, title_fontsize=5.8, loc="upper left",
           bbox_to_anchor=(1.07, 0.45), labelspacing=0.8, handletextpad=0.3, borderaxespad=0)
# D legends
p = axD.get_position()
cax = fig.add_axes([p.x1 + 0.055, p.y0 + p.height * 0.58, 0.012, p.height * 0.32])
cb = fig.colorbar(scD, cax=cax); cb.set_ticks([1, 5, 20, 100]); cb.set_ticklabels(["1", "5", "20", "100"])
cb.ax.invert_yaxis(); cb.ax.tick_params(labelsize=5.6, length=1.5); cb.outline.set_linewidth(0.4)
cb.set_label("best LISA rank\nof a family member", fontsize=6)
for k in (1, 3, 8):
    axD.scatter([], [], s=sizeD(k), c="#999999", lw=0, label=str(k))
axD.scatter([], [], s=10, c="#D8D8D8", lw=0, label=f"best rank > {TOP_N}")
axD.scatter([], [], s=sizeD(2), c="#2D0C4E", lw=0, label="number = rank ≤ 5")
if has_missing:
    axD.scatter([], [], marker="x", s=12, c="#BBBBBB", lw=0.6, label="no LISA result")
axD.legend(title=f"family members\nin the top {TOP_N}", frameon=False, fontsize=5.8, title_fontsize=5.8,
           loc="upper left", bbox_to_anchor=(1.07, 0.5), labelspacing=0.9, handletextpad=0.3, borderaxespad=0)
pd.DataFrame(rows).to_csv(FINAL / "Fig_S14_upset_intersections.tsv", sep="\t", index=False)

PANELS = {"A": axA, "B": axB, "C": [axC], "D": [axD]}
fig.canvas.draw(); rr = fig.canvas.get_renderer()
LET = []
for l, axes in PANELS.items():
    bb = mpl.transforms.Bbox.union([a.get_tightbbox(rr) for a in axes]).transformed(fig.transFigure.inverted())
    LET.append(fig.text(bb.x0 - 0.004, bb.y1 + 0.003, l, fontsize=11, fontweight="bold", ha="left", va="bottom"))
for ext in ("pdf", "png"):
    fig.savefig(FINAL / f"Fig_S14_knockdown_genes.{ext}", dpi=300, bbox_inches="tight")
for t in LET:
    t.set_visible(False)
fig.canvas.draw(); rr = fig.canvas.get_renderer()
extra = {"C": [a for a in fig.axes if a not in sum(PANELS.values(), []) and a.get_position().y0 > axD.get_position().y1],
         "D": [a for a in fig.axes if a not in sum(PANELS.values(), []) and a.get_position().y1 < axC.get_position().y0]}
for l, axes in PANELS.items():
    arts = axes + extra.get(l, [])
    bb = mpl.transforms.Bbox.union([a.get_tightbbox(rr) for a in arts]).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(FINAL / f"Fig_S14_knockdown_genes_{l}.pdf", bbox_inches=bb.padded(0.04))
for t in LET:
    t.set_visible(True)
plt.show()
print("Saved", FINAL / "Fig_S14_knockdown_genes.pdf")
