# =============================================================================
# Supplementary Figure S3 - chromatin programs (topics) of non-responsive (NR) enhancers
# Topics from scripts/11_nr_topics.py (consensus NMF, K = 5; cached loadings and usage, nothing is refitted).
# Features: 13 chromatin features before dex (EtOH or untreated tracks; GR from the dex track, since GR is hardly
# bound without hormone; FOXA1 EtOH), as percentiles within NR (same scale the topics were fitted on).
#   A  what defines each topic: features with loading >= SHOW_CUT (scaled to the topic's strongest feature)
#   C  genomic annotation per response class (% of enhancers; feature-matrix "annotation")
#   B  feature profiles: NR enhancers grouped by their main topic (largest share); median percentile
#      of each feature within NR (50 = typical NR enhancer); right: n, % GR-bound (dex ChIP peak), % CpG annotation
# For the text (not shown): K = 5 is the largest number of topics that is stable over 60 subsample fits
#   (cosine silhouette >= 0.8) without a topic made of a single feature; enhancers are mixtures (median share of the
#   main topic 0.41; only ~2 % have one topic >= 0.7), i.e. a continuum rather than discrete types; topic usage
#   follows GR binding more than response class.
# Inputs: FM; NR_TOPICS/MS_nr_topics_loadings.tsv, NR_TOPICS/MS_nr_topics_usage.tsv
# NOTE: GR is the only post-dex feature (dashed box in B): GR is barely bound without hormone. The topics are therefore
#   not purely pretreatment profiles; do not describe them as such in the text.
# Outputs (FIGURES): Fig_S3_nr_topics.pdf/png (+ _A.._C.pdf), Fig_S3_nr_topics_profile.tsv,
#   Fig_S3_annotation_per_class.tsv
# =============================================================================
# %%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

from a549_enhancers.paths import PROJECT, ML, FIGURES, NR_TOPICS   # local folders, set in paths.toml
FM = ML / "feature_matrix_all_classes_GRannotated.csv"
OUT = NR_TOPICS
FINAL = FIGURES; FINAL.mkdir(parents=True, exist_ok=True)   # final figures + their tables
SHOW_CUT = 0.25
TOPIC_NAME = {"T1": "Active (GR/EP300/FOXA1)", "T2": "Promoter-like", "T3": "Polycomb-like", "T4": "CTCF/cohesin",
              "T5": "H3K4me1-high"}
TOPIC_COL = {"T1": "#2E7D5B", "T2": "#D4A62A", "T3": "#4D4D4D", "T4": "#7A5195", "T5": "#7FA7D1"}

FONT = "Open Sans" if any("Open Sans" in f.name for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
mpl.rcParams.update({
    "font.family": [FONT, "DejaVu Sans"], "font.size": 7, "axes.titlesize": 7.4, "axes.labelsize": 6.8,
    "xtick.labelsize": 6.3, "ytick.labelsize": 6.4, "legend.fontsize": 5.8, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none",
    "axes.grid": False,                                   # override any style sheet that turns grids on
})

# ----------------------------------------------------------------- data (same features as the topic fit)
D = pd.read_csv(FM, low_memory=False)
D["GR_bound"] = D.GR_bound.astype(str).str.lower().eq("true")
col = lambda p, t: [c for c in D.columns if c.startswith(p) and t in c][0]
SF = {"ATAC": "ATAC_ctrl", "H3K27ac": col("H3K27ac_", "_eth_"), "H3K4me1": col("H3K4me1_", "_eth_"),
      "H3K4me3": col("H3K4me3_", "_eth_"), "Pol II": col("POLR2A_", "_eth_"), "EP300": "EP300_ENCFF842XYY",
      "cJUN": "JUN_ENCFF957TUK", "CTCF": "CTCF_ENCFF083AEY", "RAD21": "RAD21_ENCFF778YHU", "H2A.Z": col("H2AFZ_", "_eth_"),
      "H3K27me3": col("H3K27me3_", "_eth_"), "GR (dex)": col("NR3C1_", "_dex_"), "FOXA1": col("FOXA1_", "_eth_")}
N = D[D["class"].eq("NR")].set_index("enhancer_id")
P = pd.DataFrame({f: N[c].rank(pct=True) * 100 for f, c in SF.items()})          # percentile within NR
load = pd.read_csv(OUT / "MS_nr_topics_loadings.tsv", sep="\t", index_col=0)
use = pd.read_csv(OUT / "MS_nr_topics_usage.tsv", sep="\t").set_index("enhancer_id")
TCOLS = list(load.index); TID = {c: c.split(" ", 1)[0] for c in TCOLS}
use = use.loc[P.index]
main = use[TCOLS].idxmax(axis=1).map(TID)
FEATS = list(SF)
ANN_MAP = {"Intergenic": "Intergenic", "Intonic": "Intronic", "Intronic": "Intronic", "CpG": "CpG island", "Other": "Other"}
ANN_COL = {"Intergenic": "#8C8C8C", "Intronic": "#E1A43A", "Other": "#7A5195", "CpG island": "#3FA7A0"}
CLS = {"UP": "#D1495B", "DN": "#3B6FB6", "NR": "#5DAA82"}
D["annot"] = D.annotation.astype(str).map(ANN_MAP).fillna("Other")

prof = P.groupby(main).median().reindex([TID[c] for c in TCOLS])
stat = pd.DataFrame({"n": main.value_counts(),
                     "pct_GR_bound": N.GR_bound.groupby(main).mean() * 100,
                     "pct_CpG": N.annotation.astype(str).eq("CpG").groupby(main).mean() * 100}).reindex(prof.index)
pd.concat([prof.add_prefix("median_pct_"), stat], axis=1).round(2).to_csv(FINAL / "Fig_S3_nr_topics_profile.tsv", sep="\t")
print(pd.concat([prof.round(0), stat.round(1)], axis=1).to_string())
print(f"median share of the main topic: {use.dominant_share.median():.2f}; "
      f"share >= 0.7: {(use.dominant_share >= 0.7).mean() * 100:.1f} %")


# ----------------------------------------------------------------- panels
def plot_A(axs):
    for ax, tc in zip(axs, TCOLS):
        t = TID[tc]; v = load.loc[tc]
        v = v[v >= SHOW_CUT].sort_values()
        ax.barh(range(len(v)), v.values, color=TOPIC_COL[t], height=0.7)
        ax.set_yticks(range(len(v))); ax.set_yticklabels(v.index, fontsize=5.9)
        ax.set_xlim(0, 1.05); ax.set_xticks([0, 0.5, 1]); ax.tick_params(axis="x", labelsize=5.4)
        ax.set_title(f"{t}: {TOPIC_NAME.get(t, tc)}", fontsize=6.6, color=TOPIC_COL[t], pad=2, loc="left")
    axs[len(axs) // 2].set_xlabel("loading (relative to the topic's strongest feature)", fontsize=6.2)


def plot_B(ax, ax_stat):
    M = prof[FEATS].to_numpy()
    ax.imshow(M, cmap="RdBu_r", vmin=0, vmax=100, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=5.8,
                    color="white" if abs(M[i, j] - 50) > 28 else "#222222")
    ax.set_xticks(range(len(FEATS))); ax.set_xticklabels(FEATS, rotation=50, ha="right")
    ax.set_yticks(range(len(prof))); ax.set_yticklabels([f"{t}: {TOPIC_NAME[t]}" for t in prof.index])
    for tl, t in zip(ax.get_yticklabels(), prof.index):
        tl.set_color(TOPIC_COL[t])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Feature profiles of NR enhancers by dominant topic\n"
                 "median percentile within NR (50 = typical); dashed = post-dex feature", fontsize=7)
    ax.grid(False)                                        # no grid over the heatmap (style sheets may switch it on)
    for i in range(1, len(prof)):                         # white lines between topics
        ax.axhline(i - 0.5, color="white", lw=2.0)
    gi = FEATS.index("GR (dex)")
    ax.add_patch(mpl.patches.Rectangle((gi - 0.5, -0.5), 1, len(prof), fill=False, ec="#333333", lw=0.9, ls="--"))
    st = stat[["n", "pct_GR_bound", "pct_CpG"]].to_numpy()
    # n as text; % GR-bound and % CpG as small bars (0-100 %, bar width BAR_W in column units) with the value
    BAR_W, X0 = 1.15, {1: 0.6, 2: 2.15}                   # bar length for 100 %, left edge of each bar column
    BAR_COL = {1: "#5B4B9A", 2: "#3FA7A0"}
    ax_stat.set_xlim(-0.6, 3.75); ax_stat.set_ylim(len(prof) - 0.5, -0.5); ax_stat.axis("off")
    ax_stat.text(0, -0.6, "n", ha="center", va="bottom", fontsize=5.6)
    for j, h in ((1, "GR-bound\n(%)"), (2, "CpG island\n(%)")):
        ax_stat.text(X0[j] + BAR_W / 2, -0.6, h, ha="center", va="bottom", fontsize=5.6, linespacing=0.9)
        ax_stat.plot([X0[j], X0[j]], [-0.45, len(prof) - 0.55], color="#999999", lw=0.5)
    for i in range(len(prof)):
        ax_stat.text(0, i, f"{st[i, 0]:,.0f}", ha="center", va="center", fontsize=5.9)
        for j in (1, 2):
            w = st[i, j] / 100 * BAR_W
            ax_stat.add_patch(mpl.patches.Rectangle((X0[j], i - 0.28), w, 0.56, color=BAR_COL[j], lw=0))
            ax_stat.text(X0[j] + w + 0.04, i, f"{st[i, j]:.0f}", ha="left", va="center", fontsize=5.6)


def plot_C(ax):
    order = ["NR", "UP", "DN"]
    ct = pd.crosstab(D["class"], D["annot"], normalize="index").reindex(order)[list(ANN_COL)] * 100
    n = D["class"].value_counts()
    bottom = np.zeros(len(order))
    for a_ in ANN_COL:
        v = ct[a_].values
        ax.bar(range(len(order)), v, bottom=bottom, color=ANN_COL[a_], width=0.7, lw=0, label=a_)
        for i, (b, h) in enumerate(zip(bottom, v)):
            if h >= 5:
                ax.text(i, b + h / 2, f"{h:.0f}", ha="center", va="center", fontsize=5.8,
                        color="white" if a_ in ("Other", "Intergenic") else "#222222")
        bottom += v
    ax.set_xticks(range(len(order))); ax.set_xticklabels([f"{c}\n{n[c]:,}" for c in order])
    for tl, c in zip(ax.get_xticklabels(), order):
        tl.set_color(CLS[c])
    ax.set_ylim(0, 100); ax.set_ylabel("enhancers (%)")
    ax.set_title("Genomic annotation per class", fontsize=7)
    ax.legend(frameon=False, fontsize=5.4, loc="upper left", bbox_to_anchor=(1.0, 1.0), handlelength=0.9)
    ct.round(1).assign(n=n.reindex(order)).to_csv(FINAL / "Fig_S3_annotation_per_class.tsv", sep="\t")


# ----------------------------------------------------------------- assemble
fig = plt.figure(figsize=(8.0, 4.9))
gs = fig.add_gridspec(2, 1, height_ratios=[0.8, 1.0], hspace=0.62, left=0.06, right=0.98, top=0.93, bottom=0.12)
r1 = gs[0].subgridspec(1, len(TCOLS), wspace=0.95)
axA = [fig.add_subplot(r1[k]) for k in range(len(TCOLS))]; plot_A(axA)
r2 = gs[1].subgridspec(1, 5, width_ratios=[0.36, 1.5, 0.62, 0.16, 0.42], wspace=0.05)
axB = fig.add_subplot(r2[1]); axBs = fig.add_subplot(r2[2]); plot_B(axB, axBs)
axC = fig.add_subplot(r2[4]); plot_C(axC)
PANELS = {"A": axA, "B": [axB, axBs], "C": [axC]}
fig.canvas.draw(); rr = fig.canvas.get_renderer()
LET = []
for l, axes in PANELS.items():
    bb = mpl.transforms.Bbox.union([a.get_tightbbox(rr) for a in axes]).transformed(fig.transFigure.inverted())
    LET.append(fig.text(bb.x0 - 0.004, bb.y1 + 0.004, l, fontsize=11, fontweight="bold", ha="left", va="bottom"))
for ext in ("pdf", "png"):
    fig.savefig(FINAL / f"Fig_S3_nr_topics.{ext}", dpi=300, bbox_inches="tight")
for t in LET:
    t.set_visible(False)
fig.canvas.draw(); rr = fig.canvas.get_renderer()
for l, axes in PANELS.items():
    bb = mpl.transforms.Bbox.union([a.get_tightbbox(rr) for a in axes]).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(FINAL / f"Fig_S3_nr_topics_{l}.pdf", bbox_inches=bb.padded(0.04))
for t in LET:
    t.set_visible(True)
plt.show()
print("Saved", FINAL / "Fig_S3_nr_topics.pdf")
