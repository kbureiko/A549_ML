# =============================================================================
# Figure 1 - Machine learning classification of dex-responsive enhancers in A549 cells
#   A  workflow (drawn; edit the STEPS text below)
#   B  balanced accuracy of the three binary CatBoost models (25 outer-CV folds = 5 repeats x 5 folds)
#      next to the label-permutation null (99 permutations; mean over the 5 folds of one CV repeat; empirical p)
#   C-E  top 15 CV-stable features per model: fraction of the 25 outer folds in which the feature had a
#      positive held-out permutation importance (ordered by that fraction, then mean importance)
#   F-H  UMAP of all 8,270 enhancers (layout LAYOUT from scripts/10_feature_map_layouts.py: rank-normal union of the top features,
#      PCA, UMAP weakly supervised by class for visualisation only) coloured by F response class, G genomic
#      annotation, H main chromatin topic of NR enhancers (scripts/11_nr_topics.py; UP / DN grey)
#
# Notes for the legend:
#   - CV: 5 x 5 repeated stratified folds; preprocessing, decision threshold (max F1 on inner 3-fold CV) and
#     permutation importance (held-out fold, average-precision scoring, 10 permutations) all fold-internal.
#   - C-E show stability, not effect size: 100 % = positive held-out importance in all 25 folds. The dotted
#     line marks the 60 % stability threshold used to call a feature stable.
#   - UMAP (F-H): class separation on the map is partly imposed by the weak supervision and is not a result;
#     annotation and NR topics are computed independently of the map.
# Outputs (FIGURES): Fig_1.pdf/png (complete figure), Fig_1_<panel>.pdf (each panel, no letters),
#          Fig_1_stable_features.tsv
# Requires: pandas, numpy, matplotlib; the cached embeddings of scripts/10_feature_map_layouts.py (PREP)
# =============================================================================
# %%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from a549_enhancers.paths import PROJECT, ML, FIGURES, RUNS as RUN_DIRS, FEATURE_MAP, NR_TOPICS   # local folders, set in paths.toml
FINAL = FIGURES; FINAL.mkdir(parents=True, exist_ok=True)   # final figures + their tables
RUNS = {"NR vs UP": RUN_DIRS["nr_vs_up"],
        "UP vs DN": RUN_DIRS["up_vs_dn"],
        "NR vs DN": RUN_DIRS["nr_vs_dn"]}
FM = ML / "feature_matrix_all_classes_GRannotated.csv"
# label-permutation test (output folder of scripts/04_label_permutation_test.py)
PERM_DIR = ML / "label_permutation_test"
PERM_NULL = PERM_DIR / "null_distribution_mean_over_folds_1rep.csv"      # one row per task x permutation
PERM_RES = PERM_DIR / "label_permutation_test_results_1rep.csv"          # observed, p per task x metric
STABILITY, TOP_N = 0.60, 15
# UMAP: cached layout from scripts/10_feature_map_layouts.py (MODE = "supervised"); NR topics from scripts/11_nr_topics.py
PREP = FEATURE_MAP
EMB_FILE, LAYOUT = "umap_grid_embeddings_supervised.npz", "nn30_md0.1_cosine_sup0.5"
TOPICS_FILE = NR_TOPICS / "MS_nr_topics_usage.tsv"
UMAP_ALPHA = 0.45

needed = [FM, PREP / EMB_FILE, PREP / "umap_pcs.tsv", TOPICS_FILE] + [r / "summaries" / f for r in RUNS.values()
                 for f in ("all_cv_metrics.csv", "feature_importance_stability_summary.csv",
                           )]
missing = [str(p) for p in needed if not p.exists()]
if missing:
    raise FileNotFoundError("Not found (edit paths at top):\n  " + "\n  ".join(missing))

# ----------------------------------------------------------------- shared A549 style
FONT = "Open Sans" if any("Open Sans" in f.name for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
mpl.rcParams.update({
    "font.family": [FONT, "DejaVu Sans"], "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none",
    "axes.grid": False,
})
CLS = {"UP": "#D1495B", "DN": "#3B6FB6", "NR": "#5DAA82"}
CLS_ORDER = ["NR", "UP", "DN"]
ANNOT = {"Intergenic": "#F2B134", "Intronic": "#9B3FA0", "CpG island": "#2A9D8F", "Other": "#7F7F7F"}
ANNOT_DRAW = ["Other", "Intronic", "Intergenic", "CpG island"]   # drawing order in G (last on top)
ANNOT_MAP = {"Intergenic": "Intergenic", "Intonic": "Intronic", "Intronic": "Intronic", "CpG": "CpG island",
             "Other": "Other"}
TASKS = list(RUNS)
CAT_ORDER = ["Dex", "Control", "Genomic"]
CAT_LABEL = {"Dex": "Dex-treated (ChIP, ATAC)", "Control": "Control / untreated", "Genomic": "Sequence / genomic"}
CAT_BAR = {"Dex": "#2B4C7E", "Control": "#9DB9D5", "Genomic": "#3E9651"}
CAT_TXT = {"Dex": "#2B4C7E", "Control": "#222222", "Genomic": "#2F7D3F"}


def panel_letters(fig, axes):
    fig.canvas.draw(); r = fig.canvas.get_renderer(); out = []
    for letter, ax in axes.items():
        bb = ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
        out.append(fig.text(bb.x0 - 0.004, bb.y1 + 0.003, letter, fontsize=11, fontweight="bold",
                            ha="left", va="bottom"))
    return out


def export_panels(fig, panels, stem, letters, pad=0.04):
    """Save every panel as its own PDF (tight box around the listed artists; letters hidden)."""
    for t in letters: t.set_visible(False)
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    for name, artists in panels.items():
        boxes = [a.get_tightbbox(r) if hasattr(a, "get_tightbbox") else a.get_window_extent(r) for a in artists]
        bb = mpl.transforms.Bbox.union(boxes).transformed(fig.dpi_scale_trans.inverted()).padded(pad)
        fig.savefig(FINAL / f"{stem}_{name}.pdf", bbox_inches=bb)
    for t in letters: t.set_visible(True)


def category(f):
    if f.startswith("motif_binary::") or f in ("Enh_len", "Distance to TSS", "annotation", "gc"):
        return "Genomic"
    if "_dex_" in f or f == "ATAC_dex_1h":
        return "Dex"
    return "Control"          # "_eth_" tracks, ATAC_ctrl and untreated ENCODE ChIP-seq


RENAME = {"JUN": "cJUN", "NR3C1": "GR", "H2AFZ": "H2A.Z", "Enh_len": "Enhancer length",
          "POLR2AphosphoS2": "POLR2A pS2", "gc": "GC content"}


def short(f):
    if f.startswith("motif_binary::"):
        return f.split("::")[1].split("/")[0].split("(")[0] + " motif"
    if f in ("Enh_len", "Distance to TSS", "gc"):
        return RENAME.get(f, f)
    g = "ATAC" if f.startswith("ATAC") else f.split("_")[0]
    return RENAME.get(g, g)


# ----------------------------------------------------------------- data
metrics = pd.concat([pd.read_csv(r / "summaries" / "all_cv_metrics.csv").assign(task_display=t)
                     for t, r in RUNS.items()], ignore_index=True)
assert (metrics.groupby("task_display").size() == 25).all(), "expected 25 outer-CV folds per task"
stab, top = {}, {}
for t, r in RUNS.items():
    s = pd.read_csv(r / "summaries" / "feature_importance_stability_summary.csv")
    s["cat"] = s.feature.map(category)
    stab[t] = s[s.fraction_positive_importance >= STABILITY].copy()
    top[t] = stab[t].sort_values(["fraction_positive_importance", "mean_importance"], ascending=False).head(TOP_N)
perm = perm_res = None
if PERM_NULL.exists() and PERM_RES.exists():
    perm = pd.read_csv(PERM_NULL)
    perm_res = pd.read_csv(PERM_RES).query("metric == 'balanced_accuracy'").set_index("task_label")
else:
    print("label-permutation files not found - panel B without the null distribution:", PERM_NULL)

fm = pd.read_csv(FM, low_memory=False)
fm["annot"] = fm["annotation"].map(ANNOT_MAP).fillna("Other")
n_cls = fm["class"].value_counts()


pcs = pd.read_csv(PREP / "umap_pcs.tsv", sep="\t", usecols=["enhancer_id"])
E = np.load(PREP / EMB_FILE)[LAYOUT]
emb = pd.DataFrame({"enhancer_id": pcs.enhancer_id, "dim1": E[:, 0], "dim2": E[:, 1]}).merge(
    fm[["enhancer_id", "class", "annot", "GR_bound"]], on="enhancer_id", how="left", validate="1:1")
assert emb["class"].notna().all(), "embedding enhancers missing from the feature matrix"
emb["GRb"] = emb.GR_bound.astype(str).str.lower().eq("true")
tu = pd.read_csv(TOPICS_FILE, sep="\t")
TOPIC_COLS = [c for c in tu.columns if c[:1] == "T" and c[1:2].isdigit()]
tu["main"] = tu[TOPIC_COLS].idxmax(axis=1).str.split(" ").str[0]
emb["topic"] = emb.enhancer_id.map(tu.set_index("enhancer_id")["main"])
emb.loc[emb["class"] != "NR", "topic"] = np.nan
print(f"font: {FONT} | classes {n_cls.to_dict()} | stable features", {t: len(stab[t]) for t in TASKS},
      f"| UMAP {LAYOUT}: {len(emb):,} enhancers | permutation file:", "found" if perm is not None else "not found")


# ----------------------------------------------------------------- A: graphical abstract (drawn in inches)
# Geometry is in inches on an equal-aspect axes (AW x AH); edit the TXT strings to change wording.
AW, AH = 6.6, 2.05
TXT = {
    "cell": "A549", "pro": ("PRO-seq", "TRE detection\ndex response (45 min)"),
    "chip": ("ChIP- & ATAC-seq", "TFs & HMs, dex & ctrl"),
    "motif": "238 known motifs", "genomic": "genomic annotation, GC,\ndistance to TSS, length",
    "brk1": "genome-wide", "brk2": "genomic features",
    "ml": [("1. Cross-validation", "model performance"), ("2. Feature stability", "across folds"),
           ("3. Model training", "3 CatBoost models"), ("4. Local explanation", "LIME rules")],
    "cls": [("NR", "non-responsive"), ("UP", "activated"), ("DN", "repressed")],
    "cards": [("TF1", "+"), ("TF2", "–"), ("motif1", "–"), ("motif2", "+")],
    "val": "siRNA depletion of\nEP300, FOXA1, TEAD4\n+ PRO-seq / BRB-seq",
    "cap": ["Input data", "Feature matrix generation", "Machine learning", "Interpretation & validation"],
}
GRID_ROWS = ["NR", "NR", "UP", "DN", "⋮", "UP", "NR"]
GRID_VALS = [[1, 3, "…", 3], [2, 4, "…", 0], [7, 7, "…", 3], [1, 2, "…", 0], ["⋮", "⋮", "…", "⋮"],
             [8, 8, "…", 2], [8, 1, "…", 2]]
GRID_COLS = [("TF1", "Dex"), ("TF2", "Control"), ("…", None), ("motifX", "Genomic")]
GREEN = "#2F7D3F"


def _box(ax, x, y, w, h, fc="#EFEFEF", ec="#8C8C8C", lw=0.6, r=0.035):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                clip_on=False))


def _fat_arrow(ax, x0, x1, y, h=0.17, color="#222222", vertical=False):
    if not vertical:
        L = x1 - x0; xs_ = x0 + 0.5 * L
        pts = [(x0, y - h * .28), (xs_, y - h * .28), (xs_, y - h * .6), (x1, y), (xs_, y + h * .6),
               (xs_, y + h * .28), (x0, y + h * .28)]
    else:   # x0, x1 are y0 (start), y1 (tip); y is x
        L = x1 - x0; ys_ = x0 + 0.5 * L
        pts = [(y - h * .28, x0), (y - h * .28, ys_), (y - h * .6, ys_), (y, x1), (y + h * .6, ys_),
               (y + h * .28, ys_), (y + h * .28, x0)]
    ax.add_patch(mpl.patches.Polygon(pts, closed=True, fc=color, ec="none", clip_on=False))


def _bracket(ax, x, y0, y1, label):
    ax.plot([x + 0.04, x, x, x + 0.04], [y0, y0, y1, y1], color="#8C8C8C", lw=0.6, clip_on=False)
    ax.text(x - 0.05, (y0 + y1) / 2, label, rotation=90, ha="center", va="center", fontsize=5.8, color="#444444")


def plot_A(ax):
    ax.set_xlim(0, AW); ax.set_ylim(0, AH); ax.set_aspect("equal"); ax.axis("off")
    B, T, S, S2 = 0.30, 1.90, 6.3, 5.9          # content bottom / top, body font sizes
    # ---- input data
    x0, w = 0.26, 1.20
    ax.text(x0 + w / 2, T + 0.08, TXT["cell"], ha="center", va="bottom", fontsize=6.8, fontweight="bold")
    _box(ax, x0, 1.46, w, 0.42, fc="#6B6B6B", ec="#4A4A4A")
    ax.text(x0 + w / 2, 1.76, TXT["pro"][0], ha="center", va="center", fontsize=6.8, fontweight="bold", color="white")
    ax.text(x0 + w / 2, 1.59, TXT["pro"][1], ha="center", va="center", fontsize=S2, color="white", linespacing=1.05)
    _box(ax, x0, 1.02, w, 0.38)
    ax.text(x0 + w / 2, 1.29, TXT["chip"][0], ha="center", va="center", fontsize=6.8, fontweight="bold")
    ax.text(x0 + w / 2, 1.13, TXT["chip"][1], ha="center", va="center", fontsize=S2, color="#333333")
    _box(ax, x0, 0.66, w, 0.26)
    ax.text(x0 + w / 2, 0.79, TXT["motif"], ha="center", va="center", fontsize=S, color="#222222")
    _box(ax, x0, B, w, 0.30)
    ax.text(x0 + w / 2, B + 0.15, TXT["genomic"], ha="center", va="center", fontsize=S2, color="#222222",
            linespacing=1.05)
    _bracket(ax, x0 - 0.07, 1.02, 1.88, TXT["brk1"]); _bracket(ax, x0 - 0.07, B, 0.92, TXT["brk2"])
    _fat_arrow(ax, x0 + w + 0.08, x0 + w + 0.32, 1.10)
    # ---- feature matrix sketch
    xc, wc, yb, rh = x0 + w + 0.54, 0.26, 0.52, 0.17            # class column
    n = len(GRID_ROWS); yt = yb + n * rh
    ax.text(xc + wc / 2, yt + 0.05, "class", ha="center", va="bottom", fontsize=S)
    for i, c in enumerate(GRID_ROWS):
        y = yt - (i + 1) * rh
        if c in CLS:
            ax.add_patch(mpl.patches.Rectangle((xc, y), wc, rh, fc=CLS[c], ec="white", lw=0.5, clip_on=False))
            ax.text(xc + wc / 2, y + rh / 2, c, ha="center", va="center", fontsize=S2, color="white", fontweight="bold")
        else:
            ax.add_patch(mpl.patches.Rectangle((xc, y), wc, rh, fc="#D9D9D9", ec="white", lw=0.5, clip_on=False))
            ax.text(xc + wc / 2, y + rh / 2, c, ha="center", va="center", fontsize=S2, color="#444444")
    ax.text(xc - 0.07, (yb + yt) / 2, f"{n_cls.sum():,} enhancers", rotation=90, ha="center", va="center", fontsize=S)
    ax.text(xc + wc + 0.11, (yb + yt) / 2, "×", ha="center", va="center", fontsize=8)
    xg = xc + wc + 0.22; cw = [0.17, 0.17, 0.22, 0.19]
    ax.text(xg + sum(cw) / 2, yt + 0.22, "feature scores", ha="center", va="bottom", fontsize=S, style="italic")
    xcol = xg
    for (name, cat), cwi in zip(GRID_COLS, cw):
        ax.text(xcol + cwi / 2, yt + 0.05, name, ha="center", va="bottom", fontsize=S2 - 0.6,
                color=CAT_TXT[cat] if cat else "#444444", fontweight="bold" if cat else "normal")
        xcol += cwi
    for i, row in enumerate(GRID_VALS):
        y = yt - (i + 1) * rh; xcol = xg
        for v, cwi in zip(row, cw):
            ax.add_patch(mpl.patches.Rectangle((xcol, y), cwi, rh, fc="#EFEFEF", ec=GREEN, lw=0.6, clip_on=False))
            ax.text(xcol + cwi / 2, y + rh / 2, str(v), ha="center", va="center", fontsize=S2, color="#333333")
            xcol += cwi
    ax.text(xg + sum(cw) / 2, yb - 0.06, "340 features", ha="center", va="top", fontsize=S)
    xe = xg + sum(cw)
    _fat_arrow(ax, xe + 0.08, xe + 0.32, 1.10)
    # ---- machine learning
    xm, wm = xe + 0.41, 1.05
    k = len(TXT["ml"]); hm = (T - B - 0.04 * (k - 1)) / k
    for i, (t1, t2) in enumerate(TXT["ml"]):
        y = T - (i + 1) * hm - i * 0.04
        _box(ax, xm, y, wm, hm, fc="#DCDCDC", ec=GREEN, lw=1.0, r=0.02)
        ax.text(xm + wm / 2, y + hm * 0.64, t1, ha="center", va="center", fontsize=6.2, fontweight="bold")
        ax.text(xm + wm / 2, y + hm * 0.30, t2, ha="center", va="center", fontsize=S2, color="#333333")
    _fat_arrow(ax, xm + wm + 0.08, xm + wm + 0.32, 1.10)
    # ---- interpretation and validation
    xi = xm + wm + 0.41; wi = AW - xi
    ax.text(xi + wi / 2, T + 0.02, "stable features & LIME rules", ha="center", va="bottom", fontsize=6.4,
            fontweight="bold")
    cw_, ch_, step = 0.30, 0.15, 0.18
    for j, (name, sign) in enumerate(TXT["cards"]):
        y = T - 0.20 - j * step
        _box(ax, xi, y - ch_, cw_, ch_, fc="#EFEFEF", ec="#B0B0B0", r=0.02)
        ax.text(xi + cw_ / 2, y - ch_ / 2, name, ha="center", va="center", fontsize=S2 - 0.4)
        ax.text(xi + cw_ + 0.045, y - ch_ / 2, sign, ha="center", va="center", fontsize=6.5)
    ybr1, ybr0 = T - 0.20, T - 0.20 - 3 * step - ch_
    xb = xi + cw_ + 0.10
    ax.plot([xb, xb + 0.035, xb + 0.035, xb], [ybr0, ybr0, ybr1, ybr1], color="#444444", lw=0.6, clip_on=False)
    wcl, hcl = 0.98, 0.17; xcl = xi + wi - wcl
    ycl = [T - 0.24, T - 0.45, T - 0.66]           # top edges of the class boxes
    for (c, lab), y in zip(TXT["cls"], ycl):
        _box(ax, xcl, y - hcl, wcl, hcl, fc="white", ec=CLS[c], lw=0.9, r=0.02)
        ax.text(xcl + 0.035, y - hcl / 2, c, ha="left", va="center", fontsize=S2, fontweight="bold", color=CLS[c])
        ax.text(xcl + 0.18, y - hcl / 2, lab, ha="left", va="center", fontsize=S2 - 1.4, color="#333333")
        ax.text(xcl + wcl - 0.03, y - hcl / 2, f"{n_cls[c]:,}", ha="right", va="center", fontsize=S2 - 1.0,
                fontweight="bold", color=CLS[c])
        ax.add_patch(FancyArrowPatch((xb + 0.05, (ybr0 + ybr1) / 2), (xcl - 0.015, y - hcl / 2), arrowstyle="-|>",
                                     mutation_scale=5, lw=0.6, color="#444444", clip_on=False))
    xmid = xcl + wcl / 2
    y1 = ycl[-1] - hcl
    _fat_arrow(ax, y1 - 0.03, y1 - 0.16, xmid, h=0.16, vertical=True)
    yv = y1 - 0.18
    hv, wv = 0.42, 0.98                                   # validation box (compact, under the class boxes)
    _box(ax, xmid - wv / 2, yv - hv, wv, hv, fc="#DCDCDC", ec=GREEN, lw=0.9, r=0.02)
    ax.text(xmid, yv - hv / 2, TXT["val"], ha="center", va="center", fontsize=S2 - 0.3, linespacing=1.05)
    # ---- captions
    for cap, xc_ in zip(TXT["cap"], [x0 + w / 2, (xc - 0.1 + xe) / 2, xm + wm / 2, xi + wi / 2]):
        ax.text(xc_, 0.06, cap, ha="center", va="center", fontsize=6.8, color="#222222")


# ----------------------------------------------------------------- B: balanced accuracy
def plot_B(ax):
    """balanced accuracy of the 25 outer folds (true labels) next to the label-permutation null (as in Fig. A3)"""
    rng = np.random.default_rng(0)
    TRUE_C, NULL_C = "#333333", "#9A9A9A"
    DX = 0.2 if perm is not None else 0.0
    for i, t in enumerate(TASKS):
        v = metrics.loc[metrics.task_display == t, "balanced_accuracy"].values
        ax.boxplot(v, positions=[i - DX], widths=0.3, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor="white", edgecolor=TRUE_C, lw=0.7),
                   medianprops=dict(color=TRUE_C, lw=1.0), whiskerprops=dict(lw=0.7, color=TRUE_C),
                   capprops=dict(lw=0.7, color=TRUE_C))
        ax.scatter(i - DX + rng.uniform(-0.09, 0.09, len(v)), v, s=6, color="#666666", alpha=0.8, lw=0, zorder=3)
        ax.text(i - DX, v.max() + 0.008, f"{np.median(v):.2f}", ha="center", va="bottom", fontsize=6.2,
                color="#333333")
        if perm is None or not (perm.task_label == t).any():
            continue
        null = perm.loc[perm.task_label == t, "balanced_accuracy"].values
        ax.boxplot(null, positions=[i + DX], widths=0.3, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor="#DDDDDD", edgecolor=NULL_C, lw=0.7),
                   medianprops=dict(color=NULL_C, lw=1.0), whiskerprops=dict(lw=0.7, color=NULL_C),
                   capprops=dict(lw=0.7, color=NULL_C))
        r = perm_res.loc[t]; p, pmin = r.p_value, r.minimum_attainable_p
        ytop = v.max() + 0.062
        ax.plot([i - DX, i - DX, i + DX, i + DX], [v.max() + 0.045, ytop, ytop, null.max() + 0.012],
                lw=0.6, color="#333333", clip_on=False)
        ax.text(i, ytop + 0.008, f"p {'≤' if p <= pmin + 1e-12 else '='} {p:.2g}", ha="center", va="bottom",
                fontsize=5.9, color="#333333")
    ax.axhline(0.5, ls=":", lw=0.6, c="#888888", zorder=0)
    ax.set_xticks(range(len(TASKS))); ax.set_xticklabels(TASKS, rotation=30, ha="right")
    ax.set_xlim(-0.6, len(TASKS) - 0.4); ax.set_ylim(0.4, 1.1)
    ax.set_yticks(np.arange(0.4, 1.01, 0.1)); ax.spines["left"].set_bounds(0.4, 1.0)
    ax.set_ylabel("Balanced accuracy\n(25 outer-CV folds)")
    if perm is not None:
        n_perm = int(perm.groupby("task_label").size().min())
        h = [mpl.patches.Patch(facecolor="white", edgecolor=TRUE_C, lw=0.7, label="True labels"),
             mpl.patches.Patch(facecolor="#DDDDDD", edgecolor=NULL_C, lw=0.7, label=f"Permuted ({n_perm}×)")]
        ax.legend(handles=h, frameon=False, loc="lower left", fontsize=5.8, handlelength=1.0, borderaxespad=0.1)
    ax.set_title("Model performance", pad=4)


# ----------------------------------------------------------------- C-E: top stable features
def plot_CDE(fig, spec):
    sub = spec.subgridspec(1, 3, wspace=1.15)
    axes = []; x0 = 0.5
    for j, t in enumerate(TASKS):
        ax = fig.add_subplot(sub[0, j]); axes.append(ax)
        d = top[t].iloc[::-1]
        y = np.arange(len(d)); fr = d.fraction_positive_importance.values
        ax.hlines(y, x0, fr, color="#BBBBBB", lw=0.8, zorder=1)
        ax.scatter(fr, y, s=22, color=[CAT_BAR[c] for c in d.cat], lw=0, zorder=3)
        for yi, v in zip(y, fr):
            ax.text(v + 0.028, yi, f"{v * 100:.0f}%", va="center", ha="left", fontsize=5.4, color="#444444")
        ax.axvline(STABILITY, ls=":", lw=0.6, c="#888888", zorder=0)
        ax.set_yticks(y); ax.set_yticklabels([short(f) for f in d.feature], fontsize=6.4)
        for tl, c in zip(ax.get_yticklabels(), d.cat):
            tl.set_color(CAT_TXT[c])
        ax.tick_params(axis="y", length=0, pad=2)
        ax.set_ylim(-0.6, len(d) - 0.4)
        ax.set_xlim(x0, 1.2); ax.set_xticks([0.6, 0.8, 1.0]); ax.set_xticklabels(["60", "80", "100"])
        ax.spines["bottom"].set_bounds(x0, 1.0)
        ax.tick_params(axis="x", labelsize=6)
        ax.set_xlabel("Folds with positive held-out\nimportance (% of 25)", fontsize=6.6)
        ax.set_title(t, fontsize=7.8)
    return axes


# ----------------------------------------------------------------- F-H: UMAP
UCLS = {"NR": "tab:green", "UP": "tab:red", "DN": "tab:blue"}
UANN = {"CpG island": "#3FA7A0", "Intergenic": "#8C8C8C", "Intronic": "#E1A43A", "Other": "#7A5195"}
TOPIC_COL = {"T1": "#2E7D5B", "T2": "#D4A62A", "T3": "#4D4D4D", "T4": "#7A5195", "T5": "#7FA7D1"}
TOPIC_NAME = {"T1": "Active (GR/EP300/FOXA1)", "T2": "Promoter-like", "T3": "Polycomb-like", "T4": "CTCF/cohesin",
              "T5": "H3K4me1-high"}


def _dot(c, lab):
    return mpl.lines.Line2D([], [], marker="o", ls="", ms=4, color=c, label=lab)


def plot_umap(ax, by):
    x, y = emb.dim1.values, emb.dim2.values
    if by == "class":
        for c, s_, a in (("NR", 0.6, UMAP_ALPHA * 0.8), ("UP", 2.0, UMAP_ALPHA + 0.2), ("DN", 2.4, UMAP_ALPHA + 0.2)):
            m = (emb["class"] == c).values
            ax.scatter(x[m], y[m], s=s_, c=UCLS[c], lw=0, alpha=a, rasterized=True)
        h = [_dot(UCLS[c], f"{c} ({n_cls[c]:,})") for c in ("NR", "UP", "DN")]; title = "Response class"
    elif by == "gr":
        g = emb.GRb.values
        ax.scatter(x[~g], y[~g], s=0.6, c="#D9D9D9", lw=0, rasterized=True)
        ax.scatter(x[g], y[g], s=0.9, c="#5B4B9A", lw=0, alpha=UMAP_ALPHA, rasterized=True)
        h = [_dot("#5B4B9A", f"GR-bound ({g.sum():,})"), _dot("#D9D9D9", f"unbound ({(~g).sum():,})")]
        title = "GR binding (dex)"
    elif by == "annot":
        h = []
        for a_ in ("Intergenic", "Intronic", "Other", "CpG island"):
            m = (emb.annot == a_).values
            ax.scatter(x[m], y[m], s=0.8 if a_ != "CpG island" else 1.4, c=UANN[a_], lw=0, alpha=UMAP_ALPHA,
                       rasterized=True)
            h.append(_dot(UANN[a_], f"{a_} ({m.sum():,})"))
        title = "Genomic annotation"
    else:
        nr_ = emb.topic.notna().values
        ax.scatter(x[~nr_], y[~nr_], s=0.6, c="#E3E3E3", lw=0, rasterized=True)
        h = []
        for t in TOPIC_COL:
            m = (emb.topic == t).values
            ax.scatter(x[m], y[m], s=1.2, c=TOPIC_COL[t], lw=0, alpha=0.8, rasterized=True)   # stronger: same look as S3
            h.append(_dot(TOPIC_COL[t], f"{t} {TOPIC_NAME[t]} ({m.sum():,})"))
        h.append(_dot("#E3E3E3", "UP / DN")); title = "NR enhancers: main topic"
    ax.legend(handles=h, frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.07), handletextpad=0.1,
              labelspacing=0.25, fontsize=5.4)
    ax.set_xticks([]); ax.set_yticks([])
    # UMAP axis key: short arrows in the lower-left corner
    for dx, dy, lab, kw in ((0.16, 0, "UMAP1", dict(ha="left", va="top")), (0, 0.16, "UMAP2", dict(ha="right", va="bottom", rotation=90))):
        ax.annotate("", xy=(0.02 + dx, 0.02 + dy), xytext=(0.02, 0.02), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#333333", mutation_scale=6))
        ax.text(0.02 + (0.0 if dx else -0.012), 0.02 + (-0.015 if dx else 0.0), lab, transform=ax.transAxes,
                fontsize=5.6, color="#333333", **kw)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.set_title(title)


# ----------------------------------------------------------------- assemble
fig = plt.figure(figsize=(7.2, 9.6))
fig.subplots_adjust(left=0.085, right=0.985, top=0.985, bottom=0.03)
gs = fig.add_gridspec(4, 1, height_ratios=[0.95, 1.35, 0.1, 1.05], hspace=0.38)
axA = fig.add_subplot(gs[0]); plot_A(axA)
pA = gs[0].get_position(fig)                       # same width as the other rows, exact aspect, top-aligned
hA = pA.width * fig.get_figwidth() * AH / AW / fig.get_figheight()
axA.set_position([pA.x0, pA.y1 - hA, pA.width, hA])
row2 = gs[1].subgridspec(1, 2, width_ratios=[1.05, 3.3], wspace=0.45)
axB = fig.add_subplot(row2[0]); plot_B(axB)
axCDE = plot_CDE(fig, row2[1])
row3 = gs[3].subgridspec(1, 3, wspace=0.15)
axF, axG, axH = [fig.add_subplot(row3[k]) for k in range(3)]
for a_, by in zip((axF, axG, axH), ("class", "annot", "topic")):
    plot_umap(a_, by)
# feature-category key for C-E (dot and label colour), in the spacer row below the lollipop panels
hk = [mpl.lines.Line2D([], [], marker="o", ls="", ms=5, color=CAT_BAR[c], label=CAT_LABEL[c]) for c in CAT_ORDER]
fig.canvas.draw(); rr = fig.canvas.get_renderer()
bbC = mpl.transforms.Bbox.union([a.get_tightbbox(rr) for a in axCDE]).transformed(fig.transFigure.inverted())
lgK = fig.legend(handles=hk, loc="upper center", bbox_to_anchor=((bbC.x0 + bbC.x1) / 2, bbC.y0 - 0.004),
                 ncol=3, frameon=False, handletextpad=0.3, columnspacing=1.4, fontsize=6.6,
                 title="Feature category (dot and label colour)", title_fontsize=6.6)
for t, c in zip(lgK.get_texts(), CAT_ORDER):
    t.set_color(CAT_TXT[c])
letters = panel_letters(fig, {"A": axA, "B": axB, "C": axCDE[0], "D": axCDE[1], "E": axCDE[2],
                              "F": axF, "G": axG, "H": axH})
for ext in ("pdf", "png"):
    fig.savefig(FINAL / f"Fig_1.{ext}", dpi=300, bbox_inches="tight")
export_panels(fig, {"A": [axA], "B": [axB], "C": [axCDE[0]], "D": [axCDE[1]], "E": [axCDE[2]], "CDE_key": [lgK],
                    "F": [axF], "G": [axG], "H": [axH]}, "Fig_1", letters)
plt.show()
pd.concat([top[t].assign(task=t) for t in TASKS]).assign(feature_short=lambda d: d.feature.map(short)) \
    [["task", "feature", "feature_short", "cat", "fraction_positive_importance", "mean_importance", "top10_fraction"]] \
    .to_csv(FINAL / "Fig_1_stable_features.tsv", sep="\t", index=False)
print("Saved to", FINAL / "Fig_1.pdf", "+ per-panel PDFs Fig_1_A..H.pdf")
