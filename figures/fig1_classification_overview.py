# =============================================================================
# Figure 1 - Machine learning classification of dex-responsive enhancers in A549 cells
#   A  workflow (drawn; edit the STEPS text below)
#   B  balanced accuracy of the three binary CatBoost models (outer-CV folds = N_REPEATS x N_SPLITS of config.py).
#      The label-permutation test is reported in the text only (SHOW_PERMUTATION = False); True draws the null
#      distribution next to the boxes when the files of scripts/04_label_permutation_test.py exist.
#   C-E  held-out permutation importance per model, groups and single features in one ranking (top TOP_N rows by
#      mean decrease in average precision). A row is a feature group (family of targets or cluster of correlated
#      tracks, all members permuted together; data/feature_groups.csv of the run) or a single feature that belongs
#      to no group, so every feature appears once: inside its group(s) or as its own row.
#      dot = mean over the evaluations (circle: group, diamond: single feature; filled: positive in at least
#      STABILITY of the evaluations); line = 10th to 90th percentile; number on the right = % of evaluations with a
#      positive decrease; [n] = number of features in a group.
#      The complete ranking with the largest single member of every group: figures/figS_grouped_importance.py
#   F-H  UMAP of all 8,270 enhancers (layout LAYOUT from scripts/10_feature_map_layouts.py: rank-normal union of the top features,
#      PCA, UMAP weakly supervised by class for visualisation only) coloured by F response class, G genomic
#      annotation, H main chromatin topic of NR enhancers (scripts/11_nr_topics.py; UP / DN grey)
#
# Notes for the legend:
#   - CV: 5 folds x 10 repeats (config.py); preprocessing, decision threshold (max F1 on inner 3-fold CV) and
#     permutation importance (held-out fold, average-precision scoring, 10 permutations) all fold-internal.
#   - C-E: groups and single features are on the same scale (decrease in held-out average precision). The final
#     models and the LIME rules (Fig. 2) use the stable SINGLE features only, so a group without a stable member
#     has no LIME rule. Importance has no direction.
#   - UMAP (F-H): class separation on the map is partly imposed by the weak supervision and is not a result;
#     annotation and NR topics are computed independently of the map.
# Outputs (FIGURES): Fig_1.pdf/png (complete figure), Fig_1_<panel>.pdf (each panel, no letters),
#          Fig_1_importance_ranking.tsv (rows of C-E), Fig_1_stable_features.tsv (stable single features)
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

from a549_enhancers.config import N_REPEATS, N_SPLITS, STABLE_MIN_FRACTION_POSITIVE_IMPORTANCE
from a549_enhancers.feature_groups import feature_layer, target_of
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
SHOW_PERMUTATION = False     # panel B without the label-permutation null (reported in the text)
STABILITY, TOP_N = STABLE_MIN_FRACTION_POSITIVE_IMPORTANCE, 15     # stability threshold from config.py (0.80)
N_EVAL = N_SPLITS * N_REPEATS                                       # held-out evaluations per model (50)
# UMAP: cached layout from scripts/10_feature_map_layouts.py (MODE = "supervised"); NR topics from scripts/11_nr_topics.py
PREP = FEATURE_MAP
EMB_FILE, LAYOUT = "umap_grid_embeddings_supervised.npz", "nn50_md0.1_euclidean_sup0.5"
TOPICS_FILE = NR_TOPICS / "MS_nr_topics_usage.tsv"
UMAP_ALPHA = 0.45

needed = [FM, PREP / EMB_FILE, PREP / "umap_pcs.tsv", TOPICS_FILE] + [r / "summaries" / f for r in RUNS.values()
                 for f in ("all_cv_metrics.csv", "feature_importance_stability_summary.csv",
                           "all_cv_grouped_permutation_importance.csv")] + [r / "data" / "feature_groups.csv" for r in RUNS.values()]
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
ANNOT = {"Intergenic": "#F2B134", "Intronic": "#9B3FA0", "CpG": "#2A9D8F", "Other": "#7F7F7F"}
ANNOT_DRAW = ["Other", "Intronic", "Intergenic", "CpG"]   # drawing order in G (last on top)
ANNOT_MAP = {"Intergenic": "Intergenic", "Intonic": "Intronic", "Intronic": "Intronic", "CpG": "CpG",
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
    """Genomic / Dex / Control, with the layer rules of the pipeline (as in Fig. 2)"""
    return {"genomic": "Genomic", "dex": "Dex", "control": "Control"}[feature_layer(f)]


# names as in Fig. 2 (LIME rules)
RENAME = {"JUN": "cJUN", "NR3C1": "GR", "H2AFZ": "H2A.Z", "Enh_len": "Enhancer length",
          "POLR2AphosphoS2": "POLR2A pS2", "gc": "GC content", "Distance to TSS": "TSS distance",
          "annotation": "Annotation", "GRE_score": "GRE score",
          "nbr_enhancers_50kb": "Enhancers ≤50 kb", "nbr_enhancer_distance": "Nearest enhancer",
          "nbr_GRpeaks_50kb": "GR peaks ≤50 kb", "nbr_strongGRpeaks_50kb": "Strong GR peaks ≤50 kb",
          "nbr_GRpeak_distance": "Nearest GR peak", "nbr_strongGRpeak_distance": "Nearest strong GR peak"}
MOTIF_RENAME = {"Unknown-ESC-element": "ESC element", "c-Jun-CRE": "cJun-CRE", "RUNX-AML": "RUNX"}
FAMILY_LABEL = {      # predefined families of config.FEATURE_FAMILIES; others: the name without "family: "
    "family: all motif indicators": "All motif groups",
    "family: GRE (motif group and score)": "GRE motif and score",
    "family: neighbourhood, other enhancers": "Enhancers nearby",
    "family: neighbourhood, GR peaks after dex": "GR peaks nearby",
    "family: CTCF and cohesin (control)": "CTCF, cohesin (control)", "family: CTCF and cohesin (dex)": "CTCF, cohesin (dex)",
    "family: active histone marks (control)": "Active marks (control)", "family: active histone marks (dex)": "Active marks (dex)",
    "family: repressive marks and corepressors (control)": "Repressive marks (control)",
    "family: repressive marks and corepressors (dex)": "Repressive marks (dex)",
    "family: transcription (control)": "Transcription (control)", "family: transcription (dex)": "Transcription (dex)",
}
TYPE_COL = {"family": "#2a78d6", "correlation cluster": "#eb6834", "single feature": "#3a3a38"}
TYPE_LABEL = {"family": "Family of targets", "correlation cluster": "Cluster of correlated tracks",
              "single feature": "Single feature (in no group)"}
TYPE_MARKER = {"family": "o", "correlation cluster": "o", "single feature": "D"}
WRAP = 22                    # characters per line of a row label in C-E


def short(f):
    if f.startswith("motif_binary::"):
        m = f.split("::")[1].split("/")[0].split("(")[0].split(",")[0]
        return MOTIF_RENAME.get(m, m) + " motif"
    if f in RENAME:
        return RENAME[f]
    g = "ATAC" if f.startswith("ATAC") else f.split("_")[0]
    return RENAME.get(g, g)


def single_label(f):
    """single feature in C-E: Fig. 2 name, with the condition for signal tracks"""
    if f.startswith(("motif_binary::", "nbr_")) or f in RENAME:
        return short(f)
    return f"{short(f)} ({'dex' if feature_layer(f) == 'dex' else 'control'})"


def cluster_label(members, max_names=5):
    """'GATA3, JUND, SP1, TCF12, TEAD4 (control)'; mixed conditions: 'H3K27ac (control, dex); H3K9ac (control)';
    more than max_names targets: the first three and the number of the others ('ATF3, ELF1, ETS1 +4 (control)')"""
    cond = {}
    for f in members:
        cond.setdefault(short(f), set()).add("dex" if feature_layer(f) == "dex" else "control")
    by_cond = {}
    for t in sorted(cond, key=str.lower):
        by_cond.setdefault(", ".join(sorted(cond[t])), []).append(t)
    parts = []
    for c, names in sorted(by_cond.items(), key=lambda kv: -len(kv[1])):
        shown = names if len(names) <= max_names else names[:3] + [f"+{len(names) - 3}"]
        parts.append((", ".join(shown[:3]) + " " + shown[3] if len(names) > max_names else ", ".join(shown)) + f" ({c})")
    return "; ".join(parts)


def wrap(label, width=WRAP):
    """break a label at ', ' or ' ' so that no line is longer than width (at most three lines)"""
    words, lines, cur = label.split(" "), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur); cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    return "\n".join(lines[:3]) if len(lines) <= 3 else "\n".join(lines[:2] + [" ".join(lines[2:])])


def importance_ranking(run, model="catboost"):
    """groups and ungrouped single features of one run in one table, sorted by the mean decrease in average precision"""
    G = pd.read_csv(run / "summaries" / "all_cv_grouped_permutation_importance.csv")
    G = G[G["model"] == model]
    members = pd.read_csv(run / "data" / "feature_groups.csv")
    kind = members.drop_duplicates("group").set_index("group")["group_type"]
    rows = []
    for group, g in G.groupby("group", sort=False):
        feats = members.loc[members["group"] == group, "feature"].tolist()
        v = g["importance_mean"].to_numpy()
        lab = cluster_label(feats) if kind[group] == "correlation cluster" else FAMILY_LABEL.get(group, group.replace("family: ", ""))
        rows.append({"row": group, "row_type": kind[group], "label": lab, "n_features": len(feats),
                     "n_evaluations": len(v), "mean": v.mean(), "p10": np.percentile(v, 10), "p90": np.percentile(v, 90),
                     "share_positive": (v > 0).mean(), "members": "; ".join(feats) if len(feats) <= 12 else f"{len(feats)} features"})
    S = pd.read_csv(run / "summaries" / "feature_importance_stability_summary.csv")
    S = S[(S["model"] == model) & (S["mean_importance"] > 0) & ~S["feature"].isin(set(members["feature"]))]
    per_fold = run / "summaries" / "all_cv_permutation_importance.csv"
    pct = None
    if per_fold.exists():
        F = pd.read_csv(per_fold, usecols=["feature", "importance_mean", "model"])
        pct = F[F["model"] == model].groupby("feature")["importance_mean"].quantile([0.1, 0.9]).unstack()
    for r in S.itertuples():
        rows.append({"row": r.feature, "row_type": "single feature", "label": single_label(r.feature), "n_features": 1,
                     "n_evaluations": int(r.n_folds), "mean": r.mean_importance,
                     "p10": pct.loc[r.feature, 0.1] if pct is not None else np.nan,
                     "p90": pct.loc[r.feature, 0.9] if pct is not None else np.nan,
                     "share_positive": r.fraction_positive_importance, "members": r.feature})
    return pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------- data
metrics = pd.concat([pd.read_csv(r / "summaries" / "all_cv_metrics.csv").assign(task_display=t)
                     for t, r in RUNS.items()], ignore_index=True)
assert (metrics.groupby("task_display").size() == N_EVAL).all(), f"expected {N_EVAL} outer-CV folds per task"
N_FEATURES = int(metrics["n_features_before_preprocessing"].max()) if "n_features_before_preprocessing" in metrics else 335
stab, top, rank = {}, {}, {}
for t, r in RUNS.items():
    s = pd.read_csv(r / "summaries" / "feature_importance_stability_summary.csv")
    s["cat"] = s.feature.map(category)
    stab[t] = s[(s.fraction_positive_importance >= STABILITY) & (s.median_importance > 0)].copy()   # stable single features
    top[t] = stab[t].sort_values(["fraction_positive_importance", "mean_importance"], ascending=False)
    rank[t] = importance_ranking(r).head(TOP_N)                # rows of C-E
perm = perm_res = None
if not SHOW_PERMUTATION:
    pass
elif PERM_NULL.exists() and PERM_RES.exists():
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
    "motif": "228 motif groups", "genomic": "annotation, GC, TSS distance,\nlength, neighbourhood",
    "brk1": "genome-wide", "brk2": "genomic features",
    "ml": [("1. Cross-validation", "model performance"), ("2. Feature stability", "across folds"),
           ("3. Model training", "3 CatBoost models"), ("4. Local explanation", "LIME rules")],
    "cls": [("NR", "non-responsive"), ("UP", "induced"), ("DN", "repressed")],
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
    ax.text(xg + sum(cw) / 2, yb - 0.06, f"{N_FEATURES} features", ha="center", va="top", fontsize=S)
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
    ax.set_ylabel(f"Balanced accuracy\n({N_EVAL} outer-CV folds)")
    if perm is not None:
        n_perm = int(perm.groupby("task_label").size().min())
        h = [mpl.patches.Patch(facecolor="white", edgecolor=TRUE_C, lw=0.7, label="True labels"),
             mpl.patches.Patch(facecolor="#DDDDDD", edgecolor=NULL_C, lw=0.7, label=f"Permuted ({n_perm}×)")]
        ax.legend(handles=h, frameon=False, loc="lower left", fontsize=5.8, handlelength=1.0, borderaxespad=0.1)
    ax.set_title("Model performance", pad=4)


# ----------------------------------------------------------------- C-E: importance of groups and single features
def plot_CDE(fig, spec):
    sub = spec.subgridspec(1, 3, wspace=1.95)
    axes = []
    for j, t in enumerate(TASKS):
        ax = fig.add_subplot(sub[0, j]); axes.append(ax)
        d = rank[t]
        y = np.arange(len(d))[::-1]
        for yi, r in zip(y, d.itertuples()):
            c = TYPE_COL[r.row_type]
            if np.isfinite(r.p10):
                ax.plot([r.p10, r.p90], [yi, yi], color=c, lw=0.9, solid_capstyle="round", zorder=2)
            stable = r.share_positive >= STABILITY and r.mean > 0
            ax.plot(r.mean, yi, TYPE_MARKER[r.row_type], ms=3.4 if r.row_type == "single feature" else 4.0,
                    mfc=c if stable else "white", mec=c, mew=0.9, zorder=3)
            ax.annotate(f"{100 * r.share_positive:.0f}%", xy=(1.02, yi), xycoords=("axes fraction", "data"),
                        fontsize=5.0, color="#555555", va="center", ha="left")
        ax.axvline(0, lw=0.6, c="#888888", zorder=1)
        ax.grid(True, axis="both", color="#E6E6E6", lw=0.5, zorder=0); ax.set_axisbelow(True)
        labs = [wrap(r.label if r.row_type == "single feature" else f"{r.label}\u00a0[{r.n_features}]") for r in d.itertuples()]
        ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=5.2, linespacing=0.88)
        for tl, k in zip(ax.get_yticklabels(), d.row_type):
            tl.set_color(TYPE_COL[k])
        ax.tick_params(axis="y", length=0, pad=2)
        ax.set_ylim(-0.6, len(d) - 0.4)
        hi = float(np.nanmax(np.r_[d["p90"].to_numpy(dtype=float), d["mean"].to_numpy(dtype=float)]))
        lo = float(np.nanmin(np.r_[d["p10"].to_numpy(dtype=float), d["mean"].to_numpy(dtype=float), 0.0]))
        ax.set_xlim(lo - 0.04 * (hi - lo), hi + 0.06 * (hi - lo))
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3)); ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%g"))
        ax.tick_params(axis="x", labelsize=6)
        ax.set_xlabel("Decrease in held-out\naverage precision", fontsize=6.6)
        ax.set_title(t, fontsize=7.8)
    return axes


# ----------------------------------------------------------------- F-H: UMAP
UCLS = {"NR": "tab:green", "UP": "tab:red", "DN": "tab:blue"}
UANN = {"CpG": "#3FA7A0", "Intergenic": "#8C8C8C", "Intronic": "#E1A43A", "Other": "#7A5195"}
# Topic names and colours follow the leading features of each topic (the column names of the usage table, e.g.
# "T1 EP300/FOXA1"), not the topic number, because the numbering changes when the topics are refitted.
TOPIC_STYLE = [("EP300", "Active (GR/EP300/FOXA1)", "#2E7D5B"), ("H3K27me3", "Polycomb-like", "#4D4D4D"),
               ("CTCF", "CTCF/cohesin", "#7A5195"), ("H3K4me3", "Promoter-like", "#D4A62A"),
               ("Pol II", "Promoter-like", "#D4A62A"), ("H3K4me1", "H3K4me1-high", "#7FA7D1")]
SPARE = iter(["#B5651D", "#1F77B4", "#8C564B", "#17BECF"])
TOPIC_COL, TOPIC_NAME = {}, {}
for col_ in TOPIC_COLS:
    num, lead = col_.split(" ", 1)[0], col_.split(" ", 1)[-1]
    hit = next(((n, c) for key, n, c in TOPIC_STYLE if key in lead and n not in TOPIC_NAME.values()), None)
    TOPIC_NAME[num], TOPIC_COL[num] = hit if hit else (lead, next(SPARE))
print("NR topics:", {k: TOPIC_NAME[k] for k in TOPIC_NAME})


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
        for a_ in ("Intergenic", "Intronic", "Other", "CpG"):
            m = (emb.annot == a_).values
            ax.scatter(x[m], y[m], s=0.8 if a_ != "CpG" else 1.4, c=UANN[a_], lw=0, alpha=UMAP_ALPHA,
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
fig = plt.figure(figsize=(7.2, 10.0))
fig.subplots_adjust(left=0.085, right=0.965, top=0.985, bottom=0.03)
gs = fig.add_gridspec(4, 1, height_ratios=[0.95, 1.55, 0.16, 1.05], hspace=0.38)
axA = fig.add_subplot(gs[0]); plot_A(axA)
pA = gs[0].get_position(fig)                       # same width as the other rows, exact aspect, top-aligned
hA = pA.width * fig.get_figwidth() * AH / AW / fig.get_figheight()
axA.set_position([pA.x0, pA.y1 - hA, pA.width, hA])
row2 = gs[1].subgridspec(1, 2, width_ratios=[0.8, 3.6], wspace=0.62)
axB = fig.add_subplot(row2[0]); plot_B(axB)
axCDE = plot_CDE(fig, row2[1])
row3 = gs[3].subgridspec(1, 3, wspace=0.15)
axF, axG, axH = [fig.add_subplot(row3[k]) for k in range(3)]
for a_, by in zip((axF, axG, axH), ("class", "annot", "topic")):
    plot_umap(a_, by)
# key for C-E (marker and label colour), in the spacer row below the panels
shown = [k for k in TYPE_COL if any((rank[t].row_type == k).any() for t in TASKS)]
hk = [mpl.lines.Line2D([], [], marker=TYPE_MARKER[k], ls="-", lw=0.9, ms=4.2, color=TYPE_COL[k], label=TYPE_LABEL[k]) for k in shown]
hs = [mpl.lines.Line2D([], [], marker="o", ls="", ms=4.2, mfc="#777777", mec="#777777", label=f"positive in ≥ {STABILITY * 100:.0f}% of {N_EVAL} evaluations"),
      mpl.lines.Line2D([], [], marker="o", ls="", ms=4.2, mfc="white", mec="#777777", label="positive in fewer")]
hk = [h for pair in zip(hk, hs + [None] * len(hk)) for h in pair if h is not None] if len(hk) >= 2 else hk + hs   # rows: types / filled-open
fig.canvas.draw(); rr = fig.canvas.get_renderer()
bbC = mpl.transforms.Bbox.union([a.get_tightbbox(rr) for a in axCDE]).transformed(fig.transFigure.inverted())
lgK = fig.legend(handles=hk, loc="upper center", bbox_to_anchor=((bbC.x0 + bbC.x1) / 2, bbC.y0 - 0.004),
                 ncol=3, frameon=False, handletextpad=0.3, columnspacing=1.2, fontsize=6.0, labelspacing=0.3,
                 title="Mean and 10th to 90th percentile; [n] = features in a group; right: % of evaluations with a positive decrease",
                 title_fontsize=6.0)
letters = panel_letters(fig, {"A": axA, "B": axB, "C": axCDE[0], "D": axCDE[1], "E": axCDE[2],
                              "F": axF, "G": axG, "H": axH})
for ext in ("pdf", "png"):
    fig.savefig(FINAL / f"Fig_1.{ext}", dpi=300, bbox_inches="tight")
export_panels(fig, {"A": [axA], "B": [axB], "C": [axCDE[0]], "D": [axCDE[1]], "E": [axCDE[2]], "CDE_key": [lgK],
                    "F": [axF], "G": [axG], "H": [axH]}, "Fig_1", letters)
plt.show()
pd.concat([top[t].assign(task=t) for t in TASKS]).assign(feature_short=lambda d: d.feature.map(short)) \
    [["task", "feature", "feature_short", "cat", "fraction_positive_importance", "mean_importance", "median_importance",
      "top10_fraction"]].to_csv(FINAL / "Fig_1_stable_features.tsv", sep="\t", index=False)
pd.concat([rank[t].assign(task=t, rank=np.arange(1, len(rank[t]) + 1)) for t in TASKS]) \
    [["task", "rank", "row_type", "label", "n_features", "mean", "p10", "p90", "share_positive", "n_evaluations", "row", "members"]] \
    .round(5).to_csv(FINAL / "Fig_1_importance_ranking.tsv", sep="\t", index=False)
print("Saved to", FINAL / "Fig_1.pdf", "+ per-panel PDFs Fig_1_A..H.pdf")
