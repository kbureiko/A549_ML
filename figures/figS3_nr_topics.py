# =============================================================================
# Supplementary Figure S3 - chromatin programs (topics) of non-responsive (NR) enhancers
# Topics from scripts/11_nr_topics.py (consensus NMF; cached loadings and usage, nothing is refitted).
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
#   D-F  GR-bound enhancers: induced (UP) against non-responsive (NR), from the two cross-validation runs
#      "<date>__nr_vs_up__thr_max_f1__grbound" (all features) and "...__grbound__pretreatment" (without the features
#      measured after dex: dex tracks and GR peaks around the enhancer). Drawn only if both run folders exist.
#   D  ROC AUC and average precision of the two models over the outer-CV evaluations (dotted: chance)
#   E  model without features measured after dex: groups and single features in one ranking (as Fig. 1 C-E)
#   F  direction among GR-bound enhancers: ROC AUC of each feature alone, UP against NR (0.5 = no difference,
#      above 0.5 = higher at UP); open = before dex (EtOH / untreated track), filled = after dex. Importance (E) has
#      no direction; F gives it. Univariate and on the signal of the feature matrix.
# Inputs: FM; NR_TOPICS/MS_nr_topics_loadings.tsv, NR_TOPICS/MS_nr_topics_usage.tsv; for D-F the two run folders
#   (summaries/all_cv_metrics.csv, all_cv_grouped_permutation_importance.csv,
#   feature_importance_stability_summary.csv, data/feature_groups.csv)
# NOTE: GR is the only post-dex feature (dashed box in B): GR is barely bound without hormone. The topics are therefore
#   not purely pretreatment profiles; do not describe them as such in the text.
# Outputs (FIGURES): Fig_S3_nr_topics.pdf/png (+ one PDF per panel), Fig_S3_nr_topics_profile.tsv,
#   Fig_S3_annotation_per_class.tsv, Fig_S3_grbound_performance.tsv, Fig_S3_grbound_ranking.tsv,
#   Fig_S3_grbound_direction.tsv
# =============================================================================
# %%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

from a549_enhancers.config import STABLE_MIN_FRACTION_POSITIVE_IMPORTANCE as STABILITY
from a549_enhancers.feature_groups import feature_layer
from a549_enhancers.paths import PROJECT, ML, FIGURES, NR_TOPICS   # local folders, set in paths.toml
FM = ML / "feature_matrix_all_classes_GRannotated.csv"
OUT = NR_TOPICS
FINAL = FIGURES; FINAL.mkdir(parents=True, exist_ok=True)   # final figures + their tables
SHOW_CUT = 0.25
# GR-bound models (panels D-F): folder names inside ml_outputs; None = the newest folder with the matching ending
GRBOUND_RUN = None                  # e.g. "20261122__nr_vs_up__thr_max_f1__grbound"
GRBOUND_PRETREATMENT_RUN = None     # e.g. "20261122__nr_vs_up__thr_max_f1__grbound__pretreatment"
TOP_N = 12                          # rows in E
# Topic names and colours follow the leading features of each topic (index of the loadings table, e.g.
# "T1 EP300/FOXA1"), not the topic number, because the numbering changes when the topics are refitted. Same rule as
# figures/fig1_classification_overview.py (panel H). Filled after the loadings are read.
TOPIC_STYLE = [("EP300", "Active (GR/EP300/FOXA1)", "#2E7D5B"), ("H3K27me3", "Polycomb-like", "#4D4D4D"),
               ("CTCF", "CTCF/cohesin", "#7A5195"), ("H3K4me3", "Promoter-like", "#D4A62A"),
               ("Pol II", "Promoter-like", "#D4A62A"), ("H3K4me1", "H3K4me1-high", "#7FA7D1")]
TOPIC_NAME, TOPIC_COL = {}, {}

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
SF = {"ATAC": "ATAC_ctrl", "H3K27ac": "H3K27ac_ENCFF054OOT", "H3K4me1": "H3K4me1_ENCFF580SCK",
      "H3K4me3": "H3K4me3_ENCFF529CES", "Pol II": col("POLR2A_", "_eth_"), "EP300": "EP300_ENCFF376VNJ",
      "cJUN": "JUN_ENCFF957TUK", "CTCF": "CTCF_ENCFF083AEY", "RAD21": "RAD21_ENCFF778YHU", "H2A.Z": col("H2AFZ_", "_eth_"),
      "H3K27me3": col("H3K27me3_", "_eth_"), "GR (dex)": col("NR3C1_", "_dex_"), "FOXA1": col("FOXA1_", "_eth_")}
N = D[D["class"].eq("NR")].set_index("enhancer_id")
P = pd.DataFrame({f: N[c].rank(pct=True) * 100 for f, c in SF.items()})          # percentile within NR
load = pd.read_csv(OUT / "MS_nr_topics_loadings.tsv", sep="\t", index_col=0)
use = pd.read_csv(OUT / "MS_nr_topics_usage.tsv", sep="\t").set_index("enhancer_id")
TCOLS = list(load.index); TID = {c: c.split(" ", 1)[0] for c in TCOLS}
_spare = iter(["#B5651D", "#1F77B4", "#8C564B", "#17BECF"])
for _c in TCOLS:
    _lead = _c.split(" ", 1)[-1]
    _hit = next(((n, c) for key, n, c in TOPIC_STYLE if key in _lead and n not in TOPIC_NAME.values()), None)
    TOPIC_NAME[TID[_c]], TOPIC_COL[TID[_c]] = _hit if _hit else (_lead, next(_spare))
print("NR topics:", TOPIC_NAME)
use = use.loc[P.index]
main = use[TCOLS].idxmax(axis=1).map(TID)
FEATS = list(SF)
ANN_MAP = {"Intergenic": "Intergenic", "Intonic": "Intronic", "Intronic": "Intronic", "CpG": "CpG", "Other": "Other"}
ANN_COL = {"Intergenic": "#8C8C8C", "Intronic": "#E1A43A", "Other": "#7A5195", "CpG": "#3FA7A0"}
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


# ----------------------------------------------------------------- GR-bound models (D-F)
def find_run(name, ending):
    if name:
        return ML / name
    hits = sorted(q for q in ML.glob(f"*__nr_vs_up__*{ending}") if q.is_dir() and q.name.endswith(ending))
    return hits[-1] if hits else None


RUN_ALL, RUN_PRE = find_run(GRBOUND_RUN, "__grbound"), find_run(GRBOUND_PRETREATMENT_RUN, "__grbound__pretreatment")
HAVE_GRB = all(r is not None and (r / "summaries" / "all_cv_metrics.csv").exists() for r in (RUN_ALL, RUN_PRE))
print("GR-bound runs:", RUN_ALL.name if RUN_ALL else None, "|", RUN_PRE.name if RUN_PRE else None,
      "" if HAVE_GRB else "-> not found, panels D-F are left out")

# names as in Fig. 1 and Fig. 2
RENAME = {"JUN": "cJUN", "NR3C1": "GR", "H2AFZ": "H2A.Z", "Enh_len": "Enhancer length", "POLR2AphosphoS2": "POLR2A pS2",
          "gc": "GC content", "Distance to TSS": "TSS distance", "annotation": "Annotation", "GRE_score": "GRE score",
          "nbr_enhancers_50kb": "Enhancers ≤50 kb", "nbr_enhancer_distance": "Nearest enhancer",
          "nbr_GRpeaks_50kb": "GR peaks ≤50 kb", "nbr_strongGRpeaks_50kb": "Strong GR peaks ≤50 kb",
          "nbr_GRpeak_distance": "Nearest GR peak", "nbr_strongGRpeak_distance": "Nearest strong GR peak"}
FAMILY_LABEL = {"family: all motif indicators": "All motif groups", "family: GRE (motif group and score)": "GRE motif and score",
                "family: neighbourhood, other enhancers": "Enhancers nearby", "family: neighbourhood, GR peaks after dex": "GR peaks nearby",
                "family: CTCF and cohesin (control)": "CTCF, cohesin (control)", "family: active histone marks (control)": "Active marks (control)",
                "family: repressive marks and corepressors (control)": "Repressive marks (control)",
                "family: transcription (control)": "Transcription (control)"}
TYPE_COL = {"family": "#2a78d6", "correlation cluster": "#eb6834", "single feature": "#3a3a38"}
TYPE_LABEL = {"family": "Family of targets", "correlation cluster": "Cluster of correlated tracks", "single feature": "Single feature (in no group)"}
TYPE_MARKER = {"family": "o", "correlation cluster": "o", "single feature": "D"}


def short(f):
    if f.startswith("motif_binary::"):
        return f.split("::")[1].split("/")[0].split("(")[0].split(",")[0] + " motif"
    if f in RENAME:
        return RENAME[f]
    g = "ATAC" if f.startswith("ATAC") else f.split("_")[0]
    return RENAME.get(g, g)


def single_label(f):
    if f.startswith(("motif_binary::", "nbr_")) or f in RENAME:
        return short(f)
    return f"{short(f)} ({'dex' if feature_layer(f) == 'dex' else 'control'})"


def cluster_label(members, max_names=5):
    cond = {}
    for f in members:
        cond.setdefault(short(f), set()).add("dex" if feature_layer(f) == "dex" else "control")
    by_cond = {}
    for t in sorted(cond, key=str.lower):
        by_cond.setdefault(", ".join(sorted(cond[t])), []).append(t)
    parts = []
    for c, names in sorted(by_cond.items(), key=lambda kv: -len(kv[1])):
        txt = ", ".join(names) if len(names) <= max_names else ", ".join(names[:3]) + f" +{len(names) - 3}"
        parts.append(f"{txt} ({c})")
    return "; ".join(parts)


def wrap(label, width=24):
    lines, cur = [], ""
    for w in label.split(" "):
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur); cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    return "\n".join(lines[:2] + ([" ".join(lines[2:])] if len(lines) > 2 else []))


def importance_ranking(run, model="catboost"):
    """groups and ungrouped single features of one run, sorted by the mean decrease in average precision"""
    G = pd.read_csv(run / "summaries" / "all_cv_grouped_permutation_importance.csv")
    G = G[G["model"] == model]
    members = pd.read_csv(run / "data" / "feature_groups.csv")
    kind = members.drop_duplicates("group").set_index("group")["group_type"]
    rows = []
    for group, g in G.groupby("group", sort=False):
        feats = members.loc[members["group"] == group, "feature"].tolist()
        v = g["importance_mean"].to_numpy()
        lab = cluster_label(feats) if kind[group] == "correlation cluster" else FAMILY_LABEL.get(group, group.replace("family: ", ""))
        rows.append({"row": group, "row_type": kind[group], "label": lab, "n_features": len(feats), "n_evaluations": len(v),
                     "mean": v.mean(), "p10": np.percentile(v, 10), "p90": np.percentile(v, 90), "share_positive": (v > 0).mean(),
                     "members": "; ".join(feats) if len(feats) <= 12 else f"{len(feats)} features"})
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


def auc_up_vs_nr(x_up, x_nr):
    """ROC AUC of one feature, UP against NR (Mann-Whitney U / (n_up * n_nr)); above 0.5 = higher at UP"""
    r = pd.concat([x_up, x_nr]).rank()
    return (r.iloc[:len(x_up)].sum() - len(x_up) * (len(x_up) + 1) / 2) / (len(x_up) * len(x_nr))


def first_col(prefix, token):
    hit = [c for c in D.columns if c.startswith(prefix) and token in c]
    return hit[0] if hit else None


# target -> (column before dex, column after dex); None = not in the matrix
DIRECTION = {
    "GRE score": ("GRE_score", None),
    "GR": (first_col("NR3C1_", "_eth_"), first_col("NR3C1_", "_dex_")),
    "EP300": ("EP300_ENCFF376VNJ", first_col("EP300_", "_dex_")),
    "cJUN": ("JUN_ENCFF957TUK", first_col("JUN_", "_dex_")),
    "JUNB": (first_col("JUNB_ENCFF", ""), first_col("JUNB_", "_dex_")),
    "FOSL2": ("FOSL2_ENCFF129REO", first_col("FOSL2_", "_dex_")),
    "BCL3": (first_col("BCL3_ENCFF", ""), first_col("BCL3_", "_dex_")),
    "CREB1": (first_col("CREB1_", "_eth_"), first_col("CREB1_", "_dex_")),
    "GATA3": (first_col("GATA3_", ""), None),
    "TEAD4": (first_col("TEAD4_", ""), None),
    "H3K4me1": ("H3K4me1_ENCFF580SCK", first_col("H3K4me1_", "_dex_")),
    "H3K27ac": ("H3K27ac_ENCFF054OOT", first_col("H3K27ac_", "_dex_")),
    "Pol II": (first_col("POLR2A_", "_eth_"), first_col("POLR2A_", "_dex_")),
    "H2A.Z": (first_col("H2AFZ_", "_eth_"), first_col("H2AFZ_", "_dex_")),
}
if HAVE_GRB:
    met = {k: pd.read_csv(r / "summaries" / "all_cv_metrics.csv") for k, r in (("All features", RUN_ALL), ("Before dex only", RUN_PRE))}
    met = {k: m[m["model"].eq("catboost")] if "model" in m else m for k, m in met.items()}
    GB = D[D.GR_bound & D["class"].isin(["UP", "NR"])]
    up_, nr_ = GB[GB["class"].eq("UP")], GB[GB["class"].eq("NR")]
    CHANCE_AP = len(up_) / len(GB)
    rank_pre = importance_ranking(RUN_PRE).head(TOP_N)
    drows = []
    for name, (c0, c1) in DIRECTION.items():
        for when, c in (("before dex", c0), ("after dex", c1)):
            if c is not None and c in D.columns:
                drows.append({"feature": name, "condition": when, "column": c, "auc_UP_vs_NR": auc_up_vs_nr(up_[c], nr_[c]),
                              "median_UP": up_[c].median(), "median_NR": nr_[c].median()})
    direction = pd.DataFrame(drows)
    pd.concat([m[["roc_auc", "average_precision", "balanced_accuracy"]].agg(["mean", "std"]).T.add_prefix(f"{k}: ")
               for k, m in met.items()], axis=1).round(4).assign(chance=[0.5, round(CHANCE_AP, 4), 0.5]) \
        .to_csv(FINAL / "Fig_S3_grbound_performance.tsv", sep="\t")
    rank_pre.round(5).to_csv(FINAL / "Fig_S3_grbound_ranking.tsv", sep="\t", index=False)
    direction.round(4).to_csv(FINAL / "Fig_S3_grbound_direction.tsv", sep="\t", index=False)
    print(f"GR-bound: {len(up_)} UP, {len(nr_)} NR | evaluations {[len(m) for m in met.values()]}")
    print({k: m[["roc_auc", "average_precision"]].mean().round(3).to_dict() for k, m in met.items()})


XLAB = ["all", "before\ndex"]                     # the two models in D (features used)


def plot_D(ax_auc, ax_ap):
    for ax, col_, chance, lab in ((ax_auc, "roc_auc", 0.5, "ROC AUC"), (ax_ap, "average_precision", CHANCE_AP, "Average precision")):
        rng = np.random.default_rng(0)
        for i, (k, m) in enumerate(met.items()):
            v = m[col_].to_numpy()
            ax.boxplot(v, positions=[i], widths=0.5, showfliers=False, medianprops={"color": "#222222", "lw": 0.9},
                       boxprops={"lw": 0.7}, whiskerprops={"lw": 0.7}, capprops={"lw": 0.7})
            ax.scatter(i + rng.uniform(-0.16, 0.16, len(v)), v, s=3, c="#6E6E6E", lw=0, alpha=0.8, zorder=3)
            ax.text(i, v.max() + 0.012, f"{v.mean():.2f}", ha="center", va="bottom", fontsize=5.8)
        ax.axhline(chance, ls=":", lw=0.7, c="#888888")
        ax.text(-0.52, chance + 0.006, f"chance {chance:.3f}" if col_ != "roc_auc" else "chance", fontsize=4.8, color="#777777", va="bottom", ha="left")
        ax.set_xticks(range(len(met))); ax.set_xticklabels(XLAB, fontsize=5.4, linespacing=0.9)
        ax.set_xlim(-0.6, 1.6); ax.set_ylabel(lab, fontsize=6.4)
        ax.set_xlabel("features", fontsize=5.8, labelpad=1)
        lo = min(chance, min(m[col_].min() for m in met.values())); hi = max(m[col_].max() for m in met.values())
        ax.set_ylim(lo - 0.04, hi + 0.06)
    ax_auc.set_title(f"GR-bound enhancers:\n{len(up_)} UP against {len(nr_):,} NR", fontsize=6.6, loc="left")


def plot_E(ax):
    d = rank_pre
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
    ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=5.3, linespacing=0.88)
    for tl, k in zip(ax.get_yticklabels(), d.row_type):
        tl.set_color(TYPE_COL[k])
    ax.tick_params(axis="y", length=0, pad=2); ax.set_ylim(-0.6, len(d) - 0.4)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    ax.set_xlabel("Decrease in held-out average precision", fontsize=6.2)
    ax.set_title("Model without features measured after dex", fontsize=6.6, loc="left")
    shown = [k for k in TYPE_COL if (d.row_type == k).any()]
    ax.legend(handles=[mpl.lines.Line2D([], [], marker=TYPE_MARKER[k], ls="-", lw=0.9, ms=3.6, color=TYPE_COL[k], label=TYPE_LABEL[k]) for k in shown],
              frameon=False, fontsize=5.0, loc="upper center", bbox_to_anchor=(0.35, -0.2), ncol=3, handletextpad=0.3,
              columnspacing=0.8)


def plot_F(ax):
    names = [n for n in DIRECTION if (direction.feature == n).any()]
    y = {n: i for i, n in enumerate(names[::-1])}
    B_COL, A_COL = "#9DB7D5", "#2A4B7C"
    for n in names:
        d = direction[direction.feature == n].set_index("condition")["auc_UP_vs_NR"]
        if len(d) == 2:
            ax.annotate("", xy=(d["after dex"], y[n]), xytext=(d["before dex"], y[n]),
                        arrowprops={"arrowstyle": "-", "color": "#9A9A9A", "lw": 0.8}, zorder=1)
        if "before dex" in d:
            seq = n == "GRE score"
            ax.plot(d["before dex"], y[n], "D" if seq else "o", ms=4.2, mfc="#3E9651" if seq else "white",
                    mec="#3E9651" if seq else A_COL, mew=1.0, zorder=3)
        if "after dex" in d:
            ax.plot(d["after dex"], y[n], "o", ms=4.2, mfc=A_COL, mec=A_COL, zorder=3)
    ax.axvline(0.5, lw=0.7, c="#666666", zorder=0)
    ax.grid(True, axis="both", color="#E6E6E6", lw=0.5, zorder=0); ax.set_axisbelow(True)
    ax.set_yticks(list(y.values())); ax.set_yticklabels(list(y.keys()), fontsize=5.8)
    ax.tick_params(axis="y", length=0); ax.set_ylim(-0.7, len(names) - 0.3)
    lim = max(0.26, float((direction.auc_UP_vs_NR - 0.5).abs().max()) + 0.04)
    ax.set_xlim(0.5 - lim, 0.5 + lim)
    ax.set_xlabel("ROC AUC of the feature alone\n← higher at NR        higher at UP →", fontsize=6.0)
    ax.set_title("Direction among GR-bound enhancers", fontsize=6.6, loc="left")
    h = [mpl.lines.Line2D([], [], marker="o", ls="", ms=4, mfc="white", mec=A_COL, mew=1.0, label="before dex"),
         mpl.lines.Line2D([], [], marker="o", ls="", ms=4, mfc=A_COL, mec=A_COL, label="after dex"),
         mpl.lines.Line2D([], [], marker="D", ls="", ms=3.8, mfc="#3E9651", mec="#3E9651", label="sequence")]
    ax.legend(handles=h, frameon=False, fontsize=5.2, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3,
              handletextpad=0.2, columnspacing=0.9)


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
    for j, h in ((1, "GR-bound\n(%)"), (2, "CpG\n(%)")):
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
if HAVE_GRB:
    fig = plt.figure(figsize=(8.0, 8.4))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.8, 1.0, 1.35], hspace=0.6, left=0.06, right=0.98, top=0.96, bottom=0.09)
else:
    fig = plt.figure(figsize=(8.0, 4.9))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.8, 1.0], hspace=0.62, left=0.06, right=0.98, top=0.93, bottom=0.12)
r1 = gs[0].subgridspec(1, len(TCOLS), wspace=0.95)
axA = [fig.add_subplot(r1[k]) for k in range(len(TCOLS))]; plot_A(axA)
r2 = gs[1].subgridspec(1, 5, width_ratios=[0.36, 1.5, 0.62, 0.16, 0.42], wspace=0.05)
axB = fig.add_subplot(r2[1]); axBs = fig.add_subplot(r2[2]); plot_B(axB, axBs)
axC = fig.add_subplot(r2[4]); plot_C(axC)
PANELS = {"A": axA, "B": [axB, axBs], "C": [axC]}
if HAVE_GRB:
    r3 = gs[2].subgridspec(1, 6, width_ratios=[0.5, 0.5, 0.98, 1.3, 0.62, 1.1], wspace=0.0)
    r3d = r3[0:2].subgridspec(1, 2, wspace=1.05)
    axD1, axD2 = fig.add_subplot(r3d[0]), fig.add_subplot(r3d[1]); plot_D(axD1, axD2)
    axE = fig.add_subplot(r3[3]); plot_E(axE)
    axF = fig.add_subplot(r3[5]); plot_F(axF)
    PANELS.update({"D": [axD1, axD2], "E": [axE], "F": [axF]})
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
