# =============================================================================
# Supplementary Figure S1 - performance and validity of the three pairwise CatBoost models
#   A  ROC curves  B  precision-recall curves: out-of-fold predictions of the 5-fold x 10-repeat stratified CV;
#      line = mean over the repeats (each repeat pools its 5 held-out folds), band = range over repeats.
#      Legend: mean +/- SD over all held-out folds (N_EVAL). B: dashed = chance (share of the positive class).
#   C  ROC AUC over all held-out folds (N_EVAL) next to the label-permutation null (99 permutations; band = min to 95th
#      percentile, dashed = null mean).
#   D  simple baselines vs the full model: ROC AUC of single features (GR dex ChIP signal, GR peak, enhancer length,
#      GC content) scored on the same held-out folds; the direction of each feature is taken from the training
#      part of the fold (no fitting otherwise). Bars = mean +/- SD over folds.
#   E  stability: Spearman correlation of the out-of-fold scores of the same enhancers between the CV repeats
#      (45 pairs per task with 10 repeats).
#   F  genomic-position check (scripts/05_cv_scheme_comparison.py): the same model with random stratified folds,
#      whole chromosomes held out, or 5 Mb blocks held out; ROC AUC and AP, mean +/- SD and folds; dashed = chance.
#      For UP vs DN that script scores DN as the positive class (AP chance = DN share), unlike the published run;
#      the average-precision panel marks this under the task name ("DN+").
#   G  pre-existing context vs the state that accompanies the response (scripts/06_feature_set_models.py): the same model
#      retrained on restricted feature sets - sequence / genomic features only; these + control (EtOH or untreated)
#      signals = control-only; these + dex-treated signals = dex-only; all features. All four sets use identical
#      folds (repeated stratified CV; scripts/06 is run with fewer repeats than the published runs, e.g. 5 x 5 = 25 folds, and
#      these splits are drawn independently of the published runs in A-F, so "all features" is a retrained model and its
#      ROC AUC differs from A-F in the third decimal).
#      Left: ROC AUC, mean +/- SD over the held-out folds of that run. Right: mean paired difference in ROC AUC between
#      feature sets on the same folds, with the Nadeau-Bengio corrected resampled t-test (accounts for overlapping training
#      sets); P values adjusted over all comparisons of the panel (Benjamini-Hochberg).
# Positive class: NR vs UP -> UP, NR vs DN -> DN, UP vs DN -> UP (published runs). ROC AUC does not depend on it.
# Inputs: RUNS/<run>/summaries/{all_cv_metrics,all_cv_predictions}.csv; FM; PERM_DIR null; RES (CV-scheme TSVs);
#   feature_sets_folds.tsv (FSETS, or searched under RES / PROJECT / ML).
# Outputs (FIGURES): Fig_S1_model_performance.pdf/png (+ _A.._G.pdf), S1_model_performance_summary.tsv,
#   S1_baselines.tsv, S1_feature_sets.tsv (ROC AUC per feature set), S1_feature_sets_paired.tsv (differences, tests)
# =============================================================================
# %%
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy import stats
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score

from a549_enhancers.config import N_REPEATS, N_SPLITS
from a549_enhancers.paths import PROJECT, ML, FIGURES, RUNS as RUN_DIRS   # local folders, set in paths.toml

N_EVAL = N_SPLITS * N_REPEATS      # held-out evaluations per model (config.py)
FINAL = FIGURES; FINAL.mkdir(parents=True, exist_ok=True)   # final figures + their tables
RUNS = {"NR vs UP": RUN_DIRS["nr_vs_up"],
        "UP vs DN": RUN_DIRS["up_vs_dn"],
        "NR vs DN": RUN_DIRS["nr_vs_dn"]}
PERM_NULL = ML / "label_permutation_test" / "null_distribution_mean_over_folds_1rep.csv"
FM = ML / "feature_matrix_all_classes_GRannotated.csv"
RES = ML / "cv_scheme_comparison"   # output folder of scripts/05_cv_scheme_comparison.py (its default)
FS_MIN_FOLDS = N_SPLITS * 3   # panel G: smallest complete run accepted (scripts/06 is usually run with 5 repeats = 25 folds)
FSETS = None    # panel G: full path to feature_sets_folds.tsv; None = newest complete copy under RES / PROJECT / ML
TASKS = list(RUNS)

FONT = "Open Sans" if any("Open Sans" in f.name for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
mpl.rcParams.update({
    "font.family": [FONT, "DejaVu Sans"], "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.3, "ytick.labelsize": 6.3, "legend.fontsize": 6.0, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none",
})
TASK_COL = {"NR vs UP": "#D1495B", "UP vs DN": "#7A5195", "NR vs DN": "#3B6FB6"}
SCHEME = {"random": "#8C8C8C", "chromosome": "#2A6F97", "5 Mb block": "#E07A5F"}
SCHEME_LAB = {"random": "random", "chromosome": "chromosome held out", "5 Mb block": "5 Mb block held out"}
SETS = ["genomic", "baseline", "dex", "combined"]                       # feature sets of scripts/06_feature_set_models.py
SET_COL = {"genomic": "#3E9651", "baseline": "#9DB9D5", "dex": "#2B4C7E", "combined": "#333333"}
SET_LAB = {"genomic": "sequence / genomic", "baseline": "control-only", "dex": "dex-only", "combined": "all features"}
SET_PAIRS = [("baseline", "genomic"), ("dex", "genomic"), ("dex", "baseline"), ("combined", "baseline"),
             ("combined", "dex")]

# ----------------------------------------------------------------- data
MET, PRED = {}, {}
for t, r in RUNS.items():
    MET[t] = pd.read_csv(r / "summaries" / "all_cv_metrics.csv")
    p = pd.read_csv(r / "summaries" / "all_cv_predictions.csv")
    pos = p.positive_label.iloc[0]
    p["y"] = (p.true_label == pos).astype(int); p["score"] = p[f"proba_{pos}"]
    p["yhat"] = (p.pred_label == pos).astype(int)
    PRED[t] = p
    assert len(MET[t]) == N_EVAL, f"{t}: expected {N_EVAL} folds"
POS = {t: PRED[t].positive_label.iloc[0] for t in TASKS}
NEG = {t: PRED[t].negative_label.iloc[0] for t in TASKS}
perm = pd.read_csv(PERM_NULL) if PERM_NULL.exists() else None
# CV-scheme folds (panel F): RES first, otherwise the newest copy anywhere under PROJECT or ML
FOLDS_NAME = "cv_scheme_comparison_folds.tsv"
_cand = [RES / FOLDS_NAME] if (RES / FOLDS_NAME).exists() else sorted(
    (p for root in (PROJECT, ML) if root.exists() for p in root.rglob(FOLDS_NAME)), key=lambda p: p.stat().st_mtime)
if not _cand:
    raise FileNotFoundError(f"{FOLDS_NAME} not found in {RES} or under {PROJECT} / {ML}. "
                            "Set RES at the top to the folder holding the scripts/05_cv_scheme_comparison.py outputs.")
FOLDS_FILE = _cand[-1]
folds = pd.read_csv(FOLDS_FILE, sep="\t")
print("panel F uses", FOLDS_FILE, "| schemes:", sorted(folds.scheme.unique()), "| tasks:", sorted(folds.task.unique()))


# feature-set models (panel G): per-fold table written by scripts/06_feature_set_models.py
def load_feature_sets():
    """Newest feature_sets_folds.tsv with the same number of folds (at least FS_MIN_FOLDS) for every task x feature set."""
    name = "feature_sets_folds.tsv"

    def walk(root):                                      # like rglob, but skips environments and hidden folders
        for d, dirs, files in os.walk(root):
            dirs[:] = [x for x in dirs if not x.startswith(".") and x not in ("venv", "site-packages", "__pycache__")]
            if name in files:
                yield Path(d) / name

    cands = [Path(FSETS)] if FSETS else sorted({p for root in (RES, PROJECT, ML) if root.exists() for p in walk(root)},
                                               key=lambda p: p.stat().st_mtime, reverse=True)     # newest first
    want = pd.MultiIndex.from_product([TASKS, SETS])
    seen = []
    for f in cands:
        if not f.exists():
            seen.append(f"{f}: file not found"); continue
        d = pd.read_csv(f, sep="\t")
        d = d[d.task.isin(TASKS) & d.feature_set.isin(SETS)]
        n = d.groupby(["task", "feature_set"]).size().reindex(want, fill_value=0)
        if n.min() == n.max() and n.min() >= FS_MIN_FOLDS:
            return f, d.reset_index(drop=True)
        seen.append(f"{f}: folds per task x feature set between {n.min()} and {n.max()} (need the same number, "
                    f"at least {FS_MIN_FOLDS})")
    raise FileNotFoundError(
        f"Panel G needs {name} from scripts/06_feature_set_models.py with a complete run for each of {SETS} in each task. "
        + ("Checked: " + "; ".join(seen) if seen else f"No {name} under {RES}, {PROJECT} or {ML}.")
        + " Set FSETS at the top to the full path of the file.")


def corrected_ttest(d, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test for paired differences over repeated k-fold CV."""
    d = np.asarray(d, float); k = len(d); var = d.var(ddof=1)
    if var == 0:
        return np.nan, np.nan
    t = d.mean() / np.sqrt((1 / k + n_test / n_train) * var)
    return t, 2 * stats.t.sf(abs(t), df=k - 1)


FS_FILE, FS = load_feature_sets()
N_FS = int(FS.groupby(["task", "feature_set"]).size().iloc[0])      # held-out folds per feature set in that run
print("panel G uses", FS_FILE, f"({N_FS} folds per feature set)")
FS_SUM = FS.groupby(["task", "feature_set"]).roc_auc.agg(["mean", "std", "count"]).reindex(
    pd.MultiIndex.from_product([TASKS, SETS], names=["task", "feature_set"])).reset_index()
rows = []
for t in TASKS:
    g = FS[FS.task == t]
    for a_, b_ in SET_PAIRS:
        A = g[g.feature_set == a_].set_index(["repeat", "fold"]); B = g[g.feature_set == b_].set_index(["repeat", "fold"])
        idx = A.index.intersection(B.index)
        assert len(idx) == N_FS, f"{t}: {a_} and {b_} do not share the same {N_FS} folds"
        d = A.loc[idx, "roc_auc"] - B.loc[idx, "roc_auc"]
        tt, pp = corrected_ttest(d, A.loc[idx, "n_train"].mean(), A.loc[idx, "n_test"].mean())
        rows.append({"task": t, "comparison": f"{SET_LAB[a_]} − {SET_LAB[b_]}", "metric": "roc_auc", "n_folds": len(idx),
                     "mean_diff": d.mean(), "sd_diff": d.std(ddof=1), "folds_first_better": int((d > 0).sum()),
                     "t_corrected": tt, "p_corrected": pp})
FSP = pd.DataFrame(rows)


def bh_adjust(p):
    """Benjamini-Hochberg adjusted P values (all comparisons of panel G together)."""
    p = np.asarray(p, float); o = np.argsort(p); m = len(p)
    adj = np.empty(m); adj[o] = np.minimum.accumulate((p[o] * m / np.arange(1, m + 1))[::-1])[::-1]
    return np.clip(adj, 0, 1)


FSP["p_adjusted_BH"] = bh_adjust(FSP.p_corrected.values)
FS_SUM.round(4).to_csv(FINAL / "S1_feature_sets.tsv", sep="\t", index=False)
FSP.round({"mean_diff": 4, "sd_diff": 4, "t_corrected": 3}).to_csv(FINAL / "S1_feature_sets_paired.tsv", sep="\t", index=False)
print(FS_SUM.pivot(index="feature_set", columns="task", values="mean").reindex(index=SETS, columns=TASKS).round(3))
print(FSP.assign(p=FSP.p_corrected.map("{:.2g}".format), p_adj=FSP.p_adjusted_BH.map("{:.2g}".format))[
    ["task", "comparison", "mean_diff", "p", "p_adj"]].round(3).to_string(index=False))

METRICS = [("balanced_accuracy", "Balanced\naccuracy"), ("roc_auc", "ROC AUC"), ("average_precision", "Average\nprecision"),
           ("positive_precision", "Precision"), ("positive_recall", "Recall"), ("positive_f1", "F1")]
summ = pd.concat([MET[t][[m for m, _ in METRICS] + ["decision_threshold"]].agg(["mean", "std"]).T.assign(task=t)
                  for t in TASKS]).reset_index(names="metric")
summ.round(4).to_csv(FINAL / "S1_model_performance_summary.tsv", sep="\t", index=False)
print(summ.pivot(index="metric", columns="task", values="mean").round(3))

# ----------------------------------------------------------------- baselines on the same held-out folds
fm = pd.read_csv(FM, low_memory=False).set_index("enhancer_id")
gr_col = next(c for c in fm.columns if c.startswith("NR3C1_") and "_dex_" in c)
BASE = {"GR dex signal": fm[gr_col], "GR peak": fm.GR_bound.astype(str).str.lower().eq("true").astype(float),
        "Enhancer length": fm["Enh_len"] if "Enh_len" in fm else fm["end"] - fm["start"]}
if "gc" in fm:
    BASE["GC content"] = fm["gc"]
else:
    print("NOTE: no 'gc' column in", FM.name, "- the GC content baseline is left out of panel D")
BCOL = {"GR dex signal": "#5B4B9A", "GR peak": "#9C8FCB", "Enhancer length": "#B07A3A", "GC content": "#C9B79C",
        "Full model": "#333333"}
rows = []
for t in TASKS:
    p = PRED[t]
    for (rep, fo), g in p.groupby(["repeat", "fold"]):
        train = p[(p.repeat == rep) & (p.fold != fo)]             # the other 4 folds of the repeat = training part
        for name, v in BASE.items():
            sign = 1 if roc_auc_score(train.y, v.reindex(train.enhancer_id).values) >= 0.5 else -1
            rows.append({"task": t, "feature": name, "repeat": rep, "fold": fo,
                         "roc_auc": roc_auc_score(g.y, sign * v.reindex(g.enhancer_id).values)})
    for _, r in MET[t].iterrows():
        rows.append({"task": t, "feature": "Full model", "repeat": r["repeat"], "fold": r["fold"], "roc_auc": r.roc_auc})
BL = pd.DataFrame(rows)
BL.groupby(["task", "feature"]).roc_auc.agg(["mean", "std"]).round(3).to_csv(FINAL / "S1_baselines.tsv", sep="\t")
print(BL.groupby(["task", "feature"]).roc_auc.mean().unstack().round(3))
# ----------------------------------------------------------------- stability between CV repeats
STAB = {}
for t in TASKS:
    w = PRED[t].pivot_table(index="enhancer_id", columns="repeat", values="score")
    cc = w.corr("spearman").to_numpy(); STAB[t] = cc[np.triu_indices(len(cc), 1)]
print("repeat-to-repeat Spearman (min):", {t: round(v.min(), 3) for t, v in STAB.items()})

GRID = np.linspace(0, 1, 201)


def curves(t, kind):
    ys = []
    for _, g in PRED[t].groupby("repeat"):
        if kind == "roc":
            fpr, tpr, _ = roc_curve(g.y, g.score); ys.append(np.interp(GRID, fpr, tpr)); ys[-1][0] = 0
        else:
            prec, rec, _ = precision_recall_curve(g.y, g.score)
            ys.append(np.interp(GRID, rec[::-1], prec[::-1]))
    return np.array(ys)


# ----------------------------------------------------------------- panels
def plot_curves(ax, kind):
    for t in TASKS:
        Y = curves(t, kind); c = TASK_COL[t]
        ax.fill_between(GRID, Y.min(0), Y.max(0), color=c, alpha=0.18, lw=0)
        m = MET[t]["roc_auc" if kind == "roc" else "average_precision"]
        ax.plot(GRID, Y.mean(0), color=c, lw=1.0, label=f"{t} ({POS[t]}+): {m.mean():.2f} ± {m.std():.2f}")
        if kind == "pr":
            ax.axhline(PRED[t].y.mean(), color=c, lw=0.6, ls="--")
    if kind == "roc":
        ax.plot([0, 1], [0, 1], ls=":", lw=0.6, color="#888888")
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.set_title("ROC (AUC)")
        ax.legend(frameon=False, loc="lower right", handlelength=1.2, fontsize=5.6)
    else:
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision–recall (AP)")
        ax.legend(frameon=False, loc="upper right", handlelength=1.2, fontsize=5.6, bbox_to_anchor=(1.0, 0.95))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.set_aspect("equal")


def plot_confusion(axs):
    for ax, t in zip(axs, TASKS):
        p = PRED[t]
        cm = pd.crosstab(p.y, p.yhat).reindex(index=[0, 1], columns=[0, 1], fill_value=0).to_numpy()
        pct = cm / cm.sum(1, keepdims=True) * 100
        nrep = p.repeat.nunique()
        ax.imshow(pct, cmap="Greys", vmin=0, vmax=130, aspect="auto")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{pct[i, j]:.0f}%\n({cm[i, j] / nrep:.0f})", ha="center", va="center", fontsize=5.3, linespacing=0.95,
                        color="white" if pct[i, j] > 65 else "#222222")
        lab = [NEG[t], POS[t]]
        ax.set_xticks([0, 1]); ax.set_xticklabels(lab); ax.set_yticks([0, 1]); ax.set_yticklabels(lab)
        ax.tick_params(length=0, pad=1)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(t, fontsize=6.8, color=TASK_COL[t], pad=2)
        ax.set_xlabel("predicted", fontsize=6.2, labelpad=1)
    axs[0].set_ylabel("true", fontsize=6.2, labelpad=1)


def plot_auc_null(ax):
    rng = np.random.default_rng(0)
    for i, t in enumerate(TASKS):
        v = MET[t].roc_auc.values; c = TASK_COL[t]
        if perm is not None and (perm.task_label == t).any():
            nv = perm.loc[perm.task_label == t, "roc_auc"].values
            ax.fill_between([i - 0.4, i + 0.4], nv.min(), np.percentile(nv, 95), color="#555555", alpha=0.22, lw=0)
            ax.plot([i - 0.4, i + 0.4], [nv.mean()] * 2, color="#222222", lw=0.8, ls="--")
        ax.boxplot(v, positions=[i], widths=0.55, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor="white", edgecolor=c, lw=0.7), medianprops=dict(color=c, lw=1.1),
                   whiskerprops=dict(lw=0.6, color=c), capprops=dict(lw=0.6, color=c))
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=2.5, color=c, alpha=0.6, lw=0, zorder=3)
    ax.set_xticks(range(len(TASKS))); ax.set_xticklabels(TASKS, rotation=35, ha="right", fontsize=6)
    for tl, t in zip(ax.get_xticklabels(), TASKS):
        tl.set_color(TASK_COL[t])
    ax.set_ylim(0.3, 1.02); ax.set_ylabel(f"ROC AUC ({N_EVAL} outer-CV folds)")
    ax.set_title("vs label-permutation null", fontsize=7)
    ax.legend(handles=[mpl.patches.Patch(color="#555555", alpha=0.3, label="null (99 perm.)")], frameon=False,
              loc="lower left", fontsize=5.6, handlelength=1.0)


def plot_baselines(ax):
    feats = list(BASE) + ["Full model"]; w = 0.15
    for k, f in enumerate(feats):
        g = BL[BL.feature == f].groupby("task").roc_auc
        m, sd = g.mean().reindex(TASKS), g.std().reindex(TASKS)
        xx = np.arange(len(TASKS)) + (k - (len(feats) - 1) / 2) * w
        ax.bar(xx, m, yerr=sd, width=w * 0.92, color=BCOL[f], lw=0, label=f, error_kw=dict(lw=0.5, capsize=1))
        for x, v, e in zip(xx, m, sd):
            ax.text(x, v + e + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=4.9, rotation=90)
    ax.axhline(0.5, ls=":", lw=0.6, color="#888888")
    ax.set_xticks(range(len(TASKS))); ax.set_xticklabels(TASKS)
    for tl, t in zip(ax.get_xticklabels(), TASKS):
        tl.set_color(TASK_COL[t])
    ax.set_ylim(0.4, 1.2); ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]); ax.spines["left"].set_bounds(0.4, 1.0)
    ax.set_ylabel("ROC AUC (held-out folds)")
    ax.set_title("Single-feature baselines vs full model", fontsize=7.2)
    ax.legend(frameon=False, ncol=5, loc="upper left", fontsize=5.6, handlelength=1.0, columnspacing=0.8)


def plot_stability(ax):
    rng = np.random.default_rng(3)
    for i, t in enumerate(TASKS):
        v = STAB[t]
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=7, color=TASK_COL[t], alpha=0.8, lw=0)
        ax.text(i, v.min() - 0.012, f"min {v.min():.2f}", ha="center", va="top", fontsize=5.4)
    ax.set_xticks(range(len(TASKS))); ax.set_xticklabels(TASKS, rotation=35, ha="right", fontsize=6)
    for tl, t in zip(ax.get_xticklabels(), TASKS):
        tl.set_color(TASK_COL[t])
    ax.set_xlim(-0.65, 2.65)                                 # room for the "min" labels inside the axes
    ax.set_ylim(min(0.84, min(v.min() for v in STAB.values()) - 0.04), 1.0); ax.set_ylabel("Spearman ρ of enhancer scores\nbetween CV repeats")
    ax.set_title("Prediction stability", fontsize=7)


def plot_metrics(axs):
    rng = np.random.default_rng(0)
    for ax, (m, lab) in zip(axs, METRICS):
        for i, t in enumerate(TASKS):
            v = MET[t][m].values; c = TASK_COL[t]
            if perm is not None and m in perm.columns and (perm.task_label == t).any():
                nv = perm.loc[perm.task_label == t, m].values
                ax.fill_between([i - 0.4, i + 0.4], nv.min(), np.percentile(nv, 95), color="#555555", alpha=0.22,
                                lw=0, zorder=1)
                ax.plot([i - 0.4, i + 0.4], [nv.mean()] * 2, color="#222222", lw=0.8, ls="--", zorder=2)
            ax.boxplot(v, positions=[i], widths=0.55, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor="white", edgecolor=c, lw=0.7), medianprops=dict(color=c, lw=1.1),
                       whiskerprops=dict(lw=0.6, color=c), capprops=dict(lw=0.6, color=c))
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=2.5, color=c, alpha=0.6, lw=0, zorder=3)
        ax.set_xticks(range(len(TASKS))); ax.set_xticklabels(TASKS, rotation=40, ha="right", fontsize=5.8)
        for tl, t in zip(ax.get_xticklabels(), TASKS):
            tl.set_color(TASK_COL[t])
        ax.set_ylim(0, 1.03); ax.set_title(lab, fontsize=6.8, pad=2)
        if ax is not axs[0]:
            ax.set_yticklabels([])
    axs[0].set_ylabel(f"score ({N_EVAL} outer-CV folds)")


def plot_schemes(axs):
    rng = np.random.default_rng(1)
    schemes = [s for s in SCHEME if s in set(folds.scheme)]
    for ax, (m, lab) in zip(axs, [("roc_auc", "ROC AUC"), ("average_precision", "Average precision")]):
        for i, t in enumerate(TASKS):
            sub = folds[folds.task == t]
            for k, s in enumerate(schemes):
                v = sub.loc[sub.scheme == s, m].values; x = i + (k - 1) * 0.27
                ax.bar(x, v.mean(), yerr=v.std(), width=0.25, color=SCHEME[s], lw=0, alpha=0.9,
                       error_kw=dict(lw=0.6, capsize=1.3))
                ax.scatter(x + rng.uniform(-0.05, 0.05, len(v)), v, s=1.1, c="#222222", alpha=0.4, lw=0, zorder=3)
            chance = 0.5 if m == "roc_auc" else (sub.n_pos / sub.n_test).mean()
            ax.plot([i - 0.42, i + 0.42], [chance] * 2, ls="--", lw=0.7, color="#222222", zorder=4)
            base = sub.loc[sub.scheme == "random", m].mean()
            dmax = max(abs(sub.loc[sub.scheme == s, m].mean() - base) for s in schemes)
            ax.text(i, min(sub[m].max() + 0.02, 1.02), f"|Δ| ≤ {dmax:.3f}", ha="center", va="bottom", fontsize=5.4)
        labs = []
        for t in TASKS:                                   # average precision depends on the positive class
            sub = folds[folds.task == t]
            other = m == "average_precision" and abs((sub.n_pos / sub.n_test).mean() - PRED[t].y.mean()) > 0.05
            labs.append(f"{t}\n({NEG[t]}+)" if other else t)
        ax.set_xticks(range(len(TASKS))); ax.set_xticklabels(labs, fontsize=6.2)
        ax.set_ylim(0, 1.12); ax.set_yticks([0, 0.25, 0.5, 0.75, 1]); ax.set_ylabel(lab)
    h = [mpl.patches.Patch(color=SCHEME[s], label=SCHEME_LAB[s]) for s in schemes] + \
        [mpl.lines.Line2D([], [], ls="--", color="#222222", lw=0.7, label="chance")]
    axs[0].legend(handles=h, ncol=4, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.02), handlelength=1.0,
                  columnspacing=0.8, fontsize=5.8)


def plot_feature_sets(ax):
    w = 0.2
    for k, st in enumerate(SETS):
        g = FS[FS.feature_set == st].groupby("task").roc_auc
        m, sd = g.mean().reindex(TASKS), g.std().reindex(TASKS)
        xx = np.arange(len(TASKS)) + (k - (len(SETS) - 1) / 2) * w
        ax.bar(xx, m, yerr=sd, width=w * 0.92, color=SET_COL[st], lw=0, label=SET_LAB[st],
               error_kw=dict(lw=0.5, capsize=1))
        for x, v, e in zip(xx, m, sd):
            ax.text(x, v + e + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=4.9, rotation=90)
    ax.axhline(0.5, ls=":", lw=0.6, color="#888888")
    ax.set_xticks(range(len(TASKS))); ax.set_xticklabels(TASKS)
    for tl, t in zip(ax.get_xticklabels(), TASKS):
        tl.set_color(TASK_COL[t])
    ax.set_ylim(0.4, 1.14); ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]); ax.spines["left"].set_bounds(0.4, 1.0)
    ax.set_ylabel(f"ROC AUC ({N_FS} folds)")
    ax.legend(frameon=False, ncol=4, loc="lower left", bbox_to_anchor=(-0.02, 1.0), fontsize=5.6, handlelength=1.0,
              columnspacing=0.8, handletextpad=0.4)


def plot_feature_set_tests(ax):
    comps = [f"{SET_LAB[a_]} − {SET_LAB[b_]}" for a_, b_ in SET_PAIRS]
    M = FSP.pivot(index="comparison", columns="task", values="mean_diff").reindex(index=comps, columns=TASKS)
    Pv = FSP.pivot(index="comparison", columns="task", values="p_adjusted_BH").reindex(index=comps, columns=TASKS)
    vmax = float(np.nanmax(np.abs(M.to_numpy())))
    ax.imshow(M.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            d, pv = M.iat[i, j], Pv.iat[i, j]
            ptxt = "adj. P < 0.001" if pv < 0.001 else (f"adj. P = {pv:.3f}" if pv < 0.1 else f"adj. P = {pv:.2f}")
            ax.text(j, i, f"{d:+.3f}\n{ptxt}".replace("-", "−"), ha="center", va="center", fontsize=5.3, linespacing=1.05,
                    color="white" if abs(d) > 0.62 * vmax else "#222222")
    ax.set_xticks(range(len(TASKS))); ax.set_xticklabels(TASKS, fontsize=6.2); ax.xaxis.tick_top()
    for tl, t in zip(ax.get_xticklabels(), TASKS):
        tl.set_color(TASK_COL[t])
    ax.set_yticks(range(len(comps))); ax.set_yticklabels(comps, fontsize=5.8)
    ax.set_xticks(np.arange(-0.5, len(TASKS)), minor=True); ax.set_yticks(np.arange(-0.5, len(comps)), minor=True)
    ax.grid(which="minor", color="white", lw=1.2); ax.tick_params(which="both", length=0, pad=2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Δ ROC AUC on identical folds\n(corrected resampled t-test, BH-adjusted)", fontsize=6.8, pad=4)


def plot_thresholds(ax):
    rng = np.random.default_rng(2)
    for i, t in enumerate(TASKS):
        v = MET[t].decision_threshold.values; c = TASK_COL[t]
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=4, color=c, alpha=0.7, lw=0)
        ax.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color=c, lw=1.2)
        ax.text(i, v.max() + 0.03, f"{np.median(v):.2f}", ha="center", fontsize=5.6)
    ax.set_xticks(range(len(TASKS))); ax.set_xticklabels([f"{t}\n({POS[t]}+)" for t in TASKS], fontsize=5.8)
    ax.set_ylim(0, 0.75); ax.set_ylabel(f"tuned threshold\n(P(positive class))")
    ax.set_title("Decision thresholds", fontsize=7)


# ----------------------------------------------------------------- assemble
fig = plt.figure(figsize=(7.2, 9.6))
gs = fig.add_gridspec(4, 1, height_ratios=[1.1, 0.9, 0.95, 0.9], hspace=0.62, left=0.08, right=0.99, top=0.97,
                      bottom=0.04)
r1 = gs[0].subgridspec(1, 3, width_ratios=[1, 1, 0.55], wspace=0.42)
axA = fig.add_subplot(r1[0]); plot_curves(axA, "roc")
axB = fig.add_subplot(r1[1]); plot_curves(axB, "pr")
axC = fig.add_subplot(r1[2]); plot_auc_null(axC)
r2 = gs[1].subgridspec(1, 2, width_ratios=[1.9, 0.55], wspace=0.35)
axD = fig.add_subplot(r2[0]); plot_baselines(axD)
axF = fig.add_subplot(r2[1]); plot_stability(axF)
r3 = gs[2].subgridspec(1, 2, wspace=0.3)
axE = [fig.add_subplot(r3[0]), fig.add_subplot(r3[1])]
plot_schemes(axE)
r4 = gs[3].subgridspec(1, 2, width_ratios=[1.25, 0.9], wspace=0.8)
axG = [fig.add_subplot(r4[0]), fig.add_subplot(r4[1])]
plot_feature_sets(axG[0]); plot_feature_set_tests(axG[1])

PANELS = {"A": [axA], "B": [axB], "C": [axC], "D": [axD], "E": [axF], "F": axE, "G": axG}   # letters in reading order
fig.canvas.draw(); rr = fig.canvas.get_renderer()
LET = []
for l, axes in PANELS.items():
    bb = mpl.transforms.Bbox.union([a.get_tightbbox(rr) for a in axes]).transformed(fig.transFigure.inverted())
    LET.append(fig.text(bb.x0 - 0.005, bb.y1 + 0.004, l, fontsize=11, fontweight="bold", ha="left", va="bottom"))
for ext in ("pdf", "png"):
    fig.savefig(FINAL / f"Fig_S1_model_performance.{ext}", dpi=300, bbox_inches="tight")
for t in LET:
    t.set_visible(False)
fig.canvas.draw(); rr = fig.canvas.get_renderer()
for l, axes in PANELS.items():
    bb = mpl.transforms.Bbox.union([a.get_tightbbox(rr) for a in axes]).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(FINAL / f"Fig_S1_model_performance_{l}.pdf", bbox_inches=bb.padded(0.04))
for t in LET:
    t.set_visible(True)
plt.show()
print("Saved", FINAL / "Fig_S1_model_performance.pdf")
