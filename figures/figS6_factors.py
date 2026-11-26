# =============================================================================
# Supplementary Figure S6 - GR, EP300, FOXA1 and TEAD4 at the enhancer classes
# One row per factor (GR as reference, EP300, FOXA1, TEAD4), one column per characteristic:
#   1  mean ChIP-seq signal around the enhancer centre (+-2 kb) per class, EtOH (solid) and dex (dashed),
#      from the deepTools matrices of Fig. 2 (TEAD4: untreated only)
#      Control and dex are SEPARATE ENCODE experiments (p-value-type signal; e.g. EP300 dex is ~2x lower at weak
#      and ~15-18x lower at strong sites), so curve heights are not comparable between conditions: the dex curve
#      has its own axis (right, dashed). GR is drawn on one axis (EtOH GR is background; dex induces binding).
#   2  occupancy per enhancer as percentile rank among all 8,270 enhancers, within each experiment (scale-free)
#   3  dex-associated change per enhancer: level-adjusted rank change = dex rank minus the dex rank expected for
#      the same control rank among NR enhancers (20 bins) - removes the scale difference between experiments,
#      its signal-level dependence and regression to the mean; numbers = median (rank points)
#   4  enhancers with high signal (top quartile of all 8,270 enhancers in that condition) per class, and
#      enhancers carrying the factor's motif family (HOMER presence call), % with 95 % Wilson CI
#   p-values under the panels: UP vs NR and DN vs NR, two-sided Mann-Whitney (Wilcoxon rank-sum) test for
#   the per-enhancer values (columns 2, 3) and Fisher's exact test for the high-signal proportions (column 4).
#   Motif presence increases with enhancer length, and UP and DN enhancers are longer than NR enhancers, so the
#   motif p-values are adjusted for length: logistic regression motif ~ class + log2(length) on the enhancers of
#   the two classes, likelihood-ratio test for the class term. The unadjusted Fisher p-values and the adjusted
#   odds ratios are kept in the summary table.
# Inputs: feature matrix; HEATMAPS/matrix_sorted_<GR|EP300|FOXA1|TEAD4>.tab.gz + sorted class BEDs
# Supplementary Figure S6 (previously Fig_M_factors).
# Outputs (FIGURES): Fig_S6_factors.pdf/png, per-panel PDFs Fig_S6_factors_A..D.pdf,
#   Fig_S6_factor_class_summary.tsv
# =============================================================================
# %%
from pathlib import Path
import gzip, json, re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu, fisher_exact, chi2
from sklearn.linear_model import LogisticRegression

from a549_enhancers.paths import PROJECT, ML, FIGURES, CHIP_MATRICES, PROSEQ_MATRICES   # local folders, set in paths.toml
FINAL = FIGURES; FINAL.mkdir(parents=True, exist_ok=True)   # final figures + their tables
FM = ML / "feature_matrix_all_classes_GRannotated.csv"
HEATMAPS = CHIP_MATRICES
SORTED_BEDS = {"NR": "NR_ctrlPlusDex_sorted.bed", "UP": "UP_sorted_by_dex.bed", "DN": "DN_sorted_by_ctrl.bed"}
BEDS = {g: (HEATMAPS / f if (HEATMAPS / f).exists() else PROSEQ_MATRICES / f) for g, f in SORTED_BEDS.items()}
FACTORS = [  # (label, control column, dex column, motif regex on the HOMER motif name, matrix file)
    ("GR", "NR3C1_eth_0.02_1 hour_ENCFF914PNO", "NR3C1_dex_100_1 hour_ENCFF090XQU", r"^GRE\(", HEATMAPS / "matrix_sorted_GR.tab.gz"),
    ("EP300", "EP300_ENCFF376VNJ", "EP300_dex_100_1 hour_ENCFF876TGI", None, HEATMAPS / "matrix_sorted_EP300.tab.gz"),
    ("FOXA1", "FOXA1_eth_0.02_1 hour_ENCFF640SEZ", "FOXA1_dex_100_1 hour_ENCFF095IRQ", r"FOXA1|Foxa", HEATMAPS / "matrix_sorted_FOXA1.tab.gz"),
    ("TEAD4", "TEAD4_ENCFF186WSI", None, r"^TEAD", HEATMAPS / "matrix_sorted_TEAD4.tab.gz"),
]
HIGH_Q = 0.75              # "high signal" = above this quantile of all enhancers (per condition)
UNRELIABLE_DEX = []          # factors whose dex track is left out entirely; none
NOTES_ON_PANELS = False      # no notes are printed on the dex panels
CAVEAT_RHO = 0.3             # used only if NOTES_ON_PANELS is True
N_BINS = 20                  # baseline-rank bins for the level-adjusted change
CLS = {"UP": "#D1495B", "DN": "#3B6FB6", "NR": "#5DAA82"}
ORDER = ["NR", "UP", "DN"]

FONT = "Open Sans" if any("Open Sans" in f.name for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
mpl.rcParams.update({
    "font.family": [FONT, "DejaVu Sans"], "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.8, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "axes.grid": False,
})


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


def wilson(k, n, z=1.96):
    if n == 0: return (0, 0, 0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


# ----------------------------------------------------------------- data
fm = pd.read_csv(FM, low_memory=False)
cls = fm["class"].values
motif_cols = [c for c in fm.columns if c.startswith("motif_binary::")]
LOG_LEN = np.log2(fm["Enh_len"].to_numpy(dtype=float))


def motif_length_adjusted(has, m, ref):
    """class (mask m) vs reference class (mask ref) for a binary motif call, adjusted for enhancer length:
    logistic regression motif ~ class + log2(length); returns (odds ratio of the class term, likelihood-ratio p)"""
    sel = m | ref
    y, g, ln = has[sel].astype(int), m[sel].astype(float), LOG_LEN[sel]

    def fit(X):
        mod = LogisticRegression(C=1e9, max_iter=5000).fit(X, y)             # C large = no penalty
        pr = np.clip(mod.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
        return mod, float(np.sum(y * np.log(pr) + (1 - y) * np.log(1 - pr)))

    full, ll1 = fit(np.column_stack([g, ln])); _, ll0 = fit(ln[:, None])
    return float(np.exp(full.coef_[0][0])), float(chi2.sf(max(2 * (ll1 - ll0), 0), 1))


def read_profiles(f):
    """mean signal per bin and class for each sample of a deepTools matrix (rows matched to the class BEDs)"""
    with gzip.open(f, "rt") as fh:
        h = json.loads(fh.readline()[1:])
    d = pd.read_csv(f, sep="\t", header=None, skiprows=1)
    m = d.iloc[:, 6:].to_numpy(dtype=float); sb = h["sample_boundaries"]
    key = {k: i for i, k in enumerate(zip(d[0], d[1], d[2]))}
    labs = h["sample_labels"]
    conds = ["EtOH", "Dex"] if len(labs) == 2 else ["untreated"]
    out, med = {}, {}
    for g in ORDER:
        b = pd.read_csv(BEDS[g], sep="\t", header=None, usecols=[0, 1, 2])
        idx = [key[k] for k in zip(b[0], b[1], b[2])]
        for k, c in enumerate(conds):
            sub = m[idx, sb[k]:sb[k + 1]]
            out[(g, c)] = np.nanmean(sub, axis=0)
            med[(g, c)] = float(np.nanmedian(np.nanmean(sub, axis=1)))       # median enhancer, window mean
    bs, up = h.get("bin size", 10), h.get("upstream", 2000)
    return out, (bs[0] if isinstance(bs, list) else bs), (up[0] if isinstance(up, list) else up), med


rows = []
for lab, cc, cd, mre, mf in FACTORS:
    rho = pd.Series(fm[cc].values).rank().corr(pd.Series(fm[cd].values).rank()) if cd else np.nan
    if lab in UNRELIABLE_DEX:
        print(f"  {lab}: dex track left out (control vs dex Spearman = {rho:.2f})")
        cd = None
    x_c = 100 * pd.Series(fm[cc].values.astype(float)).rank(pct=True).to_numpy()          # percentile rank
    x_d = 100 * pd.Series(fm[cd].values.astype(float)).rank(pct=True).to_numpy() if cd else None
    if cd:
        nr_ = cls == "NR"
        b = np.clip((x_c / 100 * N_BINS).astype(int), 0, N_BINS - 1)
        arc = x_d - pd.Series(b).map(pd.Series(x_d[nr_]).groupby(b[nr_]).median()).to_numpy()
    else:
        arc = None
    mot = [c for c in motif_cols if mre and re.search(mre, c.split("::")[1])]
    has_motif = (fm[mot].values.astype(int).sum(axis=1) > 0) if mot else None
    thr_c = np.quantile(x_c, HIGH_Q); thr_d = np.quantile(x_d, HIGH_Q) if cd else None
    prof = read_profiles(mf) if Path(mf).exists() else None
    if prof is not None and not cd:                                         # drop unreliable / absent dex curves
        prof = ({k: v for k, v in prof[0].items() if k[1] != "Dex"}, prof[1], prof[2], prof[3])
    for g in ORDER:
        m = cls == g; n = int(m.sum())
        nr = cls == "NR"
        def mw(x): return np.nan if g == "NR" else mannwhitneyu(x[m], x[nr], alternative="two-sided").pvalue
        def fe(b): return np.nan if g == "NR" else fisher_exact([[int(b[m].sum()), int((~b[m]).sum())],
                                                                    [int(b[nr].sum()), int((~b[nr]).sum())]])[1]
        r = dict(factor=lab, cls=g, n=n, median_rank_control=np.median(x_c[m]),
                 median_rank_dex=np.median(x_d[m]) if cd else np.nan,
                 median_rank_change=np.median(arc[m]) if cd else np.nan,
                 frac_high_control=(x_c[m] > thr_c).mean(), frac_high_dex=(x_d[m] > thr_d).mean() if cd else np.nan,
                 frac_motif=has_motif[m].mean() if has_motif is not None else np.nan, n_motifs_in_family=len(mot),
                 p_vs_NR_control=mw(x_c), p_vs_NR_dex=mw(x_d) if cd else np.nan,
                 p_vs_NR_change=mw(arc) if cd else np.nan,
                 p_vs_NR_high_control=fe(x_c > thr_c), p_vs_NR_high_dex=fe(x_d > thr_d) if cd else np.nan,
                 p_vs_NR_motif=fe(has_motif) if has_motif is not None else np.nan)
        r["motif_odds_ratio_vs_NR_length_adjusted"], r["p_vs_NR_motif_length_adjusted"] = (
            motif_length_adjusted(has_motif, m, nr) if has_motif is not None and g != "NR" else (np.nan, np.nan))
        rows.append(r)
    FACTOR_DATA = globals().setdefault("FACTOR_DATA", {})
    FACTOR_DATA[lab] = dict(xc=x_c, xd=x_d, arc=arc, motif=has_motif, thr=(thr_c, thr_d), prof=prof, n_mot=len(mot),
                            rho=rho, unreliable=lab in UNRELIABLE_DEX)
summary = pd.DataFrame(rows)
summary.to_csv(FINAL / "Fig_S6_factor_class_summary.tsv", sep="\t", index=False, float_format="%.3g")
PV = summary.set_index(["factor", "cls"])


def fmt_p(p):
    if np.isnan(p): return "n/a"
    return "p < 1e-300" if p == 0 else (f"p = {p:.1g}" if p < 0.001 else f"p = {p:.2g}")


def p_row(ax, lab, cols, labels, head=None):
    """UP vs NR and DN vs NR p-values below the axes"""
    txt = [head] if head else []
    for c, l in zip(cols, labels):
        pu, pd_ = PV.loc[(lab, "UP"), c], PV.loc[(lab, "DN"), c]
        txt += [f"{l}UP vs NR {fmt_p(pu)}", f"{l}DN vs NR {fmt_p(pd_)}"]
    ax.text(-0.05, -0.3, "\n".join(txt), transform=ax.transAxes, ha="left", va="top", fontsize=5.2,
            color="#444444", linespacing=1.2)
print(f"font: {FONT}\n", summary.round(2).to_string(index=False))


# ----------------------------------------------------------------- one row per factor
def factor_row(fig, spec, lab, first, last):
    D = FACTOR_DATA[lab]
    sub = spec.subgridspec(1, 4, width_ratios=[1.15, 1.0, 0.8, 1.15], wspace=0.75)
    axes = [fig.add_subplot(sub[0, k]) for k in range(4)]
    # 1 metaprofile
    ax = axes[0]
    if D["prof"] is not None:
        prof, bs, up, _ = D["prof"]
        x = (np.arange(len(next(iter(prof.values())))) * bs - up) / 1000
        twin = lab != "GR" and any(c == "Dex" for _, c in prof)
        axd = ax.twinx() if twin else ax
        for (g, c), v in prof.items():
            (axd if c == "Dex" else ax).plot(x, v, color=CLS[g], lw=1.0, ls="-" if c in ("EtOH", "untreated") else "--")
        ax.set_xlim(x[0], x[-1]); ax.set_xticks([-2, 0, 2])
        if twin:
            axd.spines["right"].set_visible(True); axd.spines["right"].set_linewidth(0.6)
            axd.tick_params(labelsize=6, colors="#666666"); axd.set_ylim(bottom=0)
            axd.set_ylabel("dex", fontsize=6, color="#666666", labelpad=1)
        if D["unreliable"]:
            ax.text(0.98, 0.97, f"dex not shown\n(ρ = {D['rho']:.2f} with control)", transform=ax.transAxes,
                    ha="right", va="top", fontsize=5.4, color="#555555")
        elif NOTES_ON_PANELS and D["xd"] is not None and D["rho"] < CAVEAT_RHO and lab != "GR":
            ax.text(0.02, 0.97, f"caution: dex track\nρ = {D['rho']:.2f} with control", transform=ax.transAxes,
                    ha="left", va="top", fontsize=5.4, color="#B03A2E")
        ax.set_xlabel("Distance (kb)" if last else ""); ax.set_ylabel(f"{lab}\nmean signal", fontweight="bold")
        if first:
            h = [Line2D([], [], color="#444444", lw=1, ls="-", label="EtOH / untreated"),
                 Line2D([], [], color="#444444", lw=1, ls="--", label="dex")]
            ax.legend(handles=h, frameon=False, fontsize=6, loc="upper left", handlelength=1.6)
    # 2 per-enhancer signal
    ax = axes[1]
    pos, data, cols = [], [], []
    for i, g in enumerate(ORDER):
        m = cls == g
        series = [D["xc"][m]] + ([D["xd"][m]] if D["xd"] is not None else [])
        for k, s in enumerate(series):
            pos.append(i + (k - 0.5) * 0.36 if len(series) == 2 else i); data.append(s); cols.append((g, k))
    bp = ax.boxplot(data, positions=pos, widths=0.3, showfliers=False, patch_artist=True, whis=(5, 95),
                    medianprops=dict(color="#222222", lw=0.9), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
    for patch, (g, k) in zip(bp["boxes"], cols):
        patch.set_facecolor(CLS[g] if k == 1 or D["xd"] is None else "white"); patch.set_edgecolor(CLS[g]); patch.set_lw(0.8)
    ax.set_xticks(range(3)); ax.set_xticklabels(ORDER)
    for tl, g in zip(ax.get_xticklabels(), ORDER): tl.set_color(CLS[g]); tl.set_fontweight("bold")
    ax.set_ylabel("percentile rank"); ax.set_ylim(0, 100)
    if first:
        ax.set_title("Occupancy per enhancer, rank within\nexperiment (open = control, filled = dex)", fontsize=7)
    p_row(ax, lab, ["p_vs_NR_control"] + (["p_vs_NR_dex"] if D["xd"] is not None else []),
          ["ctrl: ", "dex: "] if D["xd"] is not None else [""])
    # 3 dex-induced change
    ax = axes[2]
    if D["xd"] is not None:
        dl = D["arc"]
        parts = ax.violinplot([dl[cls == g] for g in ORDER], positions=range(3), widths=0.8, showextrema=False)
        for b, g in zip(parts["bodies"], ORDER):
            b.set_facecolor(CLS[g]); b.set_alpha(0.55); b.set_edgecolor("none")
        for i, g in enumerate(ORDER):
            med = np.median(dl[cls == g]); ax.plot([i - 0.25, i + 0.25], [med, med], color="#222222", lw=1)
            ax.text(i, np.quantile(dl, 0.995), f"{med:+.0f}", ha="center", va="bottom", fontsize=6.3)
        ax.axhline(0, ls=":", lw=0.6, c="#888888")
        ax.set_xticks(range(3)); ax.set_xticklabels(ORDER)
        for tl, g in zip(ax.get_xticklabels(), ORDER): tl.set_color(CLS[g]); tl.set_fontweight("bold")
        ax.set_ylabel("Δ rank vs NR with\nsimilar baseline")
        if NOTES_ON_PANELS and D["rho"] < CAVEAT_RHO and lab != "GR":
            ax.text(0.5, 0.02, f"caution: ρ = {D['rho']:.2f}", transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=5.4, color="#B03A2E")
        if first: ax.set_title("Dex-associated change\nper enhancer (median)", fontsize=7)
        p_row(ax, lab, ["p_vs_NR_change"], [""])
    else:
        msg = (f"dex track\nnot comparable\n(ρ = {D['rho']:.2f})" if D["unreliable"] else "no dex\nChIP-seq")
        ax.axis("off"); ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=6.5, color="#777777",
                                transform=ax.transAxes)
    # 4 bound / motif fractions
    ax = axes[3]
    groups = [("high,\ncontrol", lambda m: D["xc"][m] > D["thr"][0])]
    if D["xd"] is not None: groups.append(("high,\ndex", lambda m: D["xd"][m] > D["thr"][1]))
    if D["motif"] is not None: groups.append((f"motif\n({D['n_mot']} PWMs)", lambda m: D["motif"][m]))
    w = 0.26
    for j, (gl, fn) in enumerate(groups):
        for i, g in enumerate(ORDER):
            m = cls == g; k = int(fn(m).sum()); n = int(m.sum())
            p, lo, hi = wilson(k, n)
            ax.bar(j + (i - 1) * w, p * 100, width=w, color=CLS[g])
            ax.errorbar(j + (i - 1) * w, p * 100, yerr=[[(p - lo) * 100], [(hi - p) * 100]], fmt="none", ecolor="#333333", lw=0.6, capsize=1.2)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g for g, _ in groups], fontsize=6.4)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylabel("Enhancers (%)"); ax.set_ylim(0, 100)
    if first: ax.set_title(f"High signal (top {100 - HIGH_Q * 100:.0f}% of all)\nor motif present, 95% CI", fontsize=7)
    pc = [("p_vs_NR_high_control", "ctrl: ")] + ([("p_vs_NR_high_dex", "dex: ")] if D["xd"] is not None else []) + \
         ([("p_vs_NR_motif_length_adjusted", "motif, length-adj.: ")] if D["motif"] is not None else [])
    p_row(ax, lab, [c for c, _ in pc], [l for _, l in pc])
    return axes


fig = plt.figure(figsize=(7.2, 2.5 * len(FACTORS)))
gs = fig.add_gridspec(len(FACTORS), 1, hspace=1.1, left=0.09, right=0.99, top=0.92, bottom=0.07)
parts, lead = {}, {}
for r, (lab, *_) in enumerate(FACTORS):
    axes = factor_row(fig, gs[r], lab, first=(r == 0), last=(r == len(FACTORS) - 1))
    L = "ABCDEFGH"[r]; parts[L] = axes; lead[L] = axes[0]
h = [mpl.patches.Patch(color=CLS[g], label=g) for g in ORDER]
fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, title="Enhancer class", title_fontsize=7)
letters = panel_letters(fig, lead)
for ext in ("pdf", "png"):
    fig.savefig(FINAL / f"Fig_S6_factors.{ext}", dpi=300, bbox_inches="tight")
export_panels(fig, parts, "Fig_S6_factors", letters)
plt.show()
print("Saved to", FINAL / "Fig_S6_factors.pdf")
