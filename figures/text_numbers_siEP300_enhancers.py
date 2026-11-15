# =============================================================================
# siEP300 PRO-seq, enhancer level: does the dex response differ after EP300 depletion?
# The numbers behind the Results statements + one panel for Figure 3 (shared A549 figure style)
#   1  siRNA x dex interaction (DESeq2, design ~ silencing + treatment + silencing:treatment): genome-wide, and
#      restricted to the enhancers that respond to dex on average over both siRNA conditions (Wald test of the mean
#      of the two dex log2FC, BH; a filter that does not depend on the difference between the conditions);
#      agreement of the dex log2FC between siNON and siEP300 over the enhancers that respond in either condition
#   2  the enhancers called in ONE condition only: their response in the other condition (direction, size, P)
#   3  enhancer classes of the public PRO-seq (UP / DN / NR from the dREG set, matched to the eRNA regions), i.e.
#      sets chosen independently of this experiment: dex response in siNON and siEP300
#        - DESeq2 log2FC (medians), the interaction (median, share > 0, Wilcoxon signed-rank vs 0) and the
#          difference siEP300 vs siNON in EtOH and in dex (the values of the dREG-vs-HOMER panel)
#        - the same response after quantile normalisation of log2(normalised count + 1) across the 8 samples
#          (removes differences in dynamic range between libraries)
#        - level-matched shift siEP300 - siNON in EtOH and in dex: enhancer minus the median of the K NR GR-unbound
#          enhancers with the closest mean level (EtOH: mean of the 4 EtOH samples; dex: of the 4 dex samples);
#          this removes differences between the libraries that depend on the signal level
#   4  EP300 and GR ChIP-seq signal (percentile rank among all eRNA regions) at the enhancers whose transcription
#      in EtOH changed after EP300 depletion, and its correlation with that change
#   Panel: dex log2FC in siNON (x) vs siEP300 (y), enhancers responsive in either condition; colour = condition(s)
#      in which the response passed the thresholds; direction is read from the quadrant. Values outside AX_LIM
#      (unshrunken log2FC of regions with zero counts in one group) are drawn at the limit.
# Sets are rebuilt here from the DESeq2 tables (padj < FDR, |log2FC| > LFC_THR); their sizes are printed first so
# they can be compared with the sets of the figure.
# Inputs (rglob under PROJECT): *_eRNA_DESeq2_*.tsv, *_eRNA_normalized_counts.tsv, *_eRNA_sample_sheet.tsv (eRNA export);
#   matched_pairs_dREG_HOMER.tsv (dREG-vs-HOMER cell; block 3); chip_signal_erna.tsv (chip_signal_server.py; block 4).
#   Blocks 3 and 4 are skipped with a message if their input is missing.
# Outputs (FIGURES): Fig_3_D_dex_concordance.pdf/png, Fig_3_siEP300_enhancer_statements.tsv
# =============================================================================
# %%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, spearmanr, wilcoxon, mannwhitneyu, norm

from a549_enhancers.paths import PROJECT, FIGURES   # local folders, set in paths.toml
FINAL = FIGURES; FINAL.mkdir(parents=True, exist_ok=True)
FDR, LFC_THR = 0.05, 0.584          # as in Methods
UNIT = "enhancer"                    # block 3: "enhancer" = one row per matched dREG enhancer, as in the dREG-vs-HOMER figure;
                                     #          "region" = each eRNA region once, only regions overlapping enhancers of one class
K_MATCH = 10                         # level-matched reference rows per row (block 3)
MEASURE = "mean_summit"              # ChIP summary of chip_signal_erna.tsv (as in the ChIP-rank panel)
MIN_MEAN = 10                        # baseMean filter for the occupancy correlations (block 4)
AX_LIM = (-8.0, 10.0)                # axis range of the panel (log2FC)
COL = {"both": "#9A9A96", "siNON only": "#EB6834", "siEP300 only": "#6250D6", "opposite": "#0B0B0B"}

FONT = "Open Sans" if any("Open Sans" in f.name for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
mpl.rcParams.update({
    "font.family": [FONT, "DejaVu Sans"], "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.8, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "axes.grid": False,
})


def find(pattern, required=True):
    hits = sorted(p for p in PROJECT.rglob(pattern) if "figures" not in p.parts)
    if not hits and required:
        raise FileNotFoundError(f"{pattern} not found under {PROJECT}")
    return hits[-1] if hits else None


def load(suffix):
    return pd.read_csv(find(f"*_eRNA_{suffix}.tsv"), sep="\t", dtype={"Peak_ID": str}).set_index("Peak_ID")


def bh(p):                                                                              # Benjamini-Hochberg, NaN kept
    p = np.asarray(p, float); ok = ~np.isnan(p); q = np.full(p.shape, np.nan); x = p[ok]
    o = np.argsort(x); adj = x[o] * len(x) / np.arange(1, len(x) + 1)
    out = np.empty(len(x)); out[o] = np.minimum.accumulate(adj[::-1])[::-1]; q[ok] = np.minimum(out, 1)
    return q


ROWS = []
def keep(block, group, n, metric, value):
    ROWS.append({"block": block, "group": group, "n": int(n), "metric": metric, "value": value})


# ----------------------------------------------------------------- data and sets
N, P = load("DESeq2_siNON_dex_vs_EtOH"), load("DESeq2_siEP300_dex_vs_EtOH")              # dex vs EtOH per siRNA
B, BD = load("DESeq2_siEP300_vs_siNON_EtOH"), load("DESeq2_siEP300_vs_siNON_dex")        # siEP300 vs siNON per treatment
I = load("DESeq2_interaction_siEP300xdex")
NC = load("normalized_counts")
for d in (P, B, BD, I, NC):
    assert d.index.equals(N.index), "eRNA export tables are not in the same row order"

sig = lambda d: (d.padj < FDR) & (d.log2FoldChange.abs() > LFC_THR)
up, dn = (lambda d: sig(d) & (d.log2FoldChange > 0)), (lambda d: sig(d) & (d.log2FoldChange < 0))
either = sig(N) | sig(P)
same_dir = np.sign(N.log2FoldChange) == np.sign(P.log2FoldChange)
cat = pd.Series(np.select([sig(N) & sig(P) & same_dir, sig(N) & sig(P), sig(N), sig(P)],
                          ["both", "opposite", "siNON only", "siEP300 only"], default="none"), index=N.index)
print(f"{len(N):,} eRNA regions; padj < {FDR}, |log2FC| > {LFC_THR}")
print(f"  EtOH, siEP300 vs siNON: up {up(B).sum()}, down {dn(B).sum()}")
print(f"  dex-induced:   both {(up(N) & up(P)).sum()}, siNON only {(up(N) & ~up(P)).sum()}, siEP300 only {(up(P) & ~up(N)).sum()}"
      f" (of these with lower EtOH signal in siEP300: {(up(P) & ~up(N) & dn(B)).sum()})")
print(f"  dex-repressed: both {(dn(N) & dn(P)).sum()}, siNON only {(dn(N) & ~dn(P)).sum()}, siEP300 only {(dn(P) & ~dn(N)).sum()}")

# ----------------------------------------------------------------- 1. interaction and agreement
x, y = N.log2FoldChange[either], P.log2FoldChange[either]
r_p, r_s, pct_same = pearsonr(x, y)[0], spearmanr(x, y)[0], 100 * same_dir[either].mean()
n_int, n_int10 = int((I.padj < FDR).sum()), int((I.padj < 0.10).sum())
print(f"\n1. siRNA x dex interaction: padj < {FDR} at {n_int} of {len(I):,} regions (padj < 0.10: {n_int10}; smallest padj {I.padj.min():.3f})")
# the same test restricted to enhancers that respond to dex ON AVERAGE over both siRNA conditions: this filter does not
# depend on the difference between the conditions (unlike "significant in either"), so BH within the set stays valid
avg = (N.log2FoldChange + P.log2FoldChange) / 2
z = avg / (np.sqrt(N.lfcSE ** 2 + P.lfcSE ** 2) / 2)
on_avg = pd.Series(bh(2 * norm.sf(z.abs())), index=N.index).lt(FDR) & (avg.abs() > LFC_THR)
q_in = pd.Series(bh(I.pvalue[on_avg].values), index=I.index[on_avg])
hits = q_in.index[q_in < FDR]
print(f"   restricted to the {on_avg.sum()} enhancers responsive on average over both conditions: BH q < {FDR} at {len(hits)}"
      f" (induced {int((avg[hits] > 0).sum())}, repressed {int((avg[hits] < 0).sum())}; larger response in siEP300: "
      f"{int((np.sign(I.log2FoldChange[hits]) == np.sign(avg[hits])).sum())}; median EtOH log2FC siEP300 vs siNON {B.log2FoldChange[hits].median():+.2f})")
print(f"   responsive in either condition: {either.sum()} | Pearson r = {r_p:.2f}, Spearman rho = {r_s:.2f}, same direction {pct_same:.1f}%")
keep("1 interaction", "all regions", len(I), f"regions with interaction padj < {FDR}", n_int)
keep("1 interaction", "all regions", len(I), "regions with interaction padj < 0.10", n_int10)
keep("1 interaction", "responsive on average over both conditions", on_avg.sum(), f"regions with interaction BH q < {FDR} (within the set)", len(hits))
keep("1 agreement", "responsive in either condition", either.sum(), "Pearson r of dex log2FC (siNON vs siEP300)", round(r_p, 3))
keep("1 agreement", "responsive in either condition", either.sum(), "Spearman rho", round(r_s, 3))
keep("1 agreement", "responsive in either condition", either.sum(), "same direction (%)", round(pct_same, 1))

# ----------------------------------------------------------------- 2. called in one condition only
print("\n2. enhancers called in one condition only: response in the OTHER condition")
for name, m, other, sgn in (("dex-induced, siEP300 only", up(P) & ~up(N), N, 1), ("dex-induced, siNON only", up(N) & ~up(P), P, 1),
                            ("dex-repressed, siEP300 only", dn(P) & ~dn(N), N, -1), ("dex-repressed, siNON only", dn(N) & ~dn(P), P, -1)):
    o = other.log2FoldChange[m] * sgn
    vals = {"median log2FC in the other condition": round(sgn * o.median(), 2), "same direction (%)": round(100 * (o > 0).mean(), 1),
            f"same direction and |log2FC| > {LFC_THR} (%)": round(100 * (o > LFC_THR).mean(), 1),
            "same direction and nominal P < 0.05 (%)": round(100 * ((o > 0) & (other.pvalue[m] < 0.05)).mean(), 1)}
    print(f"   {name:28s} n = {int(m.sum()):4d}: " + "; ".join(f"{k} {v}" for k, v in vals.items()))
    for k, v in vals.items():
        keep("2 one condition only", name, m.sum(), k, v)

# ----------------------------------------------------------------- 3. classes of the public PRO-seq
SS = pd.read_csv(find("*_eRNA_sample_sheet.tsv"), sep="\t")
cols = lambda si, tr: SS.loc[(SS.silencing == si) & (SS.treatment == tr), "sample"].tolist()
L = np.log2(NC + 1)
srt = np.sort(L.values, axis=0).mean(axis=1)                                             # quantile normalisation
QN = pd.DataFrame({c: srt[L[c].rank(method="first").astype(int).values - 1] for c in L.columns}, index=L.index)
resp = lambda X, si: X[cols(si, "dex")].mean(axis=1) - X[cols(si, "EtOH")].mean(axis=1)
lvl_e, lvl_d = L[cols("siNON", "EtOH") + cols("siEP300", "EtOH")].mean(axis=1), L[cols("siNON", "dex") + cols("siEP300", "dex")].mean(axis=1)
shift_e = L[cols("siEP300", "EtOH")].mean(axis=1) - L[cols("siNON", "EtOH")].mean(axis=1)
shift_d = L[cols("siEP300", "dex")].mean(axis=1) - L[cols("siNON", "dex")].mean(axis=1)

mp_file = find("matched_pairs_dREG_HOMER.tsv", required=False)
if mp_file is None:
    print("\n3. matched_pairs_dREG_HOMER.tsv not found - run the dREG-vs-HOMER cell first; block skipped")
else:
    mp = pd.read_csv(mp_file, sep="\t", dtype={"Peak_ID": str})
    mp = mp[mp.Peak_ID.isin(N.index)]
    if UNIT == "region":                                                                 # each eRNA region once, one class only
        g = mp.groupby("Peak_ID").agg(cls=("cls", lambda s: "/".join(sorted(set(s)))), gr=("GR_bound", "max")).reset_index()
        g = g[g.cls.isin(["UP", "DN", "NR"])].reset_index(drop=True)
    else:                                                                                # one row per matched dREG enhancer
        g = mp.rename(columns={"GR_bound": "gr"})[["Peak_ID", "cls", "gr"]].reset_index(drop=True)
    g["state"] = np.where(g.cls == "NR", np.where(g.gr.astype(bool), "NR GR-bound", "NR GR-unbound"), g.cls)
    val = lambda s, m: s.reindex(g.Peak_ID[m]).to_numpy(dtype=float)                     # values of the rows in mask m
    ref_m = (g.state == "NR GR-unbound").to_numpy()

    def excess(m, level, shift):                                                         # row minus its level-matched reference
        rl, rv, l, s = val(level, ref_m), val(shift, ref_m), val(level, m), val(shift, m)
        return np.array([s[i] - np.median(rv[np.argsort(np.abs(rl - l[i]))[:K_MATCH]]) for i in range(len(l))])

    print(f"\n3. classes of the public PRO-seq in this experiment (unit: {UNIT}; {len(g):,} rows, {g.Peak_ID.nunique():,} eRNA regions)")
    for st, sgn in (("UP", 1), ("NR GR-bound", 0), ("NR GR-unbound", 0), ("DN", -1)):
        m = (g.state == st).to_numpy(); n = int(m.sum())
        lfc_n, lfc_p, inter = val(N.log2FoldChange, m), val(P.log2FoldChange, m), val(I.log2FoldChange, m)
        inter = inter[~np.isnan(inter)]
        vals = {"median dex log2FC, siNON (DESeq2)": round(float(np.nanmedian(lfc_n)), 2),
                "median dex log2FC, siEP300 (DESeq2)": round(float(np.nanmedian(lfc_p)), 2),
                "median interaction log2FC": round(float(np.median(inter)), 2), "interaction > 0 (%)": round(100 * float((inter > 0).mean()), 1),
                "interaction vs 0, signed-rank P": float(f"{wilcoxon(inter).pvalue:.2g}"),
                "median siEP300 vs siNON log2FC in EtOH (DESeq2)": round(float(np.nanmedian(val(B.log2FoldChange, m))), 2),
                "median siEP300 vs siNON log2FC in dex (DESeq2)": round(float(np.nanmedian(val(BD.log2FoldChange, m))), 2),
                "median dex - EtOH, siNON (quantile-normalised)": round(float(np.median(val(resp(QN, "siNON"), m))), 2),
                "median dex - EtOH, siEP300 (quantile-normalised)": round(float(np.median(val(resp(QN, "siEP300"), m))), 2)}
        if sgn:
            vals["direction of the class reproduced in siNON (%)"] = round(100 * float(np.nanmean(sgn * lfc_n > 0)), 1)
            for si in ("siNON", "siEP300"):                                              # dex replicate x EtOH replicate
                vals[f"median dex - EtOH per replicate pairing, {si} (quantile-normalised)"] = ", ".join(
                    f"{np.median(val(QN[a] - QN[b], m)):+.2f}" for a in cols(si, "dex") for b in cols(si, "EtOH"))
        if st != "NR GR-unbound":
            e, d_ = excess(m, lvl_e, shift_e), excess(m, lvl_d, shift_d)
            vals.update({"level-matched shift in EtOH (log2)": round(float(np.median(e)), 2), "level-matched shift in EtOH, P": float(f"{wilcoxon(e).pvalue:.2g}"),
                         "level-matched shift in dex (log2)": round(float(np.median(d_)), 2), "level-matched shift in dex, P": float(f"{wilcoxon(d_).pvalue:.2g}")})
        print(f"   {st:14s} n = {n:4d}: " + "; ".join(f"{k} {v}" for k, v in vals.items()))
        for k, v in vals.items():
            keep("3 public classes", st, n, k, v)

# ----------------------------------------------------------------- 4. occupancy and the effect of EP300 depletion in EtOH
chip_file = find("chip_signal_erna.tsv", required=False)
if chip_file is None:
    print("\n4. chip_signal_erna.tsv not found; block skipped")
else:
    C = pd.read_csv(chip_file, sep="\t", dtype={"id": str}).set_index("id").reindex(N.index)
    R = pd.DataFrame({t: 100 * C[f"{t}|{MEASURE}"].rank(pct=True) for t in ("EP300 EtOH", "EP300 dex", "GR dex")})
    groups = {"lower in siEP300 (EtOH)": dn(B), "higher in siEP300 (EtOH)": up(B), "in no set": ~(sig(B) | sig(N) | sig(P))}
    ok = (N.baseMean >= MIN_MEAN) & R["EP300 EtOH"].notna() & B.log2FoldChange.notna()
    print(f"\n4. ChIP-seq percentile rank among all eRNA regions ({MEASURE})")
    for t in R.columns:
        med = {k: R.loc[m, t].median() for k, m in groups.items()}
        p_du = mannwhitneyu(R.loc[groups["lower in siEP300 (EtOH)"], t].dropna(), R.loc[groups["higher in siEP300 (EtOH)"], t].dropna()).pvalue
        rho, p_rho = spearmanr(R.loc[ok, t], B.log2FoldChange[ok])
        print(f"   {t:10s} median rank: " + ", ".join(f"{k} {v:.0f}" for k, v in med.items()) + f" | lower vs higher P = {p_du:.2g}"
              f" | Spearman with the EtOH log2FC (siEP300 vs siNON, baseMean >= {MIN_MEAN}, n = {int(ok.sum())}): {rho:+.2f} (P = {p_rho:.1g})")
        for k, m in groups.items():
            keep("4 occupancy", k, m.sum(), f"median {t} rank", round(med[k], 1))
        keep("4 occupancy", "lower vs higher in siEP300 (EtOH)", groups["lower in siEP300 (EtOH)"].sum() + groups["higher in siEP300 (EtOH)"].sum(), f"{t} rank, Mann-Whitney P", float(f"{p_du:.2g}"))
        keep("4 occupancy", f"baseMean >= {MIN_MEAN}", ok.sum(), f"Spearman rho, {t} rank vs EtOH log2FC (siEP300 vs siNON)", round(rho, 3))

pd.DataFrame(ROWS).to_csv(FINAL / "Fig_3_siEP300_enhancer_statements.tsv", sep="\t", index=False)

# ----------------------------------------------------------------- panel
fig = plt.figure(figsize=(4.3, 2.7))
ax = fig.add_axes([0.118, 0.155, 0.515, 0.82])                                           # legend and statistics to the right
ax.axhline(0, color="#C3C2B7", lw=0.5, zorder=1); ax.axvline(0, color="#C3C2B7", lw=0.5, zorder=1)
ax.plot(AX_LIM, AX_LIM, color="#898781", lw=0.6, ls=(0, (3, 2)), zorder=1)
xc, yc = N.log2FoldChange.clip(*AX_LIM), P.log2FoldChange.clip(*AX_LIM)
n_clip = int(((N.log2FoldChange[either] != xc[either]) | (P.log2FoldChange[either] != yc[either])).sum())
handles = []
for k in ("both", "siEP300 only", "siNON only", "opposite"):                             # smallest set drawn last
    m = (cat == k)
    if not m.any():
        continue
    ax.scatter(xc[m], yc[m], s=9, color=COL[k], edgecolor="white", linewidth=0.25, alpha=0.9, zorder=3, clip_on=False, rasterized=True)
    handles.append(Line2D([], [], marker="o", ls="", ms=4, mfc=COL[k], mec="white", mew=0.3, label=f"{k} ({int(m.sum())})"))
order = [h for k in ("both", "siNON only", "siEP300 only", "opposite") for h in handles if h.get_label().startswith(k)]
ax.legend(handles=order, title="Dex response called in", title_fontsize=6.8, loc="upper left", bbox_to_anchor=(1.03, 1.0), frameon=False,
          handletextpad=0.2, borderaxespad=0, labelspacing=0.3, alignment="left")
ax.text(1.05, 0.50, f"n = {int(either.sum())}\nPearson r = {r_p:.2f}\nsame direction: {pct_same:.0f}%\n\nsiRNA × dex interaction,\n"
        f"FDR < {FDR:g}: {n_int} of {len(I):,}", transform=ax.transAxes, ha="left", va="top", fontsize=6.8, linespacing=1.3)
ax.set_xlim(*AX_LIM); ax.set_ylim(*AX_LIM); ax.set_aspect("equal", adjustable="box", anchor="W")
ticks = np.arange(np.ceil(AX_LIM[0] / 4) * 4, AX_LIM[1] + 0.1, 4); ax.set_xticks(ticks); ax.set_yticks(ticks)
ax.set_xlabel("dex log2FC, siNON"); ax.set_ylabel("dex log2FC, siEP300")
ax.tick_params(length=2.5, width=0.6)
fig.savefig(FINAL / "Fig_3_D_dex_concordance.pdf", dpi=600); fig.savefig(FINAL / "Fig_3_D_dex_concordance.png", dpi=300)
print(f"\npanel: {either.sum()} enhancers; {n_clip} drawn at the axis limit {AX_LIM}; saved to {FINAL}")
plt.show()
