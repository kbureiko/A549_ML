# =============================================================================
# Chromatin topics of the non-responsive (NR) enhancers (consensus NMF; enhancers = mixtures of programs)
#   Writes the topic loadings and the per-enhancer topic usage that Figure 1H, Supplementary Figure S3 and
#   figures/text_numbers_feature_map.py read; the figure drawn here is a diagnostic overview.
#   input: the 13 chromatin features of the NR states (EtOH / untreated; GR in dex; FOXA1 EtOH) as percentile
#          ranks within NR (0-1: non-negative, scale-free). UP / DN are placed on the same scale (percentile of
#          their value within the NR distribution) and projected onto the NR topics (not used for fitting).
#   consensus NMF (cNMF-like): for each number of topics K, N_RUNS NMF fits on random 80 % subsamples of NR;
#          all components pooled, L2-normalised, clustered into K groups (k-means); consensus loading = median of
#          each group. Stability = cosine silhouette of the component clustering (1 = every run finds the same
#          programs); fit = relative reconstruction error of NR with the consensus loadings.
#          K_TOPICS = None -> largest K with stability >= MIN_STAB and no one-mark topics (MIN_MARKS, LOAD_CUT).
#   usage = per-enhancer topic weights (non-negative least squares on the consensus loadings), normalised to sum 1
#   A  topic loadings   B  mean topic usage by class x GR binding   C  choice of K
#   D  how mixed enhancers are: share of the dominant topic   E  dex change vs topic usage within NR (Spearman)
#   F  dominant topic vs robust NR state (if MS_nr_states_robust.tsv exists)
# Inputs : feature matrix; optional MS_nr_states_robust.tsv (an earlier table of NR states; panel F is left empty without it)
# Outputs: Fig_MS_nr_topics.pdf/png, MS_nr_topics_k_selection.tsv, MS_nr_topics_loadings.tsv,
#          MS_nr_topics_usage.tsv (all enhancers: topic weights, dominant topic)
# =============================================================================
# %%
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, non_negative_factorization
from sklearn.metrics import silhouette_score
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
from a549_enhancers.paths import PROJECT, ML, NR_TOPICS   # local folders, set in paths.toml
FM = ML / "feature_matrix_all_classes_GRannotated.csv"
OUT = NR_TOPICS; OUT.mkdir(parents=True, exist_ok=True)
K_RANGE, K_TOPICS, MIN_STAB = range(3, 10), None, 0.80
MIN_MARKS, LOAD_CUT = 2, 0.3   # a topic must combine >= MIN_MARKS features with loading >= LOAD_CUT (of its maximum);
                               # with few features, large K otherwise returns trivial one-mark "topics"
N_RUNS, FRAC, N_BINS, SEED = 60, 0.8, 20, 20261109

D = pd.read_csv(FM, low_memory=False)
D["GR_bound"] = D.GR_bound.astype(str).str.lower().eq("true")
col = lambda p, t: [c for c in D.columns if c.startswith(p) and t in c][0]
SF = {"ATAC": "ATAC_ctrl", "H3K27ac": "H3K27ac_ENCFF054OOT", "H3K4me1": "H3K4me1_ENCFF580SCK",
      "H3K4me3": "H3K4me3_ENCFF529CES", "Pol II": col("POLR2A_", "_eth_"), "EP300": "EP300_ENCFF376VNJ",
      "cJUN": "JUN_ENCFF957TUK", "CTCF": "CTCF_ENCFF083AEY", "RAD21": "RAD21_ENCFF778YHU", "H2A.Z": col("H2AFZ_", "_eth_"),
      "H3K27me3": col("H3K27me3_", "_eth_"), "GR (dex)": col("NR3C1_", "_dex_"), "FOXA1": col("FOXA1_", "_eth_")}
nr = D["class"].eq("NR").values
X = np.zeros((len(D), len(SF)))
for j, c in enumerate(SF.values()):                       # percentile within the NR distribution, for every enhancer
    ref = np.sort(D.loc[nr, c].values)
    X[:, j] = (np.searchsorted(ref, D[c].values, side="left") + np.searchsorted(ref, D[c].values, side="right")) / (2 * len(ref))
X = np.clip(X, 1e-6, 1)
Xn = X[nr]
rng = np.random.default_rng(SEED)


def consensus(k):
    comps = []
    for r in range(N_RUNS):
        idx = rng.choice(len(Xn), int(FRAC * len(Xn)), replace=False)
        m = NMF(k, init="random", random_state=int(rng.integers(1e9)), max_iter=600, tol=1e-4).fit(Xn[idx])
        comps.append(m.components_)
    C = np.vstack(comps); C = C / np.linalg.norm(C, axis=1, keepdims=True)
    km = KMeans(k, n_init=20, random_state=SEED).fit(C)
    stab = silhouette_score(C, km.labels_, metric="cosine")
    H = np.vstack([np.median(C[km.labels_ == i], axis=0) for i in range(k)])
    W, _, _ = non_negative_factorization(Xn, H=H, n_components=k, update_H=False, init="custom" if False else None,
                                         max_iter=600, random_state=SEED)
    err = np.linalg.norm(Xn - W @ H) / np.linalg.norm(Xn)
    return stab, err, H


SEL, HS = [], {}
for k in K_RANGE:
    s, e, H = consensus(k); HS[k] = H
    single = int(((H / H.max(axis=1, keepdims=True)) >= LOAD_CUT).sum(1).__lt__(MIN_MARKS).sum())
    SEL.append({"K": k, "stability_silhouette": s, "relative_error": e, "one_mark_topics": single})
    print(f"K = {k}: stability {s:.3f}, relative error {e:.3f}, one-mark topics {single}")
SEL = pd.DataFrame(SEL); SEL.to_csv(OUT / "MS_nr_topics_k_selection.tsv", sep="\t", index=False)
ok = SEL[(SEL.stability_silhouette >= MIN_STAB) & (SEL.one_mark_topics == 0)]
K = K_TOPICS or int(ok.K.max() if len(ok) else SEL.loc[SEL.stability_silhouette.idxmax(), "K"])
H = HS[K]
W, _, _ = non_negative_factorization(X, H=H, n_components=K, update_H=False, max_iter=800, random_state=SEED)
U = W / np.clip(W.sum(1, keepdims=True), 1e-12, None)                 # topic usage, sums to 1 per enhancer

# names from loadings (two strongest features relative to their mean loading)
L = pd.DataFrame(H / H.max(axis=1, keepdims=True), columns=list(SF))
spec = L / L.mean(axis=0)                                             # feature specificity across topics
names = []
for i in range(K):
    top = spec.loc[i].sort_values(ascending=False).index[:2]
    names.append("/".join(top))
order = np.argsort(-U[nr].mean(0))                                    # most used topic first
TN = [f"T{r + 1} {names[i]}" for r, i in enumerate(order)]
L = L.iloc[order].set_axis(TN); U = U[:, order]
L.to_csv(OUT / "MS_nr_topics_loadings.tsv", sep="\t")
USE = pd.DataFrame(U, columns=TN); USE.insert(0, "enhancer_id", D.enhancer_id.values)
USE.insert(1, "class", D["class"].values); USE.insert(2, "GR_bound", D.GR_bound.values)
USE["dominant_topic"] = np.array(TN)[U.argmax(1)]; USE["dominant_share"] = U.max(1)
USE.to_csv(OUT / "MS_nr_topics_usage.tsv", sep="\t", index=False)
print(f"\nchosen K = {K}\n", L.round(2).to_string())

# dex change (level-adjusted rank change; GR = dex occupancy rank)
ALL_R = lambda c: 100 * D[c].rank(pct=True)
CHG = {"GR (dex rank)": ALL_R(col("NR3C1_", "_dex_")).values}
for lab_, (c0, c1) in {"EP300": ("EP300_ENCFF376VNJ", col("EP300_", "_dex_")), "H3K27ac": ("H3K27ac_ENCFF054OOT", col("H3K27ac_", "_dex_")),
                       "Pol II": (col("POLR2A_", "_eth_"), col("POLR2A_", "_dex_")), "cohesin (SMC3)": ("SMC3_ENCFF029NYA", col("SMC3_", "_dex_"))}.items():
    r0, r1 = ALL_R(c0), ALL_R(c1)
    b = np.clip((r0 / 100 * N_BINS).astype(int), 0, N_BINS - 1)
    CHG[lab_] = (r1 - b.map(r1[nr].groupby(b[nr]).median())).values
RHO = pd.DataFrame({f: [spearmanr(U[nr, t], CHG[f][nr])[0] for t in range(K)] for f in CHG}, index=TN)

# ----------------------------------------------------------------------------- figure
FONT = "Open Sans" if any("Open Sans" in f.name for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
mpl.rcParams.update({
    "font.family": [FONT, "DejaVu Sans"], "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 6.8,
    "xtick.labelsize": 6.2, "ytick.labelsize": 6.2, "legend.fontsize": 5.8, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none",
})
TCOL = dict(zip(TN, plt.get_cmap("tab10").colors))
CLS = {"NR": "#5DAA82", "UP": "#D1495B", "DN": "#3B6FB6"}
GROUPS = [(c, g) for c in ("NR", "UP", "DN") for g in (False, True)]
st_path = next((p for p in (OUT / "MS_nr_states_robust.tsv", *PROJECT.rglob("MS_nr_states_robust.tsv")) if p.exists()), None)

fig = plt.figure(figsize=(7.2, 6.2))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.7, left=0.16, right=0.98, top=0.9, bottom=0.12)
r1 = gs[0].subgridspec(1, 2, width_ratios=[2.4, 1.3], wspace=0.3)
r2 = gs[1].subgridspec(1, 4, width_ratios=[0.8, 0.85, 1.35, 1.25], wspace=1.25)

ax = fig.add_subplot(r2[0])
ax.plot(SEL.K, SEL.stability_silhouette, "-o", ms=3, color="#333333", label="stability")
ax2 = ax.twinx(); ax2.plot(SEL.K, SEL.relative_error, "--s", ms=3, color="#B07AA1", label="error")
ax2.spines["right"].set_visible(True); ax2.tick_params(labelsize=5.6, colors="#B07AA1")
ax.axhline(MIN_STAB, ls=":", lw=0.6, color="#999999"); ax.axvline(K, color="#D1495B", lw=0.8)
ax.set_xlabel("topics (K)"); ax.set_ylabel("stability"); ax.set_ylim(0, 1.02); ax.set_xticks(list(K_RANGE))
ax.tick_params(axis="x", labelsize=5.4)
ax2.set_ylabel("error", color="#B07AA1", fontsize=6.2)
ax.set_title("Choice of K", fontsize=6.8); axA = ax

ax = fig.add_subplot(r1[0]); axB = ax
im = ax.imshow(L.values, cmap="Greys", vmin=0, vmax=1, aspect="auto")
for i in range(K):
    for j in range(L.shape[1]):
        v = L.values[i, j]
        if v >= 0.25:
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=4.8, color="white" if v > 0.6 else "black")
ax.set_xticks(range(L.shape[1])); ax.set_xticklabels(L.columns, rotation=55, ha="right", fontsize=5.8)
ax.set_yticks(range(K)); ax.set_yticklabels(TN, fontsize=5.8)
for t, n in zip(ax.get_yticklabels(), TN):
    t.set_color(TCOL[n])
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("Topic loadings (scaled to the topic maximum)", fontsize=6.8)

ax = fig.add_subplot(r1[1]); axC = ax
bottom = np.zeros(len(GROUPS))
for t in range(K):
    v = np.array([U[(D["class"].values == c) & (D.GR_bound.values == g), t].mean() for c, g in GROUPS])
    ax.bar(range(len(GROUPS)), v, bottom=bottom, color=TCOL[TN[t]], width=0.7, lw=0)
    bottom += v
ax.set_xticks(range(len(GROUPS))); ax.set_xticklabels([f"{c}\nGR{'+' if g else '−'}" for c, g in GROUPS], fontsize=5.8)
for tl, (c, _) in zip(ax.get_xticklabels(), GROUPS):
    tl.set_color(CLS[c])
ax.set_ylim(0, 1); ax.set_ylabel("mean topic usage")
ax.set_title("Topic composition by class\n(UP / DN projected onto NR topics)", fontsize=6.8)

ax = fig.add_subplot(r2[1]); axD = ax
bins = np.linspace(0, 1, 21)
for c in ("NR", "UP", "DN"):
    v = USE.loc[USE["class"] == c, "dominant_share"]
    ax.hist(v, bins=bins, density=True, histtype="step", lw=1.1, color=CLS[c], label=f"{c} (median {v.median():.2f})")
ax.set_xlabel("share of the dominant topic"); ax.set_ylabel("density")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.26), handlelength=1.0, fontsize=5.4)
ax.set_title("How mixed\nenhancers are", fontsize=6.8)

ax = fig.add_subplot(r2[2]); axE = ax
ax.imshow(RHO.values, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
for i in range(K):
    for j in range(RHO.shape[1]):
        v = RHO.values[i, j]
        ax.text(j, i, f"{v:+.1f}".replace("0.", "."), ha="center", va="center", fontsize=4.8,
                color="white" if abs(v) > 0.33 else "black")
ax.set_xticks(range(RHO.shape[1])); ax.set_xticklabels(RHO.columns, rotation=40, ha="right", fontsize=5.8)
ax.set_yticks(range(K)); ax.set_yticklabels([n.split(" ")[0] for n in TN], fontsize=5.8)
for t, n in zip(ax.get_yticklabels(), TN):
    t.set_color(TCOL[n]); t.set_fontweight("bold")
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("NR: topic usage vs\ndex change (Spearman)", fontsize=6.8)

ax = fig.add_subplot(r2[3]); axF = ax
if st_path is not None:
    st = pd.read_csv(st_path, sep="\t").set_index("enhancer_id").state_robust
    m = USE[USE["class"] == "NR"].assign(state=lambda d: d.enhancer_id.map(st))
    ct = pd.crosstab(m.state, m.dominant_topic, normalize="index")[TN] * 100
    ax.imshow(ct.values, cmap="Greys", vmin=0, vmax=100, aspect="auto")
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            v = ct.values[i, j]
            if v >= 8:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=4.8, color="white" if v > 55 else "black")
    ax.set_xticks(range(K)); ax.set_xticklabels([n.split(" ")[0] for n in TN], fontsize=5.8)
    for t, n in zip(ax.get_xticklabels(), TN):
        t.set_color(TCOL[n]); t.set_fontweight("bold")
    ax.set_yticks(range(len(ct))); ax.set_yticklabels(ct.index, fontsize=5.0); ax.tick_params(length=0)
    ax.set_title("Robust NR state vs\ndominant topic (% of state)", fontsize=6.8)
else:
    ax.axis("off"); ax.text(0.5, 0.5, "MS_nr_states_robust.tsv not found\n(panel not drawn)", ha="center", fontsize=6)
fig.canvas.draw(); rr = fig.canvas.get_renderer()
for row in (((axB, "A"), (axC, "B")), ((axA, "C"), (axD, "D"), (axE, "E"), (axF, "F"))):
    ytop = max(a.get_tightbbox(rr).transformed(fig.transFigure.inverted()).y1 for a, _ in row)
    for a, l in row:
        bb = a.get_tightbbox(rr).transformed(fig.transFigure.inverted())
        fig.text(bb.x0 - 0.005, ytop + 0.008, l, fontsize=11, fontweight="bold")
fig.text(0.01, 0.985, f"NR chromatin topics (consensus NMF, K = {K}; {N_RUNS} runs per K; n = {nr.sum():,} NR enhancers)",
         fontsize=7.6, fontweight="bold")
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"Fig_MS_nr_topics.{ext}", dpi=300, bbox_inches="tight")
plt.show()
print(RHO.round(2).to_string())
print("Saved", OUT / "Fig_MS_nr_topics.pdf")
