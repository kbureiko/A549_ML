# =============================================================================
# Supplementary Figure S2 - correlation between the top features of the three pairwise models
#   Features: union of the TOP_N features (mean held-out permutation importance over the outer-CV folds) of each
#   model (NR vs UP, UP vs DN, NR vs DN), the same selection as the UMAP feature set. Non-numeric features
#   (genomic annotation) are left out.
#   Clustermap: Spearman correlation over all 8,270 enhancers; rows/columns clustered by average linkage on
#   1 - rho. Colour bar and label colour: feature layer (dex-treated / control-untreated / sequence-genomic); the
#   condition is not written in the labels (same factor can appear twice: control and dex). Control TEAD4 (EXTRA)
#   is added to the feature set. White lines and names on the left: clusters of correlated features, defined by
#   one rule: the dendrogram is cut at 1 - rho = CUT (average linkage), and every cluster with at least two
#   features is a block. Names are attached by exact membership (BLOCK_NAMES); a cluster without a predefined
#   name is labelled with its members, so a name can never describe other features.
# Inputs: FM; RUNS/<run>/summaries/feature_importance_stability_summary.csv
# Outputs (FIGURES): Fig_S2_feature_correlation.pdf/png, Fig_S2_feature_correlation.tsv (rho matrix in
#   plot order), Fig_S2_features.tsv (feature, label, layer, rank per model), Fig_S2_blocks.tsv (cluster, members,
#   mean and minimum within-cluster rho)
# =============================================================================
# %%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

from a549_enhancers.paths import PROJECT, ML, FIGURES, RUNS as RUN_DIRS   # local folders, set in paths.toml
FM = ML / "feature_matrix_all_classes_GRannotated.csv"
RUNS = {"NR vs UP": RUN_DIRS["nr_vs_up"],
        "UP vs DN": RUN_DIRS["up_vs_dn"],
        "NR vs DN": RUN_DIRS["nr_vs_dn"]}
FINAL = FIGURES; FINAL.mkdir(parents=True, exist_ok=True)
TOP_N = 30
EXTRA = [""]
CUT = 0.5                                                # clusters: dendrogram cut at 1 - rho = CUT

FONT = "Open Sans" if any("Open Sans" in f.name for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
mpl.rcParams.update({"font.family": [FONT, "DejaVu Sans"], "font.size": 7, "pdf.fonttype": 42, "svg.fonttype": "none"})
LAYER_COL = {"Dex": "#2B4C7E", "Control": "#9DB9D5", "Genomic": "#3E9651"}
LAYER_LAB = {"Dex": "After dex (ChIP, ATAC, GR peaks nearby)", "Control": "Control / untreated", "Genomic": "Sequence / genomic"}
TASK_COL = {"NR vs UP": "#D1495B", "UP vs DN": "#7A5195", "NR vs DN": "#3B6FB6"}
RENAME = {"JUN": "cJUN", "NR3C1": "GR", "H2AFZ": "H2A.Z", "POLR2A": "Pol II", "POLR2AphosphoS2": "Pol II pS2",
          "Enh_len": "Enhancer length", "gc": "GC content", "GRE_score": "GRE score",
          "nbr_enhancers_50kb": "Enhancers ≤50 kb", "nbr_enhancer_distance": "Nearest enhancer",
          "nbr_GRpeaks_50kb": "GR peaks ≤50 kb", "nbr_strongGRpeaks_50kb": "Strong GR peaks ≤50 kb",
          "nbr_GRpeak_distance": "Nearest GR peak", "nbr_strongGRpeak_distance": "Nearest strong GR peak"}


def layer(f):
    if f.startswith("motif_binary::") or f in ("Enh_len", "Distance to TSS", "annotation", "gc", "GRE_score"):
        return "Genomic"
    if "_dex_" in f or f == "ATAC_dex_1h" or f.startswith(("nbr_GRpeak", "nbr_strongGRpeak")):
        return "Dex"
    return "Control"


def label(f):
    if f.startswith("motif_binary::"):
        return f.split("::")[1].split("/")[0].split("(")[0] + " motif"
    if f in RENAME:
        return RENAME[f]
    if f in ("Distance to TSS",):
        return f
    g = "ATAC" if f.startswith("ATAC") else f.split("_")[0]
    g = RENAME.get(g, g)
    cond = "dex" if ("_dex_" in f or f == "ATAC_dex_1h") else ("EtOH" if ("_eth_" in f or f == "ATAC_ctrl") else "untr.")
    return f"{g} ({cond})"


# ----------------------------------------------------------------- features
rows = []
for t, r in RUNS.items():
    s = pd.read_csv(r / "summaries" / "feature_importance_stability_summary.csv")
    s = s.sort_values("mean_importance", ascending=False).reset_index(drop=True)
    rows += [{"feature": f, "task": t, "rank": i + 1} for i, f in enumerate(s.feature.head(TOP_N))]
U = pd.DataFrame(rows)
U = pd.concat([U, pd.DataFrame({"feature": [f for f in EXTRA if f not in set(U.feature)], "task": list(RUNS)[0], "rank": np.nan})], ignore_index=True)
F = U.pivot_table(index="feature", columns="task", values="rank", dropna=False).reindex(columns=list(RUNS)).reset_index()
D = pd.read_csv(FM, low_memory=False)
F = F[F.feature.isin(D.columns)]
F = F[F.feature.map(lambda f: pd.api.types.is_numeric_dtype(D[f]))].reset_index(drop=True)
F["layer"] = F.feature.map(layer)
F["label"] = F.feature.map(label)
dup = F.label.duplicated(keep=False)                     # same factor + condition from two experiments
F.loc[dup, "label"] = F.loc[dup].apply(lambda r: f"{r.label} {r.feature.split('_ENCFF')[-1][:6] if '_ENCFF' in r.feature else ''}".strip(), axis=1)
print(f"union of top-{TOP_N}: {len(F)} numeric features; layers {F.layer.value_counts().to_dict()}")

# ----------------------------------------------------------------- correlation + clustering
R = D[F.feature].rank().corr()                           # Spearman = Pearson on ranks
R = R.fillna(0)
Z = linkage(squareform(1 - R.to_numpy(), checks=False), method="average")

n = len(F)
order = dendrogram(Z, no_plot=True)["leaves"]
show = [lab.rsplit(" (", 1)[0] if lab.endswith(("(dex)", "(EtOH)", "(untr.)")) else lab for lab in
        F.feature.map(label).values]                     # displayed labels, condition removed
key = F.label.values                                     # internal labels (with condition)
# ----------------------------------------------------------------- clusters of correlated features
# Rule: cut the dendrogram at 1 - rho = CUT; every cluster with >= 2 features is drawn (white lines) and named.
# BLOCK_NAMES gives a name for an exact set of members (internal labels, with condition). The name is used only
# if a cluster has exactly these members; any other cluster is labelled with its member names. Names list what
# is in the cluster and make no functional claim.
BLOCK_NAMES = [("H3K36me3, H3K79me2", ["H3K36me3 (EtOH)", "H3K79me2 (dex)", "H3K79me2 (EtOH)"]),
               ("Zic2, Zic3, ESC-element motifs", ["Zic2 motif", "Unknown-ESC-element motif", "Zic3 motif"]),
               ("H2A.Z", ["H2A.Z (dex)", "H2A.Z (EtOH)"]),
               ("GR, EP300, HES2 (dex)", ["HES2 (dex)", "EP300 (dex)", "GR (dex)"]),
               ("CTCF, RAD21, SMC3", ["CTCF (untr.)", "RAD21 (dex)", "SMC3 (dex)"]),
               ("Pol II, TAF1", ["TAF1 (EtOH)", "Pol II (dex)", "Pol II (EtOH)"]),
               ("H3K27ac, H3K9ac, H3K4me2/3",
                ["H3K27ac (dex)", "H3K27ac (EtOH)", "H3K9ac (EtOH)", "H3K4me2 (dex)", "H3K4me3 (dex)"]),
               ("ATAC, MYC", ["MYC (untr.)", "ATAC (EtOH)", "ATAC (dex)"]),
               ("TF and coregulator occupancy\n(mainly control)",
                ["FOXA2 (EtOH)", "USF1 (EtOH)", "ELF1 (EtOH)", "ZBTB33 (EtOH)", "REST (EtOH)", "SIX5 (EtOH)",
                 "BCL3 (dex)", "cJUN (dex)", "BCL3 (untr.)", "cJUN (untr.)", "PBX3 (untr.)", "JUND (EtOH)",
                 "EP300 (untr.)", "TEAD4 (untr.)", "GATA3 (untr.)", "SP1 (EtOH)", "TCF12 (EtOH)"])]
name_of = {frozenset(m): nm for nm, m in BLOCK_NAMES}
row_of = {k: r for r, k in enumerate(order)}             # feature index -> row in the plot
cl = fcluster(Z, CUT, criterion="distance")
Rv = R.to_numpy()
blocks, block_rows, used = [], [], set()
for c in np.unique(cl):
    ii = sorted(np.where(cl == c)[0], key=lambda i: row_of[i])
    if len(ii) < 2:
        continue
    p = [row_of[i] for i in ii]
    assert p == list(range(p[0], p[0] + len(p))), "a cluster from a dendrogram cut must be contiguous"
    members = [key[i] for i in ii]
    name = name_of.get(frozenset(members))
    if name is None:                                     # no predefined name for exactly these members
        u = list(dict.fromkeys(show[i] for i in ii))
        name = ", ".join(u) if len(u) <= 4 else ", ".join(u[:3]) + f" + {len(u) - 3} more"
        print(f"NOTE cluster without a predefined name, labelled with its members: {name}")
    else:
        used.add(frozenset(members))
    sub = Rv[np.ix_(ii, ii)][np.triu_indices(len(ii), 1)]
    blocks.append((p[0], p[-1] + 1, name))
    block_rows.append({"first_row": p[0], "cluster": name.replace("\n", " "), "n_features": len(ii),
                       "mean_rho": round(float(sub.mean()), 2), "min_rho": round(float(sub.min()), 2),
                       "features": ", ".join(members)})
for nm, m in BLOCK_NAMES:
    if frozenset(m) not in used:
        print("NOTE predefined name not used (no cluster with exactly these members):", nm.replace("\n", " "))
blocks.sort()
lines = sorted({x for a_, b_, _ in blocks for x in (a_, b_) if 0 < x < n})
BL = pd.DataFrame(block_rows, columns=["first_row", "cluster", "n_features", "mean_rho", "min_rho", "features"])
BL = BL.sort_values("first_row").drop(columns="first_row").reset_index(drop=True)      # in plot order
print(f"cut at 1 - rho = {CUT}: {len(BL)} clusters with >= 2 features, {int(BL.n_features.sum())} of {n} features")
print(BL[["cluster", "n_features", "mean_rho", "min_rho"]].to_string(index=False))

R.index = R.columns = range(n)
side = pd.DataFrame({"Layer": F.layer.map(LAYER_COL).values}, index=range(n))
sz = max(6.5, min(10, 0.115 * n + 1.8))
g = sns.clustermap(R, row_linkage=Z, col_linkage=Z, cmap="RdBu_r", vmin=-1, vmax=1, center=0,
                   row_colors=side, colors_ratio=0.012, dendrogram_ratio=(0.17, 0.07), linewidths=0,
                   xticklabels=True, yticklabels=True, figsize=(sz + 1.0, sz),
                   cbar_kws={"label": "Spearman ρ", "orientation": "horizontal"}, cbar_pos=(0.03, 0.915, 0.12, 0.012))
g.gs.update(top=0.88)
g.ax_cbar.set_position([0.03, 0.915, 0.12, 0.012])
ax = g.ax_heatmap
fs = max(4.2, min(6.2, 360 / n))
lay = F.layer.values
ax.set_yticks(np.arange(n) + 0.5); ax.set_yticklabels([show[k] for k in order], fontsize=fs)
ax.set_xticks(np.arange(n) + 0.5); ax.set_xticklabels([show[k] for k in order], fontsize=fs, rotation=90)
ax.tick_params(axis="both", length=0)
txtc = {"Dex": "#2B4C7E", "Control": "#333333", "Genomic": "#2F7D3F"}
for tls in (ax.get_yticklabels(), ax.get_xticklabels()):
    for tl, k in zip(tls, order):
        tl.set_color(txtc[lay[k]])
g.ax_row_colors.set_xticks([])
for b in lines:
    ax.axhline(b, color="white", lw=1.6); ax.axvline(b, color="white", lw=1.6)
    g.ax_row_colors.axhline(b, color="white", lw=1.6)
# cluster names where the row dendrogram was
g.ax_row_dendrogram.clear(); g.ax_row_dendrogram.axis("off")
g.ax_row_dendrogram.set_ylim(ax.get_ylim()); g.ax_row_dendrogram.set_xlim(0, 1)
for a_, b_, name in blocks:
    g.ax_row_dendrogram.plot([0.97, 0.97], [a_ + 0.25, b_ - 0.25], color="#555555", lw=0.8)
    g.ax_row_dendrogram.text(0.93, (a_ + b_) / 2, name, ha="right", va="center", fontsize=6.0,
                             multialignment="right", linespacing=1.15)
g.ax_cbar.tick_params(labelsize=5.6); g.ax_cbar.xaxis.label.set_size(6.2)
h = [mpl.patches.Patch(color=LAYER_COL[k], label=LAYER_LAB[k]) for k in LAYER_COL]
g.fig.legend(handles=h, loc="center left", bbox_to_anchor=(0.22, 0.921), ncol=3, frameon=False, fontsize=6.2,
             handlelength=1.0, columnspacing=1.0, title="Feature layer (bar and label colour)", title_fontsize=6.2)
g.fig.suptitle(f"Spearman correlation of {n} top-ranked model features (all {len(D):,} enhancers)", y=0.97, fontsize=8)
R.index = R.columns = key

for ext in ("pdf", "png"):
    g.savefig(FINAL / f"Fig_S2_feature_correlation.{ext}", dpi=300, bbox_inches="tight")
R.iloc[order, order].round(3).to_csv(FINAL / "Fig_S2_feature_correlation.tsv", sep="\t")
F[["feature", "label", "layer"] + list(RUNS)].rename(columns={t: f"rank_{t}" for t in RUNS}).to_csv(
    FINAL / "Fig_S2_features.tsv", sep="\t", index=False)
BL.to_csv(FINAL / "Fig_S2_blocks.tsv", sep="\t", index=False)
plt.show()
# highly correlated pairs, for the legend / text
iu = np.triu_indices(n, 1)
pairs = pd.DataFrame({"a": R.index.values[iu[0]], "b": R.columns.values[iu[1]], "rho": R.to_numpy()[iu]})
print("pairs with |rho| >= 0.8:", (pairs.rho.abs() >= 0.8).sum(), "of", len(pairs))
print(pairs.reindex(pairs.rho.abs().sort_values(ascending=False).index).head(12).round(2).to_string(index=False))
print("Saved", FINAL / "Fig_S2_feature_correlation.pdf")
