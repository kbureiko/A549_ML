# =============================================================================
# Supplementary Figure S4 - UMAP feature maps I (factors central to the GR response and enhancer activity)
#   Layout LAYOUT (same map as Figure 1 F-H). Panels: GR (dex), GRE motif, ATAC EtOH/dex; Pol II Δ, H3K27ac Δ,
#   PHF8 (untreated); H3K4me1, H3K4me3, H3K79me2 EtOH / dex / Δ.
#   occupancy panels: percentile rank among all enhancers within each experiment (EtOH, dex or untreated)
#   Δ panels: level-adjusted rank change = dex rank minus the dex rank of NR enhancers with a similar control rank
#       (20 bins), median over each enhancer's N_SMOOTH nearest neighbours in PC space. Not log2(dex/EtOH):
#       control and dex ENCODE tracks are separate experiments. Spearman(control, dex) is printed and saved; GR is
#       expected to be low (binding is induced by dex).
#   motif panels: HOMER motif call present / absent
# Rows = lists of factor groups; one panel letter per group.
# Inputs (FEATURE_MAP): embeddings file + umap_pcs.tsv; feature matrix
# Outputs (FIGURES): Fig_S4_umap_features.pdf/png, Fig_S4_umap_features_rho.tsv
# =============================================================================
# %%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.neighbors import NearestNeighbors

from a549_enhancers.paths import PROJECT, ML, FIGURES, FEATURE_MAP   # local folders, set in paths.toml
FM = ML / "feature_matrix_all_classes_GRannotated.csv"
PREP = FEATURE_MAP
FINAL = FIGURES; FINAL.mkdir(parents=True, exist_ok=True)
EMB_FILE, LAYOUT = "umap_grid_embeddings_supervised.npz", "nn30_md0.1_cosine_sup0.5"
N_SMOOTH, N_BINS, CHANGE_LIM, MIN_RHO, ALPHA = 25, 20, 30, 0.6, 0.6
NCOL, PANEL_W, PANEL_H = 4, 1.95, 1.5          # panels per row and panel size (inches)
EQUAL_ASPECT = False

# factor -> (control column, dex column); "name|tag" = first column starting with name_ and containing tag
TRACKS = {"GR": ("NR3C1|_eth_", "NR3C1|_dex_"), "H3K79me2": ("H3K79me2|_eth_", "H3K79me2|_dex_"),
          "ATAC": ("ATAC_ctrl", "ATAC_dex_1h"), "Pol II": ("POLR2A|_eth_", "POLR2A|_dex_"),
          "H3K27ac": ("H3K27ac|_eth_", "H3K27ac|_dex_"), "H3K27me3": ("H3K27me3|_eth_", "H3K27me3|_dex_"),
          "H3K4me1": ("H3K4me1|_eth_", "H3K4me1|_dex_"), "H3K4me3": ("H3K4me3|_eth_", "H3K4me3|_dex_"),
          "CTCF": ("CTCF_ENCFF083AEY", "CTCF|_dex_"), "H2A.Z": ("H2AFZ|_eth_", "H2AFZ|_dex_"),
          "FOSL2": ("FOSL2|_eth_", "FOSL2|_dex_"), "EP300": ("EP300_ENCFF842XYY", "EP300|_dex_"),
          "cJUN": ("JUN_ENCFF957TUK", "JUN|_dex_"), "BCL3": ("BCL3_ENCFF856OZJ", "BCL3|_dex_"),
          "SMC3": ("SMC3_ENCFF029NYA", "SMC3|_dex_"), "RAD21": ("RAD21_ENCFF778YHU", "RAD21|_dex_")}
SINGLE = {"PHF8": "PHF8_ENCFF690AJW", "TCF12": "TCF12|_eth_", "GATA3": "GATA3_ENCFF485KDU", "TEAD4": "TEAD4_ENCFF186WSI"}
MOTIFS = {"GRE motif": ["motif_binary::GRE(NR),IR3/A549-GR-ChIP-Seq(GSE32465)/Homer",
                        "motif_binary::GRE(NR),IR3/RAW264.7-GRE-ChIP-Seq(Unpublished)/Homer"],
          "AP-1 motif": ["motif_binary::AP-1(bZIP)/ThioMac-PU.1-ChIP-Seq(GSE21512)/Homer"]}

# panel specs: ("motif", name) | ("ctrl"/"dex", factor, cmap) | ("chg", factor) | ("single", factor, label, cmap)
M = lambda n: [("motif", n)]
TRIO = lambda f, cm: [("ctrl", f, cm), ("dex", f, cm), ("chg", f)]
DUO = lambda f, cm: [("ctrl", f, cm), ("dex", f, cm)]
FIGS = {"Fig_S4_umap_features": [  # rows of factor groups (one panel letter per group)
    [[("dex", "GR", "Purples")], M("GRE motif"), DUO("ATAC", "Reds")],
    [[("chg", "Pol II")], [("chg", "H3K27ac")], [("single", "PHF8", "PHF8 (untreated)", "Blues")]],
    [TRIO("H3K4me1", "Blues")],
    [TRIO("H3K4me3", "Blues")],
    [TRIO("H3K79me2", "Blues")],
]}
PCS = pd.read_csv(PREP / "umap_pcs.tsv", sep="\t")
E = np.load(PREP / EMB_FILE)[LAYOUT]
P = PCS.filter(regex=r"^PC\d+$").to_numpy()
D = pd.read_csv(FM, low_memory=False).set_index("enhancer_id").loc[PCS.enhancer_id].reset_index().copy()
nr = D["class"].eq("NR").values


def colname(spec):
    if spec in D.columns:
        return spec
    pre, tag = spec.split("|")
    hit = [c for c in D.columns if c.startswith(pre + "_") and tag in c]
    if not hit:
        raise KeyError(f"no column for {spec}")
    return hit[0]


rank = lambda c: 100 * D[c].rank(pct=True).to_numpy()
nbr = NearestNeighbors(n_neighbors=N_SMOOTH).fit(P).kneighbors(P, return_distance=False)
CACHE, RHO = {}, []


def change(f):
    if f not in CACHE:
        c0, c1 = map(colname, TRACKS[f])
        r0, r1 = rank(c0), rank(c1)
        rho = pd.Series(r0).corr(pd.Series(r1))
        RHO.append({"feature": f, "control": c0, "dex": c1, "spearman": rho})
        if rho < MIN_RHO and f != "GR":                  # GR: low agreement expected (binding is dex-induced)
            print(f"WARNING {f}: control vs dex rho = {rho:.2f} < {MIN_RHO}; change panel not reliable")
        b = np.clip((r0 / 100 * N_BINS).astype(int), 0, N_BINS - 1)
        arc = r1 - pd.Series(b).map(pd.Series(r1[nr]).groupby(b[nr]).median()).to_numpy()
        CACHE[f] = np.median(arc[nbr], axis=1)
    return CACHE[f]


def values(spec):
    kind = spec[0]
    if kind == "motif":
        cols = [c for c in MOTIFS[spec[1]] if c in D.columns]
        v = (D[cols].fillna(0) > 0).any(axis=1).to_numpy()
        return f"{spec[1]} (HOMER)\n{v.sum():,} enhancers", v, "motif", None
    if kind == "chg":
        return f"{spec[1]} Δ (dex change)", change(spec[1]), "chg", "RdBu_r"
    if kind in ("ctrl", "dex"):
        c = colname(TRACKS[spec[1]][0 if kind == "ctrl" else 1])
        lab = "dex" if kind == "dex" else ("EtOH" if ("_eth_" in c or c == "ATAC_ctrl") else "untreated")
        return f"{spec[1]} {lab}", rank(c), "occ", spec[2]
    if kind == "single":
        return spec[2], rank(colname(SINGLE[spec[1]])), "occ", spec[3]
    raise ValueError(spec)


FONT = "Open Sans" if any("Open Sans" in f.name for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
mpl.rcParams.update({"font.family": [FONT, "DejaVu Sans"], "font.size": 7, "axes.titlesize": 7,
                     "pdf.fonttype": 42, "svg.fonttype": "none",
                     "axes.grid": False})


def draw(rows, fname):
    nrow = len(rows); top_m, bot_m = 0.25, 0.75
    H = nrow * PANEL_H + top_m + bot_m
    fig = plt.figure(figsize=(NCOL * PANEL_W, H))
    gs = fig.add_gridspec(nrow, NCOL, left=0.035, right=0.995, top=1 - top_m / H, bottom=bot_m / H,
                          wspace=0.08, hspace=0.42)
    used, letter = {}, ord("A")
    for r, row in enumerate(rows):
        col = 0
        for group in row:
            for j, spec in enumerate(group):
                ax = fig.add_subplot(gs[r, col]); col += 1
                title, v, kind, cmap = values(spec)
                if kind == "motif":
                    ax.scatter(E[~v, 0], E[~v, 1], s=0.8, c="#DDDDDD", lw=0, rasterized=True)
                    ax.scatter(E[v, 0], E[v, 1], s=1.4, c="#2E7D5B", lw=0, alpha=0.6, rasterized=True)
                else:
                    lo, hi = (-CHANGE_LIM, CHANGE_LIM) if kind == "chg" else (0, 100)
                    o = np.argsort(np.abs(v - (lo + hi) / 2)) if kind == "chg" else np.argsort(v)  # strongest on top
                    sc = ax.scatter(E[o, 0], E[o, 1], s=0.9, c=v[o], cmap=cmap, vmin=lo, vmax=hi, lw=0, alpha=ALPHA,
                                    rasterized=True)
                    used.setdefault((kind, cmap), sc)
                ax.set_title(title, pad=2)
                ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
                if j == 0 and col - 1 == 0:          # UMAP axis key on the first panel of each row
                    for dx, dy, lab, kw in ((0.18, 0, "UMAP1", dict(ha="left", va="top")),
                                            (0, 0.18, "UMAP2", dict(ha="right", va="bottom", rotation=90))):
                        ax.annotate("", xy=(0.02 + dx, 0.02 + dy), xytext=(0.02, 0.02), xycoords="axes fraction",
                                    arrowprops=dict(arrowstyle="-|>", lw=0.6, color="#333333", mutation_scale=5))
                        ax.text(0.02 + (0.0 if dx else -0.012), 0.02 + (-0.015 if dx else 0.0), lab,
                                transform=ax.transAxes, fontsize=5.2, color="#333333", **kw)
                ax.set_aspect("equal" if EQUAL_ASPECT else "auto", adjustable="box")
                for s_ in ax.spines.values():
                    s_.set_visible(False)
                if j == 0:
                    ax.text(-0.04, 1.13, chr(letter), transform=ax.transAxes, fontsize=11, fontweight="bold",
                            ha="left", va="bottom")
                    letter += 1
    # shared colour bars at the bottom, one per colour scale used
    labels = {("chg", "RdBu_r"): "Δ: dex change vs NR at\nsimilar baseline (rank points)",
              ("occ", "Reds"): "ATAC\n(percentile rank)", ("occ", "Blues"): "ChIP-seq signal\n(percentile rank)",
              ("occ", "Purples"): "GR occupancy, dex\n(percentile rank)"}
    n = len(used); w = min(0.26, 0.9 / max(n, 1) - 0.04); x0 = 0.5 - (n * (w + 0.04) - 0.04) / 2
    for k, (key, sc) in enumerate(used.items()):
        cax = fig.add_axes([x0 + k * (w + 0.04), 0.32 / H, w, 0.06 / H])
        cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
        cb.ax.tick_params(labelsize=5.6, length=1.5, pad=1); cb.outline.set_linewidth(0.4)
        cb.set_label(labels.get(key, key[1]), fontsize=5.8, labelpad=1)
    for ext in ("pdf", "png"):
        fig.savefig(FINAL / f"{fname}.{ext}", dpi=300, bbox_inches="tight")
    plt.show()
    print("Saved", FINAL / f"{fname}.pdf")


for name, rows in FIGS.items():
    draw(rows, name)
R = pd.DataFrame(RHO); R.to_csv(FINAL / "Fig_S4_umap_features_rho.tsv", sep="\t", index=False)
print(R[["feature", "spearman"]].round(2).to_string(index=False))
