# =============================================================================
# Feature map, step 2: UMAP layouts on the principal components of step 1 (scripts/09_feature_map_prep.py)
#   The figures use the layout nn30_md0.1_cosine_sup0.5 of MODE = "supervised" (n_neighbors 30, min_dist 0.1,
#   cosine distance, target_weight 0.5, seed SEED), stored in umap_grid_embeddings_supervised.npz.
#   MODE = "unsupervised": n_neighbors x min_dist x metric (+ one weakly supervised layout, target_weight 0.1)
#   MODE = "supervised":   target_weight x n_neighbors x min_dist x metric (euclidean, cosine) - class labels shape the layout,
#     so use it for visualisation only (clusters and change scores are computed in PC space and do not change);
#     cached separately (umap_grid_embeddings_supervised.npz) - the unsupervised results are kept
#   scores per layout (computed in a fixed random subsample, so layouts are comparable):
#     knn_kept      fraction of each enhancer's K_EVAL nearest neighbours in PC space that are also among its
#                   K_EVAL nearest neighbours on the map (local structure kept; higher = better)
#     trustworthiness  sklearn; penalises map neighbours that were far apart in PC space (higher = better)
#     nr_spread     NR spread on the map relative to all enhancers (check that NR is not squeezed into a blob)
#   NR k-means clusters (k in K_SHOW, union features, NR only) are computed once and coloured on every layout;
#   the existing A2 states are kept as a separate column (never overwritten).
# Inputs : FEATURE_MAP/umap_pcs.tsv (step 1), A2_enhancer_states.tsv (optional)
# Outputs (FEATURE_MAP): umap_grid_embeddings.npz (cached; delete to recompute), umap_grid_scores.tsv,
#   umap_nr_clusters.tsv, Fig_umap_grid_class.png, Fig_umap_grid_clusters_k<k>.png
# =============================================================================
# %%
from pathlib import Path
import itertools
import time
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors
import umap

from a549_enhancers.paths import PROJECT, FEATURE_MAP   # local folders, set in paths.toml
PREP = FEATURE_MAP
N_NEIGHBORS = (15, 30, 50)
MIN_DIST = (0.1, 0.3, 0.5)
METRICS = ("euclidean", "correlation")
SUPERVISED = dict(n_neighbors=30, min_dist=0.3, metric="euclidean", target_weight=0.1)   # comparison only
MODE = "supervised"          # "unsupervised" (grid above) or "supervised" (visualisation-only grid below)
SUP_TARGET_WEIGHT = (0.1, 0.3, 0.5)
SUP_N_NEIGHBORS = (15, 30, 50)
SUP_MIN_DIST = (0.1, 0.3)
SUP_METRICS = ("euclidean", "cosine")
K_SHOW = (4, 7)
K_EVAL, N_EVAL, SEED = 15, 3000, 20261109

PCS = pd.read_csv(PREP / "umap_pcs.tsv", sep="\t")
PCS["GR_bound"] = PCS.GR_bound.astype(str).str.lower().eq("true")
P = PCS.filter(regex=r"^PC\d+$").to_numpy()
cls = PCS["class"].to_numpy()
print(f"{len(PCS):,} enhancers x {P.shape[1]} PCs")

# ----------------------------------------------------------------------------- NR clusters (kept separate)
nr = cls == "NR"
CL = PCS[["enhancer_id", "class", "GR_bound"]].copy()
for k in K_SHOW:
    km = KMeans(k, n_init=20, random_state=SEED).fit(P[nr])
    lab = np.full(len(P), -1); lab[nr] = km.labels_
    CL[f"nr_kmeans_k{k}"] = lab                         # -1 = UP / DN (not clustered)
old = [p for p in (PROJECT / "figures" / "A2_enhancer_states.tsv", *PROJECT.rglob("A2_enhancer_states.tsv")) if p.exists()]
if old:
    st = pd.read_csv(old[0], sep="\t").set_index("enhancer_id").state
    CL["A2_state"] = CL.enhancer_id.map(st).where(nr)
    print("A2 states read from", old[0])
CL.to_csv(PREP / "umap_nr_clusters.tsv", sep="\t", index=False)

# ----------------------------------------------------------------------------- embeddings (cached)
if MODE == "supervised":     # class labels shape the layout -> for visualisation only; say so in the legend
    GRID = [dict(n_neighbors=n, min_dist=d, metric=m, target_weight=w)
            for w, n, d, m in itertools.product(SUP_TARGET_WEIGHT, SUP_N_NEIGHBORS, SUP_MIN_DIST, SUP_METRICS)]
else:
    GRID = [dict(n_neighbors=n, min_dist=d, metric=m) for n, d, m in itertools.product(N_NEIGHBORS, MIN_DIST, METRICS)]
    GRID.append(SUPERVISED)
key = lambda g: f"nn{g['n_neighbors']}_md{g['min_dist']}_{g['metric']}" + (
    f"_sup{g['target_weight']}" if "target_weight" in g else "")
SFX = "_supervised" if MODE == "supervised" else ""
cache = PREP / f"umap_grid_embeddings{SFX}.npz"
EMB = dict(np.load(cache)) if cache.exists() else {}
y_sup = pd.Categorical(cls).codes
t0 = time.time()
for i, g in enumerate(GRID):
    if key(g) in EMB:
        continue
    t = time.time()
    kw = {k: v for k, v in g.items() if k != "target_weight"}
    red = umap.UMAP(n_components=2, random_state=SEED, **kw,
                    **({"target_weight": g["target_weight"]} if "target_weight" in g else {}))
    EMB[key(g)] = red.fit_transform(P, y=y_sup if "target_weight" in g else None).astype(np.float32)
    np.savez_compressed(cache, **EMB)                   # save after each layout (resumable)
    done = i + 1
    print(f"  [{done}/{len(GRID)}] {key(g)}  {time.time() - t:.0f} s  (elapsed {time.time() - t0:.0f} s)")

# ----------------------------------------------------------------------------- scores
rng = np.random.default_rng(SEED)
idx = rng.choice(len(P), min(N_EVAL, len(P)), replace=False)
nn_hi = NearestNeighbors(n_neighbors=K_EVAL + 1).fit(P[idx]).kneighbors(P[idx], return_distance=False)[:, 1:]


def spread(E, mask):
    tot = np.median(np.linalg.norm(E - E.mean(0), axis=1))
    return np.median(np.linalg.norm(E[mask] - E[mask].mean(0), axis=1)) / tot


rows = []
for g in GRID:
    E = EMB[key(g)]
    nn_lo = NearestNeighbors(n_neighbors=K_EVAL + 1).fit(E[idx]).kneighbors(E[idx], return_distance=False)[:, 1:]
    kept = np.mean([len(set(a) & set(b)) / K_EVAL for a, b in zip(nn_hi, nn_lo)])
    rows.append({"layout": key(g), **{k: g.get(k) for k in ("n_neighbors", "min_dist", "metric", "target_weight")},
                 "knn_kept": kept, "trustworthiness": trustworthiness(P[idx], E[idx], n_neighbors=K_EVAL),
                 "nr_spread": spread(E, nr), "up_spread": spread(E, cls == "UP"), "dn_spread": spread(E, cls == "DN")})
S = pd.DataFrame(rows).sort_values("knn_kept", ascending=False)
S.to_csv(PREP / f"umap_grid_scores{SFX}.tsv", sep="\t", index=False)
print("\nlayouts ranked by kNN preservation (higher = keeps local structure better):")
print(S.drop(columns=["target_weight"]).round(3).to_string(index=False))

# ----------------------------------------------------------------------------- grid figures
mpl.rcParams.update({"font.size": 6.5, "axes.spines.top": False, "axes.spines.right": False})
CLS = {"NR": "#5DAA82", "UP": "#D1495B", "DN": "#3B6FB6"}
order = S.layout.tolist()
ncol = 5; nrow = int(np.ceil(len(order) / ncol))


def grid_fig(colour_fn, fname, title):
    fig, axs = plt.subplots(nrow, ncol, figsize=(2.1 * ncol, 2.15 * nrow))
    for ax, lay in zip(axs.ravel(), order):
        E = EMB[lay]; colour_fn(ax, E)
        s = S.set_index("layout").loc[lay]
        ax.set_title(f"{lay}\nkNN {s.knn_kept:.2f} | trust {s.trustworthiness:.3f} | NR spread {s.nr_spread:.2f}",
                     fontsize=5.6)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    for ax in axs.ravel()[len(order):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=8, y=1.0)
    fig.tight_layout()
    fig.savefig(PREP / fname, dpi=170, bbox_inches="tight")
    plt.show()


def by_class(ax, E):
    for c, s in (("NR", 1.0), ("UP", 2.5), ("DN", 3.0)):           # UP / DN on top
        m = cls == c
        ax.scatter(E[m, 0], E[m, 1], s=s, c=CLS[c], lw=0, alpha=0.5 if c == "NR" else 0.9, rasterized=True)


grid_fig(by_class, f"Fig_umap_grid_class{SFX}.png", f"UMAP grid ({MODE}) coloured by class (NR green, UP red, "
         "DN blue); ordered by kNN preservation")
TAB = plt.get_cmap("tab10")
for k in K_SHOW:
    lab = CL[f"nr_kmeans_k{k}"].to_numpy()

    def by_cluster(ax, E, lab=lab):
        ax.scatter(E[~nr, 0], E[~nr, 1], s=1.5, c="#DDDDDD", lw=0, rasterized=True)
        ax.scatter(E[nr, 0], E[nr, 1], s=1.0, c=[TAB(int(v)) for v in lab[nr]], lw=0, alpha=0.7, rasterized=True)

    grid_fig(by_cluster, f"Fig_umap_grid_clusters_k{k}{SFX}.png",
             f"UMAP grid coloured by NR k-means cluster (k = {k}; UP / DN grey)")
print("Saved to", PREP)
