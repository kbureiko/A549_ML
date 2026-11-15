# =============================================================================
# Feature map, step 1: feature set, transform and principal components (no UMAP yet)
#   1  combined feature set = union of the top-N features (mean CV permutation importance) of the three pairwise
#      models (nr_vs_up, up_vs_dn, nr_vs_dn); report layers, heavy tails
#   2  rank -> normal-score transform per feature (scale-free; spreads the low-signal bulk that holds most NR),
#      drop near-duplicates (|Spearman| > MAX_RHO, keep the more important), PCA; how many PCs carry the signal
#   3  does the transform un-squeeze NR?  per-class spread in PC space, old scaling (log1p + z) vs rank-normal
#   4  NR clustering: k-means on the PCs (NR only) for k = 3..9 - silhouette and bootstrap stability (ARI);
#      agreement with the existing A2 NR states, separating the effect of the TRANSFORM (A2 features, new
#      transform) from the effect of the FEATURE SET (union features)
# Inputs : feature matrix; ML/<run>/summaries/feature_importance_stability_summary.csv for the three runs;
#          A2_enhancer_states.tsv (optional - recomputed with the A2 recipe if absent)
# Outputs (FEATURE_MAP): umap_features.tsv, umap_redundancy.tsv, umap_pcs.tsv (enhancer_id, class,
#          GR_bound, PC1..), umap_prep_checks.tsv, umap_nr_kmeans.tsv, Fig_umap_prep_checks.png
# =============================================================================
# %%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import norm, spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from a549_enhancers.paths import PROJECT, ML, FEATURE_MAP   # local folders, set in paths.toml
FM = ML / "feature_matrix_all_classes_GRannotated.csv"
OUT = FEATURE_MAP; OUT.mkdir(parents=True, exist_ok=True)
TASKS = ("nr_vs_up", "up_vs_dn", "nr_vs_dn")
TOP_N = 30
INCLUDE_MOTIFS = False     # binary motif calls add two-level jumps to distances; the map is meant to show chromatin
INCLUDE_LENGTH = False     # Enh_len tracks activity / detectability (see length analysis); keep out of the layout
MAX_RHO = 0.90
VAR_KEEP = 0.90            # PCs kept = enough for this share of variance (capped at MAX_PC)
MAX_PC = 30
K_RANGE = range(3, 10)
N_BOOT, BOOT_FRAC, SEED = 20, 0.8, 20261109
rng = np.random.default_rng(SEED)


def find(pattern, root=PROJECT):
    hits = sorted(p for p in root.rglob(pattern) if "figures" not in p.parts)
    return hits[-1] if hits else None


# ----------------------------------------------------------------------------- 1 feature set
imp = {}
for task in TASKS:
    runs = sorted(d for d in ML.glob(f"*__{task}__*") if (d / "summaries" / "feature_importance_stability_summary.csv").exists())
    if not runs:
        raise FileNotFoundError(f"no run with summaries/feature_importance_stability_summary.csv for {task} in {ML}")
    s = pd.read_csv(runs[-1] / "summaries" / "feature_importance_stability_summary.csv")
    imp[task] = s.sort_values("mean_importance", ascending=False).reset_index(drop=True)
    print(f"{task}: {runs[-1].name}  ({len(s)} features)")

FMX = pd.read_csv(FM, low_memory=False)
FMX["GR_bound"] = FMX.GR_bound.astype(str).str.lower().eq("true")


def layer(f):
    if f.startswith("motif_binary::"):
        return "motif"
    if f in ("annotation", "Distance to TSS", "Enh_len", "gc"):
        return "genomic"
    if "_dex_" in f or f == "ATAC_dex_1h":
        return "dex"
    return "baseline"


rows = []
for task, s in imp.items():
    for r, f in enumerate(s.feature.head(TOP_N), 1):
        rows.append({"feature": f, "task": task, "rank": r})
U = pd.DataFrame(rows)
F = (U.groupby("feature").agg(best_rank=("rank", "min"), n_models=("task", "nunique"),
                              models=("task", lambda t: ",".join(sorted(t)))).reset_index())
F["layer"] = F.feature.map(layer)
F["in_matrix"] = F.feature.isin(FMX.columns)
F["numeric"] = F.feature.map(lambda f: f in FMX.columns and pd.api.types.is_numeric_dtype(FMX[f]))
keep = F.numeric & ~((F.layer == "motif") & (not INCLUDE_MOTIFS)) & ~((F.feature == "Enh_len") & (not INCLUDE_LENGTH))
keep &= ~F.feature.isin(["Distance to TSS", "gc"])          # genomic position / composition, not chromatin
F["used_before_dedup"] = keep
print(f"\nunion of top-{TOP_N}: {len(F)} features; by layer {F.layer.value_counts().to_dict()}; "
      f"in models: {F.n_models.value_counts().sort_index().to_dict()}")
if (~F.in_matrix).any():
    print("WARNING not in feature matrix:", F.loc[~F.in_matrix, "feature"].tolist())

X = FMX[F.loc[keep, "feature"]].astype(float)
tails = pd.DataFrame({
    "feature": X.columns,
    "frac_missing": X.isna().mean().values,
    "p99_over_median": (X.quantile(0.99) / X.median().replace(0, np.nan)).values,
    "skew_log1p": np.log1p(X.clip(lower=0)).skew().values,
})
print("\nheaviest tails (99th percentile / median):")
print(tails.sort_values("p99_over_median", ascending=False).head(8).round(2).to_string(index=False))


# ----------------------------------------------------------------------------- 2 transform, dedup, PCA
def rank_normal(df):
    r = df.rank(method="average", na_option="keep")
    return pd.DataFrame(norm.ppf((r - 0.5) / r.notna().sum()), index=df.index, columns=df.columns).fillna(0.0)


R = rank_normal(X)
rho = R.corr(method="pearson")                     # Pearson on normal scores ~ Spearman on raw values
order = F.loc[keep].sort_values(["best_rank", "n_models"], ascending=[True, False]).feature.tolist()
kept, dropped = [], []
for f in order:
    hit = [k for k in kept if abs(rho.loc[f, k]) > MAX_RHO]
    if hit:
        dropped.append({"feature": f, "dropped_for": hit[0], "rho": round(rho.loc[f, hit[0]], 3)})
    else:
        kept.append(f)
RED = pd.DataFrame(dropped)
F["used"] = F.feature.isin(kept)
F.to_csv(OUT / "umap_features.tsv", sep="\t", index=False)
RED.to_csv(OUT / "umap_redundancy.tsv", sep="\t", index=False)
print(f"\nredundancy |rho| > {MAX_RHO}: dropped {len(RED)}, kept {len(kept)}")
if len(RED):
    print(RED.to_string(index=False))


def pca(M):
    M = (M - M.mean(axis=0)) / M.std(axis=0, ddof=0).replace(0, 1)
    u, s, vt = np.linalg.svd(M.values, full_matrices=False)
    ev = s ** 2 / (s ** 2).sum()
    return u * s, ev, pd.DataFrame(vt.T, index=M.columns)


PC_new, ev_new, LOAD = pca(R[kept])
n_pc = int(min(MAX_PC, np.searchsorted(np.cumsum(ev_new), VAR_KEEP) + 1))
Xold = np.log1p(X[kept].clip(lower=0)).fillna(0)
PC_old, ev_old, _ = pca(Xold)
print(f"\nPCA (rank-normal, {len(kept)} features): {n_pc} PCs for {VAR_KEEP:.0%} variance; "
      f"PC1 {ev_new[0]:.0%}, PC2 {ev_new[1]:.0%}")
top_load = {f"PC{i + 1}": ", ".join(LOAD[i].abs().sort_values(ascending=False).head(4).index.str.split("_ENCFF").str[0])
            for i in range(min(5, n_pc))}
for k, v in top_load.items():
    print(f"  {k}: {v}")
PCS = pd.DataFrame(PC_new[:, :n_pc], columns=[f"PC{i + 1}" for i in range(n_pc)])
PCS.insert(0, "enhancer_id", FMX.enhancer_id.values); PCS.insert(1, "class", FMX["class"].values)
PCS.insert(2, "GR_bound", FMX.GR_bound.values)
PCS.to_csv(OUT / "umap_pcs.tsv", sep="\t", index=False)


# ----------------------------------------------------------------------------- 3 is NR squeezed?
def spread(P, cls):
    """median distance to the class centroid / median distance of all points to the global centroid"""
    tot = np.median(np.linalg.norm(P - P.mean(0), axis=1))
    return {c: np.median(np.linalg.norm(P[cls == c] - P[cls == c].mean(0), axis=1)) / tot for c in ("NR", "UP", "DN")}


cls = FMX["class"].values
sp_old, sp_new = spread(PC_old[:, :n_pc], cls), spread(PC_new[:, :n_pc], cls)
checks = [{"check": f"spread {c} (old log1p+z)", "value": sp_old[c]} for c in sp_old] + \
         [{"check": f"spread {c} (rank-normal)", "value": sp_new[c]} for c in sp_new]
print("\nrelative spread (1 = as spread as all enhancers):")
for c in ("NR", "UP", "DN"):
    print(f"  {c}: old {sp_old[c]:.2f} -> rank-normal {sp_new[c]:.2f}")

# ----------------------------------------------------------------------------- 4 NR clustering
nr = FMX["class"].eq("NR").values
old_path = find("A2_enhancer_states.tsv")
if old_path is not None:
    OLD = pd.read_csv(old_path, sep="\t").set_index("enhancer_id").state
    print(f"\nexisting NR states: {old_path}")
else:                                                   # recipe of the earlier NR states
    col = lambda p, t: [c for c in FMX.columns if c.startswith(p) and t in c][0]
    SF = [col("H3K27ac_", "eth"), col("H3K4me1_", "eth"), col("H3K4me3_", "eth"), col("POLR2A_", "eth"),
          "EP300_ENCFF842XYY", "JUN_ENCFF957TUK", "CTCF_ENCFF083AEY", "RAD21_ENCFF778YHU", col("H2AFZ_", "eth"),
          col("H3K27me3_", "eth"), col("NR3C1_", "dex"), col("FOXA1_", "dex"), "ATAC_ctrl"]
    Xa = np.log1p(FMX[SF].clip(lower=0)); Za = (Xa - Xa[nr].mean()) / Xa[nr].std()
    OLD = pd.Series(KMeans(6, n_init=20, random_state=0).fit(Za[nr]).labels_.astype(str),
                    index=FMX.enhancer_id[nr])
    print("\nA2_enhancer_states.tsv not found - recomputed the A2 states (labels unnamed)")
old_nr = OLD.reindex(FMX.enhancer_id[nr]).values


def km_stats(P, label):
    out = []
    for k in K_RANGE:
        full = KMeans(k, n_init=10, random_state=SEED).fit(P)
        sil = silhouette_score(P, full.labels_, sample_size=min(3000, len(P)), random_state=SEED)
        aris = []
        for _ in range(N_BOOT):
            idx = rng.choice(len(P), int(BOOT_FRAC * len(P)), replace=False)
            m = KMeans(k, n_init=5, random_state=int(rng.integers(1e9))).fit(P[idx])
            aris.append(adjusted_rand_score(full.labels_, m.predict(P)))
        ari_old = adjusted_rand_score(old_nr, full.labels_) if k == 6 else np.nan
        out.append({"features": label, "k": k, "silhouette": sil, "stability_ARI_mean": np.mean(aris),
                    "stability_ARI_min": np.min(aris), "ARI_vs_A2_states": ari_old})
    return out


# (i) A2 state features with the new transform -> effect of the transform alone
A2_FEATS = [c for c in FMX.columns if any(c.startswith(p) for p in ("H3K27ac_", "H3K4me1_", "H3K4me3_", "POLR2A_",
            "H2AFZ_", "H3K27me3_")) and "_eth_" in c] + ["EP300_ENCFF842XYY", "JUN_ENCFF957TUK", "CTCF_ENCFF083AEY",
            "RAD21_ENCFF778YHU", "ATAC_ctrl"] + [c for c in FMX.columns if c.startswith(("NR3C1_", "FOXA1_")) and "_dex_" in c]
PA, eva, _ = pca(rank_normal(FMX.loc[nr, A2_FEATS].astype(float)))
na = int(np.searchsorted(np.cumsum(eva), VAR_KEEP) + 1)
# (ii) union features (NR rows of the rank-normal PCs computed on all enhancers)
KM = pd.DataFrame(km_stats(PA[:, :na], "A2 features, rank-normal") + km_stats(PC_new[nr, :n_pc], "union features, rank-normal"))
KM.to_csv(OUT / "umap_nr_kmeans.tsv", sep="\t", index=False)
print("\nNR k-means (stability = ARI of 80% subsamples vs full fit; >0.8 stable):")
print(KM.round(3).to_string(index=False))

pd.DataFrame(checks + [{"check": "n_features_used", "value": len(kept)}, {"check": "n_pcs", "value": n_pc}]).to_csv(
    OUT / "umap_prep_checks.tsv", sep="\t", index=False)

# ----------------------------------------------------------------------------- diagnostic figure
mpl.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False})
CLS = {"UP": "#D1495B", "DN": "#3B6FB6", "NR": "#5DAA82"}
fig, ax = plt.subplots(1, 4, figsize=(11, 2.8), gridspec_kw={"width_ratios": [1.2, 1, 1, 1.4], "wspace": 0.45})
t = tails.set_index("feature").p99_over_median.sort_values(ascending=False).head(15)
ax[0].barh(range(len(t)), t.values, color="#888888"); ax[0].set_yticks(range(len(t)))
ax[0].set_yticklabels(t.index.str.split("_ENCFF").str[0], fontsize=5.5); ax[0].invert_yaxis()
ax[0].set_xlabel("99th percentile / median"); ax[0].set_title("Heaviest tails (raw)")
ax[1].plot(np.arange(1, len(ev_new) + 1), np.cumsum(ev_new), "-o", ms=2, color="#333333")
ax[1].axhline(VAR_KEEP, ls=":", color="#999999"); ax[1].axvline(n_pc, ls=":", color="#999999")
ax[1].set_xlabel("PCs"); ax[1].set_ylabel("cumulative variance"); ax[1].set_title(f"PCA ({len(kept)} features)")
x = np.arange(3)
ax[2].bar(x - 0.18, [sp_old[c] for c in ("NR", "UP", "DN")], 0.36, color="white",
          edgecolor=[CLS[c] for c in ("NR", "UP", "DN")], label="log1p + z")
ax[2].bar(x + 0.18, [sp_new[c] for c in ("NR", "UP", "DN")], 0.36, color=[CLS[c] for c in ("NR", "UP", "DN")],
          label="rank-normal")
ax[2].set_xticks(x); ax[2].set_xticklabels(["NR", "UP", "DN"]); ax[2].axhline(1, ls=":", color="#999999")
ax[2].set_ylabel("spread vs all enhancers"); ax[2].legend(frameon=False, fontsize=6); ax[2].set_title("Class spread in PC space")
for lab, ls in (("A2 features, rank-normal", "--"), ("union features, rank-normal", "-")):
    k = KM[KM.features == lab]
    ax[3].plot(k.k, k.stability_ARI_mean, ls, marker="o", ms=3, color="#333333", label=f"stability: {lab}")
    ax[3].plot(k.k, k.silhouette, ls, marker="s", ms=3, color="#B07AA1", label=f"silhouette: {lab}")
ax[3].set_xlabel("k (NR enhancers)"); ax[3].set_ylim(0, 1.02); ax[3].legend(frameon=False, fontsize=5)
ax[3].set_title("NR k-means")
fig.savefig(OUT / "Fig_umap_prep_checks.png", dpi=200, bbox_inches="tight")
plt.show()
print("Saved to", OUT)
