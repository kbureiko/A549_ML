# =============================================================================
# Matched comparison of UP and DN enhancers with NR enhancers (numbers for the text; no figure is drawn)
# Question: is the dex-associated change of a factor at UP / DN enhancers explained by GR signal?
# For every factor measured in both conditions and every UP or DN enhancer, the K NR enhancers of the same
# GR-binding status are taken that are nearest in
#   "control":      the factor's percentile rank in control cells
#   "control + GR": the factor's percentile rank in control cells and the GR percentile rank after dex
# Matched difference = percentile rank after dex of the enhancer minus the median rank after dex of its K matched
# NR enhancers. Per group (UP / DN x GR-bound / not bound): median matched difference, bootstrap 95 % CI of the
# median, Wilcoxon signed-rank p against zero, and the median distance to the K-th matched enhancer (match quality,
# in rank points). A difference that disappears with "control + GR" is accounted for by GR signal.
# Also printed: GR percentile after dex per group, and Spearman(factor after dex, GR after dex).
# FOXA1: control and dex tracks are uncorrelated (rho ~0.05), so only its "control + GR" column is informative.
# Input: FM (same feature matrix as the class summary cell)
# Output (FIGURES): S4_S5_matched_to_NR.tsv (columns matched_control_* and matched_control_GR_*:
#   median, ci_low, ci_high, p_signed_rank, pct_below_0, dist_kth)
# =============================================================================
# %%
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon
from sklearn.neighbors import NearestNeighbors

from a549_enhancers.paths import PROJECT, ML, FIGURES   # local folders, set in paths.toml
FM = ML / "feature_matrix_all_classes_GRannotated.csv"
FINAL = FIGURES; FINAL.mkdir(parents=True, exist_ok=True)
K = 20                      # matched NR enhancers per UP / DN enhancer
N_BOOT, SEED = 2000, 0      # bootstrap of the median
GR_DEX = "NR3C1|_dex_"
TRACKS = {"ATAC": ("ATAC_ctrl", "ATAC_dex_1h"), "Pol II": ("POLR2A|_eth_", "POLR2A|_dex_"),
          "H3K27ac": ("H3K27ac|_eth_", "H3K27ac|_dex_"), "FOSL2": ("FOSL2|_eth_", "FOSL2|_dex_"),
          "EP300": ("EP300_ENCFF842XYY", "EP300|_dex_"), "cJUN": ("JUN_ENCFF957TUK", "JUN|_dex_"),
          "BCL3": ("BCL3_ENCFF856OZJ", "BCL3|_dex_"), "SMC3": ("SMC3_ENCFF029NYA", "SMC3|_dex_"),
          "RAD21": ("RAD21_ENCFF778YHU", "RAD21|_dex_"), "CTCF": ("CTCF_ENCFF083AEY", "CTCF|_dex_"),
          "FOXA1": ("FOXA1|_eth_", "FOXA1|_dex_"), "H2A.Z": ("H2AFZ|_eth_", "H2AFZ|_dex_")}

D = pd.read_csv(FM, low_memory=False).set_index("enhancer_id")
cls = D["class"].to_numpy()
grb = D.GR_bound.astype(str).str.lower().eq("true").to_numpy()
grp = np.where(grb, cls + " GR+", cls + " GR-")
PAIRS = [("UP GR+", "NR GR+"), ("DN GR+", "NR GR+"), ("UP GR-", "NR GR-"), ("DN GR-", "NR GR-")]


def colname(spec):
    if spec in D.columns:
        return spec
    pre, tag = spec.split("|")
    hit = [c for c in D.columns if c.startswith(pre + "_") and tag in c]
    if not hit:
        raise KeyError(f"no column for {spec}")
    return hit[0]


rank = lambda c: 100 * D[c].rank(pct=True).to_numpy()
gr_dex = rank(colname(GR_DEX))
rng = np.random.default_rng(SEED)


def matched(X, r_dex, group, reference):
    ia, ib = np.where(grp == group)[0], np.where(grp == reference)[0]
    dist, idx = NearestNeighbors(n_neighbors=K).fit(X[ib]).kneighbors(X[ia])
    diff = r_dex[ia] - np.median(r_dex[ib][idx], axis=1)
    boot = np.median(rng.choice(diff, (N_BOOT, len(diff))), axis=1)
    return {"median": np.median(diff), "ci_low": np.percentile(boot, 2.5), "ci_high": np.percentile(boot, 97.5),
            "p_signed_rank": wilcoxon(diff).pvalue, "pct_below_0": 100 * (diff < 0).mean(),
            "dist_kth": np.median(dist[:, -1])}


rows = []
for f, (a, b) in TRACKS.items():
    r0, r1 = rank(colname(a)), rank(colname(b))
    for group, reference in PAIRS:
        row = {"factor": f, "group": group, "reference": reference, "n": int((grp == group).sum()),
               "control_percentile": np.median(r0[grp == group]), "dex_percentile": np.median(r1[grp == group]),
               "spearman_dex_vs_GR_dex_all": spearmanr(r1, gr_dex)[0],
               "spearman_dex_vs_GR_dex_GRbound": spearmanr(r1[grb], gr_dex[grb])[0]}
        for name, X in (("matched_control", r0[:, None]), ("matched_control_GR", np.column_stack([r0, gr_dex]))):
            row.update({f"{name}_{k}": v for k, v in matched(X, r1, group, reference).items()})
        rows.append(row)
M = pd.DataFrame(rows)
M.to_csv(FINAL / "S4_S5_matched_to_NR.tsv", sep="\t", index=False, float_format="%.4g")

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
GROUPS = ["UP GR+", "NR GR+", "DN GR+", "UP GR-", "NR GR-", "DN GR-"]
print("enhancers per group:", {g: int((grp == g).sum()) for g in GROUPS})
print("GR percentile after dex (median):", {g: round(float(np.median(gr_dex[grp == g])), 1) for g in GROUPS})
print("   GR-bound DN vs GR-bound NR: Mann-Whitney p = %.1e; GR-bound UP vs GR-bound NR: p = %.1e" % (
    mannwhitneyu(gr_dex[grp == "DN GR+"], gr_dex[grp == "NR GR+"]).pvalue,
    mannwhitneyu(gr_dex[grp == "UP GR+"], gr_dex[grp == "NR GR+"]).pvalue))
fmt = lambda r, k: f"{r[k + '_median']:+.1f} ({r[k + '_ci_low']:+.1f} to {r[k + '_ci_high']:+.1f})"
show = pd.DataFrame({"factor": M.factor, "group": M.group, "n": M.n,
                     "matched on control": [fmt(r, "matched_control") for _, r in M.iterrows()],
                     "matched on control + GR after dex": [fmt(r, "matched_control_GR") for _, r in M.iterrows()],
                     "p (control + GR)": M.matched_control_GR_p_signed_rank.map("{:.0e}".format),
                     "rho with GR after dex (all / GR-bound)": M.spearman_dex_vs_GR_dex_all.map("{:.2f}".format) + " / "
                     + M.spearman_dex_vs_GR_dex_GRbound.map("{:.2f}".format)})
print(f"\nmedian matched difference in percentile rank after dex (95 % CI), {K} matched NR enhancers of the same GR status:")
print(show.to_string(index=False))
print("\nSaved:", FINAL / "S4_S5_matched_to_NR.tsv")
