# =============================================================================
# Numbers for the text of the feature-map section (Figure 1 F-H, Supplementary Figures S3-S5; no figure is drawn)
#   1. Class summary (unchanged). For every factor shown in S4 / S5: Spearman(control, dex), median percentile rank
#      in control and after dex per response class, and the level-adjusted rank change (delta = dex rank minus the
#      median dex rank of NR enhancers with a similar control rank, 20 bins - the same measure as the delta panels,
#      here per enhancer, not smoothed) per class and per class x GR binding; share of UP above / DN below zero;
#      Mann-Whitney p against NR enhancers of the same GR status. Single-condition tracks and motifs: median
#      percentile / % with motif.
#   2. Neighbourhoods in the feature space the map is built from (N_NB nearest neighbours in PC space; does not
#      depend on the layout): how mixed the classes are (and, if the embeddings are found, the same on the 2-D map,
#      which shows how much of the class separation comes from the supervision); how mixed the NR topics are.
#   3. NR subsets: GR binding, top 10% of GR signal, main topic, annotation, and NR enhancers with at least MIN_NB
#      UP (or DN) enhancers among their neighbours. Per subset: n, % GR-bound, % GRE, median delta per factor,
#      median control / dex percentile of key factors, PRO-seq log2 fold change (if the PRO-seq table is found).
#   4. Within NR, per factor: Spearman(delta, PRO-seq log2FC), Spearman(delta, GC content), and the share of
#      enhancers with |delta| > 10 after smoothing as in the panels (median over N_NB nearest neighbours).
# Inputs: FM; NR_TOPICS/MS_nr_topics_usage.tsv; FEATURE_MAP/umap_pcs.tsv
#   optional: FEATURE_MAP/<EMB_FILE> (the map), PROSEQ_MA_TABLE (PRO-seq log2FC per enhancer)
# Outputs (FIGURES): S4_S5_class_summary.tsv, S4_S5_neighbourhoods.tsv, S4_S5_nr_subsets.tsv,
#   S4_S5_delta_within_NR.tsv
# =============================================================================
# %%
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, binom
from sklearn.neighbors import NearestNeighbors

from a549_enhancers.paths import PROJECT, ML, FIGURES, PROSEQ_MA_TABLE, FEATURE_MAP, NR_TOPICS   # local folders, set in paths.toml
FM = ML / "feature_matrix_all_classes_GRannotated.csv"
TOPICS_FILE = NR_TOPICS / "MS_nr_topics_usage.tsv"
PREP = FEATURE_MAP                                           # umap_pcs.tsv and the embeddings of the map
EMB_FILE, LAYOUT = "umap_grid_embeddings_supervised.npz", "nn30_md0.1_cosine_sup0.5"
PROSEQ = PROSEQ_MA_TABLE                                      # per-enhancer PRO-seq log2FC (optional)
FINAL = FIGURES; FINAL.mkdir(parents=True, exist_ok=True)
N_BINS = 20
N_NB, MIN_NB = 25, 5                # neighbourhood size (as in the delta panels); "close to UP / DN" = >= MIN_NB of N_NB
TRACKS = {"GR": ("NR3C1|_eth_", "NR3C1|_dex_"), "ATAC": ("ATAC_ctrl", "ATAC_dex_1h"),
          "Pol II": ("POLR2A|_eth_", "POLR2A|_dex_"), "H3K27ac": ("H3K27ac|_eth_", "H3K27ac|_dex_"),
          "H3K4me1": ("H3K4me1|_eth_", "H3K4me1|_dex_"), "H3K4me3": ("H3K4me3|_eth_", "H3K4me3|_dex_"),
          "H3K79me2": ("H3K79me2|_eth_", "H3K79me2|_dex_"), "FOSL2": ("FOSL2|_eth_", "FOSL2|_dex_"),
          "EP300": ("EP300_ENCFF842XYY", "EP300|_dex_"), "cJUN": ("JUN_ENCFF957TUK", "JUN|_dex_"),
          "BCL3": ("BCL3_ENCFF856OZJ", "BCL3|_dex_"), "SMC3": ("SMC3_ENCFF029NYA", "SMC3|_dex_"),
          "RAD21": ("RAD21_ENCFF778YHU", "RAD21|_dex_"), "CTCF": ("CTCF_ENCFF083AEY", "CTCF|_dex_"),
          "FOXA1": ("FOXA1|_eth_", "FOXA1|_dex_"), "H2A.Z": ("H2AFZ|_eth_", "H2AFZ|_dex_")}
SINGLE = {"PHF8": "PHF8_ENCFF690AJW", "TCF12": "TCF12|_eth_", "GATA3": "GATA3_ENCFF485KDU", "TEAD4": "TEAD4_ENCFF186WSI"}
MOTIFS = {"GRE motif": ["motif_binary::GRE(NR),IR3/A549-GR-ChIP-Seq(GSE32465)/Homer",
                        "motif_binary::GRE(NR),IR3/RAW264.7-GRE-ChIP-Seq(Unpublished)/Homer"],
          "AP-1 motif": ["motif_binary::AP-1(bZIP)/ThioMac-PU.1-ChIP-Seq(GSE21512)/Homer"]}
KEY = ["ATAC", "Pol II", "H3K27ac", "EP300", "cJUN", "BCL3"]   # control / dex percentiles reported per NR subset

D = pd.read_csv(FM, low_memory=False).set_index("enhancer_id")
cls = D["class"]; nr = cls.eq("NR").to_numpy()
grb = D.GR_bound.astype(str).str.lower().eq("true")
grp = pd.Series(np.where(grb, cls + " GR+", cls + " GR-"), index=D.index)
CLASSES, GROUPS = ["UP", "DN", "NR"], ["UP GR+", "UP GR-", "DN GR+", "DN GR-", "NR GR+", "NR GR-"]


def colname(spec):
    if spec in D.columns:
        return spec
    pre, tag = spec.split("|")
    hit = [c for c in D.columns if c.startswith(pre + "_") and tag in c]
    if not hit:
        raise KeyError(f"no column for {spec}")
    return hit[0]


rank = lambda c: pd.Series(100 * D[c].rank(pct=True).to_numpy(), index=D.index)

# ----------------------------------------------------------------- 1. class summary
R0, R1, DELTA, rows = {}, {}, {}, []
for f, (a, b) in TRACKS.items():
    r0, r1 = rank(colname(a)), rank(colname(b))
    bins = pd.Series(np.clip((r0.to_numpy() / 100 * N_BINS).astype(int), 0, N_BINS - 1), index=D.index)
    delta = r1 - bins.map(r1[nr].groupby(bins[nr]).median())          # level-adjusted rank change
    R0[f], R1[f], DELTA[f] = r0, r1, delta
    row = {"feature": f, "kind": "control and dex", "spearman_control_dex": r0.corr(r1)}
    for c in CLASSES:
        m = cls.eq(c)
        row.update({f"control_{c}": r0[m].median(), f"dex_{c}": r1[m].median(), f"delta_{c}": delta[m].median()})
    for g in GROUPS:
        row[f"delta_{g}"] = delta[grp == g].median()
        if not g.startswith("NR"):
            row[f"p_vs_NR_{g}"] = mannwhitneyu(delta[grp == g], delta[grp == "NR " + g.split()[1]]).pvalue
    row["pct_UP_delta_above_0"] = 100 * (delta[cls.eq("UP")] > 0).mean()
    row["pct_DN_delta_below_0"] = 100 * (delta[cls.eq("DN")] < 0).mean()
    rows.append(row)
for f, spec in SINGLE.items():
    r = rank(colname(spec))
    row = {"feature": f, "kind": "control only"}
    row.update({f"control_{c}": r[cls.eq(c)].median() for c in CLASSES})
    row.update({f"control_{g}": r[grp == g].median() for g in GROUPS})
    row["p_DN_vs_NR"] = mannwhitneyu(r[cls.eq("DN")], r[nr]).pvalue
    rows.append(row)
for f, cols in MOTIFS.items():
    v = (D[[c for c in cols if c in D.columns]].fillna(0) > 0).any(axis=1)
    row = {"feature": f, "kind": "motif, % of enhancers"}
    row.update({f"control_{c}": 100 * v[cls.eq(c)].mean() for c in CLASSES})
    row.update({f"control_{g}": 100 * v[grp == g].mean() for g in GROUPS})
    rows.append(row)
S = pd.DataFrame(rows)
S.to_csv(FINAL / "S4_S5_class_summary.tsv", sep="\t", index=False, float_format="%.4g")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 40)
print("=" * 30, "1. class summary")
show = ["feature", "spearman_control_dex"] + [f"{k}_{c}" for c in CLASSES for k in ("control", "dex", "delta")]
print(S[S.kind == "control and dex"][show].round(1).to_string(index=False))
print(S[S.kind == "control and dex"][["feature"] + [f"delta_{g}" for g in GROUPS] +
                                     ["pct_UP_delta_above_0", "pct_DN_delta_below_0"]].round(1).to_string(index=False))
print(S[S.kind != "control and dex"][["feature", "kind"] + [f"control_{c}" for c in CLASSES] +
                                     [f"control_{g}" for g in GROUPS]].round(1).to_string(index=False))

# marks that follow genomic annotation: median control percentile per annotation
ann = D.annotation.astype(str)
for f, spec in (("PHF8", SINGLE["PHF8"]), ("H3K4me1", TRACKS["H3K4me1"][0]), ("H3K4me3", TRACKS["H3K4me3"][0]),
                ("H3K79me2", TRACKS["H3K79me2"][0])):
    print(f"{f:9s} control percentile per annotation:", rank(colname(spec)).groupby(ann).median().round(0).to_dict())

# GR binding and topics
print("\n% GR-bound per class:", (100 * grb.groupby(cls).mean()).round(1).to_dict())
print("% GR-bound among NR, per annotation:", (100 * grb[nr].groupby(D.annotation[nr]).mean()).round(1).to_dict())
top = R1["GR"] >= 90
print(f"top 10% of GR signal after dex: {int(top.sum())} enhancers; " +
      ", ".join(f"{c} {int((top & cls.eq(c)).sum())} ({100 * (top & cls.eq(c)).sum() / top.sum():.1f}%)" for c in CLASSES))
U = pd.read_csv(TOPICS_FILE, sep="\t").set_index("enhancer_id").reindex(D.index[nr])
tc = [c for c in U.columns if c[:1] == "T" and c[1:2].isdigit()]
main, share = U[tc].idxmax(axis=1).str.split(" ").str[0], U[tc].max(axis=1)
print("NR enhancers per main topic:", main.value_counts().sort_index().to_dict())
print("share of GR-bound NR enhancers per main topic (%):",
      (100 * main[grb[nr]].value_counts(normalize=True)).round(1).sort_index().to_dict())
print("main topic of CpG-annotated NR enhancers (%):",
      (100 * main[ann[nr].eq("CpG")].value_counts(normalize=True)).round(1).sort_index().to_dict())
print(f"main-topic share: median {share.median():.2f}; {100 * (share >= 0.7).mean():.1f}% of NR enhancers at >= 0.7")
for f, spec in SINGLE.items():
    print(f"{f:6s} control percentile per main topic:", rank(colname(spec))[nr].groupby(main).median().round(0).to_dict())
for f, cols in MOTIFS.items():
    v = (D[[c for c in cols if c in D.columns]].fillna(0) > 0).any(axis=1)
    print(f"{f}: % of NR enhancers per main topic:", (100 * v[nr].groupby(main).mean()).round(0).to_dict())

# ----------------------------------------------------------------- 2. neighbourhoods in feature space
print("\n" + "=" * 30, f"2. neighbourhoods ({N_NB} nearest neighbours)")
PCS = pd.read_csv(PREP / "umap_pcs.tsv", sep="\t").set_index("enhancer_id")
Pm = PCS.filter(regex=r"^PC\d+$").reindex(D.index).to_numpy()
if np.isnan(Pm).any():
    raise ValueError("umap_pcs.tsv does not contain every enhancer of the feature matrix")


def neighbours(X, k):                                    # k nearest neighbours, the enhancer itself left out
    idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X, return_distance=False)
    return np.array([row[row != i][:k] for i, row in enumerate(idx)])


c_arr = cls.to_numpy()
nb = neighbours(Pm, N_NB)
n_up = pd.Series((c_arr[nb] == "UP").sum(axis=1), index=D.index)
n_dn = pd.Series((c_arr[nb] == "DN").sum(axis=1), index=D.index)
spaces = {f"feature space ({Pm.shape[1]} PCs)": nb}
emb = PREP / EMB_FILE
if emb.exists() and LAYOUT in np.load(emb).files:        # embeddings are stored in the row order of umap_pcs.tsv
    E = pd.DataFrame(np.load(emb)[LAYOUT], index=PCS.index).reindex(D.index).to_numpy()
    spaces["map (2-D layout)"] = neighbours(E, N_NB)
else:
    print(f"(map embeddings {emb.name} / {LAYOUT} not found - class mixing is given for the feature space only)")
rows = []
for space, nbx in spaces.items():
    for c in CLASSES:
        m = c_arr == c
        own, nrs = (c_arr[nbx][m] == c).mean(axis=1), (c_arr[nbx][m] == "NR").mean(axis=1)
        rows.append({"space": space, "class": c, "n": int(m.sum()), "class_frequency_pct": 100 * m.mean(),
                     "mean_own_class_pct": 100 * own.mean(), "median_own_class_pct": 100 * np.median(own),
                     "n_majority_own_class": int((own > 0.5).sum()), "pct_majority_own_class": 100 * (own > 0.5).mean(),
                     "n_majority_NR": int((nrs > 0.5).sum()), "pct_majority_NR": 100 * (nrs > 0.5).mean()})
NBH = pd.DataFrame(rows)
NBH.to_csv(FINAL / "S4_S5_neighbourhoods.tsv", sep="\t", index=False, float_format="%.4g")
print("class of the neighbours (own class = same class as the enhancer):")
print(NBH.round(1).to_string(index=False))
for c, cnt in (("UP", n_up), ("DN", n_dn)):
    freq = (c_arr == c).mean()
    expected = nr.sum() * (1 - binom.cdf(MIN_NB - 1, N_NB, freq))
    print(f"NR enhancers with >= {MIN_NB} {c} enhancers among their {N_NB} neighbours: {int((cnt[nr] >= MIN_NB).sum())} "
          f"({100 * (cnt[nr] >= MIN_NB).mean():.1f}% of NR; {expected:.0f} expected if the classes were mixed at random)")
t_arr = main.to_numpy()
same = (t_arr[neighbours(Pm[nr], N_NB)] == t_arr[:, None]).mean(axis=1)
tf = main.value_counts(normalize=True)
print(f"NR: {100 * same.mean():.0f}% of the {N_NB} nearest NR neighbours have the same main topic "
      f"({100 * (tf ** 2).sum():.0f}% expected by chance); per topic:",
      {t: round(100 * same[t_arr == t].mean()) for t in sorted(tf.index)})

# ----------------------------------------------------------------- 3. NR subsets
print("\n" + "=" * 30, "3. NR subsets (medians)")
LFC = None
if PROSEQ.exists():
    Mx = pd.read_csv(PROSEQ, sep="\t")
    for shift in (1, 0):                                 # HOMER positions are 1-based, the enhancer IDs 0-based
        ids = Mx.Chr.astype(str) + "_" + (Mx.Start - shift).astype(str) + "_" + Mx.End.astype(str)
        if ids.isin(D.index).mean() > 0.99:
            LFC = pd.Series(Mx.log2FC.to_numpy(), index=ids.to_numpy()).reindex(D.index)
            break
    if LFC is None:
        print(f"({PROSEQ.name}: coordinates do not match the enhancer IDs - PRO-seq columns left out)")
else:
    print(f"({PROSEQ} not found - PRO-seq columns left out)")
is_nr = pd.Series(nr, index=D.index)
main_all = main.reindex(D.index)
gre = (D[[c for c in MOTIFS["GRE motif"] if c in D.columns]].fillna(0) > 0).any(axis=1)
UP_LIKE, DN_LIKE = f"NR, >= {MIN_NB} UP neighbours", f"NR, >= {MIN_NB} DN neighbours"
SUB = {"UP": cls.eq("UP"), "DN": cls.eq("DN"), "NR": is_nr, "NR, GR-bound": is_nr & grb, "NR, not GR-bound": is_nr & ~grb,
       "NR, GR signal in top 10%": is_nr & top, UP_LIKE: is_nr & (n_up >= MIN_NB), DN_LIKE: is_nr & (n_dn >= MIN_NB)}
SUB.update({f"NR, topic {t}": is_nr & main_all.eq(t) for t in sorted(main.unique())})
SUB.update({f"NR, {a}": is_nr & ann.eq(a) for a in sorted(ann.unique())})
rows = []
for name, m in SUB.items():
    row = {"subset": name, "n": int(m.sum()), "pct_GR_bound": 100 * grb[m].mean(), "pct_GRE": 100 * gre[m].mean(),
           "GR_dex_percentile": R1["GR"][m].median()}
    if LFC is not None:
        v = LFC[m].dropna()
        row.update({"proseq_log2FC": v.median(), "proseq_log2FC_q25": v.quantile(0.25),
                    "proseq_log2FC_q75": v.quantile(0.75), "pct_log2FC_above_0": 100 * (v > 0).mean()})
    row.update({f"delta_{f}": DELTA[f][m].median() for f in TRACKS if f != "GR"})
    for f in KEY:
        row[f"control_{f}"], row[f"dex_{f}"] = R0[f][m].median(), R1[f][m].median()
    rows.append(row)
NS = pd.DataFrame(rows)
NS.to_csv(FINAL / "S4_S5_nr_subsets.tsv", sep="\t", index=False, float_format="%.4g")
first = ["subset", "n", "pct_GR_bound", "pct_GRE", "GR_dex_percentile"] + [c for c in NS.columns if "log2FC" in c]
print(NS[first].round(2).to_string(index=False))
print(NS[["subset"] + [c for c in NS.columns if c.startswith("delta_")]].round(1).rename(
    columns=lambda c: c.replace("delta_", "")).to_string(index=False))
print("control -> dex percentile:")
print(pd.DataFrame({"subset": NS.subset, **{f: NS[f"control_{f}"].round(0).astype(int).astype(str) + " -> " +
                                           NS[f"dex_{f}"].round(0).astype(int).astype(str) for f in KEY}}).to_string(index=False))
if LFC is not None:
    for name in (UP_LIKE, DN_LIKE):
        a, b = LFC[SUB[name]].dropna(), LFC[is_nr & ~SUB[name]].dropna()
        print(f"PRO-seq log2FC, {name} vs the other NR enhancers: Mann-Whitney p = {mannwhitneyu(a, b).pvalue:.1e}")

# ----------------------------------------------------------------- 4. delta within NR
print("\n" + "=" * 30, "4. delta within the NR class")
nb_self = NearestNeighbors(n_neighbors=N_NB).fit(Pm).kneighbors(Pm, return_distance=False)   # as in the delta panels
SM = {f: pd.Series(np.median(DELTA[f].to_numpy()[nb_self], axis=1), index=D.index) for f in TRACKS}   # as drawn
rows = []
for f in TRACKS:
    d = DELTA[f][nr]
    row = {"feature": f}
    if LFC is not None:
        ok = LFC[nr].notna()
        row["spearman_delta_vs_log2FC"] = spearmanr(d[ok], LFC[nr][ok])[0]
        row["spearman_delta_vs_log2FC_GR_bound_NR"] = spearmanr(d[ok & grb[nr]], LFC[nr][ok & grb[nr]])[0]
    row["spearman_delta_vs_GC"] = spearmanr(d, D["gc"][nr])[0]
    row["pct_enhancers_abs_smoothed_delta_above_10"] = 100 * (SM[f].abs() > 10).mean()
    rows.append(row)
W = pd.DataFrame(rows)
W.to_csv(FINAL / "S4_S5_delta_within_NR.tsv", sep="\t", index=False, float_format="%.4g")
print(W.round(3).to_string(index=False))
print(f"NR enhancers drawn with delta >= 25 in the panels (smoothed), and how many of them have >= {MIN_NB} UP "
      f"neighbours (all NR: {100 * (n_up[nr] >= MIN_NB).mean():.0f}%):")
for f in ("Pol II", "FOSL2", "EP300", "cJUN", "BCL3"):
    hi = is_nr & (SM[f] >= 25)
    print(f"   {f:7s} n = {int(hi.sum()):4d}; with >= {MIN_NB} UP neighbours: {100 * (n_up[hi] >= MIN_NB).mean():.0f}%")
if LFC is not None:
    ok = LFC[nr].notna()
    print(f"Spearman, number of UP minus number of DN neighbours vs PRO-seq log2FC (NR): "
          f"{spearmanr((n_up - n_dn)[nr][ok], LFC[nr][ok])[0]:.2f}")
print("Saved", ", ".join(str(FINAL / n) for n in ("S4_S5_class_summary.tsv", "S4_S5_neighbourhoods.tsv",
                                                 "S4_S5_nr_subsets.tsv", "S4_S5_delta_within_NR.tsv")))
