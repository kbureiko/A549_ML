# =============================================================================
# Figure 2 - LIME rules of the three final CatBoost models and signal around the enhancer classes
#   A-C  LIME butterflies: rules favouring each class of a model (definitions: header of scripts/03_lime_rules_final_models.py).
#        Bar = mean contribution of the displayed rule per enhancer of the class; label = rule (quartile bin) and %
#        of the class's enhancers whose explanation contains it with a weight towards the class.
#   D-Q  heatmaps from the deepTools matrices (one lettered panel per track): rows = NR / UP / DN enhancers (NR sorted
#        by EtOH + dex PRO-seq, UP by dex, DN by EtOH), columns = EtOH and dex (or the single untreated track),
#        +-2 kb, 10-bp bins. Colour scale per column = 0 .. 97th percentile of that experiment's signal (EtOH and dex
#        ENCODE tracks are separate experiments; same colour = same rank within the experiment); GR on one shared
#        scale. PRO-seq = plus (magenta) / minus (cyan) strand overlay.
# Inputs: LIME outputs (<date>__lime_final_models/, under PROJECT or ML_ROOT); feature matrix;
#   HEAT/matrix_sorted_<track>.tab.gz; PRO-seq matrices + sorted class BEDs in PRO. Missing tracks are skipped.
# Outputs (FIGURES): Fig_2.pdf/png, Fig_2_<letter>.pdf (one per panel, no letters), Fig_2_key.pdf,
#   Fig_2_lime_dominant_rules.tsv
# =============================================================================
# %%
from pathlib import Path
import gzip, json, math, re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

from a549_enhancers.paths import PROJECT, ML, FIGURES, CHIP_MATRICES, PROSEQ_MATRICES   # local folders, set in paths.toml
HEAT = CHIP_MATRICES                        # ChIP / ATAC matrices (regions: NR_ctrlPlusDex_sorted, UP_sorted_by_dex, DN_sorted_by_ctrl)
PRO = PROSEQ_MATRICES                        # PRO-seq matrices: ctrl/dex x plus/minus + sorted BEDs
ML_ROOT = ML
FM = ML_ROOT / "feature_matrix_all_classes_GRannotated.csv"
FINAL = FIGURES; FINAL.mkdir(parents=True, exist_ok=True)
OUT = FINAL
TRACKS = [                 # (label, matrix file, colormap); order = panel order
    ("PRO-seq", "PROSEQ", None),
    ("GR", HEAT / "matrix_sorted_GR.tab.gz", "Purples"),
    ("POLR2A", HEAT / "matrix_sorted_POL2RA.tab.gz", "RdPu"),
    ("ATAC", HEAT / "matrix_sorted_ATAC.tab.gz", "Blues"),
    ("H3K27ac", HEAT / "matrix_sorted_H3K27ac.tab.gz", "GnBu"),
    ("EP300", HEAT / "matrix_sorted_EP300.tab.gz", "Oranges"),
    ("FOXA1", HEAT / "matrix_sorted_FOXA1.tab.gz", "Reds"),
    ("cJUN", HEAT / "matrix_sorted_cJUN.tab.gz", "Greens"),
    ("BCL3", HEAT / "matrix_sorted_BCL3.tab.gz", "OrRd"),
    ("CREB1", HEAT / "matrix_sorted_CREB1.tab.gz", "BuPu"),
    ("H2A.Z", HEAT / "matrix_sorted_H2AFZ.tab.gz", "Greys"),
    ("TCF12", HEAT / "matrix_sorted_TCF12.tab.gz", "YlOrBr"),
    ("GATA3", HEAT / "matrix_sorted_GATA3.tab.gz", "PuBu"),
    ("TEAD4", HEAT / "matrix_sorted_TEAD4.tab.gz", "PuRd"),
]
PRO_FILES = {("EtOH", "plus"): PRO / "matrix_enh_ctrl_plus.tab.gz", ("EtOH", "minus"): PRO / "matrix_enh_ctrl_minus.tab.gz",
             ("Dex", "plus"): PRO / "matrix_enh_dex_plus.tab.gz", ("Dex", "minus"): PRO / "matrix_enh_dex_minus.tab.gz"}
SORTED_BEDS = {"NR": "NR_ctrlPlusDex_sorted.bed", "UP": "UP_sorted_by_dex.bed", "DN": "DN_sorted_by_ctrl.bed"}
# reference row order of ALL heatmaps: the sorted class BEDs (looked up in HEAT, then in PRO)
PRO_SORTED = {g: (HEAT / f if (HEAT / f).exists() else PRO / f) for g, f in SORTED_BEDS.items()}
ROWS = [["PRO-seq", "GR", "POLR2A", "ATAC", "H3K27ac", "EP300"],        # layout of the earlier figure (Fig. 2 D-Q)
        ["FOXA1", "cJUN", "BCL3", "CREB1", "H2A.Z", "TCF12", "GATA3", "TEAD4"]]
FIRST_LETTER = "D"                             # panel letters continue after the LIME panels A-C
VMAX_PCT = 97                                  # colour scale upper limit (percentile of positive values)
VMAX = {}                                      # per-track override, e.g. {"GR": 8}
SCALE_MODE = "per_condition"                   # "per_condition" (each column 0..VMAX_PCT of its own experiment), "nr_reference" or "shared"
DEX_CAUTION = {}                               # no caution notes in the main figure (see S6 / legend)
SCALE_EXCLUDE = ["GR"]                         # always "shared": GR binding itself is dex-induced, also at sub-threshold (GR-unbound) sites
FLAG_FACTOR = 3.0
GROUP_HEIGHT = {"NR": 3.0, "UP": 1.3, "DN": 0.9}
GROUPS = ["NR", "UP", "DN"]
N_TOP = 20           # features per butterfly (ranked by the sum of both sides' contributions)
MIN_SUPPORT = 0.05   # a side's bar/label is drawn only if the feature supports the class in >= 5% of its enhancers
TASKS = [("nr_vs_up", "NR", "UP", "NR vs UP model"), ("up_vs_dn", "DN", "UP", "UP vs DN model"),
         ("nr_vs_dn", "NR", "DN", "NR vs DN model")]     # (task, left class, right class, title)


def latest_lime_dir():
    hits = [p.parent for root in (PROJECT, ML_ROOT) if root.exists() for p in root.rglob("lime_run_metadata.json")]
    if not hits:
        raise FileNotFoundError("lime_run_metadata.json not found under PROJECT or ML_ROOT - run scripts/03_lime_rules_final_models.py")
    return sorted(hits, key=lambda p: p.stat().st_mtime)[-1]


LIME_DIR = latest_lime_dir()
_missing = [t for t, *_ in TASKS if not (LIME_DIR / t / "lime_summary_by_rule.csv").exists()]
if _missing:
    print("no LIME output yet for:", _missing, "- these panels are left out")
    TASKS = [t for t in TASKS if t[0] not in _missing]
if not FM.exists():
    hits = sorted(p for root in (PROJECT, ML_ROOT) for p in root.rglob("feature_matrix_all_classes_GRannotated.csv"))
    if not hits:
        raise FileNotFoundError(f"{FM} not found")
    FM = hits[-1]
print("LIME folder:", LIME_DIR)

# ----------------------------------------------------------------- shared A549 style
FONT = "Open Sans" if any("Open Sans" in f.name for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
mpl.rcParams.update({
    "font.family": [FONT, "DejaVu Sans"], "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none",
    "axes.grid": False,
})
CLS = {"UP": "#D1495B", "DN": "#3B6FB6", "NR": "#5DAA82"}
CAT_ORDER = ["Dex", "Control", "Genomic"]
CAT_LABEL = {"Dex": "Dex-treated (ChIP, ATAC)", "Control": "Control / untreated", "Genomic": "Sequence / genomic"}
CAT_BAR = {"Dex": "#2B4C7E", "Control": "#9DB9D5", "Genomic": "#3E9651"}
CAT_TXT = {"Dex": "#2B4C7E", "Control": "#222222", "Genomic": "#2F7D3F"}
LEVELS = ["low (Q1, Q1–2)", "intermediate", "high (Q4, Q3–4)"]
STRAND = {"plus": np.array([0.80, 0.10, 0.55]), "minus": np.array([0.05, 0.60, 0.75])}


def panel_letters(fig, axes):
    fig.canvas.draw(); r = fig.canvas.get_renderer(); out = []
    for letter, ax in axes.items():
        bb = ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
        out.append(fig.text(bb.x0 - 0.004, bb.y1 + 0.003, letter, fontsize=11, fontweight="bold",
                            ha="left", va="bottom"))
    return out


def export_panels(fig, panels, stem, letters, pad=0.04):
    for t in letters: t.set_visible(False)
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    for name, artists in panels.items():
        boxes = [a.get_tightbbox(r) if hasattr(a, "get_tightbbox") else a.get_window_extent(r) for a in artists]
        bb = mpl.transforms.Bbox.union(boxes).transformed(fig.dpi_scale_trans.inverted()).padded(pad)
        fig.savefig(OUT / f"{stem}_{name}.pdf", bbox_inches=bb)
    for t in letters: t.set_visible(True)


def category(f):
    if f.startswith("motif_binary::") or f in ("Enh_len", "Distance to TSS", "annotation", "gc"):
        return "Genomic"
    if "_dex_" in f or f == "ATAC_dex_1h":
        return "Dex"
    return "Control"


RENAME = {"JUN": "cJUN", "NR3C1": "GR", "H2AFZ": "H2A.Z", "Enh_len": "Enhancer length",
          "POLR2AphosphoS2": "POLR2A pS2", "Distance to TSS": "TSS distance", "gc": "GC content",
          "annotation": "Annotation"}


MOTIF_RENAME = {"Unknown-ESC-element": "ESC element", "c-Jun-CRE": "cJun-CRE", "RUNX-AML": "RUNX"}


def short(f):
    if f.startswith("motif_binary::"):
        m = f.split("::")[1].split("/")[0].split("(")[0].split(",")[0]
        return MOTIF_RENAME.get(m, m) + " motif"
    if f in RENAME:
        return RENAME[f]
    g = "ATAC" if f.startswith("ATAC") else f.split("_")[0]
    if not g:                                   # feature named only by its ENCODE accession
        m = re.search(r"(ENCFF\w+)", f); g = m.group(1) if m else f
    return RENAME.get(g, g)


# ----------------------------------------------------------------- load LIME summaries
NUM = r"-?\d+(?:\.\d+)?(?:e-?\d+)?"


def parse(rule):
    """-> (feature, lo, hi, level) ; level only for categorical rules 'feature=level'"""
    m = re.match(rf"^({NUM}) < (.+) <= ({NUM})$", rule)
    if m: return m.group(2), float(m.group(1)), float(m.group(3)), None
    m = re.match(rf"^(.+) <= ({NUM})$", rule)
    if m: return m.group(1), -np.inf, float(m.group(2)), None
    m = re.match(rf"^(.+) > ({NUM})$", rule)
    if m: return m.group(1), float(m.group(2)), np.inf, None
    m = re.match(r"^(.+?)=(.*)$", rule)
    if m:
        lev = m.group(2)
        if lev in ("0", "1"):
            v = float(lev); return m.group(1), v - 0.5, v, lev     # binary: absent (-0.5, 0] / present (0.5, 1]
        return m.group(1), np.nan, np.nan, lev
    raise ValueError(f"cannot parse rule: {rule}")


meta = json.load(open(LIME_DIR / "lime_run_metadata.json", encoding="utf-8"))
rules, feats, stab, n_expl = [], [], [], {}
for task, left, right, _ in TASKS:
    d = LIME_DIR / task
    r = pd.read_csv(d / "lime_summary_by_rule.csv"); f = pd.read_csv(d / "lime_summary_by_feature.csv")
    rules.append(r); feats.append(f)
    n_expl[task] = f.groupby("side").n_class.first().to_dict()          # explained enhancers per class
    if (d / "lime_stability.csv").exists():
        stab.append(pd.read_csv(d / "lime_stability.csv"))
rules = pd.concat(rules, ignore_index=True); feats = pd.concat(feats, ignore_index=True)
rules[["feature", "lo", "hi", "level"]] = pd.DataFrame(rules.rule.map(parse).tolist(), index=rules.index)
rules["fcat"] = rules.feature.map(category); feats["fcat"] = feats.feature.map(category)
print("explained enhancers:", n_expl)
if stab:
    stab = pd.concat(stab, ignore_index=True)
    print("stability (2nd seed): Spearman of feature contributions / top-20 overlap:",
          [(f"{s.task}:{s.side}", round(s.spearman_all_features, 2), int(s.top20_overlap)) for s in stab.itertuples()])

fm = pd.read_csv(FM, low_memory=False)
missing_f = sorted(set(rules.feature) - set(fm.columns))
if missing_f:
    print("features in LIME rules missing from the feature matrix (ignored in D-E):", missing_f[:10])


# ----------------------------------------------------------------- dominant rule per feature and side
def dominant(df):
    """supporting bins of one feature for one side -> dominant contiguous block (merged range + quartile tag)"""
    task, side, f = df.name
    df = df[df.expected_contribution > 0]
    if df.empty:
        return None
    if df.level.notna().any():                            # categorical feature: best level
        r = df.sort_values("expected_contribution").iloc[-1]
        lab = ("present" if r.level == "1" else "absent") if f.startswith("motif_binary::") else str(r.level)
        return pd.Series({"w": r.expected_contribution, "frac": r.frac_support, "lo": r.lo, "hi": r.hi,
                          "level": r.level, "label": lab, "qtag": "", "qbin": np.nan, "n_bins": 1})
    df = df.sort_values("lo")
    blocks, cur = [], [df.index[0]]
    for i0, i1 in zip(df.index[:-1], df.index[1:]):
        (cur.append(i1) if np.isclose(df.at[i0, "hi"], df.at[i1, "lo"]) else (blocks.append(cur), cur := [i1]))
    blocks.append(cur)
    best = max(blocks, key=lambda b: df.loc[b, "expected_contribution"].sum())
    b = df.loc[best]
    lo, hi = b.lo.min(), b.hi.max()
    lab = f"≤ {hi:.3g}" if np.isinf(lo) else f"> {lo:.3g}" if np.isinf(hi) else f"{lo:.3g}–{hi:.3g}"
    q0, q1 = int(b.bin.min()) + 1, int(b.bin.max()) + 1
    qtag = f"Q{q0}" if q0 == q1 else f"Q{q0}–{q1}"
    return pd.Series({"w": b.expected_contribution.sum(), "frac": b.frac_support.sum(), "lo": lo, "hi": hi,
                      "level": None, "label": lab, "qtag": qtag, "qbin": (q0 + q1) / 2, "n_bins": len(b)})


dom = (rules.groupby(["task", "side", "feature"], group_keys=True).apply(dominant, include_groups=False)
       .dropna(how="all").reset_index())
dom["fcat"] = dom.feature.map(category)
dom["level_q"] = pd.cut(dom.qbin, [0, 1.5, 3.49, 5], labels=LEVELS).astype(object)   # midpoint of the merged bins
dom.loc[dom.fcat == "Genomic", "level_q"] = np.nan
dom.to_csv(FINAL / "Fig_2_lime_dominant_rules.tsv", sep="\t", index=False)


# ----------------------------------------------------------------- A-C: butterflies
def cond_suffix(f):
    c = category(f)
    return " (dex)" if c == "Dex" else (" (EtOH)" if "_eth_" in f else " (untr.)" if c == "Control" else "")


def row_names(feats_):
    """short names; the condition (dex-treated vs control) is given by the label colour only"""
    return [short(f) for f in feats_]


def butterfly(fig, spec, task, left, right, title):
    d = dom[dom.task == task]
    L = d[d.side == left].set_index("feature"); R = d[d.side == right].set_index("feature")
    tot = L.w.reindex(L.index.union(R.index)).fillna(0) + R.w.reindex(L.index.union(R.index)).fillna(0)
    feats_ = list(tot.sort_values(ascending=False).head(N_TOP).index)
    names = row_names(feats_)
    n = len(feats_)
    sub = spec.subgridspec(1, 3, width_ratios=[1, 1.0, 1], wspace=0.0)
    axL, axN, axR = (fig.add_subplot(sub[0, k]) for k in range(3))
    xmax = max(L.w.max(), R.w.max()) * 2.0
    for ax, T, cls, sgn in ((axL, L, left, -1), (axR, R, right, 1)):
        for i, f in enumerate(feats_):
            if f not in T.index or T.at[f, "frac"] < MIN_SUPPORT: continue
            r = T.loc[f]
            ax.barh(i, r.w, color=CAT_BAR[r.fcat], height=0.7, zorder=2)
            txt = (r.qtag if r.qtag else r.label) + f" · {r.frac * 100:.0f}%"
            ax.text(r.w + xmax * 0.03, i, txt, va="center", ha="left" if sgn > 0 else "right", fontsize=5.2,
                    color="#444444")
        ax.set_ylim(n - 0.4, -0.7); ax.set_yticks([]); ax.set_xlim(0, xmax)
        if sgn < 0:
            ax.invert_xaxis(); ax.spines["left"].set_visible(False); ax.spines["right"].set_visible(True)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
        ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%g"))
        ax.tick_params(axis="x", labelsize=5.6)
        ax.set_title(f"favours {cls}", color=CLS[cls], fontsize=7, fontweight="bold", pad=2)
    axN.axis("off"); axN.set_ylim(n - 0.4, -0.7); axN.set_xlim(0, 1)
    for i, (f, nm) in enumerate(zip(feats_, names)):
        axN.text(0.5, i, nm, ha="center", va="center", fontsize=5.4 if len(nm) <= 13 else (4.9 if len(nm) <= 16 else 4.4),
                 color=CAT_TXT[category(f)])
    ne = n_expl[task]
    axN.set_title(f"{title}\n({left}: {ne[left]:,}, {right}: {ne[right]:,} enhancers)", fontsize=7.2, pad=10,
                  linespacing=1.15)
    axR.set_xlabel("LIME weight per enhancer", fontsize=6.2, labelpad=2)
    axR.xaxis.set_label_coords(-0.45, -0.06)
    return axN, [axL, axN, axR]


# ----------------------------------------------------------------- data (heatmaps)
def read_matrix(f):
    with gzip.open(f, "rt") as fh:
        h = json.loads(fh.readline()[1:])
    d = pd.read_csv(f, sep="\t", header=None, skiprows=1)
    m = d.iloc[:, 6:].to_numpy(dtype=float)
    sb, gb = h["sample_boundaries"], h["group_boundaries"]
    samples, labs = {}, h["sample_labels"]
    if len(labs) == 2:            # control first, dex second (computeMatrix sample order); checked on the labels
        assert "dex" not in labs[0].lower() and "dex" in labs[1].lower() or "dex" not in "".join(labs).lower(), \
            f"{f}: expected samples in the order control, dex - got {labs}"
        conds = ["EtOH", "Dex"]
    elif len(labs) == 1:
        conds = ["Dex" if "dex" in labs[0].lower() else "ctrl"]
    else:
        raise ValueError(f"{f}: {len(labs)} samples; expected 1 (control) or 2 (control, dex)")
    for k, cond in enumerate(conds):
        samples[cond] = m[:, sb[k]:sb[k + 1]]
    glabs = [l.split("_")[0].replace("A549", "").strip("_") for l in h["group_labels"]]
    groups = dict(zip(GROUPS, [slice(gb[i], gb[i + 1]) for i in range(3)]))
    assert len(gb) == 4, f"{f}: expected 3 region groups (NR, UP, DN), got {h['group_labels']}"
    if glabs[:3] != GROUPS:
        print(f"  note: {Path(f).name} group labels are {h['group_labels']} - taken as NR, UP, DN in this order")
    return samples, groups, d[3].to_numpy()


def reorder_by_beds(f, samples):
    """rows of every matrix in the order of the sorted class BEDs (matched on chrom, start, end)"""
    d = pd.read_csv(f, sep="\t", header=None, skiprows=1, usecols=[0, 1, 2])
    key = {k: i for i, k in enumerate(zip(d[0], d[1], d[2]))}
    order, groups, pos = [], {}, 0
    coords = list(zip(d[0], d[1], d[2]))
    for gname in GROUPS:
        b = pd.read_csv(PRO_SORTED[gname], sep="\t", header=None, usecols=[0, 1, 2])
        idx = [key.get(k) for k in zip(b[0], b[1], b[2])]
        miss = sum(i is None for i in idx)
        if miss:
            raise ValueError(f"{Path(f).name}: {miss} regions of {PRO_SORTED[gname].name} not found in the matrix")
        order += idx; groups[gname] = slice(pos, pos + len(idx)); pos += len(idx)
    return {c: m[order] for c, m in samples.items()}, groups, [f"{coords[i][0]}_{coords[i][1]}_{coords[i][2]}" for i in order]


_st = sorted(p for p in PROJECT.rglob("MS_chromhmm_enhancer_states.tsv"))
GRB = None
if _st:
    _t = pd.read_csv(_st[-1], sep="\t")
    GRB = dict(zip(_t.enhancer_id, _t.GR_bound.astype(str).str.lower().eq("true")))
elif SCALE_MODE == "nr_reference":
    print("  WARNING: MS_chromhmm_enhancer_states.tsv not found - reference = all NR enhancers (GR-bound included)")

heat, missing = {}, []
for label, f, cmap in TRACKS:
    if f == "PROSEQ":
        if not all(p.exists() for p in [*PRO_FILES.values(), *PRO_SORTED.values()]):
            missing.append(label); continue
        mats, ids0 = {}, None
        for key, p in PRO_FILES.items():
            s, g, ids = read_matrix(p)
            mats[key] = next(iter(s.values()))
            ids0 = ids if ids0 is None else ids0
            assert (ids == ids0).all(), "PRO-seq matrices are not in the same region order"
        row_of = {rid: i for i, rid in enumerate(ids0)}
        order = {c: [row_of[r] for r in pd.read_csv(p, sep="\t", header=None)[3]] for c, p in PRO_SORTED.items()}
        vmax = {s: float(np.ceil(np.percentile(np.concatenate([mats[(c, s)].ravel() for c in ("EtOH", "Dex")])
                                               [np.concatenate([mats[(c, s)].ravel() for c in ("EtOH", "Dex")]) > 0], 95)))
                for s in ("plus", "minus")}
        heat[label] = {"proseq": True, "mats": mats, "order": order, "vmax": vmax, "conds": ["EtOH", "Dex"]}
    elif Path(f).exists():
        s, g, _ = read_matrix(f)
        s, g, ids = reorder_by_beds(f, s)
        conds = [c for c in ("EtOH", "Dex", "ctrl") if c in s]
        mode = SCALE_MODE if (len(conds) == 2 and label not in SCALE_EXCLUDE) else "shared"
        factor = 1.0
        if mode == "nr_reference":
            nr_rows = np.arange(g["NR"].start, g["NR"].stop)
            if GRB is not None:
                unb = np.array([GRB.get(ids[i]) is False for i in nr_rows])
                if unb.sum() < 100:
                    print(f"  {label}: only {unb.sum()} GR-unbound NR rows matched by coordinates - using all NR")
                else:
                    nr_rows = nr_rows[unb]
            factor = float(np.nanmean(s["EtOH"][nr_rows]) / np.nanmean(s["Dex"][nr_rows]))
            s = {**s, "Dex": s["Dex"] * factor}
        if mode == "per_condition":
            vmax = {c: VMAX.get(label, float(np.percentile(s[c][s[c] > 0], VMAX_PCT))) for c in conds}
        else:
            allv = np.concatenate([v.ravel() for v in s.values()])
            vmax = VMAX.get(label, float(np.percentile(allv[allv > 0], VMAX_PCT)))
        if mode == "nr_reference":
            flag = "  <-- dex has far less signal; pattern less reliable" if factor > FLAG_FACTOR or factor < 1 / FLAG_FACTOR else ""
            print(f"  {label}: dex x {factor:.2f} (reference: {len(nr_rows):,} NR rows){flag}")
        heat[label] = {"proseq": False, "samples": s, "groups": g, "vmax": vmax, "cmap": cmap, "conds": conds,
                       "mode": mode, "factor": factor}
    else:
        missing.append(label)
if missing:
    print("matrices not found - panels skipped:", missing)
tracks = [t for t in TRACKS if t[0] in heat]
print(f"font: {FONT} | tracks: {[t[0] for t in tracks]}")


def blend(plus, minus, vmax):
    p = np.clip(plus / vmax["plus"], 0, 1)[..., None]; m = np.clip(minus / vmax["minus"], 0, 1)[..., None]
    return np.clip(1 - p * (1 - STRAND["plus"]) - m * (1 - STRAND["minus"]), 0, 1)


# ----------------------------------------------------------------- one track
def plot_track(fig, spec, label, first_in_row):
    H = heat[label]; conds = H["conds"]
    inner = spec.subgridspec(3, len(conds), height_ratios=[GROUP_HEIGHT[g] for g in GROUPS], hspace=0.05,
                             wspace=0.05)
    axes = []
    for c, cond in enumerate(conds):
        for i, g in enumerate(GROUPS):
            ax = fig.add_subplot(inner[i, c]); axes.append(ax)
            if H["proseq"]:
                rows = H["order"][g]
                sub = blend(H["mats"][(cond, "plus")][rows], H["mats"][(cond, "minus")][rows], H["vmax"])
                ax.imshow(sub, aspect="auto", interpolation="antialiased")
                nreg, nb = len(rows), sub.shape[1]
            else:
                sub = H["samples"][cond][H["groups"][g]]
                vm = H["vmax"][cond] if isinstance(H["vmax"], dict) else H["vmax"]
                ax.imshow(sub, aspect="auto", cmap=H["cmap"], vmin=0, vmax=vm, interpolation="antialiased")
                nreg, nb = sub.shape
            ax.set_xticks([]); ax.set_yticks([])
            for s_ in ax.spines.values():
                s_.set_visible(True); s_.set_linewidth(0.4); s_.set_color("#777777")
            if i == 0:
                ax.set_title({"ctrl": "untreated"}.get(cond, cond), fontsize=6.5, pad=2)
            if first_in_row and c == 0:
                ax.set_ylabel(f"{g}\n{nreg:,}", rotation=0, ha="right", va="center", fontsize=6.5, color=CLS[g],
                              fontweight="bold", labelpad=3)
            if i == 2:
                ax.set_xticks([0, nb / 2, nb - 1]); ax.set_xticklabels(["", "0", ""], fontsize=5.6)
                ax.tick_params(length=1.5, pad=1)
    tax = fig.add_subplot(spec); tax.axis("off"); axes.insert(0, tax)   # first: letter goes above the title
    tax.set_title(label, fontsize=8, fontweight="bold", pad=13)
    pos = tax.get_position()
    if H["proseq"]:
        w = pos.width * 0.36
        for k, s in enumerate(["plus", "minus"]):
            cax = fig.add_axes([pos.x0 + pos.width * 0.1 + k * (w + pos.width * 0.08), pos.y0 - 0.33 / fig.get_figheight(), w, 0.06 / fig.get_figheight()]); axes.append(cax)
            grad = np.linspace(0, 1, 256)[None, :, None]
            cax.imshow(np.clip(1 - grad * (1 - STRAND[s]), 0, 1), aspect="auto", extent=(0, H["vmax"][s], 0, 1))
            cax.set_yticks([]); cax.set_xticks([0, H["vmax"][s]]); cax.set_xticklabels(["0", f"{H['vmax'][s]:g}"], fontsize=5.5)
            cax.tick_params(length=1.5, pad=1)
            for sp in cax.spines.values(): sp.set_linewidth(0.4)
            cax.set_title(s, fontsize=4.8, pad=1.5)
    else:
        fmt = lambda v: f"{v:.0f}" if v >= 2 else f"{v:.1f}"
        vm = H["vmax"] if isinstance(H["vmax"], dict) else {c: H["vmax"] for c in conds}
        bars = conds if H.get("mode") == "per_condition" else [None]
        w = pos.width * (0.5 if bars == [None] else 0.34)
        for k, c in enumerate(bars):
            x0 = pos.x0 + (pos.width * 0.25 if c is None else pos.width * 0.1 + k * (w + pos.width * 0.13))
            cax = fig.add_axes([x0, pos.y0 - 0.33 / fig.get_figheight(), w, 0.06 / fig.get_figheight()]); axes.append(cax)
            v = vm[c if c is not None else conds[0]]
            cb = fig.colorbar(mpl.cm.ScalarMappable(mpl.colors.Normalize(0, v), H["cmap"]), cax=cax,
                              orientation="horizontal")
            cb.set_ticks([0, v]); cb.set_ticklabels(["0", fmt(v)])
            cb.ax.tick_params(labelsize=5.5, length=1.5, pad=1); cb.outline.set_linewidth(0.4)
            if c is not None:
                cax.set_title({"ctrl": "untreated"}.get(c, c), fontsize=4.8, pad=1.5)
        if label in DEX_CAUTION and "Dex" in conds:
            cax.text(0.5, -3.2, f"dex: ρ = {DEX_CAUTION[label]:.2f} with control", transform=cax.transAxes,
                     ha="center", va="top", fontsize=4.8, color="#B03A2E")
        if H.get("mode") == "nr_reference" and abs(H["factor"] - 1) > 0.05:
            cax.text(0.5, -3.2, f"dex × {H['factor']:.2f}", transform=cax.transAxes, ha="center",
                     va="top", fontsize=4.8, color="#B03A2E" if (H["factor"] > FLAG_FACTOR or H["factor"] < 1 / FLAG_FACTOR) else "#555555")
    return axes


# ----------------------------------------------------------------- assemble
rows = [[t for t in row if t in heat] for row in ROWS]
rows = [r for r in rows if r]
extra = [t[0] for t in tracks if not any(t[0] in r for r in rows)]
if extra:
    rows.append(extra)                         # tracks not listed in ROWS go to a last row
fig = plt.figure(figsize=(7.2, 4.8 + 2.9 * len(rows)))
hr = [4.3, 0.25]
for r_ in range(len(rows)):
    hr += [2.7] + ([0.12] if r_ < len(rows) - 1 else [])   # spacer below each heatmap row for its colour bars
outer = fig.add_gridspec(len(hr), 1, height_ratios=hr, hspace=0.42,
                         left=0.08, right=0.99, top=0.975, bottom=0.04)
parts, lead = {}, {}
letters_ = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
row1 = outer[0].subgridspec(1, len(TASKS), wspace=0.16)
for k, t in enumerate(TASKS):
    lead[letters_[k]], parts[letters_[k]] = butterfly(fig, row1[k], *t[:3], t[3])
kax = fig.add_subplot(outer[1]); kax.axis("off")
h = [mpl.patches.Patch(color=CAT_BAR[c], label=CAT_LABEL[c]) for c in CAT_ORDER]
lg = kax.legend(handles=h, loc="center", ncol=3, frameon=False, fontsize=6.6, handlelength=1.2, columnspacing=1.6,
                title="Feature category (bar and label colour)", title_fontsize=6.6)
for tx, c in zip(lg.get_texts(), CAT_ORDER):
    tx.set_color(CAT_TXT[c])
parts["key"] = [kax]
k = letters_.index(FIRST_LETTER)
for r, row in enumerate(rows):
    widths = [len(heat[t]["conds"]) for t in row]                # paired tracks twice as wide as single ones
    sub = outer[2 + 2 * r].subgridspec(1, len(row), width_ratios=widths, wspace=0.3 if max(widths) == 2 else 0.25)
    for c, label in enumerate(row):
        axes = plot_track(fig, sub[0, c], label, first_in_row=(c == 0))
        parts[letters_[k]] = axes; lead[letters_[k]] = axes[0]; k += 1
        if c == 0:
            fig.text(axes[0].get_position().x0 - 0.065, axes[-1].get_position().y0 - 0.004,
                     "±2 kb around\nenhancer centre", fontsize=5.4, color="#555555", va="top")
letters = panel_letters(fig, {l: a for l, a in lead.items()})
for ext in ("pdf", "png"):
    fig.savefig(FINAL / f"Fig_2.{ext}", dpi=300, bbox_inches="tight")
export_panels(fig, parts, "Fig_2", letters)
plt.show()
print("Saved to", FINAL / "Fig_2.pdf", "+ per-panel PDFs")
