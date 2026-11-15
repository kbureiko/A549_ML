# =============================================================================
# Collect the LISA results of the knockdown gene sets (EP300, FOXA1, TEAD4) into one long table
# (input of Supplementary Figure S14D, figures/figS14_knockdown_genes.py)
# Input: the LISA result folders (lisa "multi" runs; each run has two gene sets, gs1 / gs2, and a
#        ChIP-seq and a motif model; the *_cauchy_combine_dedup.csv files are used). They are found by
#        searching the whole project folder, so it does not matter where the run folders were unzipped.
# Output: A6_exports/A6_LISA_all_sets.tsv  (set, knockdown, group, direction, model, TF, pval, rank)
#         A6_exports/A6_LISA_top_per_set.tsv (top 15 TFs per set and model, for checking)
# RUN_MAP assigns the two gene sets of every LISA run (gs1 = KD-unique / shared UP, gs2 = siNON-unique /
# shared DN). p-values are read with exact parsing (see the comment at read_csv).
# Note: gs2 of the two siTEAD4 "Unique" runs is the siNON-unique list of the siFOXA1 comparison, so the
#   TEAD4_siNON_specific_* rows repeat FOXA1_siNON_specific_* until those two LISA jobs are rerun.
# =============================================================================
# %%
from pathlib import Path
import re
import numpy as np
import pandas as pd

from a549_enhancers.paths import PROJECT, KNOCKDOWN   # local folders, set in paths.toml
OUT = KNOCKDOWN / "A6_exports"; OUT.mkdir(parents=True, exist_ok=True)

# run-name prefix -> {gs: (knockdown, group, direction)}
RUN_MAP = {
    "siEP300siNONdexUP_Unique":  {"gs1": ("EP300", "KD_specific", "up"),   "gs2": ("EP300", "siNON_specific", "up")},
    "siEP300siNONdexDN_Unique":  {"gs1": ("EP300", "KD_specific", "down"), "gs2": ("EP300", "siNON_specific", "down")},
    "siEP300_shared_UPvsDN":     {"gs1": ("EP300", "shared", "up"),        "gs2": ("EP300", "shared", "down")},
    "siEP300_silencing":         {"gs1": ("EP300", "basal", "up"),         "gs2": ("EP300", "basal", "down")},
    "siFOXAsiNON_dex_UP_Unique": {"gs1": ("FOXA1", "KD_specific", "up"),   "gs2": ("FOXA1", "siNON_specific", "up")},
    "siFOXAsiNON_dex_DN_Unique": {"gs1": ("FOXA1", "KD_specific", "down"), "gs2": ("FOXA1", "siNON_specific", "down")},
    "siFOXA1_shared_UpvsDN":     {"gs1": ("FOXA1", "shared", "up"),        "gs2": ("FOXA1", "shared", "down")},
    "siTEAD4siNON_dex_UP_Unique": {"gs1": ("TEAD4", "KD_specific", "up"),  "gs2": ("TEAD4", "siNON_specific", "up")},
    "siTEAD4siNON_dex_DN_Unique": {"gs1": ("TEAD4", "KD_specific", "down"), "gs2": ("TEAD4", "siNON_specific", "down")},
    "siTEAD4_shared_UpvsDN":     {"gs1": ("TEAD4", "shared", "up"),        "gs2": ("TEAD4", "shared", "down")},
}
# older non-"Unique" runs (siFOXAsiNON_dex_UP, ..._DN, siTEAD4siNON_dex_UP, ..._DN) are skipped

pat = re.compile(r"^(?P<run>.+?)_\d{4}_\d{2}_\d{2}_[\d.]+__hg38_(?P<gs>gs[12])\.txt_(?P<model>chipseq|motif)_cauchy_combine_dedup\.csv$")
files = sorted(p for p in PROJECT.rglob("*_cauchy_combine_dedup.csv") if "figures" not in p.parts)
if not files:
    raise FileNotFoundError(f"no *_cauchy_combine_dedup.csv files found under {PROJECT} - unzip the LISA results anywhere inside it")
print(f"{len(files)} LISA result files found under {PROJECT}")
rows, skipped = [], set()
if True:
    for f in files:
        m = pat.match(f.name)
        if not m: skipped.add(f.name); continue
        run, gs, model = m["run"], m["gs"], m["model"]
        if run not in RUN_MAP: skipped.add(run); continue
        kd, grp, dirn = RUN_MAP[run][gs]
        # float_precision="round_trip" is required: LISA writes p-values as long decimal strings, and the default
        # pandas parser returns 0 for those below ~1e-16, which made all of them tie at rank 1
        t = pd.read_csv(f, index_col=0, float_precision="round_trip")
        p = pd.to_numeric(t["pval"], errors="coerce").clip(lower=1e-300)
        tf = t["TF"].astype(str).str.upper()
        x = (pd.DataFrame({"TF": tf, "pval": p, "record": t.index.astype(str),
                           "cell_line": t.get("cell_line", t.get("cellline", pd.Series(index=t.index, dtype=str)))})
             .sort_values("pval").drop_duplicates("TF"))                # best record per TF
        # exact ties share the best rank (with exact parsing there should be none at the top; checked below)
        x["rank"] = x.pval.rank(method="min").astype(int)
        x["at_floor"] = x.pval <= 1e-299
        x = x.assign(set=f"{kd}_{grp}_{dirn}", knockdown=kd, group=grp, direction=dirn, model=model,
                     run=run, gs=gs, file=f.name)
        rows.append(x)
found_runs = {r["run"] for r in rows for r in [r.iloc[0]]} if False else set()
for x in rows: found_runs.add(x["run"].iloc[0])
missing_runs = [r for r in RUN_MAP if r not in found_runs]
print(f"runs found: {len(found_runs)} of {len(RUN_MAP)} expected" + (f" | MISSING: {missing_runs}" if missing_runs else ""))
if missing_runs:
    print("  -> the result folders of the missing runs are not under PROJECT (or their names differ from RUN_MAP);"
          " the figure will mark these sets as 'no LISA result'")
lisa = pd.concat(rows, ignore_index=True)
lisa.to_csv(OUT / "A6_LISA_all_sets.tsv", sep="\t", index=False)
top = lisa[lisa["rank"] <= 15].sort_values(["knockdown", "group", "direction", "model", "rank"])
top.to_csv(OUT / "A6_LISA_top_per_set.tsv", sep="\t", index=False)
print("sets x models:", lisa.groupby(["set", "model"]).size().unstack().fillna(0).astype(int).to_string())
ties = lisa[lisa["rank"] == 1].groupby(["set", "model"]).TF.nunique()
print("check - TFs sharing rank 1 per set and model (must all be 1):",
      "OK" if (ties == 1).all() else "\n" + ties[ties > 1].to_string())
print("skipped:", sorted(skipped))
for (s, mdl), g in top[top.model == "motif"].groupby(["set", "model"]):
    print(f"{s:28s} {mdl}: {', '.join(g.TF.head(8))}")
print("Saved:", OUT / "A6_LISA_all_sets.tsv")
