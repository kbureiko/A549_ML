"""Supplementary Table S1 - ENCODE datasets used as signal features and other ENCODE files.

The table is built from the column names of the feature matrix, which carry the ENCODE target, the treatment
and the file accession (for example "H3K27ac_dex_100_1 hour_ENCFF530RMD"; no treatment = untreated cells).
The two ATAC-seq tracks have no accession in their column name: fill ATAC_ACCESSIONS below.

ENCODE metadata (experiment accession, output type, assembly, treatment, project, replicates, release date) is
added from a metadata.tsv downloaded from the ENCODE portal (--metadata FILE; "Download" on a file search writes
it). Files that are not in that table (ATAC-seq, peak files) can be looked up with --fetch, which asks the
portal directly and needs internet access. Both are optional; without them those columns are empty.

Input:  <run folder>/summaries/feature_importance_stability_summary.csv of a main run (the features the models used);
        without a run folder: <ml_outputs>/feature_matrix_all_classes_GRannotated.csv (scripts/08)
Output: <figures>/Supp_Table_S1_ENCODE_datasets.xlsx and .tsv

Run:  uv run python scripts/16_supp_table_encode_datasets.py --metadata /path/to/metadata.tsv
      (add --fetch to look up the files that the metadata table does not contain)
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request

import pandas as pd

from a549_enhancers.paths import FIGURES, ML, RUNS   # local folders, set in paths.toml

# ENCODE file accessions of the two ATAC-seq signal tracks (not part of the column names) - fill in.
ATAC_ACCESSIONS = {"ATAC_ctrl": "ENCFF214BCI", "ATAC_dex_1h": "ENCFF255ZEC"}
# Experiment accession, used when the file accession above is still empty (dex 0 h = control).
ATAC_EXPERIMENTS = {"ATAC_ctrl": "ENCSR220ASC"}

# ENCODE files used outside the feature matrix: (accession, target, treatment, use)
OTHER_FILES = [
    ("ENCFF395FIP", "NR3C1", "see ENCODE", "GR ChIP-seq peaks; definition of GR-bound enhancers"),
    ("ENCFF951EGL", "NR3C1", "EtOH", "Supp. Figure S7D (signal at siEP300 eRNA regions)"),
    ("ENCFF090XQU", "NR3C1", "dex", "Supp. Figure S7D (signal at siEP300 eRNA regions)"),
    ("ENCFF788DYT", "EP300", "EtOH", "Supp. Figure S7D (signal at siEP300 eRNA regions)"),
    ("ENCFF876TGI", "EP300", "dex", "Supp. Figure S7D (signal at siEP300 eRNA regions)"),
]

NOT_SIGNAL = {"enhancer_id", "class", "annotation", "Distance to TSS", "Enh_len", "gc", "GR_bound", "GRE_score",
              "chr", "start", "end"}
TREATMENT = {
    "dex": ("100 nM dexamethasone", "1 hour", "dex"),
    "eth": ("0.02% ethanol", "1 hour", "control"),
}


def find_matrix():
    for cand in [ML / "feature_matrix_all_classes_GRannotated.csv",
                 ML / "full_feature_matrix" / "feature_matrix_all_classes.csv"]:
        if cand.exists():
            return cand
    hits = sorted(ML.glob("**/feature_matrix_all_classes*.csv"))
    if not hits:
        raise FileNotFoundError(f"No feature matrix under {ML}. Run scripts/07 (and 08) first.")
    return hits[0]


def model_features():
    """Features of the models, from the importance summary of a main run (the list the models were trained on)."""
    for task in ("nr_vs_up", "up_vs_dn", "nr_vs_dn"):
        if task in RUNS:
            f = RUNS[task] / "summaries" / "feature_importance_stability_summary.csv"
            if f.exists():
                return pd.read_csv(f, usecols=["feature"])["feature"].drop_duplicates().tolist(), f.parent.parent
    return None, None


def describe(feature: str) -> dict:
    row = {"feature": feature, "target": "", "assay": "", "treatment": "", "duration": "", "condition": "",
           "ENCODE_file": ""}
    if feature in ATAC_ACCESSIONS:
        dex = feature == "ATAC_dex_1h"
        row.update(target="chromatin accessibility", assay="ATAC-seq",
                   treatment="100 nM dexamethasone",
                   duration="1 hour" if dex else "0 hours", condition="dex" if dex else "control",
                   ENCODE_file=ATAC_ACCESSIONS[feature])
        return row
    m = re.match(r"^(?P<target>[^_]+)(?:_(?P<trt>dex|eth)_(?P<dose>[0-9.]+)_(?P<dur>[^_]+))?_(?P<acc>ENCFF[0-9A-Z]{6}(?:_ENCFF[0-9A-Z]{6})*)(?:_(?P<tag>CRISPR))?$", feature)
    if not m:
        raise ValueError(f"Unexpected feature name: {feature!r}")
    row["target"] = m["target"] + (" (CRISPR-tagged)" if m["tag"] else "")   # name ends in _CRISPR: tagged protein
    row["assay"] = "Histone ChIP-seq" if re.match(r"^H[234]", m["target"]) else "TF ChIP-seq"
    row["ENCODE_file"] = m["acc"]          # "ENCFF..._ENCFF..." = mean of two tracks; split into rows in main()
    if m["trt"]:
        row["treatment"], _, row["condition"] = TREATMENT[m["trt"]]
        row["duration"] = m["dur"]
    else:
        row["treatment"], row["condition"] = "untreated", "control"
    return row


def fetch(accession: str) -> dict:
    """Metadata of one ENCODE file; empty values when the portal cannot be reached."""
    empty = {"ENCODE_experiment": "", "output_type": "", "file_format": "", "assembly": "", "lab": "",
             "biosample_summary": ""}
    if not accession:
        return empty
    url = f"https://www.encodeproject.org/files/{accession}/?format=json"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "a549-enhancers"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            j = json.load(resp)
    except Exception as err:   # no network, unknown accession, portal down
        print(f"  {accession}: not fetched ({err})")
        return empty

    def name(v):   # "/labs/tim-reddy/" -> "tim-reddy"; {"title": ...} -> title
        if isinstance(v, dict):
            return str(v.get("title") or v.get("accession") or v.get("@id") or "")
        return str(v or "").strip("/").split("/")[-1]

    return {"ENCODE_experiment": name(j.get("dataset")), "output_type": str(j.get("output_type", "")),
            "file_format": str(j.get("file_format", "")), "assembly": str(j.get("assembly", "")),
            "lab": name(j.get("lab")), "biosample_summary": str(j.get("simple_biosample_summary", ""))}


META_COLUMNS = {   # ENCODE metadata.tsv column -> column of the table
    "Experiment accession": "ENCODE_experiment", "Output type": "output_type", "File format": "file_format",
    "File assembly": "assembly", "Biosample treatments": "ENCODE_treatment",
    "Biosample treatments amount": "ENCODE_treatment_amount", "Biosample treatments duration": "ENCODE_treatment_duration",
    "Project": "project", "Biological replicate(s)": "biological_replicates",
    "Experiment date released": "date_released", "File analysis title": "analysis",
}
META_OUT = list(META_COLUMNS.values()) + ["lab", "biosample_summary"]


def read_metadata(path) -> pd.DataFrame:
    """ENCODE metadata.tsv -> one row per file accession with the columns of META_COLUMNS."""
    m = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    missing = [c for c in ["File accession", *META_COLUMNS] if c not in m.columns]
    if missing:
        raise ValueError(f"{path}: columns not found: {missing}")
    m = m.drop_duplicates("File accession").set_index("File accession")
    return m[list(META_COLUMNS)].rename(columns=META_COLUMNS)


README = [
    ("Supplementary Table S1", "ENCODE datasets used in this study."),
    ("feature", "Column name in the feature matrix (as in Supplementary Table S2)."),
    ("target", "ENCODE target (NR3C1 = GR, POLR2A = Pol II, H2AFZ = H2A.Z, JUN = cJUN)."),
    ("assay", "TF ChIP-seq (transcription factors and coregulators), Histone ChIP-seq or ATAC-seq."),
    ("treatment, duration", "Treatment of the A549 cells as annotated by ENCODE; untreated = no treatment annotated."),
    ("condition", "control (EtOH-treated or untreated) or dex, as used in the text."),
    ("ENCODE_file", "ENCODE file accession of the signal p-value bigWig."),
    ("ENCODE_experiment, output_type, file_format, assembly", "From the ENCODE portal."),
    ("ENCODE_treatment, _amount, _duration", "Treatment as annotated by ENCODE (empty = no treatment annotated)."),
    ("project", "ENCODE project of the experiment: ENCODE, or GGR (Genomics of Gene Regulation, the dexamethasone time course)."),
    ("biological_replicates, date_released, analysis", "Replicates pooled in the file, release date of the experiment and ENCODE processing pipeline version."),
    ("use", "model feature, or the analysis in which a file outside the feature matrix was used."),
    ("Mean of two files", "A feature with two file accessions in its name (CREB1 control) is the mean of two experiments; "
                          "each file has its own row."),
    ("Time courses", "The ChIP-seq tracks of the dexamethasone time courses are listed in Supplementary Table S4."),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", help="metadata.tsv downloaded from the ENCODE portal")
    ap.add_argument("--fetch", action="store_true",
                    help="ask the ENCODE portal for files that are not in --metadata (needs internet)")
    args = ap.parse_args()

    cols, matrix = model_features()
    if cols is None:                      # no run folder: fall back to the saved matrix (may hold tracks the models do not use)
        matrix = find_matrix()
        cols = pd.read_csv(matrix, nrows=0).columns
        print(f"NOTE: no run summary found - features read from {matrix.name}, which can contain tracks that are not model features")
    signal = [c for c in cols if not c.startswith("motif") and c not in NOT_SIGNAL and not c.startswith("nbr_")]
    tab = pd.DataFrame([describe(c) for c in signal])
    tab["ENCODE_file"] = tab["ENCODE_file"].str.split("_")     # a feature that is the mean of two tracks: one row per file
    tab = tab.explode("ENCODE_file", ignore_index=True)
    tab["use"] = "model feature"
    other = pd.DataFrame([{"feature": "", "target": t, "assay": "TF ChIP-seq", "treatment": trt, "duration": "",
                           "condition": "", "ENCODE_file": acc, "use": use} for acc, t, trt, use in OTHER_FILES])
    # a file that is also a model feature is listed once, with both uses
    dup = other["ENCODE_file"].isin(tab["ENCODE_file"])
    for _, r in other[dup].iterrows():
        tab.loc[tab["ENCODE_file"] == r["ENCODE_file"], "use"] += "; " + r["use"]
    tab = pd.concat([tab, other[~dup]], ignore_index=True)

    for c in META_OUT:
        tab[c] = ""
    if args.metadata:
        meta = read_metadata(args.metadata)
        hit = tab["ENCODE_file"].isin(meta.index)
        for c in META_COLUMNS.values():
            tab.loc[hit, c] = tab.loc[hit, "ENCODE_file"].map(meta[c])
        print(f"{args.metadata}: {int(hit.sum())} of {int(tab['ENCODE_file'].astype(bool).sum())} files matched")
        # the treatment in the column name must agree with the ENCODE annotation
        feat = tab[hit & tab["use"].str.startswith("model feature")]
        name_trt = feat["treatment"].map({"100 nM dexamethasone": "dexamethasone", "0.02% ethanol": "ethanol", "untreated": ""})
        bad = feat[(name_trt != feat["ENCODE_treatment"]) |
                   (feat["duration"].ne("") & feat["duration"].ne(feat["ENCODE_treatment_duration"]))]
        print("treatment in the feature name differs from ENCODE:",
              "none" if bad.empty else bad[["feature", "ENCODE_treatment", "ENCODE_treatment_amount",
                                            "ENCODE_treatment_duration"]].to_string(index=False))
        wrong = feat[(feat["output_type"] != "signal p-value") | (feat["assembly"] != "GRCh38")]
        print("not a GRCh38 signal p-value file:", "none" if wrong.empty else wrong["ENCODE_file"].tolist())
    if args.fetch:
        todo = tab.index[tab["ENCODE_file"].astype(bool) & (tab["ENCODE_experiment"] == "")]
        print(f"asking the ENCODE portal for {len(todo)} files ...")
        for i in todo:
            got = fetch(tab.at[i, "ENCODE_file"])
            for c, v in got.items():
                if v:
                    tab.at[i, c] = v
            time.sleep(0.2)
    for feature, exp in ATAC_EXPERIMENTS.items():   # known experiment for a track without ENCODE metadata
        hit = (tab["feature"] == feature) & (tab["ENCODE_experiment"] == "")
        tab.loc[hit, "ENCODE_experiment"] = exp
    meta_cols = META_OUT
    order = {"dex": 1, "control": 0, "": 2}
    tab["_o"] = tab["use"].str.startswith("model feature").map({True: 0, False: 1})
    tab["_c"] = tab["condition"].map(order)
    tab = tab.sort_values(["_o", "assay", "target", "_c", "ENCODE_file"], kind="mergesort").drop(columns=["_o", "_c"])
    tab = tab[["feature", "target", "assay", "condition", "treatment", "duration", "ENCODE_file",
               *meta_cols, "use"]].reset_index(drop=True)

    feat = tab[tab["use"].str.startswith("model feature")]
    print(f"{matrix.name}: {feat['feature'].nunique()} signal features from {len(feat)} files "
          f"({(feat['condition'] == 'control').sum()} control: {(feat['treatment'] == '0.02% ethanol').sum()} EtOH, "
          f"{(feat['treatment'] == 'untreated').sum()} untreated, "
          f"{(feat['assay'].eq('ATAC-seq') & feat['condition'].eq('control')).sum()} ATAC; "
          f"{(feat['condition'] == 'dex').sum()} dex), {feat['target'].nunique()} targets; "
          f"dex durations: {sorted(set(feat.loc[feat['condition'] == 'dex', 'duration']))}")
    missing = feat.loc[feat["ENCODE_file"] == "", "feature"].tolist()
    if missing:
        print("no accession yet (fill ATAC_ACCESSIONS at the top of the script):", missing)

    FIGURES.mkdir(parents=True, exist_ok=True)
    tab.to_csv(FIGURES / "Supp_Table_S1_ENCODE_datasets.tsv", sep="\t", index=False)
    try:
        with pd.ExcelWriter(FIGURES / "Supp_Table_S1_ENCODE_datasets.xlsx", engine="openpyxl") as writer:
            pd.DataFrame(README, columns=["Item", "Description"]).to_excel(writer, sheet_name="README", index=False)
            tab.to_excel(writer, sheet_name="ENCODE datasets", index=False, freeze_panes=(1, 1))
        print("written: Supp_Table_S1_ENCODE_datasets.xlsx and .tsv in the figures folder")
    except ImportError:
        print("openpyxl is not installed - only the TSV file was written (uv add openpyxl, then rerun).")


if __name__ == "__main__":
    main()
