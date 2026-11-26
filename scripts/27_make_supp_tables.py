"""All supplementary tables in one run.

    S1  ENCODE datasets of the feature matrix             scripts/16_supp_table_encode_datasets.py
    S2  permutation importance, features and groups       scripts/15_supp_table_feature_importance.py
    S3  ChIP-seq / ATAC-seq signal by enhancer class      scripts/17_supp_table_signal_changes.py
    S4  dex time course by enhancer class                 scripts/26_supp_table_timecourse.py
                                                          (scripts/25_timecourse_by_class.py is run first if its
                                                          output is not there yet)
    S5  RT-qPCR primer sequences                          built here from <analysis>/primers.tsv

Every table is written to the figures folder as Supp_Table_S<n>_*.xlsx (with TSV copies of the sheets). A table
that fails does not stop the others; the summary at the end lists what was written and what was not.

    uv run python scripts/27_make_supp_tables.py
    uv run python scripts/27_make_supp_tables.py --metadata C:/path/to/metadata.tsv     # ENCODE metadata for S1
    uv run python scripts/27_make_supp_tables.py --fetch                                # S1: ask the ENCODE portal
    uv run python scripts/27_make_supp_tables.py --only S2 S5

Primers (S5). The sequences are not in the repository. The first run writes a template, <analysis>/primers.tsv, with
the columns gene, forward_primer, reverse_primer and note. Fill in the sequences (5' to 3'), add or remove rows,
and run the script again.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from a549_enhancers import paths

HERE = Path(__file__).resolve().parent
TABLES = ["S1", "S2", "S3", "S4", "S5"]
PRIMER_GENES = ["GAPDH", "EP300", "FOXA1", "TEAD4"]        # rows of the template; edit primers.tsv, not this list
PRIMER_COLUMNS = ["gene", "forward_primer", "reverse_primer", "note"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="+", choices=TABLES, default=TABLES, help="tables to make (default: all)")
    p.add_argument("--metadata", default=None, help="S1: metadata.tsv downloaded from the ENCODE portal")
    p.add_argument("--fetch", action="store_true", help="S1: look up missing files on the ENCODE portal (needs internet)")
    return p.parse_args()


def run(script: str, *args: str) -> bool:
    """Run one script of this folder with the same interpreter; True if it finished without error."""
    cmd = [sys.executable, str(HERE / script), *args]
    print(f"\n{'=' * 78}\n{script} {' '.join(args)}\n{'=' * 78}", flush=True)
    return subprocess.run(cmd).returncode == 0


def primers(figures: Path, analysis: Path) -> tuple[bool, str]:
    """Supplementary Table S5 from <analysis>/primers.tsv; writes a template when the file is not there."""
    src = analysis / "primers.tsv"
    if not src.exists():
        analysis.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"gene": PRIMER_GENES, "forward_primer": "", "reverse_primer": "", "note": ""})[PRIMER_COLUMNS].to_csv(
            src, sep="\t", index=False)
        return False, f"template written: {src} - fill in the sequences (5' to 3') and run again"
    t = pd.read_csv(src, sep="\t", dtype=str).fillna("")
    missing = [c for c in PRIMER_COLUMNS[:3] if c not in t.columns]
    if missing:
        return False, f"{src}: columns not found: {missing}"
    if "note" not in t.columns:
        t["note"] = ""
    for c in ("forward_primer", "reverse_primer"):
        t[c] = t[c].str.replace(r"\s|-|5'|3'", "", regex=True).str.upper()
    t = t[t["gene"].str.strip() != ""]
    empty = t.loc[(t.forward_primer == "") | (t.reverse_primer == ""), "gene"].tolist()
    if empty:
        return False, f"{src}: no sequence yet for {empty} - fill in and run again"
    bad = [f"{r.gene} ({c.split('_')[0]})" for r in t.itertuples() for c in ("forward_primer", "reverse_primer")
           if not re.fullmatch(r"[ACGT]+", getattr(r, c))]
    if bad:
        return False, f"{src}: letters other than A, C, G, T in {bad}"
    out = t[PRIMER_COLUMNS].rename(columns={"gene": "Gene", "forward_primer": "Forward primer (5'-3')",
                                            "reverse_primer": "Reverse primer (5'-3')", "note": "Note"})
    out.insert(3, "Length forward / reverse (nt)", [f"{len(f)} / {len(r)}" for f, r in zip(t.forward_primer, t.reverse_primer)])
    readme = [("Supplementary Table S5", "Primer sequences used for RT-qPCR."),
              ("Gene", "Target gene (HGNC symbol). GAPDH was the reference gene."),
              ("Forward / Reverse primer", "Sequences in 5' to 3' direction."),
              ("Use", "Knockdown efficiency of the siRNAs was measured by RT-qPCR (see Methods).")]
    figures.mkdir(parents=True, exist_ok=True)
    out.to_csv(figures / "Supp_Table_S5_primers.tsv", sep="\t", index=False)
    try:
        with pd.ExcelWriter(figures / "Supp_Table_S5_primers.xlsx", engine="openpyxl") as w:
            pd.DataFrame(readme, columns=["Item", "Description"]).to_excel(w, sheet_name="README", index=False)
            out.to_excel(w, sheet_name="Primers", index=False, freeze_panes=(1, 1))
    except ImportError:
        return False, "openpyxl is not installed - only the TSV was written (uv add openpyxl, then rerun)"
    return True, f"{len(out)} primer pairs"


def main() -> None:
    a = parse_args()
    cfg = paths.load()
    figures, analysis = cfg.require("figures"), cfg.require("analysis")
    started = datetime.now().timestamp()
    status = {}

    if "S1" in a.only:
        extra = (["--metadata", a.metadata] if a.metadata else []) + (["--fetch"] if a.fetch else [])
        status["S1"] = (run("16_supp_table_encode_datasets.py", *extra),
                        "" if a.metadata or a.fetch else "written without ENCODE metadata (--metadata or --fetch adds it)")
    if "S2" in a.only:
        status["S2"] = (run("15_supp_table_feature_importance.py"), "")
    if "S3" in a.only:
        status["S3"] = (run("17_supp_table_signal_changes.py"), "")
    if "S4" in a.only:
        tc = analysis / "class_analyses" / "timecourse_by_class.tsv"
        ok = tc.exists() or run("25_timecourse_by_class.py")
        status["S4"] = (ok and run("26_supp_table_timecourse.py", "--table-number", "S4"), "")
    if "S5" in a.only:
        print(f"\n{'=' * 78}\nS5 primers\n{'=' * 78}")
        status["S5"] = primers(figures, analysis)
        print(status["S5"][1])

    print(f"\n{'=' * 78}\nSummary  (folder: figures, as set in paths.toml)\n{'=' * 78}")
    for t in TABLES:
        if t not in status:
            continue
        ok, note = status[t]
        new = sorted(f.name for f in figures.glob(f"Supp_Table_{t}_*.xlsx") if f.stat().st_mtime >= started - 1)
        state = "written" if ok and new else ("NOT written" if not ok else "finished, but no new .xlsx found")
        print(f"  {t}: {state:12s} {', '.join(new)}{'  | ' + note if note else ''}")
    if not all(ok for ok, _ in status.values()):
        print("\nAt least one table was not written; the messages of that script are above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
