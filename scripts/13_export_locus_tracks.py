# =============================================================================
# Locus tracks of Figure 4E (DUSP1, FKBP5, CYP24A1, LIF): export published A549 PRO-seq signal for four regions
# from bigWig files to one small binned table (all tracks), which figures/fig4_enhancer_gene_links.py reads.
# Requires: numpy, pandas only (a small built-in bigWig reader; pyBigWig is not needed,
# it does not install on Windows)
# Published PRO-seq (hg38; EtOH and dex 45 min; plus and minus strand); replicates are
# averaged per condition and strand. Minus-strand bigWigs may store negative values.
# =============================================================================
# %%
from pathlib import Path
import numpy as np
import pandas as pd
import struct, zlib


# ----------------------------------------------------------------- minimal bigWig reader
class BigWig:
    def __init__(s, path):
        s.f=open(path,'rb'); b=s.f.read(64)
        (s.magic,s.version,s.nzoom,s.ctree,s.fdata,s.findex,s.fc,s.dfc,s.asql,s.tsum,s.ubuf)=struct.unpack('<IHHQQQHHQQI',b[:56])
        assert s.magic==0x888FFC26, hex(s.magic)
        s.f.seek(s.tsum); s.summary=struct.unpack('<Qdddd',s.f.read(40))   # basesCovered,min,max,sum,sumSq
        s.chroms=s._chroms()
    def _chroms(s):
        s.f.seek(s.ctree); m,bs,ks,vs,ic,_=struct.unpack('<IIIIQQ',s.f.read(32)); out={}
        def node(off):
            s.f.seek(off); leaf,_,n=struct.unpack('<BBH',s.f.read(4))
            for i in range(n):
                key=s.f.read(ks).rstrip(b'\0').decode()
                if leaf: cid,size=struct.unpack('<II',s.f.read(8)); out[key]=(cid,size)
                else:
                    child,=struct.unpack('<Q',s.f.read(8)); pos=s.f.tell(); node(child); s.f.seek(pos)
        node(s.ctree+32); return out
    def _blocks(s,cid,start,end):
        s.f.seek(s.findex); hdr=s.f.read(48); res=[]
        def node(off):
            s.f.seek(off); leaf,_,n=struct.unpack('<BBH',s.f.read(4))
            items=[struct.unpack('<IIIIQQ' if leaf else '<IIIIQ',s.f.read(32 if leaf else 24)) for _ in range(n)]
            for it in items:
                sc,sb,ec,eb=it[:4]
                if (sc,sb)<(cid,end) and (ec,eb)>(cid,start):
                    if leaf: res.append((it[4],it[5]))
                    else: node(it[4])
        node(s.findex+48); return res
    def values(s,chrom,start,end):
        cid=s.chroms[chrom][0]; arr=np.zeros(end-start)
        for off,size in s._blocks(cid,start,end):
            s.f.seek(off); d=s.f.read(size)
            if s.ubuf: d=zlib.decompress(d)
            c,st,en,step,span,typ,_,n=struct.unpack('<IIIIIBBH',d[:24]); p=24
            for i in range(n):
                if typ==1: a,b,v=struct.unpack('<IIf',d[p:p+12]); p+=12
                elif typ==2: a,v=struct.unpack('<If',d[p:p+8]); p+=8; b=a+span
                else: v,=struct.unpack('<f',d[p:p+4]); p+=4; a=st+i*step; b=a+span
                if c!=cid: continue
                a2,b2=max(a,start),min(b,end)
                if a2<b2: arr[a2-start:b2-start]=v
        return arr


from a549_enhancers.paths import ENHANCER_GENE   # local folders, set in paths.toml
BW_DIR = ENHANCER_GENE
OUT = ENHANCER_GENE; OUT.mkdir(parents=True, exist_ok=True)

BIGWIGS = {   # (condition, strand): [bigWig files, one per replicate]
    ("EtOH", "plus"):  ["A549_PROseq_ctrl_rep1_plus.bw",  "A549_PROseq_ctrl_rep2_plus.bw"],
    ("EtOH", "minus"): ["A549_PROseq_ctrl_rep1_minus.bw", "A549_PROseq_ctrl_rep2_minus.bw"],
    ("Dex", "plus"):   ["A549_PROseq_dex-100nM-45min_rep1_plus.bw",
                        "A549_PROseq_dex-100nM-45min_rep2_plus.bw"],
    ("Dex", "minus"):  ["A549_PROseq_dex-100nM-45min_rep1_minus.bw",
                        "A549_PROseq_dex-100nM-45min_rep2_minus.bw"],
}
BIGWIGS = {k: [BW_DIR / f for f in v] for k, v in BIGWIGS.items()}
missing = [str(f) for v in BIGWIGS.values() for f in v if not f.exists()]
if missing:
    raise FileNotFoundError("Not found:\n  " + "\n  ".join(missing))

REGIONS = {"DUSP1": ("chr5", 172_700_000, 172_930_000),
           "FKBP5": ("chr6", 35_540_000, 35_760_000),
           "CYP24A1": ("chr20", 54_095_000, 54_190_000),
           "LIF": ("chr22", 30_185_000, 30_268_000)}
BIN = 100   # bp


def is_bigwig(path):
    with open(path, "rb") as fh:
        return struct.unpack("<I", fh.read(4))[0] == 0x888FFC26


def region_values(path):
    """per-base values for every region; works for bigWig and for plain bedGraph text files"""
    out = {}
    if is_bigwig(path):
        bw = BigWig(str(path))
        print(f"{path.name}: bigWig, bases covered {bw.summary[0]:,}, total signal {bw.summary[3]:,.0f}")
        for gene, (chrom, start, end) in REGIONS.items():
            ch = chrom if chrom in bw.chroms else chrom.replace("chr", "")
            out[gene] = bw.values(ch, start, end) if ch in bw.chroms else np.zeros(end - start)
        bw.f.close()
        return out
    # bedGraph (text): stream in chunks, keep only intervals overlapping the regions
    arrs = {g: np.zeros(e - s) for g, (c, s, e) in REGIONS.items()}
    covered, total = 0, 0.0
    for ch in pd.read_csv(path, sep="\t", header=None, comment="t", usecols=[0, 1, 2, 3],
                          names=["chr", "s", "e", "v"], dtype={"chr": str}, chunksize=2_000_000):
        covered += int((ch.e - ch.s).sum()); total += float(((ch.e - ch.s) * ch.v).sum())
        for gene, (chrom, start, end) in REGIONS.items():
            d = ch[((ch.chr == chrom) | (ch.chr == chrom.replace("chr", ""))) & (ch.e > start) & (ch.s < end)]
            for a, b, v in zip(d.s.values, d.e.values, d.v.values):
                arrs[gene][max(a, start) - start:min(b, end) - start] = v
    print(f"{path.name}: bedGraph text (not a bigWig), bases covered {covered:,}, total signal {total:,.0f}")
    return arrs


per_file = {f: region_values(f) for v in BIGWIGS.values() for f in v}

rows = []
for gene, (chrom, start, end) in REGIONS.items():
    edges = np.arange(start, end + 1, BIN)
    for (cond, strand), files in BIGWIGS.items():
        vals = [per_file[f][gene][:len(edges[:-1]) * BIN].reshape(-1, BIN).mean(axis=1) for f in files]
        m = np.mean(vals, axis=0)
        rows.append(pd.DataFrame({"gene": gene, "chrom": chrom, "start": edges[:-1], "end": edges[1:],
                                  "condition": cond, "strand": strand, "signal": m}))
tab = pd.concat(rows, ignore_index=True)
tab["signal"] = tab.signal.abs()
tab.to_csv(OUT / "A5_locus_PROseq_tracks.tsv.gz", sep="\t", index=False)
print(tab.groupby(["gene", "condition", "strand"]).signal.agg(["min", "max", "mean"]))
print("Saved:", OUT / "A5_locus_PROseq_tracks.tsv.gz")
