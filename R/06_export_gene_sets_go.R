# =============================================================================
# Knockdown gene sets + GO:BP enrichment for EP300, FOXA1 and TEAD4
#
# Self-contained: works from the DESeq2 tables written by R/04_export_proseq_siEP300_tables.R and R/05_export_brbseq_tables.R
#   panel_export_PROseq_siEP300/        *_PROseq_DESeq2_siNON_dex_vs_EtOH.tsv
#                                       *_PROseq_DESeq2_siEP300_dex_vs_EtOH.tsv
#   panel_export_BRBseq_siFOXA1_siTEAD4/*_BRBseq_DESeq2_siNON_Dex_vs_Veh.tsv
#                                       *_BRBseq_DESeq2_siFOXA1_Dex_vs_Veh.tsv
#                                       *_BRBseq_DESeq2_siTEAD4_Dex_vs_Veh.tsv
# Gene sets (padj < 0.05 & |log2FC| > 0.5):
#   shared       = intersect(siNON, KD)        per direction
#   KD_specific  = setdiff(KD, siNON)
#   siNON_specific = setdiff(siNON, KD)
# GO: clusterProfiler::enrichGO, BP, BH; universe = all genes tested in that assay
# (padj not NA), which is more appropriate than the whole genome. Cut-offs are set to 1
# so every set gets a value for every term that reaches nominal p < 0.2 anywhere;
# that lets the figure show -log10(FDR) for a theme across all sets, not only where it
# passed. Sets are named <KD>_<group>_<direction>, e.g. EP300_KD_specific_up.
#
# Outputs (OUTDIR, date-prefixed):
#   A6_gene_sets_long.tsv        set, knockdown, group, direction, gene
#   A6_gene_set_sizes.tsv
#   A6_GO_BP_all_sets.tsv        all terms with p < 0.2 in a set (one row per set x term)
#   lisa_inputs/<set>.txt        one gene per line (same sets, for re-running LISA)
# =============================================================================
suppressPackageStartupMessages({
  library(clusterProfiler)
  library(org.Hs.eg.db)
  library(dplyr)
})

source(if (file.exists("R/paths.R")) "R/paths.R" else "paths.R")
P <- read_paths()
PRO_DIR <- file.path(P$knockdown, "panel_export_PROseq_siEP300")
BRB_DIR <- file.path(P$knockdown, "panel_export_BRBseq_siFOXA1_siTEAD4")
OUTDIR  <- file.path(P$knockdown, "A6_exports")
PADJ <- 0.05; LFC <- 0.5
PREFIX <- format(Sys.Date(), "%Y%m%d")
dir.create(file.path(OUTDIR, "lisa_inputs"), recursive = TRUE, showWarnings = FALSE)
out <- function(x, name) write.table(x, file.path(OUTDIR, paste0(PREFIX, "_", name)),
                                     sep = "\t", quote = FALSE, row.names = FALSE)

latest <- function(dir, pattern) {
  f <- sort(list.files(dir, pattern = pattern, full.names = TRUE))
  if (!length(f)) stop("No file matching ", pattern, " in ", dir)
  f[length(f)]
}
read_res <- function(f) {
  d <- read.delim(f, check.names = FALSE)
  d[!is.na(d$gene) & d$gene != "", ]
}
calls <- function(d) list(
  up   = sort(unique(d$gene[!is.na(d$padj) & d$padj < PADJ & d$log2FoldChange >  LFC])),
  down = sort(unique(d$gene[!is.na(d$padj) & d$padj < PADJ & d$log2FoldChange < -LFC])),
  tested = sort(unique(d$gene[!is.na(d$padj)])))

pro_non <- read_res(latest(PRO_DIR, "PROseq_DESeq2_siNON_dex_vs_EtOH\\.tsv$"))
pro_ep  <- read_res(latest(PRO_DIR, "PROseq_DESeq2_siEP300_dex_vs_EtOH\\.tsv$"))
brb_non <- read_res(latest(BRB_DIR, "BRBseq_DESeq2_siNON_Dex_vs_Veh\\.tsv$"))
brb_fox <- read_res(latest(BRB_DIR, "BRBseq_DESeq2_siFOXA1_Dex_vs_Veh\\.tsv$"))
brb_tea <- read_res(latest(BRB_DIR, "BRBseq_DESeq2_siTEAD4_Dex_vs_Veh\\.tsv$"))

KD <- list(
  EP300 = list(non = calls(pro_non), kd = calls(pro_ep),  assay = "PRO-seq 1 h"),
  FOXA1 = list(non = calls(brb_non), kd = calls(brb_fox), assay = "BRB-seq 6 h"),
  TEAD4 = list(non = calls(brb_non), kd = calls(brb_tea), assay = "BRB-seq 6 h"))

# ----------------------------------------------------------------- gene sets
sets <- list(); universe <- list(); meta <- list()
for (k in names(KD)) {
  x <- KD[[k]]
  universe[[k]] <- union(x$non$tested, x$kd$tested)
  for (dirn in c("up", "down")) {
    a <- x$non[[dirn]]; b <- x$kd[[dirn]]
    grp <- list(shared = intersect(a, b), KD_specific = setdiff(b, a), siNON_specific = setdiff(a, b))
    for (g in names(grp)) {
      nm <- paste(k, g, dirn, sep = "_")
      sets[[nm]] <- sort(grp[[g]])
      meta[[nm]] <- data.frame(set = nm, knockdown = k, group = g, direction = dirn, assay = x$assay)
      writeLines(sets[[nm]], file.path(OUTDIR, "lisa_inputs", paste0(nm, ".txt")))
    }
  }
}
meta <- bind_rows(meta)
long <- bind_rows(lapply(names(sets), function(nm) if (length(sets[[nm]]))
  data.frame(set = nm, gene = sets[[nm]]))) %>% left_join(meta, by = "set")
out(long, "A6_gene_sets_long.tsv")
sizes <- meta %>% mutate(n_genes = sapply(set, function(s) length(sets[[s]])))
out(sizes, "A6_gene_set_sizes.tsv")
print(sizes[, c("set", "n_genes")], row.names = FALSE)

# ----------------------------------------------------------------- GO:BP
go <- list()
for (nm in names(sets)) {
  g <- sets[[nm]]; k <- meta$knockdown[meta$set == nm]
  if (length(g) < 5) { message(nm, ": <5 genes, skipped"); next }
  ego <- tryCatch(enrichGO(gene = g, universe = universe[[k]], OrgDb = org.Hs.eg.db, keyType = "SYMBOL",
                           ont = "BP", pAdjustMethod = "BH", pvalueCutoff = 1, qvalueCutoff = 1,
                           minGSSize = 10, maxGSSize = 500, readable = FALSE),
                  error = function(e) { message(nm, ": ", conditionMessage(e)); NULL })
  if (is.null(ego)) next
  d <- as.data.frame(ego)
  if (!nrow(d)) next
  d <- d[d$pvalue < 0.2, ]
  d$set <- nm
  go[[nm]] <- d
  message(nm, ": ", length(g), " genes, ", sum(d$p.adjust < 0.05), " terms at FDR < 0.05")
}
go <- bind_rows(go) %>% left_join(meta, by = "set") %>%
  select(set, knockdown, group, direction, assay, ID, Description, GeneRatio, BgRatio,
         pvalue, p.adjust, qvalue, Count, geneID)
out(go, "A6_GO_BP_all_sets.tsv")

sink(file.path(OUTDIR, paste0(PREFIX, "_A6_exports_info.txt")))
cat("Gene sets: padj <", PADJ, "& |log2FC| >", LFC, "; shared / KD_specific / siNON_specific per direction\n")
cat("GO: enrichGO BP, BH, universe = genes tested in the assay, minGSSize 10, maxGSSize 500;",
    "terms kept if nominal p < 0.2\n")
cat("Inputs:\n ", paste(c(latest(PRO_DIR, "siNON_dex_vs_EtOH"), latest(PRO_DIR, "siEP300_dex_vs_EtOH"),
                          latest(BRB_DIR, "siNON_Dex_vs_Veh"), latest(BRB_DIR, "siFOXA1_Dex_vs_Veh"),
                          latest(BRB_DIR, "siTEAD4_Dex_vs_Veh")), collapse = "\n  "), "\n\n")
print(sizes, row.names = FALSE); cat("\n"); print(sessionInfo())
sink()
cat("\nExported to:", normalizePath(OUTDIR), "\n")
