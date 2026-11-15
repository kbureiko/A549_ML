# =============================================================================
# Export eRNA (HOMER-defined enhancer) results from the siEP300 PRO-seq
# (tables read by figures/text_numbers_siEP300_enhancers.py)
#
# Same input, model and thresholds as R/01_proseq_siEP300_enhancers.R:
#   raw.eRNA_combined_A549_PROseq_siEP300.txt, no count pre-filter,
#   design ~ silencing + treatment + silencing:treatment (siNON, EtOH reference),
#   significance padj < 0.05 & |log2FC| > 0.5
# The response categories are rebuilt with exactly the original set logic, and
# the script prints their sizes so they can be checked against the
# manuscript (e.g. 191 shared / 71 siNON-specific / 330 + 43 siEP300-specific
# dex-induced; 72 / 60 / 151 dex-repressed; 34 up / 122 down basal).
#
# Outputs (OUTDIR, date-prefixed):
#   eRNA_regions.tsv                  ID, coordinates, strand, HOMER annotation
#   eRNA_DESeq2_siNON_dex_vs_EtOH.tsv
#   eRNA_DESeq2_siEP300_dex_vs_EtOH.tsv
#   eRNA_DESeq2_siEP300_vs_siNON_EtOH.tsv     (basal EP300 effect)
#   eRNA_DESeq2_siEP300_vs_siNON_dex.tsv      (EP300 effect under dex)
#   eRNA_DESeq2_interaction_siEP300xdex.tsv
#   eRNA_normalized_counts.tsv, eRNA_raw_counts.tsv, eRNA_sample_sheet.tsv
#   eRNA_categories.tsv               0/1 membership in every original category
#   eRNA_category_sizes.tsv, eRNA_model_info.txt
# =============================================================================

suppressPackageStartupMessages({
  library(DESeq2)
  library(dplyr)
})

# ----------------------------- Settings --------------------------------------
source(if (file.exists("R/paths.R")) "R/paths.R" else "paths.R")
P <- read_paths()
counts_file <- file.path(P$proseq_counts, "raw.eRNA_combined_A549_PROseq_siEP300.txt")
padj_cutoff <- 0.05
lfc_cutoff  <- 0.5            # as in the original eRNA script
PREFIX      <- format(Sys.Date(), "%Y%m%d")
OUTDIR      <- file.path(P$knockdown, "eRNA_export_siEP300")
dir.create(OUTDIR, showWarnings = FALSE, recursive = TRUE)
out <- function(x, name) write.table(x, file.path(OUTDIR, paste0(PREFIX, "_", name)),
                                     sep = "\t", quote = FALSE, row.names = FALSE)

# ----------------------------- 1. Read ----------------------------------------
raw <- read.delim(counts_file, check.names = TRUE)
names(raw)[1] <- "Peak_ID"
raw$Peak_ID <- as.character(raw$Peak_ID)
if (anyDuplicated(raw$Peak_ID)) {
  stop(sum(duplicated(raw$Peak_ID)), " duplicated Peak_IDs in ", counts_file,
       ": ", paste(head(unique(raw$Peak_ID[duplicated(raw$Peak_ID)]), 10), collapse = ", "))
}

count_cols <- grep("Tag\\.Count\\.in\\.given\\.bp", names(raw), value = TRUE)
stopifnot(length(count_cols) == 8)
counts <- as.matrix(raw[, count_cols])
rownames(counts) <- raw$Peak_ID
colnames(counts) <- sub("\\.Tag\\.Count\\.in\\.given\\.bp.*", "", count_cols)
cat("Regions:", nrow(counts), " Samples:", ncol(counts), "\n")
print(colnames(counts))

# Region table (coordinates as counted; no window adjustment)
annot_cols <- intersect(c("Chr", "Start", "End", "Strand", "Annotation", "Distance.to.TSS",
                          "Nearest.PromoterID", "Gene.Name", "Gene.Type"), names(raw))
out(raw[, c("Peak_ID", annot_cols)], "eRNA_regions.tsv")

# ----------------------------- 2. Samples -------------------------------------
parts <- strsplit(colnames(counts), "_")
st <- lapply(parts, function(x) strsplit(x[3], "\\.")[[1]])
sampleData <- data.frame(sample    = colnames(counts),
                         silencing = sapply(st, `[`, 1),
                         treatment = sapply(st, `[`, 2),
                         replicate = sapply(parts, `[`, 4))
sampleData$silencing <- relevel(factor(sampleData$silencing), ref = "siNON")
sampleData$treatment <- relevel(factor(sampleData$treatment), ref = "EtOH")
rownames(sampleData) <- sampleData$sample
print(table(sampleData$silencing, sampleData$treatment))

# ----------------------------- 3. DESeq2 (original model) ---------------------
dds <- DESeqDataSetFromMatrix(countData = round(counts), colData = sampleData,
                              design = ~ silencing + treatment + silencing:treatment)
dds <- DESeq(dds)
print(resultsNames(dds))
# "Intercept" "silencing_siEP300_vs_siNON" "treatment_dex_vs_EtOH" "silencingsiEP300.treatmentdex"

res <- list(
  siNON_dex_vs_EtOH    = results(dds, name = "treatment_dex_vs_EtOH"),
  siEP300_dex_vs_EtOH  = results(dds, contrast = list(c("treatment_dex_vs_EtOH",
                                                        "silencingsiEP300.treatmentdex"))),
  siEP300_vs_siNON_EtOH = results(dds, name = "silencing_siEP300_vs_siNON"),
  siEP300_vs_siNON_dex  = results(dds, contrast = list(c("silencing_siEP300_vs_siNON",
                                                         "silencingsiEP300.treatmentdex"))),
  interaction_siEP300xdex = results(dds, name = "silencingsiEP300.treatmentdex")
)
for (nm in names(res)) {
  df <- as.data.frame(res[[nm]])
  out(data.frame(Peak_ID = rownames(df), df, row.names = NULL), paste0("eRNA_DESeq2_", nm, ".tsv"))
}

nc <- counts(dds, normalized = TRUE)
out(data.frame(Peak_ID = rownames(nc), nc, row.names = NULL, check.names = FALSE), "eRNA_normalized_counts.tsv")
out(data.frame(Peak_ID = rownames(counts), counts, row.names = NULL, check.names = FALSE), "eRNA_raw_counts.tsv")
out(data.frame(sampleData, sizeFactor = sizeFactors(dds), row.names = NULL), "eRNA_sample_sheet.tsv")

# ----------------------------- 4. Original categories -------------------------
sig <- function(r, dir) {
  df <- as.data.frame(r)
  keep <- !is.na(df$padj) & df$padj < padj_cutoff &
          (if (dir == "up") df$log2FoldChange > lfc_cutoff else df$log2FoldChange < -lfc_cutoff)
  rownames(df)[keep]
}
non_up  <- sig(res$siNON_dex_vs_EtOH, "up");    non_dn <- sig(res$siNON_dex_vs_EtOH, "dn")
ep_up   <- sig(res$siEP300_dex_vs_EtOH, "up");  ep_dn  <- sig(res$siEP300_dex_vs_EtOH, "dn")
bas_up  <- sig(res$siEP300_vs_siNON_EtOH, "up"); bas_dn <- sig(res$siEP300_vs_siNON_EtOH, "dn")

# identical set logic to the original script
cats <- list(
  siNON_Dex_Up              = setdiff(non_up, union(bas_dn, ep_up)),
  siEP300_Dex_Up            = setdiff(ep_up, union(bas_dn, non_up)),
  Shared_Dex_Up             = setdiff(intersect(non_up, ep_up), bas_dn),
  siEP300_Dex_Up_siEP300_Dn = intersect(ep_up, bas_dn),
  siNON_Dex_Down            = setdiff(non_dn, ep_dn),
  siEP300_Dex_Down          = setdiff(ep_dn, non_dn),
  Shared_Dex_Down           = intersect(non_dn, ep_dn),
  siEP300_Dn                = setdiff(bas_dn, ep_up),
  siEP300_Up                = setdiff(bas_up, non_up),
  # plain significance sets (no exclusions), for reference
  all_siNON_Dex_Up = non_up, all_siNON_Dex_Down = non_dn,
  all_siEP300_Dex_Up = ep_up, all_siEP300_Dex_Down = ep_dn,
  all_basal_siEP300_Up = bas_up, all_basal_siEP300_Dn = bas_dn
)
memb <- data.frame(Peak_ID = rownames(counts))
for (nm in names(cats)) memb[[nm]] <- as.integer(memb$Peak_ID %in% cats[[nm]])
out(memb, "eRNA_categories.tsv")

sizes <- data.frame(category = names(cats), n = sapply(cats, length))
print(sizes, row.names = FALSE)
out(sizes, "eRNA_category_sizes.tsv")

# ----------------------------- 5. Info ----------------------------------------
sink(file.path(OUTDIR, paste0(PREFIX, "_eRNA_model_info.txt")))
cat("Input:", counts_file, "\nRegions:", nrow(counts), "(no count pre-filter)\n")
cat("Design: ~ silencing + treatment + silencing:treatment (ref siNON, EtOH)\n")
cat("Significance: padj <", padj_cutoff, "& |log2FC| >", lfc_cutoff, "(unshrunken MLE)\n")
cat("resultsNames:", paste(resultsNames(dds), collapse = ", "), "\n\n")
print(sizes, row.names = FALSE)
cat("\n"); print(sessionInfo())
sink()
cat("\nExported to:", normalizePath(OUTDIR), "\n")
