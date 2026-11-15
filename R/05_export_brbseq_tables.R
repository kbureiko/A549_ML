# =============================================================================
# Export full DESeq2 results + counts for the Python figures (Supplementary Figure S14)
# A549 BRB-seq, siNON / siFOXA1 / siTEAD4, Dex 6 h vs Veh (exon counts)
#
# Same samples and model as R/03_brbseq_siFOXA1_siTEAD4.R (the model also contains the TNF and DexTNF
# samples), but genes with a row sum of counts <= 10 are removed first, so the Dex-vs-Veh gene sets differ
# from that script by a few genes.
#
# Outputs (in OUTDIR, prefixed with the date):
#   BRBseq_DESeq2_siNON_Dex_vs_Veh.tsv          all tested genes, unfiltered
#   BRBseq_DESeq2_siFOXA1_Dex_vs_Veh.tsv
#   BRBseq_DESeq2_siTEAD4_Dex_vs_Veh.tsv
#   BRBseq_DESeq2_interaction_siFOXA1xDex.tsv   (siFOXA1 Dex-Veh) - (siNON Dex-Veh)
#   BRBseq_DESeq2_interaction_siTEAD4xDex.tsv   (siTEAD4 Dex-Veh) - (siNON Dex-Veh)
#   BRBseq_normalized_counts.tsv                all samples in the model
#   BRBseq_raw_counts_all_genes.tsv             ALL genes before the min_counts
#                                               filter + coordinates + pass_filter
#   BRBseq_sample_sheet.tsv                     sample, siRNA, treatment, size factor
#   BRBseq_model_info.txt                       design, contrasts, sessionInfo
# =============================================================================

suppressPackageStartupMessages({
  library(DESeq2)
  library(dplyr)
})

# ----------------------------- Settings --------------------------------------
source(if (file.exists("R/paths.R")) "R/paths.R" else "paths.R")
P <- read_paths()
counts_file    <- file.path(P$brbseq_counts, "raw.counts.A549.BRBseq.exons.txt")
min_counts     <- 10          # same as the main analysis scripts
PREFIX         <- format(Sys.Date(), "%Y%m%d")
OUTDIR         <- file.path(P$knockdown, "panel_export_BRBseq_siFOXA1_siTEAD4")

siRNA_list     <- c("siNoN", "siFoxA1", "siTEAD4")
treatment_list <- c("Veh", "Dex", "TNF", "DexTNF")

# ----------------------------- 1. Read counts --------------------------------
if (!file.exists(counts_file)) stop("File not found: ", counts_file)
rawCounts <- read.delim(counts_file)

normal_chrs <- c(paste0("chr", 1:23), "chrX", "chrY", "chrM")

count_table <- rawCounts %>%
  mutate(gene_name = sub("\\|.*", "", Annotation.Divergence)) %>%
  filter(chr %in% normal_chrs) %>%
  filter(gene_name != "0.000")

# Count columns are 9:44 in the original HOMER table
counts_mat <- as.data.frame(count_table[, 9:44])
gene_names <- count_table$gene_name

# Gene annotation for the export (first column = HOMER transcript ID)
annot_cols <- intersect(c("chr", "start", "end", "strand", "Length"), colnames(count_table))
annot <- data.frame(
  gene    = gene_names,
  gene_id = count_table[[1]],
  count_table[, annot_cols, drop = FALSE],
  check.names = FALSE
)

keep_first <- !duplicated(gene_names)
if (any(!keep_first)) {
  dups <- unique(gene_names[!keep_first])
  warning(length(dups), " duplicated gene names; keeping first occurrence: ",
          paste(head(dups, 10), collapse = ", "))
}
counts_mat <- counts_mat[keep_first, ]
annot      <- annot[keep_first, ]
gene_names <- gene_names[keep_first]
rownames(counts_mat) <- gene_names

colnames(counts_mat) <- sapply(strsplit(colnames(counts_mat), "_"),
                               function(x) paste(x[1:4], collapse = "_"))

pattern <- paste0("(", paste(siRNA_list, collapse = "|"), ")_(",
                  paste(treatment_list, collapse = "|"), ")")
selected_cols <- grep(pattern, colnames(counts_mat), value = TRUE, ignore.case = TRUE)
counts_mat <- counts_mat[, selected_cols]

counts_all  <- counts_mat
pass_filter <- rowSums(counts_mat) > min_counts
counts_mat  <- counts_mat[pass_filter, ]
cat("Genes before filter:", nrow(counts_all), " after filter:", nrow(counts_mat), "\n")
cat("Samples:", ncol(counts_mat), "\n")

# ----------------------------- 2. Sample table -------------------------------
parts <- strsplit(colnames(counts_mat), "_")
sampleData <- data.frame(
  sample    = colnames(counts_mat),
  silencing = sapply(parts, `[`, 2),
  treatment = sapply(parts, `[`, 3)
)
sampleData$group <- factor(paste(sampleData$silencing, sampleData$treatment, sep = "_"))

# Find the actual level names (sample names were matched case-insensitively)
lvl <- function(si, tr) {
  hit <- levels(sampleData$group)[tolower(levels(sampleData$group)) ==
                                    tolower(paste(si, tr, sep = "_"))]
  if (length(hit) != 1) stop("Group not found (or ambiguous): ", si, "_", tr,
                             "\nLevels: ", paste(levels(sampleData$group), collapse = ", "))
  hit
}
L <- list(
  NON_Veh   = lvl("siNoN",   "Veh"), NON_Dex   = lvl("siNoN",   "Dex"),
  FOXA1_Veh = lvl("siFoxA1", "Veh"), FOXA1_Dex = lvl("siFoxA1", "Dex"),
  TEAD4_Veh = lvl("siTEAD4", "Veh"), TEAD4_Dex = lvl("siTEAD4", "Dex")
)

# siNON Veh as reference: does not change the fit, only makes the
# interaction contrasts readable
sampleData$group <- relevel(sampleData$group, ref = L$NON_Veh)
rownames(sampleData) <- sampleData$sample
print(table(sampleData$group))

# ----------------------------- 3. DESeq2 -------------------------------------
dds <- DESeqDataSetFromMatrix(countData = round(as.matrix(counts_mat)),
                              colData   = sampleData,
                              design    = ~ group)
dds <- DESeq(dds)
print(resultsNames(dds))

# Per-siRNA Dex effects (identical to the main analysis scripts)
get_res <- function(dex, veh) results(dds, contrast = c("group", dex, veh))
res_siNON   <- get_res(L$NON_Dex,   L$NON_Veh)
res_siFOXA1 <- get_res(L$FOXA1_Dex, L$FOXA1_Veh)
res_siTEAD4 <- get_res(L$TEAD4_Dex, L$TEAD4_Veh)

# Interactions: (siKD Dex - siKD Veh) - (siNON Dex - siNON Veh)
# Coefficients are differences from siNON_Veh, so
#   interaction = b[KD_Dex] - b[KD_Veh] - b[NON_Dex]
# log2FC > 0: Dex effect larger (more positive) after knockdown
# log2FC < 0: Dex effect smaller (more negative) after knockdown
coef_name <- function(g) make.names(paste0("group_", g, "_vs_", L$NON_Veh))
int_contrast <- function(kd_dex, kd_veh) {
  num <- coef_name(kd_dex)
  den <- c(coef_name(kd_veh), coef_name(L$NON_Dex))
  stopifnot(all(c(num, den) %in% resultsNames(dds)))
  list(num, den)
}
c_FOXA1 <- int_contrast(L$FOXA1_Dex, L$FOXA1_Veh)
c_TEAD4 <- int_contrast(L$TEAD4_Dex, L$TEAD4_Veh)
res_int_FOXA1 <- results(dds, contrast = c_FOXA1)
res_int_TEAD4 <- results(dds, contrast = c_TEAD4)

# Optional shrunken LFCs (plotting only; classification uses unshrunken LFCs)
add_shrunk <- function(res, dex, veh) {
  out <- as.data.frame(res)
  if (requireNamespace("ashr", quietly = TRUE)) {
    shr <- lfcShrink(dds, contrast = c("group", dex, veh), res = res,
                     type = "ashr", quiet = TRUE)
    out$log2FoldChange_ashr <- shr$log2FoldChange
  } else {
    message("ashr not installed; skipping shrunken LFCs")
  }
  data.frame(gene = rownames(out), out, check.names = FALSE, row.names = NULL)
}
as_tab <- function(res) data.frame(gene = rownames(res), as.data.frame(res),
                                   check.names = FALSE, row.names = NULL)

tab_siNON   <- add_shrunk(res_siNON,   L$NON_Dex,   L$NON_Veh)
tab_siFOXA1 <- add_shrunk(res_siFOXA1, L$FOXA1_Dex, L$FOXA1_Veh)
tab_siTEAD4 <- add_shrunk(res_siTEAD4, L$TEAD4_Dex, L$TEAD4_Veh)

# ----------------------------- 4. Sanity check -------------------------------
gr_controls <- c("TSC22D3", "PER1", "FKBP5", "KLF9", "ZBTB16", "DUSP1")
cat("\nGR positive controls, siNON Dex vs Veh:\n")
print(as.data.frame(res_siNON)[intersect(gr_controls, rownames(res_siNON)),
                               c("baseMean", "log2FoldChange", "padj")])

# ----------------------------- 5. Export -------------------------------------
dir.create(OUTDIR, showWarnings = FALSE, recursive = TRUE)
out <- function(x, name) {
  write.table(x, file.path(OUTDIR, paste0(PREFIX, "_", name)),
              sep = "\t", quote = FALSE, row.names = FALSE)
}

out(tab_siNON,   "BRBseq_DESeq2_siNON_Dex_vs_Veh.tsv")
out(tab_siFOXA1, "BRBseq_DESeq2_siFOXA1_Dex_vs_Veh.tsv")
out(tab_siTEAD4, "BRBseq_DESeq2_siTEAD4_Dex_vs_Veh.tsv")
out(as_tab(res_int_FOXA1), "BRBseq_DESeq2_interaction_siFOXA1xDex.tsv")
out(as_tab(res_int_TEAD4), "BRBseq_DESeq2_interaction_siTEAD4xDex.tsv")

norm_counts <- counts(dds, normalized = TRUE)
out(data.frame(gene = rownames(norm_counts), norm_counts,
               check.names = FALSE, row.names = NULL),
    "BRBseq_normalized_counts.tsv")

raw_all <- data.frame(annot, pass_filter = pass_filter, counts_all,
                      check.names = FALSE, row.names = NULL)
out(raw_all, "BRBseq_raw_counts_all_genes.tsv")

out(data.frame(sampleData, sizeFactor = sizeFactors(dds), row.names = NULL),
    "BRBseq_sample_sheet.tsv")

info_file <- file.path(OUTDIR, paste0(PREFIX, "_BRBseq_model_info.txt"))
sink(info_file)
cat("Dataset: A549 BRB-seq, siNON/siFOXA1/siTEAD4, Dex 6 h vs Veh; exon counts\n")
cat("Counts file:", counts_file, "\n")
cat("Samples in model: all siRNA x {Veh, Dex, TNF, DexTNF} (as in R/03_brbseq_siFOXA1_siTEAD4.R)\n")
cat("Filter: rowSums >", min_counts, "; duplicated gene names: first kept\n")
cat("Design: ~ group (reference", L$NON_Veh, ")\n")
cat("resultsNames:", paste(resultsNames(dds), collapse = ", "), "\n")
cat("Interaction siFOXA1 x Dex: list(", c_FOXA1[[1]], ", c(",
    paste(c_FOXA1[[2]], collapse = ", "), "))\n")
cat("Interaction siTEAD4 x Dex: list(", c_TEAD4[[1]], ", c(",
    paste(c_TEAD4[[2]], collapse = ", "), "))\n")
cat("LFCs in result tables: unshrunken MLE (log2FoldChange); ashr column if present\n\n")
print(table(sampleData$group))
cat("\n")
print(sessionInfo())
sink()

cat("\nExported to:", normalizePath(OUTDIR), "\n")
print(list.files(OUTDIR))
