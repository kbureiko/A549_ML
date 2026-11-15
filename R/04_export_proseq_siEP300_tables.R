# =============================================================================
# Export full DESeq2 results + counts for the Python figures (Figure 4, Supplementary Figure S14)
# A549 PRO-seq, siNON / siEP300, dex 1 h vs EtOH
#
# Reading, filtering and duplicate handling are identical to the main analysis scripts
# script, so siNON / siEP300 dex-vs-EtOH results match the existing gene sets.
#
# Outputs (in OUTDIR, prefixed with the date):
#   PROseq_DESeq2_siNON_dex_vs_EtOH.tsv        all tested genes, unfiltered
#   PROseq_DESeq2_siEP300_dex_vs_EtOH.tsv      all tested genes, unfiltered
#   PROseq_DESeq2_interaction_siEP300xdex.tsv  (siEP300 dex-EtOH) - (siNON dex-EtOH)
#   PROseq_normalized_counts.tsv               DESeq2 size-factor normalized
#   PROseq_raw_counts_all_genes.tsv            ALL genes before the min_counts
#                                              filter + coordinates + pass_filter
#                                              flag (to tell "not expressed" apart
#                                              from "expressed but unchanged")
#   PROseq_sample_sheet.tsv                    sample, group, siRNA, treatment,
#                                              size factor
#   PROseq_model_info.txt                      design, contrasts, settings,
#                                              sessionInfo
# =============================================================================

suppressPackageStartupMessages({
  library(DESeq2)
  library(dplyr)
})

source(if (file.exists("R/paths.R")) "R/paths.R" else "paths.R")
P <- read_paths()
# ----------------------------- Settings --------------------------------------
counts_file       <- file.path(P$proseq_counts, "raw.genes.A549_PROseq_siEP300.txt")
count_col_pattern <- "(siNON|siEP300).*rep[0-9]+"
min_counts        <- 10          # same as the main analysis scripts
PREFIX            <- format(Sys.Date(), "%Y%m%d")
OUTDIR            <- file.path(P$knockdown, "panel_export_PROseq_siEP300")

# ----------------------------- 1. Read counts --------------------------------
if (!file.exists(counts_file)) stop("File not found: ", counts_file)
rawCounts <- read.delim(counts_file, check.names = TRUE)

normal_chrs <- c(paste0("chr", 1:23), "chrX", "chrY", "chrM")

count_table <- rawCounts %>%
  mutate(gene_name = sub("\\|.*", "", Annotation.Divergence)) %>%
  filter(chr %in% normal_chrs) %>%
  filter(gene_name != "0.000")

count_cols_idx <- grep(count_col_pattern, colnames(count_table))
if (length(count_cols_idx) == 0) stop("No count columns matched '", count_col_pattern, "'")
cat("Count columns found (", length(count_cols_idx), "):\n", sep = "")
print(colnames(count_table)[count_cols_idx])

counts_mat <- as.data.frame(count_table[, count_cols_idx, drop = FALSE])
non_numeric <- names(counts_mat)[!sapply(counts_mat, is.numeric)]
if (length(non_numeric) > 0) {
  counts_mat[non_numeric] <- lapply(counts_mat[non_numeric],
                                    function(x) as.numeric(as.character(x)))
}
if (anyNA(counts_mat)) {
  warning(sum(is.na(counts_mat)), " NA values after conversion; setting them to 0")
  counts_mat[is.na(counts_mat)] <- 0
}

# Gene annotation kept for the export (first column = HOMER transcript ID;
# coordinates/strand/length are used later for TSS-to-GR-peak distances)
annot_cols <- intersect(c("chr", "start", "end", "strand", "Length"), colnames(count_table))
annot <- data.frame(
  gene    = count_table$gene_name,
  gene_id = count_table[[1]],
  count_table[, annot_cols, drop = FALSE],
  check.names = FALSE
)

gene_names <- count_table$gene_name
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
colnames(counts_mat) <- sub("(.*rep[0-9]+).*", "\\1", colnames(counts_mat))

# Keep the pre-filter matrix for export, then filter as in the original
counts_all  <- counts_mat
pass_filter <- rowSums(counts_mat) > min_counts
counts_mat  <- counts_mat[pass_filter, ]

cat("Genes before filter:", nrow(counts_all), " after filter:", nrow(counts_mat), "\n")

# ----------------------------- 2. Sample table -------------------------------
grp <- sapply(strsplit(colnames(counts_mat), "_"), `[`, 3)
sampleData <- data.frame(
  sample    = colnames(counts_mat),
  group     = grp,
  silencing = sub("\\..*", "", grp),
  treatment = sub(".*\\.", "", grp)
)
expected <- c("siNON.EtOH", "siNON.dex", "siEP300.EtOH", "siEP300.dex")
missing_grp <- setdiff(expected, unique(sampleData$group))
if (length(missing_grp) > 0) stop("Missing groups: ", paste(missing_grp, collapse = ", "))

# siNON.EtOH as reference level. This does not change the fit (same model as
# ~group with any reference), it only makes the interaction contrast readable.
sampleData$group <- relevel(factor(sampleData$group), ref = "siNON.EtOH")
rownames(sampleData) <- sampleData$sample
print(table(sampleData$group))

# ----------------------------- 3. DESeq2 -------------------------------------
dds <- DESeqDataSetFromMatrix(countData = round(as.matrix(counts_mat)),
                              colData   = sampleData,
                              design    = ~ group)
dds <- DESeq(dds)
print(resultsNames(dds))

# Per-siRNA dex effects (identical to the main analysis scripts)
res_siNON   <- results(dds, contrast = c("group", "siNON.dex",   "siNON.EtOH"))
res_siEP300 <- results(dds, contrast = c("group", "siEP300.dex", "siEP300.EtOH"))

# Interaction: (siEP300.dex - siEP300.EtOH) - (siNON.dex - siNON.EtOH)
# With siNON.EtOH as reference, coefficients are differences from siNON.EtOH, so
#   interaction = b[siEP300.dex] - b[siEP300.EtOH] - b[siNON.dex]
# log2FC > 0: the dex effect is larger (more positive) after siEP300
# log2FC < 0: the dex effect is smaller (more negative) after siEP300
int_num <- "group_siEP300.dex_vs_siNON.EtOH"
int_den <- c("group_siEP300.EtOH_vs_siNON.EtOH", "group_siNON.dex_vs_siNON.EtOH")
stopifnot(all(c(int_num, int_den) %in% resultsNames(dds)))
res_int <- results(dds, contrast = list(int_num, int_den))

# Optional shrunken LFCs (for plotting only; classification uses unshrunken
# LFCs, as in the main analysis scripts). ashr is used because it accepts contrasts.
add_shrunk <- function(res, contrast) {
  out <- as.data.frame(res)
  if (requireNamespace("ashr", quietly = TRUE)) {
    shr <- lfcShrink(dds, contrast = contrast, res = res, type = "ashr", quiet = TRUE)
    out$log2FoldChange_ashr <- shr$log2FoldChange
  } else {
    message("ashr not installed; skipping shrunken LFCs")
  }
  cbind(gene = rownames(out), out)
}

tab_siNON   <- add_shrunk(res_siNON,   c("group", "siNON.dex",   "siNON.EtOH"))
tab_siEP300 <- add_shrunk(res_siEP300, c("group", "siEP300.dex", "siEP300.EtOH"))
tab_int     <- cbind(gene = rownames(res_int), as.data.frame(res_int))

# ----------------------------- 4. Sanity check -------------------------------
gr_controls <- c("TSC22D3", "PER1", "FKBP5", "KLF9", "ZBTB16", "DUSP1")
cat("\nGR positive controls, siNON dex vs EtOH:\n")
print(tab_siNON[intersect(gr_controls, rownames(tab_siNON)),
                c("baseMean", "log2FoldChange", "padj")])

# ----------------------------- 5. Export -------------------------------------
dir.create(OUTDIR, showWarnings = FALSE, recursive = TRUE)
out <- function(x, name) {
  write.table(x, file.path(OUTDIR, paste0(PREFIX, "_", name)),
              sep = "\t", quote = FALSE, row.names = FALSE)
}

out(tab_siNON,   "PROseq_DESeq2_siNON_dex_vs_EtOH.tsv")
out(tab_siEP300, "PROseq_DESeq2_siEP300_dex_vs_EtOH.tsv")
out(tab_int,     "PROseq_DESeq2_interaction_siEP300xdex.tsv")

norm_counts <- counts(dds, normalized = TRUE)
out(cbind(gene = rownames(norm_counts), as.data.frame(norm_counts)),
    "PROseq_normalized_counts.tsv")

raw_all <- data.frame(annot, pass_filter = pass_filter, counts_all,
                      check.names = FALSE, row.names = NULL)
out(raw_all, "PROseq_raw_counts_all_genes.tsv")

out(cbind(sampleData, sizeFactor = sizeFactors(dds)), "PROseq_sample_sheet.tsv")

info_file <- file.path(OUTDIR, paste0(PREFIX, "_PROseq_model_info.txt"))
sink(info_file)
cat("Dataset: A549 PRO-seq, siNON/siEP300, dex 1 h vs EtOH; whole-gene strand-specific counts\n")
cat("Counts file:", counts_file, "\n")
cat("Filter: rowSums >", min_counts, "; duplicated gene names: first kept\n")
cat("Design: ~ group (reference siNON.EtOH)\n")
cat("resultsNames:", paste(resultsNames(dds), collapse = ", "), "\n")
cat("Interaction contrast: list(", int_num, ", c(", paste(int_den, collapse = ", "), "))\n")
cat("LFCs in result tables: unshrunken MLE (log2FoldChange); ashr column if present\n\n")
print(sessionInfo())
sink()

cat("\nExported to:", normalizePath(OUTDIR), "\n")
print(list.files(OUTDIR))
