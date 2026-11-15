# =============================================================================
# Transcript -> gene table for GSE104714 (Ensembl transcript IDs, hg19 / Ensembl 75 = GENCODE 19)
# Output: tx2gene_GSE104714.tsv  (tx_id without version, gene_id, gene_name, gene_biotype)
# Requires: BiocManager::install("EnsDb.Hsapiens.v75")
# =============================================================================
source(if (file.exists("R/paths.R")) "R/paths.R" else "paths.R")
P <- read_paths()
out_file <- file.path(P$analysis, "enhancer_gene", "tx2gene_GSE104714.tsv")   # adjust if [layout] enhancer_gene is renamed
dir.create(dirname(out_file), showWarnings = FALSE, recursive = TRUE)
suppressPackageStartupMessages(library(EnsDb.Hsapiens.v75))
edb <- EnsDb.Hsapiens.v75
tx <- transcripts(edb, columns = c("tx_id", "gene_id", "gene_name", "gene_biotype"),
                  return.type = "data.frame")
tx <- unique(tx[, c("tx_id", "gene_id", "gene_name", "gene_biotype")])
write.table(tx, out_file, sep = "\t", quote = FALSE, row.names = FALSE)
cat("Transcripts:", nrow(tx), " genes:", length(unique(tx$gene_id)), "\n")
