# =============================================================================
# BRB-seq after FOXA1 or TEAD4 depletion: dex response (6 h) in siNON, siFOXA1 and siTEAD4 cells
#
#   Figure 6B, 7A               heatmaps of the dex-responsive genes, split by category
#   Figure 6C, 7B               GO Biological Process terms of the categories (selected terms)
#   Supplementary Fig. S10B, S12B   proportional Venn (Euler) diagrams of the six gene sets
#   Supplementary Fig. S10C,D, S12C,D   log2 fold changes of the dex-up / dex-down categories
#   gene lists for LISA         (Supplementary Figures S11 and S13; LISA itself was run outside R)
#
# Model: DESeq2 on all 36 samples (siNON, siFOXA1, siTEAD4 x Veh, Dex, TNF, DexTNF x 3 replicates).
#   Dex effect in siNON cells and the siRNA effects in Veh from ~ silencing + treatment + silencing:treatment
#   (reference levels siNoN, Veh); dex effect in siFOXA1 / siTEAD4 cells from the same samples with one
#   twelve-level group factor and siFoxA1_Veh / siTEAD4_Veh as reference. The TNF samples take part in the
#   dispersion estimates only.
# Genes: chr1-22, X, Y, M; unnamed transcripts removed.
#   No count pre-filter: in the working script the line "rowSums(...) > 10" stood before the count table was
#   built, so it did not act on the table that was analysed. R/05_export_brbseq_tables.R applies that filter,
#   which is why its set sizes differ by a few genes.
# Sets: adjusted p < 0.05 and |log2 fold change| > 0.5.
# GO: clusterProfiler::enrichGO, ontology BP, Benjamini-Hochberg, p and q cut-off 0.05; background = the genes
#   tested in this experiment.
#
# Input : <brbseq_counts>/raw.counts.A549.BRBseq.exons.txt (HOMER count table; columns 9-44 = samples)
# Output: <knockdown>/brbseq_siFOXA1_siTEAD4/
# Run from the repository root:  Rscript R/03_brbseq_siFOXA1_siTEAD4.R
# =============================================================================
library(DESeq2)
library(dplyr)
library(tidyr)
library(eulerr)
library(ComplexHeatmap)
library(circlize)
library(ggplot2)        # stat_summary(fun.data = mean_sdl) also needs the Hmisc package to be installed
library(clusterProfiler)

source(if (file.exists("R/paths.R")) "R/paths.R" else "paths.R")
P <- read_paths()
out_dir <- output_dir(P, "brbseq_siFOXA1_siTEAD4")

clean_genes <- function(x) {
  x <- as.character(x)
  x <- x[!is.na(x)]
  x <- trimws(x)
  x <- x[nzchar(x)]
  sort(unique(x))
}

# ---- Preparing data for DESeq2 ----
rawCounts_exons <- read.delim(file.path(P$brbseq_counts, "raw.counts.A549.BRBseq.exons.txt"))

# Extract gene names from Homer "Annotation.Divergence" column
count_table_exons <- rawCounts_exons %>%
  mutate(gene_name = sub("\\|.*", "", Annotation.Divergence))

# Define normal chromosomes
normal_chrs <- c(paste0("chr", 1:23), "chrX", "chrY", "chrM")

# Add a column to indicate if the transcript is from a normal chromosome
count_table_exons$is_normal_chr <- ifelse(count_table_exons$chr %in% normal_chrs, 1, 0)
count_table_exons <- count_table_exons %>%
  filter(is_normal_chr == 1)

# Calculate the total number of transcripts with a symbol of "0.000" in count_table
symbol_count_exons <- length(count_table_exons[count_table_exons$gene_name == "0.000", ])
symbol_count_exons
# Remove "0.000" symbol containing transcripts
count_table_exons <- count_table_exons %>%
  filter(!gene_name == "0.000")

# Gene list for DGE
genes <- count_table_exons$gene_name
print(paste0("Number of genes:", length(genes)))

# Check that every gene is unique
unique_genes <- unique(count_table_exons$gene_name)
print(paste0("Number of unique genes:", length(genes)))

# Select the right c0lumns
gene_level_counts_deseq_exons <- count_table_exons %>%
  dplyr::select(gene_name, 9:44)

# Change the "format"
gene_level_counts_deseq_exons <- as.data.frame(gene_level_counts_deseq_exons)

# The genes c0lumn to row names!
rownames(gene_level_counts_deseq_exons) <- gene_level_counts_deseq_exons[, 1]

# Remove the first column
gene_level_counts_deseq_exons <- gene_level_counts_deseq_exons[, -1]

### Making sample names look better
original_sample_names <- colnames(gene_level_counts_deseq_exons)

# Split each sample name by underscore and keep only the first 4 parts
shortened_sample_names <- sapply(strsplit(original_sample_names, "_"), function(x) paste(x[1:4], collapse = "_"))

# Set the shortened sample names as column names
colnames(gene_level_counts_deseq_exons) <- shortened_sample_names

selected_cols <- grep("siNoN|siFoxA1|siTEAD4", names(gene_level_counts_deseq_exons), value = TRUE)

siRNA_list <- c("siNoN", "siFoxA1", "siTEAD4") # list of siRNAs
treatment_list <- c("Veh", "Dex", "TNF", "dexTNF")  # list of treatments

pattern <- paste0("(", paste(siRNA_list, collapse = "|"), ")_(", paste(treatment_list, collapse = "|"), ")")

selected_cols <- grep(pattern, names(gene_level_counts_deseq_exons), value = TRUE, ignore.case = TRUE)

name_ending <- paste0("", paste(siRNA_list, collapse = "_"), "_", paste(treatment_list, collapse = "_"))
# Extract the selected columns from the dataframe

gene_level_counts_expressed_deseq_exons <- gene_level_counts_deseq_exons[, selected_cols]

print(paste0("Genes in the count table: ", nrow(gene_level_counts_deseq_exons)))

# ---- Sample table and DESeq2: interaction design (reference siNoN, Veh) ----

sampleData_siRNA <- data.frame(sample = colnames(gene_level_counts_expressed_deseq_exons))

sampleParts <- strsplit(sampleData_siRNA$sample, "_")

sampleData_siRNA$silencing <- sapply(sampleParts, function(x) x[2])
sampleData_siRNA$treatment <- sapply(sampleParts, function(x) x[3])

sampleData_siRNA$silencing <- factor(sampleData_siRNA$silencing)
sampleData_siRNA$treatment <- factor(sampleData_siRNA$treatment)

# Relevel the factors to set the reference levels
sampleData_siRNA$silencing <- relevel(sampleData_siRNA$silencing, ref = "siNoN")
sampleData_siRNA$treatment <- relevel(sampleData_siRNA$treatment, ref = "Veh")

dds_siRNA <- DESeqDataSetFromMatrix(countData = gene_level_counts_expressed_deseq_exons, colData = sampleData_siRNA, design = ~silencing + treatment + silencing:treatment)
dds_siRNA <- DESeq(dds_siRNA)

# ---- DESeq2: dex effect in siFOXA1 cells (group design, reference siFoxA1_Veh) ----
sampleData_siFOXA1_veh <- data.frame(sample = colnames(gene_level_counts_expressed_deseq_exons))

sampleParts_siFOXA1_veh <- strsplit(sampleData_siFOXA1_veh$sample, "_")

sampleData_siFOXA1_veh$treatment <- sapply(sampleParts_siFOXA1_veh, function(x) paste(x[2], x[3], sep = "_"))

sampleData_siFOXA1_veh$treatment <- factor(sampleData_siFOXA1_veh$treatment)

# Relevel the factors to set the reference levels
sampleData_siFOXA1_veh$treatment <- relevel(sampleData_siFOXA1_veh$treatment, ref = "siFoxA1_Veh")

dds_siFOXA1_veh <- DESeqDataSetFromMatrix(countData = gene_level_counts_expressed_deseq_exons, colData = sampleData_siFOXA1_veh, design = ~treatment)
dds_siFOXA1_veh <- DESeq(dds_siFOXA1_veh)

# ---- DESeq2: dex effect in siTEAD4 cells (group design, reference siTEAD4_Veh) ----
sampleData_siTEAD4_veh <- data.frame(sample = colnames(gene_level_counts_expressed_deseq_exons))

sampleParts_siTEAD4_veh <- strsplit(sampleData_siTEAD4_veh$sample, "_")

sampleData_siTEAD4_veh$treatment <- sapply(sampleParts_siTEAD4_veh, function(x) paste(x[2], x[3], sep = "_"))

sampleData_siTEAD4_veh$treatment <- factor(sampleData_siTEAD4_veh$treatment)

# Relevel the factors to set the reference levels
sampleData_siTEAD4_veh$treatment <- relevel(sampleData_siTEAD4_veh$treatment, ref = "siTEAD4_Veh")

dds_siTEAD4_veh <- DESeqDataSetFromMatrix(countData = gene_level_counts_expressed_deseq_exons, colData = sampleData_siTEAD4_veh, design = ~treatment)
dds_siTEAD4_veh <- DESeq(dds_siTEAD4_veh)

# =============================== siFOXA1 ===============================
# ---- siFOXA1: gene sets and Venn diagram (Supplementary Figure S10B) ----
res_silencing_siFoxA1_vs_siNoN <- results(dds_siRNA, name="silencing_siFoxA1_vs_siNoN")
res_treatment_Dex_vs_Veh <- results(dds_siRNA, name="treatment_Dex_vs_Veh")
res_treatment_siFoxA1_Dex_vs_siFoxA1_Veh <- results(dds_siFOXA1_veh, name="treatment_siFoxA1_Dex_vs_siFoxA1_Veh")

# silencing_siFoxA1_vs_siNoN, treatment_Dex_vs_Veh, treatment_siFoxA1_Dex_vs_siFoxA1_Veh
# 1. siFoxA1 vs siNoN
upregulated_genes_siFoxA1_vs_siNoN <- as.data.frame(res_silencing_siFoxA1_vs_siNoN) %>%
  mutate(gene = rownames(res_silencing_siFoxA1_vs_siNoN)) %>%
  filter(padj < 0.05 & log2FoldChange > 0.5) %>%
  pull(gene)

downregulated_genes_siFoxA1_vs_siNoN <- as.data.frame(res_silencing_siFoxA1_vs_siNoN) %>%
  mutate(gene = rownames(res_silencing_siFoxA1_vs_siNoN)) %>%
  filter(padj < 0.05 & log2FoldChange < -0.5) %>%
  pull(gene)

# 2. Dex vs Veh
upregulated_genes_Dex_vs_Veh <- as.data.frame(res_treatment_Dex_vs_Veh) %>%
  mutate(gene = rownames(res_treatment_Dex_vs_Veh)) %>%
  filter(padj < 0.05 & log2FoldChange > 0.5) %>%
  pull(gene)

downregulated_genes_Dex_vs_Veh <- as.data.frame(res_treatment_Dex_vs_Veh) %>%
  mutate(gene = rownames(res_treatment_Dex_vs_Veh)) %>%
  filter(padj < 0.05 & log2FoldChange < -0.5) %>%
  pull(gene)

# 3. siFoxA1 Dex vs siFoxA1 Veh
upregulated_genes_siFoxA1_Dex_vs_siFoxA1_Veh <- as.data.frame(res_treatment_siFoxA1_Dex_vs_siFoxA1_Veh) %>%
  mutate(gene = rownames(res_treatment_siFoxA1_Dex_vs_siFoxA1_Veh)) %>%
  filter(padj < 0.05 & log2FoldChange > 0.5) %>%
  pull(gene)

downregulated_genes_siFoxA1_Dex_vs_siFoxA1_Veh <- as.data.frame(res_treatment_siFoxA1_Dex_vs_siFoxA1_Veh) %>%
  mutate(gene = rownames(res_treatment_siFoxA1_Dex_vs_siFoxA1_Veh)) %>%
  filter(padj < 0.05 & log2FoldChange < -0.5) %>%
  pull(gene)

# Prepare the data with all six categories
venn_data <- list(
  "siNON dex Up" = upregulated_genes_Dex_vs_Veh,
  "siNON dex Dn" = downregulated_genes_Dex_vs_Veh,
  "siFoxA1 Dex Up" = upregulated_genes_siFoxA1_Dex_vs_siFoxA1_Veh,
  "siFoxA1 Dex Down" = downregulated_genes_siFoxA1_Dex_vs_siFoxA1_Veh,
  "siFoxA1 vs siNoN Up" = upregulated_genes_siFoxA1_vs_siNoN,
  "siFoxA1 vs siNoN Down" = downregulated_genes_siFoxA1_vs_siNoN
)

# Load the `eulerr` package
library(eulerr)

# Fit the Euler diagram using `eulerr`
fit <- euler(venn_data)

# Save the plot with counts as labels
png(file.path(out_dir, "20260200_proportional_venn_siFoxA1_vs_siNoN_Dex_vs_Veh_with_counts.png"), width = 800, height = 600)

# Plot with labels showing the number of genes in each section using `quantities`
plot(fit,
     fills = c("#ea0934", "#0b21f7", "#eea9a9", "#9992f7", "#1e792a", "#59f70b"),  # Use reddish and bluish colors for up and down categories
     alpha = 0.6,
     quantities = list(type = c("counts", "percent"), font = 3, round = 2, cex = 0.8), # Display counts explicitly
     main = "Proportional Venn Diagram with Gene Counts\nUp and Down Regulated Genes in Multiple Comparisons"
)

# Close the device to save the file
dev.off()

# Save a clean plot without counts, names, or conditions
png(file.path(out_dir, "20260200_proportional_venn_siFoxA1_vs_siNoN_Dex_vs_Veh_clean.png"), width = 800, height = 600)

# Plot without labels or main title
plot(fit,
     fills = c("#ea0934", "#0b21f7", "#eea9a9", "#9992f7", "#1e792a", "#59f70b"),  # Reddish and bluish colors
     alpha = 0.5,
     quantities = FALSE,  # No counts
     labels = FALSE,
     edges = list(lty = 1)  # Solid edges, no labels
)

# Close the device to save the file
dev.off()

# ---- siFOXA1: heatmap (Figure 6B) ----
# Step 1: Extract normalized counts from DESeq2 object
normalized_counts <- counts(dds_siRNA, normalized = TRUE)
normalized_counts <- as.data.frame(normalized_counts)

# 1. siFoxA1 Dex Up excluding Down in siFoxA1 vs siNoN and excluding Up in Dex vs Veh
siFoxA1_dex_up_excluding_others <- setdiff(
  upregulated_genes_siFoxA1_Dex_vs_siFoxA1_Veh,
  upregulated_genes_Dex_vs_Veh)

name1 <- paste("siFoxA1 Dex Up (n =", length(siFoxA1_dex_up_excluding_others), ")")

# 2. siNON Dex Up excluding Down in siFoxA1 vs siNoN and excluding Up in siFoxA1 Dex
Dex_up_excluding_others <- setdiff(
  upregulated_genes_Dex_vs_Veh,
  upregulated_genes_siFoxA1_Dex_vs_siFoxA1_Veh
)
name2 <- paste("siNON Dex Up (n =", length(Dex_up_excluding_others), ")")

# 3. Shared siFoxA1 Dex Up and siNON Dex Up
shared_dex_up <- intersect(upregulated_genes_siFoxA1_Dex_vs_siFoxA1_Veh, upregulated_genes_Dex_vs_Veh)

name3 <- paste("siFoxA1 Dex Up & siNON Dex Up  (n =", length(shared_dex_up), ")")

# 6. siFoxA1 Dex Down excluding Down in Dex and siFoxA1 Dex Down
siFoxA1_dex_down_excluding_others <- setdiff(
  downregulated_genes_siFoxA1_Dex_vs_siFoxA1_Veh,
  downregulated_genes_Dex_vs_Veh
)
name4 <- paste("siFoxA1 Dex Down (n =", length(siFoxA1_dex_down_excluding_others), ")")

# 7. Dex Down excluding Down in siFoxA1 Dex
Dex_down_excluding_others <- setdiff(
  downregulated_genes_Dex_vs_Veh,
  downregulated_genes_siFoxA1_Dex_vs_siFoxA1_Veh
)
name5 <- paste("siNON Dex Down (n =", length(Dex_down_excluding_others), ")")

# 8. Shared siFoxA1 Dex Down & siNON Dex Down
shared_dex_down <- intersect(
  downregulated_genes_siFoxA1_Dex_vs_siFoxA1_Veh,
  downregulated_genes_Dex_vs_Veh
)
name6 <- paste("Shared siFoxA1 Dex Down & siNON Dex Down (n =", length(shared_dex_down), ")")

# Combine all updated categories into a named list
gene_categories <- list(
  name1 = siFoxA1_dex_up_excluding_others,
  name2 = Dex_up_excluding_others,
  name3 = shared_dex_up,
  name4 = siFoxA1_dex_down_excluding_others,
  name5 = Dex_down_excluding_others,
  name6 = shared_dex_down
)

# Assign descriptive names to the categories
names(gene_categories) <- c(
  paste("siFoxA1 Dex Up (n =", length(siFoxA1_dex_up_excluding_others), ")"),
  paste("siNON Dex Up (n =", length(Dex_up_excluding_others), ")"),
  paste("siFoxA1 Dex Up & siNON Dex Up (n =", length(shared_dex_up), ")"),
  paste("siFoxA1 Dex Down (n =", length(siFoxA1_dex_down_excluding_others), ")"),
  paste("siNON Dex Down (n =", length(Dex_down_excluding_others), ")"),
  paste("siFoxA1 Dex Down & siNON Dex Down (n =", length(shared_dex_down), ")")
)

# Print the length of each category
category_lengths <- sapply(gene_categories, length)
print(category_lengths)

# Prepare the data for the heatmap
# Combine all unique genes from the categories
all_genes <- unique(unlist(gene_categories))

# Extract the normalized counts for these genes
heatmap_data <- normalized_counts[all_genes, ]

# Subset the normalized counts to include only the required columns and exclude DexTNF
columns_to_include <- grep(
  pattern = "siNoN_Veh|siNoN_Dex|siFoxA1_Veh|siFoxA1_Dex",
  x = colnames(normalized_counts),
  value = TRUE
)
columns_to_exclude <- grep(
  pattern = "DexTNF",
  x = columns_to_include,
  value = TRUE,
  invert = TRUE  # Exclude matches to "DexTNF"
)

# Define the desired order of columns
desired_order <- c(
  grep("siNoN_Veh", columns_to_exclude, value = TRUE),
  grep("siNoN_Dex", columns_to_exclude, value = TRUE),
  grep("siFoxA1_Veh", columns_to_exclude, value = TRUE),
  grep("siFoxA1_Dex", columns_to_exclude, value = TRUE)
)

# Subset and reorder the heatmap data
heatmap_data <- normalized_counts[all_genes, desired_order]

# Step 1: Scale the data as z-scores across rows
z_scored_heatmap_data <- t(scale(t(heatmap_data)))  # Calculate z-scores by rows
z_scored_heatmap_data[is.na(z_scored_heatmap_data)] <- 0  # Replace any NaNs with 0 for proper visualization

# Create an annotation data frame to separate rows by category with descriptive names
annotation_row <- data.frame(Category = rep(names(gene_categories), sapply(gene_categories, length)))
rownames(annotation_row) <- unlist(gene_categories)

# Assign colors to each unique category
unique_categories <- unique(annotation_row$Category)
category_colors <- setNames(
  c("red", "blue", "purple", "orange", "green", "yellow")[1:length(unique_categories)],
  unique_categories
)

# Plot the heatmap with the specified column order
png(file.path(out_dir, "20241126_Ordered_siFOXA1_Differentially_Regulated_Enhancers_Heatmap.png"), width = 1200, height = 1200, res = 150)
Heatmap(
  z_scored_heatmap_data,
  name = "Z-Score",
  row_names_side = "left",
  cluster_rows = FALSE,
  cluster_columns = FALSE,
  show_row_names = FALSE,
  show_column_names = TRUE,
  row_split = annotation_row$Category,  # Split rows by category
  left_annotation = rowAnnotation(Category = annotation_row$Category, col = list(Category = category_colors)),
  col = colorRamp2(c(-2, 0, 2), c("blue", "white", "red")),
  heatmap_legend_param = list(title = "Z-Score")
)
dev.off()

# Gene lists for LISA (siFOXA1 comparison), one symbol per line
lisa_dir <- file.path(out_dir, "lisa_inputs_siFOXA1_vs_siNON")
dir.create(lisa_dir, showWarnings = FALSE, recursive = TRUE)
lisa_sets <- list(
  siFOXA1_Dex_Up_Unique          = siFoxA1_dex_up_excluding_others,
  Shared_siNON_siFOXA1_Dex_Up    = shared_dex_up,
  siNON_Dex_Up_Unique         = Dex_up_excluding_others,
  siFOXA1_Dex_Down_Unique        = siFoxA1_dex_down_excluding_others,
  Shared_siNON_siFOXA1_Dex_Down  = shared_dex_down,
  siNON_Dex_Down_Unique       = Dex_down_excluding_others
)
for (nm in names(lisa_sets)) {
  writeLines(clean_genes(lisa_sets[[nm]]), file.path(lisa_dir, paste0(nm, ".txt")))
  message(nm, ": ", length(clean_genes(lisa_sets[[nm]])), " genes")
}

# ---- siFOXA1: GO (Figure 6C) ----
# org.Hs.eg.db is attached only here: AnnotationDbi masks dplyr::select().
library(org.Hs.eg.db)
# Background of every GO test: the genes tested in this experiment (rows of the DESeq2 model).
go_universe <- rownames(dds_siRNA)

# Define gene categories for siFOXA1
gene_categories <- list(
  siFoxA1_dex_up________________ = siFoxA1_dex_up_excluding_others,
  Dex_up = Dex_up_excluding_others,
  shared_dex_up = shared_dex_up,
  siFoxA1_dex_down = siFoxA1_dex_down_excluding_others,
  Dex_down = Dex_down_excluding_others,
  shared_dex_down = shared_dex_down
)

# Initialize an empty data frame to store all enrichment results
all_enrich_results <- data.frame()

# Define a function to perform GO enrichment and store results in a table
run_enrichGO <- function(gene_symbols, condition_name) {
  if (length(gene_symbols) > 0) {
    ego <- enrichGO(
      gene = gene_symbols,
      universe = go_universe,
      OrgDb = org.Hs.eg.db,
      keyType = "SYMBOL",
      ont = "BP",
      pAdjustMethod = "BH",
      pvalueCutoff = 0.05,
      qvalueCutoff = 0.05,
      readable = TRUE
    )

    # Check if `ego` has any results before converting it to a data frame
    if (!is.null(ego) && nrow(as.data.frame(ego)) > 0) {
      ego_df <- as.data.frame(ego)
      ego_df$Condition <- condition_name
      return(ego_df)
    } else {
      cat("\nCondition:", condition_name, "has no significant GO terms.\n")
      return(NULL)
    }
  } else {
    cat("\nCondition:", condition_name, "has no genes for enrichment analysis.\n")
    return(NULL)
  }
}

# Run enrichment analysis for each category and combine results
for (condition_name in names(gene_categories)) {
  gene_symbols <- gene_categories[[condition_name]]
  result <- run_enrichGO(gene_symbols, condition_name)
  if (!is.null(result)) {
    all_enrich_results <- bind_rows(all_enrich_results, result)
  }
}

# Exclude conditions with no significant GO terms
significant_conditions <- unique(all_enrich_results$Condition)
gene_categories <- gene_categories[names(gene_categories) %in% significant_conditions]

# Modify the `Description` column to shorten terms
all_enrich_results$Description <- all_enrich_results$Description %>%
  gsub("\\bregulation\\b", "reg.", .) %>%
  gsub("\\bof biological process\\b", "of bio. proc.", .)

# Select top 10 GO terms for each condition based on p.adjust
selected_terms_ordered <- all_enrich_results %>%
  filter(p.adjust <= 0.05) %>%
  group_by(Condition) %>%
  arrange(p.adjust) %>%
  slice_head(n = 15) %>%
  ungroup() %>%
  distinct(Description) %>%
  pull(Description)

# Expand data to ensure all terms and conditions are included
expanded_data <- expand.grid(Condition = names(gene_categories), Description = selected_terms_ordered) %>%
  left_join(all_enrich_results %>% filter(Description %in% selected_terms_ordered),
            by = c("Condition", "Description"))

# Remove rows where `Condition` is not in `gene_categories`
expanded_data <- expanded_data %>% filter(Condition %in% names(gene_categories))

# Calculate bubble sizes from GeneRatio and BgRatio
expanded_data <- expanded_data %>%
  separate(GeneRatio, into = c("GeneRatio_Num", "GeneRatio_Denom"), sep = "/", fill = "right") %>%
  separate(BgRatio, into = c("BgRatio_Num", "BgRatio_Denom"), sep = "/", fill = "right") %>%
  mutate(
    GeneRatio_Num = as.numeric(GeneRatio_Num),
    GeneRatio_Denom = as.numeric(GeneRatio_Denom),
    BgRatio_Num = as.numeric(BgRatio_Num),
    BgRatio_Denom = as.numeric(BgRatio_Denom),
    BubbleSize = ifelse(!is.na(GeneRatio_Num) & !is.na(GeneRatio_Denom),
                        (GeneRatio_Num / GeneRatio_Denom) * 100, NA)
  )

# Replace "regulation" and "of biological process" in GO terms
expanded_data$Description <- expanded_data$Description %>%
  gsub("\\bregulation\\b", "reg.", .) %>%
  gsub("\\bof biological process\\b", "of bio. proc.", .)

# Update factor levels for `Description`
expanded_data$Description <- factor(expanded_data$Description, levels = unique(expanded_data$Description))
write.table(expanded_data, file = file.path(out_dir, "A549_BRBseq_siFOXA1_genes_expanded_data_GO.txt"), sep = "\t", row.names = TRUE, col.names = NA, quote = FALSE)

biological_processes <- list(
  "negative reg. of apoptotic signaling pathway",
  "cellular response to chemical stress",
  "intracellular receptor signaling pathway",
  "positive reg. of canonical NF-kappaB signal transduction",
  "epithelial cell migration",
  "insulin receptor signaling pathway",
  "negative reg. of hormone secretion",
  "retinoid metabolic process",
  "mitotic metaphase chromosome alignment",
  "mitotic cell cycle checkpoint signaling",
  "negative reg. of cell cycle",
  "positive reg. of fat cell differentiation"
)

filtered_data <- expanded_data[grepl(paste(biological_processes, collapse = "|"), expanded_data$Description), ]
# Create a horizontal bubble plot
bubble_plot <- ggplot(filtered_data, aes(x = Description, y = Condition)) +
  geom_point(aes(size = BubbleSize, color = -log10(p.adjust)), na.rm = TRUE) +
  scale_size_continuous(name = "Gene Ratio %", range = c(1, 5)) +
  scale_color_gradient(name = "-log10(p.adjust)", low = "#B3CDE3", high = "#011F4B") +
  labs(
    title = "Top 15 Enrichment Bubble Plot for GO:BP Categories (siFOXA1)",
    x = "GO Term",
    y = "Condition"
  ) +
  theme_classic() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 10),
    axis.text.y = element_text(size = 10)
  )

# Display the plot
print(bubble_plot)

# Save the plot to a file
ggsave(file.path(out_dir, "20241216_GO_enrichment_bubble_plot_horizontal_siFOXA1_filtered_filtered.png"),
  plot = bubble_plot,
  width = 8,
  height = 5,
  dpi = 600
)

# ---- siFOXA1: log2 fold changes per category (Supplementary Figure S10C, D) ----
# Define gene categories for upregulated genes
gene_categories <- list(
  siNON = Dex_up_excluding_others,
  siFOXA1 = siFoxA1_dex_up_excluding_others,
  shared = shared_dex_up
)

# Initialize an empty data frame for storing log2FC values and categories
log2fc_data <- data.frame()

# Extract log2FC values for each category and bind them into a single data frame
for (category_name in names(gene_categories)) {
  genes <- gene_categories[[category_name]]

  # Get log2FC values for siFOXA1 Dex vs Veh
  siFOXA1_dex_log2fc <- as.data.frame(res_treatment_siFoxA1_Dex_vs_siFoxA1_Veh) %>%
    mutate(gene = rownames(res_treatment_siFoxA1_Dex_vs_siFoxA1_Veh)) %>%
    filter(gene %in% genes) %>%
    dplyr::select(gene, log2FoldChange) %>%
    rename(log2FC = log2FoldChange) %>%
    mutate(Comparison = "siFOXA1_Dex", Category = category_name)

  # Get log2FC values for siNON Dex vs Veh
  siNON_dex_log2fc <- as.data.frame(res_treatment_Dex_vs_Veh) %>%
    mutate(gene = rownames(res_treatment_Dex_vs_Veh)) %>%
    filter(gene %in% genes) %>%
    dplyr::select(gene, log2FoldChange) %>%
    rename(log2FC = log2FoldChange) %>%
    mutate(Comparison = "siNON_Dex", Category = category_name)

  # Combine log2FC data for all comparisons
  category_data <- bind_rows(siFOXA1_dex_log2fc, siNON_dex_log2fc)
  log2fc_data <- bind_rows(log2fc_data, category_data)
}

# Update `Category` labels in the `log2fc_data` data frame with the desired order
log2fc_data$Category <- factor(log2fc_data$Category, levels = c("siNON", "siFOXA1", "shared"))

# Set the factor levels for `Comparison`
log2fc_data$Comparison <- factor(log2fc_data$Comparison, levels = c("siNON_Dex", "siFOXA1_Dex"))

# Plot the box plots for upregulated genes
boxplot_plot <- ggplot(log2fc_data, aes(x = Category, y = log2FC, fill = Comparison)) +
  geom_boxplot(alpha = 0.6) +  # Boxplot with transparency
  stat_summary(fun = mean, geom = "point", shape = 18, size = 3, color = "black", position = position_dodge(0.75)) +
  stat_summary(fun.data = mean_sdl, fun.args = list(mult = 1), geom = "errorbar", width = 0.2, color = "black", position = position_dodge(0.75)) +
  geom_hline(yintercept = 0, color = "black", size = 1, linetype = "dashed") +
  labs(title = "Log2 Fold Change Box Plots for Upregulated Gene Categories",
       x = "Gene Category",
       y = "Log2 Fold Change") +
  scale_fill_manual(values = c("siNON_Dex" = "#ea0934", "siFOXA1_Dex" = "#eea9a9")) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.background = element_rect(fill = "white", color = "white"),
    plot.background = element_rect(fill = "white", color = "white")
  )

# Display the plot
print(boxplot_plot)

# Save the plot as a PNG file
ggsave(file.path(out_dir, "20260200_log2FC_boxplot_with_mean_sd_by_category_genes_up_updated_siFOXA1.png"), plot = boxplot_plot, width = 8, height = 8, dpi = 300)

# Define gene categories for downregulated genes
gene_categories <- list(
  siNON = Dex_down_excluding_others,
  siFOXA1 = siFoxA1_dex_down_excluding_others,
  shared = shared_dex_down
)

# Initialize an empty data frame for storing log2FC values and categories
log2fc_data <- data.frame()

# Extract log2FC values for each category and bind them into a single data frame
for (category_name in names(gene_categories)) {
  genes <- gene_categories[[category_name]]

  # Get log2FC values for siFOXA1 Dex vs Veh
  siFOXA1_dex_log2fc <- as.data.frame(res_treatment_siFoxA1_Dex_vs_siFoxA1_Veh) %>%
    mutate(gene = rownames(res_treatment_siFoxA1_Dex_vs_siFoxA1_Veh)) %>%
    filter(gene %in% genes) %>%
    dplyr::select(gene, log2FoldChange) %>%
    rename(log2FC = log2FoldChange) %>%
    mutate(Comparison = "siFOXA1_Dex", Category = category_name)

  # Get log2FC values for siNON Dex vs Veh
  siNON_dex_log2fc <- as.data.frame(res_treatment_Dex_vs_Veh) %>%
    mutate(gene = rownames(res_treatment_Dex_vs_Veh)) %>%
    filter(gene %in% genes) %>%
    dplyr::select(gene, log2FoldChange) %>%
    rename(log2FC = log2FoldChange) %>%
    mutate(Comparison = "siNON_Dex", Category = category_name)

  # Combine log2FC data for all comparisons
  category_data <- bind_rows(siFOXA1_dex_log2fc, siNON_dex_log2fc)
  log2fc_data <- bind_rows(log2fc_data, category_data)
}

# Update `Category` labels in the `log2fc_data` data frame with the desired order
log2fc_data$Category <- factor(log2fc_data$Category, levels = c("siNON", "siFOXA1", "shared"))

# Set the factor levels for `Comparison`
log2fc_data$Comparison <- factor(log2fc_data$Comparison, levels = c("siNON_Dex", "siFOXA1_Dex"))

# Plot the box plots for downregulated genes
boxplot_plot <- ggplot(log2fc_data, aes(x = Category, y = log2FC, fill = Comparison)) +
  geom_boxplot(alpha = 0.6) +  # Boxplot with transparency
  stat_summary(fun = mean, geom = "point", shape = 18, size = 3, color = "black", position = position_dodge(0.75)) +
  stat_summary(fun.data = mean_sdl, fun.args = list(mult = 1), geom = "errorbar", width = 0.2, color = "black", position = position_dodge(0.75)) +
  geom_hline(yintercept = 0, color = "black", size = 1, linetype = "dashed") +
  labs(title = "Log2 Fold Change Box Plots for Downregulated Gene Categories",
       x = "Gene Category",
       y = "Log2 Fold Change") +
  scale_fill_manual(values = c("siNON_Dex" = "#0b21f7", "siFOXA1_Dex" = "#9992f7")) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.background = element_rect(fill = "white", color = "white"),
    plot.background = element_rect(fill = "white", color = "white")
  )

# Display the plot
print(boxplot_plot)

# Save the plot as a PNG file
ggsave(file.path(out_dir, "20250200_log2FC_boxplot_with_mean_sd_for_downregulated_categories_updated_siFOXA1.png"), plot = boxplot_plot, width = 8, height = 8, dpi = 300)

# =============================== siTEAD4 ===============================
# ---- siTEAD4: gene sets and Venn diagram (Supplementary Figure S12B) ----
# Extract DESeq2 results for siTEAD4
res_silencing_siTEAD4_vs_siNoN <- results(dds_siRNA, name = "silencing_siTEAD4_vs_siNoN")
res_treatment_Dex_vs_Veh <- results(dds_siRNA, name = "treatment_Dex_vs_Veh")
res_treatment_siTEAD4_Dex_vs_siTEAD4_Veh <- results(dds_siTEAD4_veh, name = "treatment_siTEAD4_Dex_vs_siTEAD4_Veh")

# 1. siTEAD4 vs siNoN
upregulated_genes_siTEAD4_vs_siNoN <- as.data.frame(res_silencing_siTEAD4_vs_siNoN) %>%
  mutate(gene = rownames(res_silencing_siTEAD4_vs_siNoN)) %>%
  filter(padj < 0.05 & log2FoldChange > 0.5) %>%
  pull(gene)

downregulated_genes_siTEAD4_vs_siNoN <- as.data.frame(res_silencing_siTEAD4_vs_siNoN) %>%
  mutate(gene = rownames(res_silencing_siTEAD4_vs_siNoN)) %>%
  filter(padj < 0.05 & log2FoldChange < -0.5) %>%
  pull(gene)

# 2. Dex vs Veh
upregulated_genes_Dex_vs_Veh <- as.data.frame(res_treatment_Dex_vs_Veh) %>%
  mutate(gene = rownames(res_treatment_Dex_vs_Veh)) %>%
  filter(padj < 0.05 & log2FoldChange > 0.5) %>%
  pull(gene)

downregulated_genes_Dex_vs_Veh <- as.data.frame(res_treatment_Dex_vs_Veh) %>%
  mutate(gene = rownames(res_treatment_Dex_vs_Veh)) %>%
  filter(padj < 0.05 & log2FoldChange < -0.5) %>%
  pull(gene)

# 3. siTEAD4 Dex vs siTEAD4 Veh
upregulated_genes_siTEAD4_Dex_vs_siTEAD4_Veh <- as.data.frame(res_treatment_siTEAD4_Dex_vs_siTEAD4_Veh) %>%
  mutate(gene = rownames(res_treatment_siTEAD4_Dex_vs_siTEAD4_Veh)) %>%
  filter(padj < 0.05 & log2FoldChange > 0.5) %>%
  pull(gene)

downregulated_genes_siTEAD4_Dex_vs_siTEAD4_Veh <- as.data.frame(res_treatment_siTEAD4_Dex_vs_siTEAD4_Veh) %>%
  mutate(gene = rownames(res_treatment_siTEAD4_Dex_vs_siTEAD4_Veh)) %>%
  filter(padj < 0.05 & log2FoldChange < -0.5) %>%
  pull(gene)

# Prepare the data with all six categories
venn_data <- list(
  "siNON Dex Up" = upregulated_genes_Dex_vs_Veh,
  "siNON Dex Down" = downregulated_genes_Dex_vs_Veh,
  "siTEAD4 Dex Up" = upregulated_genes_siTEAD4_Dex_vs_siTEAD4_Veh,
  "siTEAD4 Dex Down" = downregulated_genes_siTEAD4_Dex_vs_siTEAD4_Veh,
  "siTEAD4 vs siNoN Up" = upregulated_genes_siTEAD4_vs_siNoN,
  "siTEAD4 vs siNoN Down" = downregulated_genes_siTEAD4_vs_siNoN
)

fit <- euler(venn_data)
png(file.path(out_dir, "20260200_proportional_venn_siTEAD4_vs_siNoN_Dex_vs_Veh_with_counts.png"), width = 800, height = 600)
plot(fit,
     fills = c("#ea0934", "#0b21f7", "#eea9a9", "#9992f7", "#1e792a", "#59f70b"),
     alpha = 0.6,
     quantities = list(type = c("counts", "percent"), font = 3, round = 2, cex = 0.8),
     main = "Proportional Venn Diagram with Gene Counts\nUp and Down Regulated Genes in Multiple Comparisons"
)
dev.off()

# ---- siTEAD4: heatmap (Figure 7A) ----
# Step 1: Extract normalized counts from DESeq2 object
normalized_counts <- counts(dds_siRNA, normalized = TRUE)
normalized_counts <- as.data.frame(normalized_counts)

# 1. siTEAD4 Dex Up excluding Down in siTEAD4 vs siNoN and excluding Up in Dex vs Veh
siTEAD4_dex_up_excluding_others <- setdiff(
  upregulated_genes_siTEAD4_Dex_vs_siTEAD4_Veh,
  upregulated_genes_Dex_vs_Veh
)
name1 <- paste("siTEAD4 Dex Up (n =", length(siTEAD4_dex_up_excluding_others), ")")

# 2. siNON Dex Up excluding Down in siTEAD4 vs siNoN and excluding Up in siTEAD4 Dex
Dex_up_excluding_others <- setdiff(
  upregulated_genes_Dex_vs_Veh,
  upregulated_genes_siTEAD4_Dex_vs_siTEAD4_Veh
)
name2 <- paste("siNON Dex Up (n =", length(Dex_up_excluding_others), ")")

# 3. Shared siTEAD4 Dex Up and siNON Dex Up
shared_dex_up <- intersect(upregulated_genes_siTEAD4_Dex_vs_siTEAD4_Veh, upregulated_genes_Dex_vs_Veh)
name3 <- paste("siTEAD4 Dex Up & siNON Dex Up (n =", length(shared_dex_up), ")")

# 4. siTEAD4 Dex Down excluding Down in Dex and siTEAD4 Dex Down
siTEAD4_dex_down_excluding_others <- setdiff(
  downregulated_genes_siTEAD4_Dex_vs_siTEAD4_Veh,
  downregulated_genes_Dex_vs_Veh
)
name4 <- paste("siTEAD4 Dex Down (n =", length(siTEAD4_dex_down_excluding_others), ")")

# 5. Dex Down excluding Down in siTEAD4 Dex
Dex_down_excluding_others <- setdiff(
  downregulated_genes_Dex_vs_Veh,
  downregulated_genes_siTEAD4_Dex_vs_siTEAD4_Veh
)
name5 <- paste("siNON Dex Down (n =", length(Dex_down_excluding_others), ")")

# 6. Shared siTEAD4 Dex Down & siNON Dex Down
shared_dex_down <- intersect(
  downregulated_genes_siTEAD4_Dex_vs_siTEAD4_Veh,
  downregulated_genes_Dex_vs_Veh
)
name6 <- paste("siTEAD4 Dex Down & siNON Dex Down (n =", length(shared_dex_down), ")")

# Step 1: Combine all updated categories into a named list
gene_categories <- list(
  siTEAD4_dex_up_excluding_others,
  Dex_up_excluding_others,
  shared_dex_up,
  siTEAD4_dex_down_excluding_others,
  Dex_down_excluding_others,
  shared_dex_down
)

# Assign descriptive names to the categories
names(gene_categories) <- c(
  paste("siTEAD4 Dex Up (n =", length(siTEAD4_dex_up_excluding_others), ")"),
  paste("siNON Dex Up (n =", length(Dex_up_excluding_others), ")"),
  paste("siTEAD4 Dex Up & siNON Dex Up (n =", length(shared_dex_up), ")"),
  paste("siTEAD4 Dex Down (n =", length(siTEAD4_dex_down_excluding_others), ")"),
  paste("siNON Dex Down (n =", length(Dex_down_excluding_others), ")"),
  paste("siTEAD4 Dex Down & siNON Dex Down (n =", length(shared_dex_down), ")")
)

# Step 2: Remove duplicates (HOXA-AS3 and HOXA7) from gene categories
gene_categories <- lapply(gene_categories, function(category) {
  category[!category %in% c("HOXA-AS3", "HOXA7")]
})

# Print the length of each category after removing duplicates
category_lengths <- sapply(gene_categories, length)
print(category_lengths)

# Step 3: Combine all unique genes from the updated categories
all_genes <- unique(unlist(gene_categories))

# Step 4: Extract normalized counts for these genes
heatmap_data <- normalized_counts[all_genes, ]

# Step 5: Subset the normalized counts to include only required columns (exclude DexTNF)
columns_to_include <- grep(
  pattern = "siNoN_Veh|siNoN_Dex|siTEAD4_Veh|siTEAD4_Dex",
  x = colnames(normalized_counts),
  value = TRUE
)
columns_to_exclude <- grep(
  pattern = "DexTNF",
  x = columns_to_include,
  value = TRUE,
  invert = TRUE  # Exclude matches to "DexTNF"
)

# Step 6: Define the desired order of columns
desired_order <- c(
  grep("siNoN_Veh", columns_to_exclude, value = TRUE),
  grep("siNoN_Dex", columns_to_exclude, value = TRUE),
  grep("siTEAD4_Veh", columns_to_exclude, value = TRUE),
  grep("siTEAD4_Dex", columns_to_exclude, value = TRUE)
)

# Subset and reorder the heatmap data
heatmap_data <- heatmap_data[, desired_order]

# Step 7: Scale the data as z-scores across rows
z_scored_heatmap_data <- t(scale(t(heatmap_data)))  # Calculate z-scores by rows
z_scored_heatmap_data[is.na(z_scored_heatmap_data)] <- 0  # Replace any NaNs with 0

# Step 8: Create an annotation data frame to separate rows by category
annotation_row <- data.frame(Category = rep(names(gene_categories), sapply(gene_categories, length)))
rownames(annotation_row) <- unlist(gene_categories)

# Ensure annotation_row matches z_scored_heatmap_data
annotation_row <- annotation_row[rownames(z_scored_heatmap_data), , drop = FALSE]

# Step 9: Assign colors to each unique category
unique_categories <- unique(annotation_row$Category)
category_colors <- setNames(
  c("red", "blue", "purple", "orange", "green", "yellow")[1:length(unique_categories)],
  unique_categories
)

# Step 10: Plot the heatmap with the specified column order
png(file.path(out_dir, "20241126_Ordered_siTEAD4_Differentially_Regulated_Enhancers_Heatmap.png"), width = 1200, height = 1200, res = 150)
Heatmap(
  z_scored_heatmap_data,
  name = "Z-Score",
  row_names_side = "left",
  cluster_rows = FALSE,
  cluster_columns = FALSE,
  show_row_names = FALSE,
  show_column_names = TRUE,
  row_split = annotation_row$Category,  # Split rows by category
  left_annotation = rowAnnotation(Category = annotation_row$Category, col = list(Category = category_colors)),
  col = colorRamp2(c(-2, 0, 2), c("blue", "white", "red")),
  heatmap_legend_param = list(title = "Z-Score")
)
dev.off()

# Gene lists for LISA (siTEAD4 comparison), one symbol per line
lisa_dir <- file.path(out_dir, "lisa_inputs_siTEAD4_vs_siNON")
dir.create(lisa_dir, showWarnings = FALSE, recursive = TRUE)
lisa_sets <- list(
  siTEAD4_Dex_Up_Unique          = siTEAD4_dex_up_excluding_others,
  Shared_siNON_siTEAD4_Dex_Up    = shared_dex_up,
  siNON_Dex_Up_Unique         = Dex_up_excluding_others,
  siTEAD4_Dex_Down_Unique        = siTEAD4_dex_down_excluding_others,
  Shared_siNON_siTEAD4_Dex_Down  = shared_dex_down,
  siNON_Dex_Down_Unique       = Dex_down_excluding_others
)
for (nm in names(lisa_sets)) {
  writeLines(clean_genes(lisa_sets[[nm]]), file.path(lisa_dir, paste0(nm, ".txt")))
  message(nm, ": ", length(clean_genes(lisa_sets[[nm]])), " genes")
}

# ---- siTEAD4: GO (Figure 7B) ----

# Define gene categories for siTEAD4
gene_categories <- list(
  siTEAD4_dex_up_________ = siTEAD4_dex_up_excluding_others,
  Dex_up = Dex_up_excluding_others,
  shared_dex_up = shared_dex_up,
  siTEAD4_dex_down_____________ = siTEAD4_dex_down_excluding_others,
  Dex_down = Dex_down_excluding_others,
  shared_dex_down = shared_dex_down
)

# Initialize an empty data frame to store all enrichment results
all_enrich_results <- data.frame()

# Define a function to perform GO enrichment and store results in a table
run_enrichGO <- function(gene_symbols, condition_name) {
  if (length(gene_symbols) > 0) {
    ego <- enrichGO(
      gene = gene_symbols,
      universe = go_universe,
      OrgDb = org.Hs.eg.db,
      keyType = "SYMBOL",
      ont = "BP",
      pAdjustMethod = "BH",
      pvalueCutoff = 0.05,
      qvalueCutoff = 0.05,
      readable = TRUE
    )

    # Check if `ego` has any results before converting it to a data frame
    if (!is.null(ego) && nrow(as.data.frame(ego)) > 0) {
      ego_df <- as.data.frame(ego)
      ego_df$Condition <- condition_name
      return(ego_df)
    } else {
      cat("\nCondition:", condition_name, "has no significant GO terms.\n")
      return(NULL)
    }
  } else {
    cat("\nCondition:", condition_name, "has no genes for enrichment analysis.\n")
    return(NULL)
  }
}

# Run enrichment analysis for each category and combine results
for (condition_name in names(gene_categories)) {
  gene_symbols <- gene_categories[[condition_name]]
  result <- run_enrichGO(gene_symbols, condition_name)
  if (!is.null(result)) {
    all_enrich_results <- bind_rows(all_enrich_results, result)
  }
}

# Exclude conditions with no significant GO terms
significant_conditions <- unique(all_enrich_results$Condition)
gene_categories <- gene_categories[names(gene_categories) %in% significant_conditions]

# Modify the `Description` column to shorten terms
all_enrich_results$Description <- all_enrich_results$Description %>%
  gsub("\\bregulation\\b", "reg.", .) %>%
  gsub("\\bof biological process\\b", "of bio. proc.", .)

# Select top 10 GO terms for each condition based on p.adjust
selected_terms_ordered <- all_enrich_results %>%
  filter(p.adjust <= 0.05) %>%
  group_by(Condition) %>%
  arrange(p.adjust) %>%
  slice_head(n = 15) %>%
  ungroup() %>%
  distinct(Description) %>%
  pull(Description)

# Expand data to ensure all terms and conditions are included
expanded_data <- expand.grid(Condition = names(gene_categories), Description = selected_terms_ordered) %>%
  left_join(all_enrich_results %>% filter(Description %in% selected_terms_ordered),
            by = c("Condition", "Description"))

# Remove rows where `Condition` is not in `gene_categories`
expanded_data <- expanded_data %>% filter(Condition %in% names(gene_categories))

# Calculate bubble sizes from GeneRatio and BgRatio
expanded_data <- expanded_data %>%
  separate(GeneRatio, into = c("GeneRatio_Num", "GeneRatio_Denom"), sep = "/", fill = "right") %>%
  separate(BgRatio, into = c("BgRatio_Num", "BgRatio_Denom"), sep = "/", fill = "right") %>%
  mutate(
    GeneRatio_Num = as.numeric(GeneRatio_Num),
    GeneRatio_Denom = as.numeric(GeneRatio_Denom),
    BgRatio_Num = as.numeric(BgRatio_Num),
    BgRatio_Denom = as.numeric(BgRatio_Denom),
    BubbleSize = ifelse(!is.na(GeneRatio_Num) & !is.na(GeneRatio_Denom),
                        (GeneRatio_Num / GeneRatio_Denom) * 100, NA)
  )

# Replace "regulation" and "of biological process" in GO terms
expanded_data$Description <- expanded_data$Description %>%
  gsub("\\bregulation\\b", "reg.", .) %>%
  gsub("\\bof biological process\\b", "of bio. proc.", .)

# Update factor levels for `Description`
expanded_data$Description <- factor(expanded_data$Description, levels = unique(expanded_data$Description))

write.table(expanded_data, file = file.path(out_dir, "A549_BRBseq_siTEAD4_genes_expanded_data_GO.txt"), sep = "\t", row.names = TRUE, col.names = NA, quote = FALSE)

biological_processes <- list(
  "negative reg. of apoptotic signaling pathway",
  "response to oxidative stress",
  "intracellular receptor signaling pathway",
  "positive reg. of canonical NF-kappaB signal transduction",
  "epithelial cell migration",
  "cellular response to starvation",
  "insulin receptor signaling pathway",
  "tissue migration",
  "negative reg. of hormone secretion",
  "negative reg. of cell cycle",
  "reg. of chromosome organization",
  "mitotic sister chromatid segregation"
)
filtered_data <- expanded_data[grepl(paste(biological_processes, collapse = "|"), expanded_data$Description), ]
# Create a horizontal bubble plot
bubble_plot <- ggplot(filtered_data, aes(x = Description, y = Condition)) +
  geom_point(aes(size = BubbleSize, color = -log10(p.adjust)), na.rm = TRUE) +
  scale_size_continuous(name = "Gene Ratio %", range = c(1, 5)) +
  scale_color_gradient(name = "-log10(p.adjust)", low = "#B3CDE3", high = "#011F4B") +
  labs(
    title = "Top 10 Enrichment Bubble Plot for GO:BP Categories (siTEAD4)",
    x = "GO Term",
    y = "Condition"
  ) +
  theme_classic() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 10),
    axis.text.y = element_text(size = 10)
  )

# Display the plot
print(bubble_plot)

# Save the plot to a file
ggsave(file.path(out_dir, "20241216_GO_enrichment_bubble_plot_horizontal_siTEAD4_filtered.png"),
  plot = bubble_plot,
  width = 8,
  height = 4.5,
  dpi = 600
)

# ---- siTEAD4: log2 fold changes per category (Supplementary Figure S12C, D) ----
# Define gene categories for upregulated genes
gene_categories <- list(
  siNON = Dex_up_excluding_others,
  siTEAD4 = siTEAD4_dex_up_excluding_others,
  shared = shared_dex_up
)

# Initialize an empty data frame for storing log2FC values and categories
log2fc_data <- data.frame()

# Extract log2FC values for each category and bind them into a single data frame
for (category_name in names(gene_categories)) {
  genes <- gene_categories[[category_name]]

  # Get log2FC values for siTEAD4 Dex vs Veh
  siTEAD4_dex_log2fc <- as.data.frame(res_treatment_siTEAD4_Dex_vs_siTEAD4_Veh) %>%
    mutate(gene = rownames(res_treatment_siTEAD4_Dex_vs_siTEAD4_Veh)) %>%
    filter(gene %in% genes) %>%
    dplyr::select(gene, log2FoldChange) %>%
    rename(log2FC = log2FoldChange) %>%
    mutate(Comparison = "siTEAD4_Dex", Category = category_name)

  # Get log2FC values for siNON Dex vs Veh
  siNON_dex_log2fc <- as.data.frame(res_treatment_Dex_vs_Veh) %>%
    mutate(gene = rownames(res_treatment_Dex_vs_Veh)) %>%
    filter(gene %in% genes) %>%
    dplyr::select(gene, log2FoldChange) %>%
    rename(log2FC = log2FoldChange) %>%
    mutate(Comparison = "siNON_Dex", Category = category_name)

  # Combine log2FC data for all comparisons
  category_data <- bind_rows(siTEAD4_dex_log2fc, siNON_dex_log2fc)
  log2fc_data <- bind_rows(log2fc_data, category_data)
}

# Update `Category` labels in the `log2fc_data` data frame with the desired order
log2fc_data$Category <- factor(log2fc_data$Category, levels = c("siNON", "siTEAD4", "shared"))

# Set the factor levels for `Comparison`
log2fc_data$Comparison <- factor(log2fc_data$Comparison, levels = c("siNON_Dex", "siTEAD4_Dex"))

# Plot the box plots for upregulated genes
boxplot_plot <- ggplot(log2fc_data, aes(x = Category, y = log2FC, fill = Comparison)) +
  geom_boxplot(alpha = 0.6) +  # Boxplot with transparency
  stat_summary(fun = mean, geom = "point", shape = 18, size = 3, color = "black", position = position_dodge(0.75)) +
  stat_summary(fun.data = mean_sdl, fun.args = list(mult = 1), geom = "errorbar", width = 0.2, color = "black", position = position_dodge(0.75)) +
  geom_hline(yintercept = 0, color = "black", size = 1, linetype = "dashed") +
  labs(title = "Log2 Fold Change Box Plots for Upregulated Gene Categories",
       x = "Gene Category",
       y = "Log2 Fold Change") +
  scale_fill_manual(values = c("siNON_Dex" = "#ea0934", "siTEAD4_Dex" = "#eea9a9")) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.background = element_rect(fill = "white", color = "white"),
    plot.background = element_rect(fill = "white", color = "white")
  )

# Display the plot
print(boxplot_plot)

# Save the plot as a PNG file
ggsave(file.path(out_dir, "20260200_log2FC_boxplot_with_mean_sd_by_category_genes_up_updated_siTEAD4.png"), plot = boxplot_plot, width = 8, height = 8, dpi = 300)

# Define gene categories for downregulated genes
gene_categories <- list(
  siNON = Dex_down_excluding_others,
  siTEAD4 = siTEAD4_dex_down_excluding_others,
  shared = shared_dex_down
)

# Initialize an empty data frame for storing log2FC values and categories
log2fc_data <- data.frame()

# Extract log2FC values for each category and bind them into a single data frame
for (category_name in names(gene_categories)) {
  genes <- gene_categories[[category_name]]

  # Get log2FC values for siTEAD4 Dex vs Veh
  siTEAD4_dex_log2fc <- as.data.frame(res_treatment_siTEAD4_Dex_vs_siTEAD4_Veh) %>%
    mutate(gene = rownames(res_treatment_siTEAD4_Dex_vs_siTEAD4_Veh)) %>%
    filter(gene %in% genes) %>%
    dplyr::select(gene, log2FoldChange) %>%
    rename(log2FC = log2FoldChange) %>%
    mutate(Comparison = "siTEAD4_Dex", Category = category_name)

  # Get log2FC values for siNON Dex vs Veh
  siNON_dex_log2fc <- as.data.frame(res_treatment_Dex_vs_Veh) %>%
    mutate(gene = rownames(res_treatment_Dex_vs_Veh)) %>%
    filter(gene %in% genes) %>%
    dplyr::select(gene, log2FoldChange) %>%
    rename(log2FC = log2FoldChange) %>%
    mutate(Comparison = "siNON_Dex", Category = category_name)

  # Combine log2FC data for all comparisons
  category_data <- bind_rows(siTEAD4_dex_log2fc, siNON_dex_log2fc)
  log2fc_data <- bind_rows(log2fc_data, category_data)
}

# Update `Category` labels in the `log2fc_data` data frame with the desired order
log2fc_data$Category <- factor(log2fc_data$Category, levels = c("siNON", "siTEAD4", "shared"))

# Set the factor levels for `Comparison`
log2fc_data$Comparison <- factor(log2fc_data$Comparison, levels = c("siNON_Dex", "siTEAD4_Dex"))

# Plot the box plots for downregulated genes
boxplot_plot <- ggplot(log2fc_data, aes(x = Category, y = log2FC, fill = Comparison)) +
  geom_boxplot(alpha = 0.6) +  # Boxplot with transparency
  stat_summary(fun = mean, geom = "point", shape = 18, size = 3, color = "black", position = position_dodge(0.75)) +
  stat_summary(fun.data = mean_sdl, fun.args = list(mult = 1), geom = "errorbar", width = 0.2, color = "black", position = position_dodge(0.75)) +
  geom_hline(yintercept = 0, color = "black", size = 1, linetype = "dashed") +
  labs(title = "Log2 Fold Change Box Plots for Downregulated Gene Categories",
       x = "Gene Category",
       y = "Log2 Fold Change") +
  scale_fill_manual(values = c("siNON_Dex" = "#0b21f7", "siTEAD4_Dex" = "#9992f7")) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.background = element_rect(fill = "white", color = "white"),
    plot.background = element_rect(fill = "white", color = "white")
  )

# Display the plot
print(boxplot_plot)

# Save the plot as a PNG file
ggsave(file.path(out_dir, "20260200_log2FC_boxplot_with_mean_sd_for_downregulated_categories_updated_siTEAD4.png"), plot = boxplot_plot, width = 8, height = 8, dpi = 300)

writeLines(capture.output(sessionInfo()), file.path(out_dir, "sessionInfo.txt"))
