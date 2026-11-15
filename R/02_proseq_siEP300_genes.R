# =============================================================================
# siEP300 PRO-seq, gene level: dex response in siNON and siEP300 cells
#
#   Figure 5A                  heatmap of the differentially transcribed genes, split by category
#   Figure 5B                  GO Biological Process terms of the dex-response categories (selected terms)
#   Supplementary Figure S8A   proportional Venn (Euler) diagram of the six gene sets
#   Supplementary Figure S8B,C log2 fold changes of the dex-up / dex-down categories
#   Supplementary Figure S8D   GO Biological Process terms of the genes lower after EP300 depletion
#   gene lists for LISA        (Supplementary Figure S9; LISA itself was run outside R)
#
# Model: DESeq2. Dex effect in siNON cells and siEP300 effect in EtOH from
#   ~ silencing + treatment + silencing:treatment (reference levels siNON, EtOH); dex effect in siEP300 cells
#   from the same samples with one four-level group factor and siEP300.EtOH as reference.
# Genes: chr1-22, X, Y, M; unnamed transcripts removed; row sum of counts > 10.
# Sets: adjusted p < 0.05 and |log2 fold change| > 0.5.
# GO: clusterProfiler::enrichGO, ontology BP, Benjamini-Hochberg, p and q cut-off 0.05; background = the genes
#   tested in this experiment.
#
# Input : <proseq_counts>/raw.genes.A549_PROseq_siEP300.txt (HOMER count table)
# Output: <knockdown>/proseq_siEP300_genes/
# Run from the repository root:  Rscript R/02_proseq_siEP300_genes.R
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
out_dir <- output_dir(P, "proseq_siEP300_genes")

# ---- Preparing data for DESeq2 ----
# Read in the raw read counts
rawCounts_exons <- read.delim(file.path(P$proseq_counts, "raw.genes.A549_PROseq_siEP300.txt"))

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
print(paste0("Number of unique genes:", length(unique_genes)))

# Select the right c0lumns
gene_level_counts_deseq_exons <- count_table_exons %>%
  dplyr::select(gene_name, 9:17)

# Change the "format"
gene_level_counts_deseq_exons <- as.data.frame(gene_level_counts_deseq_exons)

# The genes c0lumn to row names!
rownames(gene_level_counts_deseq_exons) <- gene_level_counts_deseq_exons[, 1]

# Remove the first column
gene_level_counts_deseq_exons <- gene_level_counts_deseq_exons[, -1]
# Filter out not expressed genes
gene_level_counts_deseq_exons <- gene_level_counts_deseq_exons[rowSums(gene_level_counts_deseq_exons) > 10, ]

### Making sample names look better
original_sample_names <- colnames(gene_level_counts_deseq_exons)
# Split each sample name by underscore and keep only the first 4 parts
shortened_sample_names <- sub("(.*rep[0-9]+).*", "\\1", original_sample_names)
# Set the shortened sample names as column names
colnames(gene_level_counts_deseq_exons) <- shortened_sample_names

# ---- DESeq2: interaction design (reference siNON, EtOH) ----
sampleData <- data.frame(sample = colnames(gene_level_counts_deseq_exons))

sampleParts <- strsplit(sampleData$sample, "_")
silencing_treatment <- lapply(sampleParts, function(x) strsplit(x[3], "\\."))

# Step 3: Extract silencing and treatment parts
sampleData$silencing <- sapply(silencing_treatment, function(x) x[[1]][1])
sampleData$treatment <- sapply(silencing_treatment, function(x) x[[1]][2])

sampleData$silencing <- factor(sampleData$silencing)
sampleData$treatment <- factor(sampleData$treatment)

# Relevel the factors to set the reference levels
sampleData$silencing <- relevel(sampleData$silencing, ref = "siNON")
sampleData$treatment <- relevel(sampleData$treatment, ref = "EtOH")

dds <- DESeqDataSetFromMatrix(countData = gene_level_counts_deseq_exons, colData = sampleData, design = ~silencing + treatment + silencing:treatment)
dds <- DESeq(dds)

resultsNames(dds)
# [1] "Intercept"                     "silencing_siEP300_vs_siNON"    "treatment_dex_vs_EtOH"         "silencingsiEP300.treatmentdex"

res_treatment_dex_vs_EtOH <- results(dds, name="treatment_dex_vs_EtOH")
res_silencing_siEP300_vs_siNON <- results(dds, name="silencing_siEP300_vs_siNON")

# ---- DESeq2: dex effect in siEP300 cells (group design, reference siEP300.EtOH) ----
sampleData_siep300_ETOH <- data.frame(sample = colnames(gene_level_counts_deseq_exons))

sampleParts_siep300_ETOH  <- strsplit(sampleData_siep300_ETOH$sample, "_")

sampleData_siep300_ETOH $treatment <- sapply(sampleParts_siep300_ETOH, function(x) x[3])

sampleData_siep300_ETOH$treatment <- factor(sampleData_siep300_ETOH$treatment)
# Relevel the factors to set the reference levels
sampleData_siep300_ETOH$treatment <- relevel(sampleData_siep300_ETOH$treatment, ref = "siEP300.EtOH")

dds_siep300_ETOH <- DESeqDataSetFromMatrix(countData = gene_level_counts_deseq_exons, colData = sampleData_siep300_ETOH, design = ~treatment)
dds_siep300_ETOH <- DESeq(dds_siep300_ETOH)

result_names_siep300_ETOH <- resultsNames(dds_siep300_ETOH)
# [1] "Intercept"                             "treatment_siEP300.dex_vs_siEP300.EtOH" "treatment_siNON.dex_vs_siEP300.EtOH"   "treatment_siNON.EtOH_vs_siEP300.EtOH"
res_treatment_siEP300_dex_vs_siEP300_EtOH <- results(dds_siep300_ETOH, name="treatment_siEP300.dex_vs_siEP300.EtOH")

# ---- Gene sets and Venn diagram (Supplementary Figure S8A) ----
# Prepare upregulated and downregulated gene sets for each comparison as before
# siNON dex vs veh
upregulated_genes_siNON_dex_vs_veh <- as.data.frame(res_treatment_dex_vs_EtOH) %>%
  mutate(gene = rownames(res_treatment_dex_vs_EtOH)) %>%
  filter(padj < 0.05 & log2FoldChange > 0.5) %>%
  pull(gene)

downregulated_genes_siNON_dex_vs_veh <- as.data.frame(res_treatment_dex_vs_EtOH) %>%
  mutate(gene = rownames(res_treatment_dex_vs_EtOH)) %>%
  filter(padj < 0.05 & log2FoldChange < -0.5) %>%
  pull(gene)

# siEP300 dex vs veh
upregulated_genes_siEP300_dex_vs_veh <- as.data.frame(res_treatment_siEP300_dex_vs_siEP300_EtOH) %>%
  mutate(gene = rownames(res_treatment_siEP300_dex_vs_siEP300_EtOH)) %>%
  filter(padj < 0.05 & log2FoldChange > 0.5) %>%
  pull(gene)

downregulated_genes_siEP300_dex_vs_veh <- as.data.frame(res_treatment_siEP300_dex_vs_siEP300_EtOH) %>%
  mutate(gene = rownames(res_treatment_siEP300_dex_vs_siEP300_EtOH)) %>%
  filter(padj < 0.05 & log2FoldChange < -0.5) %>%
  pull(gene)

# siNON vs siEP300 (no dex treatment)
upregulated_genes_siNON_vs_siEP300 <- as.data.frame(res_silencing_siEP300_vs_siNON) %>%
  mutate(gene = rownames(res_silencing_siEP300_vs_siNON)) %>%
  filter(padj < 0.05 & log2FoldChange > 0.5) %>%
  pull(gene)

downregulated_genes_siNON_vs_siEP300 <- as.data.frame(res_silencing_siEP300_vs_siNON) %>%
  mutate(gene = rownames(res_silencing_siEP300_vs_siNON)) %>%
  filter(padj < 0.05 & log2FoldChange < -0.5) %>%
  pull(gene)

# Prepare the data with all six categories
venn_data <- list(
  "siNON Dex Up" = upregulated_genes_siNON_dex_vs_veh,
  "siNON Dex Down" = downregulated_genes_siNON_dex_vs_veh,
  "siEP300 Dex Up" = upregulated_genes_siEP300_dex_vs_veh,
  "siEP300 Dex Down" = downregulated_genes_siEP300_dex_vs_veh,
  "siNON vs siEP300 Up" = upregulated_genes_siNON_vs_siEP300,
  "siNON vs siEP300 Down" = downregulated_genes_siNON_vs_siEP300
)

# Fit the Euler diagram using `eulerr`
fit <- euler(venn_data)

# Save the plot with counts as labels
png(file.path(out_dir, "20250800_proportional_venn_siNON_siEP300_dex_vs_veh_with_counts_siNON.png"), width = 800, height = 600)

# Plot with labels showing the number of genes in each section using `quantities`
plot(fit,
     fills = c("#ff0000", "blue", "#eb5109", "purple", "#ff6347", "#4682b4"),  # Use reddish and bluish colors for up and down categories
     alpha = 0.6,
     quantities = list(type = c("counts", "percent"), font = 3, round = 2, cex = 0.8), # Display counts explicitly
     main = "Proportional Venn Diagram with Gene Counts\nUp and Down Regulated Genes in Multiple Comparisons")

# Close the device to save the file
dev.off()

# Save a clean plot without counts, names, or conditions
png(file.path(out_dir, "20250800_proportional_venn_siNON_siEP300_dex_vs_veh_clean_siNON_new.png"), width = 800, height = 600)

# Plot without labels or main title
plot(fit,
     fills = c("#ff0000", "blue", "#eb5109", "purple", "#ff6347", "#4682b4"),  # Reddish and bluish colors
     alpha = 0.5,
     quantities = FALSE,  # No counts
     labels = FALSE,
     edges = list(lty = 1))      # No labels or conditions

# Close the device to save the file
dev.off()

# ---- Categories ----
# Define gene categories with descriptive names and counts

siNON_dex_up_excluding_others <- setdiff(
  upregulated_genes_siNON_dex_vs_veh,
  union(
    downregulated_genes_siNON_vs_siEP300, upregulated_genes_siEP300_dex_vs_veh)
)
# Update category1 description with the new length
category1 <- paste("siNON Dex Up (n =", length(siNON_dex_up_excluding_others), ")")

# 2. siEP300 Dex Up excluding Down in siNON vs siEP300 and excluding Up in siNON Dex
siEP300_dex_up_excluding_others <- setdiff(
  upregulated_genes_siEP300_dex_vs_veh,
  union(downregulated_genes_siNON_vs_siEP300, upregulated_genes_siNON_dex_vs_veh)
)
category2 <- paste("siEP300 Dex Up (n =", length(siEP300_dex_up_excluding_others), ")")

# 3. Shared siNON Dex Up and siEP300 Dex Up excluding siNON vs siEP300 Down
shared_dex_up_excluding_siNON_siEP300_down <- setdiff(
  intersect(upregulated_genes_siNON_dex_vs_veh, upregulated_genes_siEP300_dex_vs_veh),
  downregulated_genes_siNON_vs_siEP300
)
category3 <- paste("Shared siNON & siEP300 Dex Up - siNON vs siEP300 Down (n =", length(shared_dex_up_excluding_siNON_siEP300_down), ")")

# 4. siEP300 Dex Up - siNON vs siEP300 Down excluding siNON Dex Up
siEP300_dex_up_down_in_siNON_vs_siEP300 <- setdiff(
  intersect(upregulated_genes_siEP300_dex_vs_veh, downregulated_genes_siNON_vs_siEP300),
  upregulated_genes_siNON_dex_vs_veh
)
category4 <- paste("siEP300 Dex Up & siNON vs siEP300 Down (n =", length(siEP300_dex_up_down_in_siNON_vs_siEP300), ")")

# 5. Shared siNON Dex Up, siEP300 Dex Up, and siNON vs siEP300 Down
shared_dex_up_down_in_siNON_vs_siEP300 <- intersect(
  intersect(upregulated_genes_siNON_dex_vs_veh, upregulated_genes_siEP300_dex_vs_veh),
  downregulated_genes_siNON_vs_siEP300
)
category5 <- paste("Shared siNON Dex Up & siEP300 Dex Up & siNON vs siEP300 Down (n =", length(shared_dex_up_down_in_siNON_vs_siEP300), ")")

# Repeat similar naming for the downregulated categories

# 6. siNON Dex Down - Excluding Up in siNON vs siEP300 and Down in siEP300 Dex
siNON_dex_down_excluding_others <- setdiff(
  downregulated_genes_siNON_dex_vs_veh,
  union(downregulated_genes_siNON_vs_siEP300, downregulated_genes_siEP300_dex_vs_veh)
)
category6 <- paste("siNON Dex Down (n =", length(siNON_dex_down_excluding_others), ")")

# 7. siEP300 Dex Down - Excluding Up in siNON vs siEP300 and Down in siNON Dex
siEP300_dex_down_excluding_others <- setdiff(
  downregulated_genes_siEP300_dex_vs_veh,
  union(downregulated_genes_siNON_vs_siEP300, downregulated_genes_siNON_dex_vs_veh)
)
category7 <- paste("siEP300 Dex Down (n =", length(siEP300_dex_down_excluding_others), ")")

# 8. Shared siNON Dex Down & siEP300 Dex Down - Excluding Up in siNON vs siEP300 Dn
shared_dex_down_excluding_siNON_siEP300_up <- setdiff(
  intersect(downregulated_genes_siNON_dex_vs_veh, downregulated_genes_siEP300_dex_vs_veh),
  downregulated_genes_siNON_vs_siEP300
)
category8 <- paste("Shared siNON Dex Down & siEP300 Dex Down (n =", length(shared_dex_down_excluding_siNON_siEP300_up), ")")

# 9. siEP300 Dex Down - siNON vs siEP300 Dn, Excluding siNON Dex Down
siEP300_dex_down_up_in_siNON_vs_siEP300 <- setdiff(
  intersect(downregulated_genes_siEP300_dex_vs_veh, downregulated_genes_siNON_vs_siEP300),
  downregulated_genes_siNON_dex_vs_veh
)
category9 <- paste("siEP300 Dex Down & siNON vs siEP300 Dn (n =", length(siEP300_dex_down_up_in_siNON_vs_siEP300), ")")

# 10. Shared siNON Dex Down & siEP300 Dex Down & siNON vs siEP300 Dn
shared_dex_down_up_in_siNON_vs_siEP300 <- intersect(
  intersect(downregulated_genes_siNON_dex_vs_veh, downregulated_genes_siEP300_dex_vs_veh),
  downregulated_genes_siNON_vs_siEP300
)
category10 <- paste("Shared siNON Dex Down & siEP300 Dex Down & siNON vs siEP300 Down (n =", length(shared_dex_down_up_in_siNON_vs_siEP300), ")")
# 11. Shared between siNON Dex Down and siNON vs siEP300 Down, Excluding siEP300 Dex Down
siNON_dex_down_sinon_vs_siep300_down_excluding_siep300_dex_down <- setdiff(
  intersect(downregulated_genes_siNON_dex_vs_veh, downregulated_genes_siNON_vs_siEP300),
  shared_dex_down_up_in_siNON_vs_siEP300
)
category11 <- paste("Shared siNON Dex Down & siNON vs siEP300 Down (n =",
                    length(siNON_dex_down_sinon_vs_siep300_down_excluding_siep300_dex_down), ")")

# siNON vs siEP300 Up
siNON_vs_siEP300_up <- upregulated_genes_siNON_vs_siEP300
category12 <- paste("siNON vs siEP300 Up (n =", length(siNON_vs_siEP300_up), ")")

# siNON vs siEP300 Down excluding siNON Dex Down, siNON Dex Up, siEP300 Dex Up, and siEP300 Dex Down
siNON_vs_siEP300_down_excluding_others <- setdiff(
  downregulated_genes_siNON_vs_siEP300,
  union(
    union(downregulated_genes_siNON_dex_vs_veh, upregulated_genes_siNON_dex_vs_veh),
    union(upregulated_genes_siEP300_dex_vs_veh, downregulated_genes_siEP300_dex_vs_veh)
  )
)
category13 <- paste("siNON vs siEP300 Down excluding others (n =",
                    length(siNON_vs_siEP300_down_excluding_others), ")")

# Dex-response sets without reference to the siEP300 effect (used for Figure 5A, the GO analysis and LISA)
siNON_dex_up_all <- unique(na.omit(upregulated_genes_siNON_dex_vs_veh))
siEP300_dex_up_all <- unique(na.omit(upregulated_genes_siEP300_dex_vs_veh))
siNON_dex_down_all <- unique(na.omit(downregulated_genes_siNON_dex_vs_veh))
siEP300_dex_down_all <- unique(na.omit(downregulated_genes_siEP300_dex_vs_veh))
# =========================
siNON_dex_up_unique <- setdiff(
  upregulated_genes_siNON_dex_vs_veh,
  upregulated_genes_siEP300_dex_vs_veh
)

siEP300_dex_up_unique <- setdiff(
  upregulated_genes_siEP300_dex_vs_veh,
  upregulated_genes_siNON_dex_vs_veh
)

shared_dex_up <- intersect(
  upregulated_genes_siNON_dex_vs_veh,
  upregulated_genes_siEP300_dex_vs_veh
)

category_up1 <- paste0("siNON Dex Up Unique (n = ", length(siNON_dex_up_unique), ")")
category_up2 <- paste0("siEP300 Dex Up Unique (n = ", length(siEP300_dex_up_unique), ")")
category_up3 <- paste0("Shared siNON & siEP300 Dex Up (n = ", length(shared_dex_up), ")")

# =========================
# 3 DOWN categories (same logic)
# =========================
siNON_dex_down_unique <- setdiff(
  downregulated_genes_siNON_dex_vs_veh,
  downregulated_genes_siEP300_dex_vs_veh
)

siEP300_dex_down_unique <- setdiff(
  downregulated_genes_siEP300_dex_vs_veh,
  downregulated_genes_siNON_dex_vs_veh
)

shared_dex_down <- intersect(
  downregulated_genes_siNON_dex_vs_veh,
  downregulated_genes_siEP300_dex_vs_veh
)

siNON_vs_siEP300_down <- downregulated_genes_siNON_vs_siEP300

# ---- Heatmap (Figure 5A) ----
normalized_counts <- as.data.frame(counts(dds, normalized = TRUE))

# The ten blocks of Figure 5A, top to bottom. They are built from the sets defined above.
# (The working script drew thirteen blocks; this is the selection and order of the published panel.
#  Sizes expected with the published data: 161, 804, 105, 152, 235, 340, 153, 199, 128, 133.)
gene_categories <- list(
  "siEP300 Up" = siNON_vs_siEP300_up,
  "siEP300 Dn" = siNON_vs_siEP300_down_excluding_others,
  "siEP300 dex Up & siEP300 Dn" = siEP300_dex_up_down_in_siNON_vs_siEP300,
  "siNON dex Up & siEP300 dex Up & siEP300 Dn" = shared_dex_up_down_in_siNON_vs_siEP300,
  "siNON dex Up" = siNON_dex_up_unique,
  "siNON dex Up & siEP300 dex Up" = shared_dex_up_excluding_siNON_siEP300_down,
  "siEP300 dex Up" = siEP300_dex_up_excluding_others,
  "siNON dex Dn" = siNON_dex_down_excluding_others,
  "siNON dex Dn & siEP300 dex Dn" = shared_dex_down_excluding_siNON_siEP300_up,
  "siEP300 dex Dn" = siEP300_dex_down_excluding_others
)
print(sapply(gene_categories, length))

# Genes that fall into two blocks would give duplicated row names; they are removed from all blocks.
all_members <- unlist(gene_categories, use.names = FALSE)
genes_to_remove <- unique(all_members[duplicated(all_members)])
message("genes in more than one block (removed from the heatmap): ", paste(genes_to_remove, collapse = ", "))
gene_categories <- lapply(gene_categories, function(category) setdiff(category, genes_to_remove))
names(gene_categories) <- paste0(names(gene_categories), " (n = ", sapply(gene_categories, length), ")")

ordered_genes <- unlist(gene_categories, use.names = FALSE)
heatmap_data <- normalized_counts[ordered_genes, ]
z_scored_heatmap_data <- t(scale(t(heatmap_data)))            # z-scores by row
z_scored_heatmap_data[is.na(z_scored_heatmap_data)] <- 0
annotation_row <- data.frame(
  Gene = ordered_genes,
  Category = factor(rep(names(gene_categories), sapply(gene_categories, length)), levels = names(gene_categories))
)
category_colors <- setNames(
  c("red", "blue", "purple", "orange", "green", "yellow", "cyan", "pink", "brown", "gray")[seq_along(gene_categories)],
  names(gene_categories)
)

png(file.path(out_dir, "20250800_Differentially_Regulated_Genes_ComplexHeatmap_Split_Zscore_Ordered.png"), width = 1400, height = 1200, res = 200)
Heatmap(
  z_scored_heatmap_data,
  name = "Z-Score",
  row_names_side = "left",
  cluster_rows = FALSE,
  cluster_columns = FALSE,
  show_row_names = FALSE,
  show_column_names = TRUE,
  row_split = annotation_row$Category,
  cluster_row_slices = FALSE,
  left_annotation = rowAnnotation(
    Category = annotation_row$Category,
    col = list(Category = category_colors),
    show_annotation_name = FALSE
  ),
  col = colorRamp2(c(-2, 0, 2), c("blue", "white", "red")),
  heatmap_legend_param = list(title = "Z-Score")
)
dev.off()

# ---- Log2 fold changes per category (Supplementary Figure S8B, C) ----

# Update descriptive names for gene categories
gene_categories <- list(
  siNON_dex_up_excluding_others,
  siEP300_dex_up_excluding_others,
  shared_dex_up_excluding_siNON_siEP300_down,
  siEP300_dex_up_down_in_siNON_vs_siEP300,
  shared_dex_up_down_in_siNON_vs_siEP300
)
names(gene_categories) <- paste0(
  c("siNON dex up", "siEP300 dex up", "siNON & siEP300 dex up", "siEP300 dex up & siNON vs siEP300 Dn",
    "siNON & siEP300 dex up, siNON vs siEP300 Dn"),
  " (n=", sapply(gene_categories, length), ")"
)

# Initialize an empty data frame to store the log2FC values and category
log2fc_data <- data.frame()

# Extract log2FC values for each category and bind them into a single data frame
for (category_name in names(gene_categories)) {
  genes <- gene_categories[[category_name]]

  # Skip empty categories
  if (length(genes) == 0) next

  # Get log2FC values for siNON_dex
  siNON_dex_log2fc <- as.data.frame(res_treatment_dex_vs_EtOH) %>%
    mutate(gene = rownames(res_treatment_dex_vs_EtOH)) %>%
    filter(gene %in% genes) %>%
    dplyr::select(gene, log2FoldChange) %>%
    rename(log2FC = log2FoldChange) %>%
    mutate(Comparison = "siNON_dex", Category = category_name)

  # Get log2FC values for siEP300_dex
  siEP300_dex_log2fc <- as.data.frame(res_treatment_siEP300_dex_vs_siEP300_EtOH) %>%
    mutate(gene = rownames(res_treatment_siEP300_dex_vs_siEP300_EtOH)) %>%
    filter(gene %in% genes) %>%
    dplyr::select(gene, log2FoldChange) %>%
    rename(log2FC = log2FoldChange) %>%
    mutate(Comparison = "siEP300_dex", Category = category_name)

  # Get log2FC values for siNON_vs_siEP300
  siNON_vs_siEP300_log2fc <- as.data.frame(res_silencing_siEP300_vs_siNON) %>%
    mutate(gene = rownames(res_silencing_siEP300_vs_siNON)) %>%
    filter(gene %in% genes) %>%
    dplyr::select(gene, log2FoldChange) %>%
    rename(log2FC = log2FoldChange) %>%
    mutate(Comparison = "siNON_vs_siEP300", Category = category_name)

  # Combine all comparisons for this category
  category_data <- bind_rows(siNON_dex_log2fc, siEP300_dex_log2fc, siNON_vs_siEP300_log2fc)
  log2fc_data <- bind_rows(log2fc_data, category_data)
}

# Check for empty `log2fc_data`
if (nrow(log2fc_data) == 0) {
  stop("No data available for plotting. Check input categories and gene lists.")
}

# Update `Category` labels in the `log2fc_data` data frame
log2fc_data$Category <- factor(log2fc_data$Category, levels = names(gene_categories))

# Set the factor levels for `Comparison` to enforce the specific order
log2fc_data$Comparison <- factor(log2fc_data$Comparison, levels = c("siNON_dex", "siEP300_dex", "siNON_vs_siEP300"))

# Plot the box plots with white background, mean, and standard deviation
boxplot_plot <- ggplot(log2fc_data, aes(x = Category, y = log2FC, fill = Comparison)) +
  geom_boxplot(alpha = 0.6) +  # Boxplot with some transparency
  # Add points for mean with error bars for standard deviation
  stat_summary(fun = mean, geom = "point", shape = 18, size = 3, color = "black", position = position_dodge(0.75)) +
  stat_summary(fun.data = mean_sdl, fun.args = list(mult = 1), geom = "errorbar", width = 0.2, color = "black", position = position_dodge(0.75)) +
  labs(title = "Log2 Fold Change Box Plots for Each Gene Category",
       x = "Gene Category",
       y = "Log2 Fold Change") +
  scale_fill_manual(values = c("siNON_dex" = "#ea0934", "siEP300_dex" = "#eea9a9", "siNON_vs_siEP300" = "green")) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.background = element_rect(fill = "white", color = "white"),
    plot.background = element_rect(fill = "white", color = "white")
  )

# Display the plot
print(boxplot_plot)

# Save the plot as a PNG file with white background
ggsave(file.path(out_dir, "20260200_log2FC_boxplot_with_mean_sd_by_category_fixed.png"), plot = boxplot_plot, width = 8, height = 8, dpi = 300)

# Initialize the gene categories as a named list
gene_categories <- list(
  siNON_dex_down_excluding_others,
  siEP300_dex_down_excluding_others,
  shared_dex_down_excluding_siNON_siEP300_up,
  siEP300_dex_down_up_in_siNON_vs_siEP300,
  shared_dex_down_up_in_siNON_vs_siEP300,
  siNON_dex_down_sinon_vs_siep300_down_excluding_siep300_dex_down
)

# Dynamically assign names to the list elements
names(gene_categories) <- c(
  paste0("siNON dex down (n=", length(siNON_dex_down_excluding_others), ")"),
  paste0("siEP300 dex down (n=", length(siEP300_dex_down_excluding_others), ")"),
  paste0("Shared dex down (n=", length(shared_dex_down_excluding_siNON_siEP300_up), ")"),
  paste0("siEP300 dex down & siNON vs siEP300 dn (n=", length(siEP300_dex_down_up_in_siNON_vs_siEP300), ")"),
  paste0("Shared dex down & siNON vs siEP300 dn (n=", length(shared_dex_down_up_in_siNON_vs_siEP300), ")"),
  paste0("siNON dex down & siNON vs siEP300 dn (n=", length(siNON_dex_down_sinon_vs_siep300_down_excluding_siep300_dex_down), ")")
)

# Initialize an empty data frame to store the log2FC values and category
log2fc_data <- data.frame()

# Extract log2FC values for each category and bind them into a single data frame
for (category_name in names(gene_categories)) {
  genes <- gene_categories[[category_name]]

  # Skip empty categories
  if (length(genes) == 0) next

  # Get log2FC values for siNON_dex
  siNON_dex_log2fc <- as.data.frame(res_treatment_dex_vs_EtOH) %>%
    mutate(gene = rownames(res_treatment_dex_vs_EtOH)) %>%
    filter(gene %in% genes) %>%
    dplyr::select(gene, log2FoldChange) %>%
    rename(log2FC = log2FoldChange) %>%
    mutate(Comparison = "siNON_dex", Category = category_name)

  # Get log2FC values for siEP300_dex
  siEP300_dex_log2fc <- as.data.frame(res_treatment_siEP300_dex_vs_siEP300_EtOH) %>%
    mutate(gene = rownames(res_treatment_siEP300_dex_vs_siEP300_EtOH)) %>%
    filter(gene %in% genes) %>%
    dplyr::select(gene, log2FoldChange) %>%
    rename(log2FC = log2FoldChange) %>%
    mutate(Comparison = "siEP300_dex", Category = category_name)

  # Get log2FC values for siNON_vs_siEP300
  siNON_vs_siEP300_log2fc <- as.data.frame(res_silencing_siEP300_vs_siNON) %>%
    mutate(gene = rownames(res_silencing_siEP300_vs_siNON)) %>%
    filter(gene %in% genes) %>%
    dplyr::select(gene, log2FoldChange) %>%
    rename(log2FC = log2FoldChange) %>%
    mutate(Comparison = "siNON_vs_siEP300", Category = category_name)

  # Combine all log2FC data for this category
  category_data <- bind_rows(siNON_dex_log2fc, siEP300_dex_log2fc, siNON_vs_siEP300_log2fc)
  log2fc_data <- bind_rows(log2fc_data, category_data)
}

# Ensure `log2fc_data` is not empty
if (nrow(log2fc_data) == 0) {
  stop("No data available for plotting. Please check your gene categories and DESeq2 results.")
}

# Update `Category` labels in the `log2fc_data` data frame
log2fc_data$Category <- factor(log2fc_data$Category, levels = names(gene_categories))

# Set the factor levels for `Comparison` to enforce the specific order
log2fc_data$Comparison <- factor(log2fc_data$Comparison, levels = c("siNON_dex", "siEP300_dex", "siNON_vs_siEP300"))

# Plot the box plots with white background, mean, and standard deviation
boxplot_plot <- ggplot(log2fc_data, aes(x = Category, y = log2FC, fill = Comparison)) +
  geom_boxplot(alpha = 0.6) +  # Boxplot with some transparency
  # Add points for mean with error bars for standard deviation
  stat_summary(fun = mean, geom = "point", shape = 18, size = 3, color = "black", position = position_dodge(0.75)) +
  stat_summary(fun.data = mean_sdl, fun.args = list(mult = 1), geom = "errorbar", width = 0.2, color = "black", position = position_dodge(0.75)) +
  geom_hline(yintercept = 0, color = "black", size = 1, linetype = "dashed") +  # Add a bold horizontal line at y=0
  labs(title = "Log2 Fold Change Box Plots for Downregulated Gene Categories",
       x = "Gene Category",
       y = "Log2 Fold Change") +
  scale_fill_manual(values = c("siNON_dex" = "#0b21f7", "siEP300_dex" = "#9992f7", "siNON_vs_siEP300" = "green")) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    axis.text.y = element_text(size = 12),  # Add y-axis labels
    axis.line.y = element_line(color = "black", size = 0.8),  # Add y-axis line
    panel.background = element_rect(fill = "white", color = "white"),
    plot.background = element_rect(fill = "white", color = "white")
  )

# Display the plot
print(boxplot_plot)

# Save the plot as a PNG file with white background
ggsave(file.path(out_dir, "20260200_log2FC_boxplot_with_mean_sd_for_downregulated_categories_fixed_0_line_dn.png"), plot = boxplot_plot, width = 8, height = 8, dpi = 300)

# ---- GO: dex-response categories (Figure 5B) ----
# Background of every GO test: the genes tested in this experiment (rows of the DESeq2 model).
go_universe <- rownames(dds)
# org.Hs.eg.db is attached only here: AnnotationDbi masks dplyr::select().
library(org.Hs.eg.db)
# ---- Output folder ----
OUTDIR <- file.path(out_dir, "20260224_siEP300_dex_GO")
dir.create(OUTDIR, showWarnings = FALSE, recursive = TRUE)

# ---- Gene sets (6 categories) ----
gene_categories <- list(
  siNON_Dex_Up_Unique     = siNON_dex_up_unique,
  siEP300_Dex_Up_Unique   = siEP300_dex_up_unique,
  Shared_Dex_Up           = shared_dex_up,
  siNON_Dex_Down_Unique   = siNON_dex_down_unique,
  siEP300_Dex_Down_Unique = siEP300_dex_down_unique,
  Shared_Dex_Down         = shared_dex_down
)

category_labels <- setNames(
  paste0(names(gene_categories), " (n = ", sapply(gene_categories, length), ")"),
  names(gene_categories)
)

conditions <- gene_categories
names(conditions) <- category_labels[names(gene_categories)]

print(sapply(gene_categories, length))

# ---- Helper functions ----
run_enrichGO <- function(gene_symbols, condition_name) {
  if (length(gene_symbols) == 0) {
    message("Condition ", condition_name, " has no genes.")
    return(NULL)
  }

  ego <- enrichGO(
    gene          = gene_symbols,
    universe      = go_universe,
    OrgDb         = org.Hs.eg.db,
    keyType       = "SYMBOL",
    ont           = "BP",
    pAdjustMethod = "BH",
    pvalueCutoff  = 0.05,
    qvalueCutoff  = 0.05,
    readable      = TRUE
  )

  ego_df <- as.data.frame(ego)
  if (nrow(ego_df) == 0) {
    message("Condition ", condition_name, " has no significant GO terms.")
    return(NULL)
  }

  ego_df$Condition <- condition_name
  ego_df
}

calc_bubble_size <- function(df) {
  df %>%
    separate(GeneRatio, into = c("GeneRatio_Num", "GeneRatio_Denom"),
             sep = "/", fill = "right", remove = FALSE) %>%
    mutate(
      GeneRatio_Num   = as.numeric(GeneRatio_Num),
      GeneRatio_Denom = as.numeric(GeneRatio_Denom),
      BubbleSize = ifelse(!is.na(GeneRatio_Num) & !is.na(GeneRatio_Denom) & GeneRatio_Denom > 0,
                          (GeneRatio_Num / GeneRatio_Denom) * 100, NA_real_)
    )
}

# ---- Run enrichment + save full table ----
all_enrich_results <- bind_rows(lapply(names(conditions), function(nm) {
  run_enrichGO(conditions[[nm]], nm)
}))

if (nrow(all_enrich_results) == 0) {
  stop("No enrichment results across all conditions (nothing passed cutoffs).")
}

write.table(all_enrich_results,
            file = file.path(OUTDIR, "GO_enrichment_results_all_6cats.tsv"),
            sep = "\t", row.names = FALSE, quote = FALSE)

# ---- Select TOP terms per condition ----
TOP_N <- 15

selected_terms_ordered <- all_enrich_results %>%
  filter(p.adjust <= 0.05) %>%
  group_by(Condition) %>%
  arrange(p.adjust, .by_group = TRUE) %>%
  slice_head(n = TOP_N) %>%
  ungroup() %>%
  distinct(Description) %>%
  pull(Description)

expanded_data <- expand.grid(
  Condition   = names(conditions),
  Description = selected_terms_ordered,
  stringsAsFactors = FALSE
) %>%
  left_join(
    all_enrich_results %>% filter(Description %in% selected_terms_ordered),
    by = c("Condition", "Description")
  ) %>%
  calc_bubble_size()

expanded_data$Condition <- factor(expanded_data$Condition, levels = names(conditions), ordered = TRUE)
expanded_data$Description <- factor(expanded_data$Description, levels = selected_terms_ordered)

# ---- Bubble plot: vertical ----
p_vertical <- ggplot(expanded_data, aes(x = Condition, y = Description)) +
  geom_point(aes(size = BubbleSize, color = -log10(p.adjust)), na.rm = TRUE) +
  scale_size_continuous(name = "Gene Ratio %", range = c(1, 3)) +
  scale_color_gradient(name = "-log10(p.adjust)", low = "#B3CDE3", high = "#011F4B") +
  labs(
    title = paste0("Top ", TOP_N, " GO:BP terms per category (Dex)"),
    x = "Category",
    y = "GO term"
  ) +
  theme_classic() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    axis.text.y = element_text(size = 8)
  )

ggsave(file.path(OUTDIR, "20260224_GO_bubble_vertical_siEP300_dex_6cats.png"),
       plot = p_vertical, width = 9, height = 10, dpi = 300)

# ---- Bubble plot: horizontal ----
p_horizontal <- ggplot(expanded_data, aes(x = Description, y = Condition)) +
  geom_point(aes(size = BubbleSize, color = -log10(p.adjust)), na.rm = TRUE) +
  scale_size_continuous(name = "Gene Ratio %", range = c(1, 5)) +
  scale_color_gradient(name = "-log10(p.adjust)", low = "#B3CDE3", high = "#011F4B") +
  labs(
    title = paste0("Top ", TOP_N, " GO:BP terms per category (Dex)"),
    x = "GO term",
    y = "Category"
  ) +
  theme_classic() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
    axis.text.y = element_text(size = 9)
  )

ggsave(file.path(OUTDIR, "20260224_GO_bubble_horizontal_siEP300_dex_6cats.png"),
       plot = p_horizontal, width = 14, height = 6, dpi = 300)

cat("\nSaved to:", OUTDIR, "\n")

# ---- GO dex effect 2026: plot only significant categories + selected pathways ----

library(dplyr)
library(tidyr)
library(ggplot2)

OUTDIR <- file.path(out_dir, "20260224_siEP300_dex_GO")
infile <- file.path(OUTDIR, "GO_enrichment_results_all_6cats.tsv")

all_enrich_results <- read.delim(infile, stringsAsFactors = FALSE, check.names = FALSE)

# ---- helper: shorten terms to match your "reg." style ----
shorten_go <- function(x) {
  x %>%
    gsub("\\bregulation\\b", "reg.", ., perl = TRUE) %>%
    gsub("\\bof biological process\\b", "of bio. proc.", ., perl = TRUE)
}

# ---- helper: bubble size from GeneRatio ----
calc_bubble_size <- function(df) {
  df %>%
    separate(GeneRatio, into = c("GeneRatio_Num", "GeneRatio_Denom"),
             sep = "/", fill = "right", remove = FALSE) %>%
    mutate(
      GeneRatio_Num   = as.numeric(GeneRatio_Num),
      GeneRatio_Denom = as.numeric(GeneRatio_Denom),
      BubbleSize = ifelse(!is.na(GeneRatio_Num) & !is.na(GeneRatio_Denom) & GeneRatio_Denom > 0,
                          (GeneRatio_Num / GeneRatio_Denom) * 100, NA_real_)
    )
}

# =========================================================
# 1) Keep ONLY categories with at least one significant GO term
# =========================================================
sig_conditions <- all_enrich_results %>%
  group_by(Condition) %>%
  summarise(any_sig = any(!is.na(p.adjust) & p.adjust <= 0.05), .groups = "drop") %>%
  filter(any_sig) %>%
  pull(Condition)

all_sig <- all_enrich_results %>% filter(Condition %in% sig_conditions)

message("Significant categories kept (", length(sig_conditions), "):")
print(sig_conditions)

# =========================================================
# 2) Two bubble plots (Top N per significant category)
# =========================================================
TOP_N <- 25

selected_terms_ordered <- all_sig %>%
  filter(p.adjust <= 0.05) %>%
  group_by(Condition) %>%
  arrange(p.adjust, .by_group = TRUE) %>%
  slice_head(n = TOP_N) %>%
  ungroup() %>%
  distinct(Description) %>%
  pull(Description)

expanded_top <- expand.grid(
  Condition   = sig_conditions,
  Description = selected_terms_ordered,
  stringsAsFactors = FALSE
) %>%
  left_join(all_sig, by = c("Condition", "Description")) %>%
  calc_bubble_size()

expanded_top$Condition   <- factor(expanded_top$Condition, levels = sig_conditions)
expanded_top$Description <- factor(expanded_top$Description, levels = selected_terms_ordered)

# --- vertical ---
p_vertical <- ggplot(expanded_top, aes(x = Condition, y = Description)) +
  geom_point(aes(size = BubbleSize, color = -log10(p.adjust)), na.rm = TRUE) +
  scale_size_continuous(name = "Gene Ratio %", range = c(1, 3)) +
  scale_color_gradient(name = "-log10(p.adjust)", low = "#B3CDE3", high = "#011F4B") +
  labs(title = paste0("Top ", TOP_N, " GO:BP terms per significant category (Dex)"),
       x = "Category", y = "GO term") +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        axis.text.y = element_text(size = 8))

ggsave(file.path(OUTDIR, "20260224_GO_bubble_vertical_sigCats_siEP300_dex.png"),
       p_vertical, width = 9, height = 10, dpi = 300)

# --- horizontal ---
p_horizontal <- ggplot(expanded_top, aes(x = Description, y = Condition)) +
  geom_point(aes(size = BubbleSize, color = -log10(p.adjust)), na.rm = TRUE) +
  scale_size_continuous(name = "Gene Ratio %", range = c(1, 5)) +
  scale_color_gradient(name = "-log10(p.adjust)", low = "#B3CDE3", high = "#011F4B") +
  labs(title = paste0("Top ", TOP_N, " GO:BP terms per significant category (Dex)"),
       x = "GO term", y = "Category") +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
        axis.text.y = element_text(size = 9))

ggsave(file.path(OUTDIR, "20260224_GO_bubble_horizontal_sigCats_siEP300_dex.png"),
       p_horizontal, width = 14, height = 6, dpi = 300)

# =========================================================
# 3) Selected pathways plot (horizontal) using SHORTENED names
#    -> matches your list like "reg. of ..." / "positive reg. of ..."
# =========================================================
# ---- Selected pathways plot (horizontal) using SHORTENED names ----
selected_terms_short <- c(
  "wound healing",
  "positive reg. of interleukin-2 production",
  "negative reg. of cell-cell adhesion mediated by cadherin",
  "response to glucocorticoid",
  "negative reg. of epithelial cell differentiation",
  "epithelial cell development",
  "reg. of extracellular matrix organization",
  "receptor signaling pathway via STAT",
  "positive reg. of cell-cell adhesion",
  "glandular epithelial cell differentiation"
)

all_sig2 <- all_sig %>%
  dplyr::mutate(Description_short = shorten_go(Description))

sel_df <- all_sig2 %>%
  dplyr::filter(Description_short %in% selected_terms_short)

missing <- setdiff(selected_terms_short, unique(all_sig2$Description_short))
if (length(missing) > 0) {
  message("Selected terms NOT found (check spelling):\n  ", paste(missing, collapse = "\n  "))
}

expanded_sel <- expand.grid(
  Condition         = sig_conditions,
  Description_short = selected_terms_short,
  stringsAsFactors  = FALSE
) %>%
  dplyr::left_join(
    sel_df %>% dplyr::select(Condition, Description_short, p.adjust, GeneRatio),
    by = c("Condition", "Description_short")
  ) %>%
  calc_bubble_size()

# ---- enforce category order: siNON DN -> siEP300 DN -> siEP300 UP (then anything else) ----
desired_order_keys <- c("siNON_Dex_Down_Unique", "siEP300_Dex_Down_Unique", "siEP300_Dex_Up_Unique")
desired_levels <- unique(unlist(lapply(desired_order_keys, function(k) {
  sig_conditions[grepl(k, sig_conditions, fixed = TRUE)]
})))
other_levels <- setdiff(sig_conditions, desired_levels)
sig_conditions_ordered <- c(desired_levels, other_levels)

expanded_sel$Condition <- factor(expanded_sel$Condition, levels = sig_conditions_ordered)
expanded_sel$Description_short <- factor(expanded_sel$Description_short, levels = selected_terms_short)

p_selected <- ggplot(expanded_sel, aes(x = Description_short, y = Condition)) +
  geom_point(aes(size = BubbleSize, color = -log10(p.adjust)), na.rm = TRUE) +
  scale_size_continuous(name = "Gene Ratio %", range = c(2, 6)) +
  scale_color_gradient(name = "-log10(p.adjust)", low = "#B3CDE3", high = "#011F4B") +
  labs(title = "Selected GO:BP terms (Dex; significant categories only)",
       x = "GO term (short)", y = "Category") +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 9),
        axis.text.y = element_text(size = 9))

ggsave(
  file.path(OUTDIR, "20260224_GO_selected_terms_horizontal_short_sigCats_siEP300_dex.png"),
  p_selected, width = 8, height = 4.5, dpi = 300
)

# ---- GO: siEP300 effect (Supplementary Figure S8D) ----
# =========================

library(ggplot2)
library(clusterProfiler)
library(org.Hs.eg.db)
library(dplyr)
library(tidyr)

# -------------------------
# 0) INPUTS: gene sets
# -------------------------
if (!exists("siNON_vs_siEP300_down") && exists("downregulated_genes_siNON_vs_siEP300")) {
  siNON_vs_siEP300_down <- downregulated_genes_siNON_vs_siEP300
}

gene_categories <- list(
  siNON_vs_siEP300_Up   = siNON_vs_siEP300_up,
  siNON_vs_siEP300_Down = siNON_vs_siEP300_down
)

# -------------------------
# 1) OUTPUT FOLDER
# -------------------------
OUTDIR <- file.path(out_dir, "20260224_siEP300_silencing_GO")
dir.create(OUTDIR, showWarnings = FALSE, recursive = TRUE)

# -------------------------
# 2) Enrichment function
# -------------------------
run_enrichGO <- function(gene_symbols, condition_name) {
  if (length(gene_symbols) == 0) {
    message("Condition ", condition_name, " has no genes.")
    return(NULL)
  }

  ego <- enrichGO(
    gene          = gene_symbols,
    universe      = go_universe,
    OrgDb         = org.Hs.eg.db,
    keyType       = "SYMBOL",
    ont           = "BP",
    pAdjustMethod = "BH",
    pvalueCutoff  = 0.05,
    qvalueCutoff  = 0.05,
    readable      = TRUE
  )

  ego_df <- as.data.frame(ego)
  if (nrow(ego_df) == 0) {
    message("Condition ", condition_name, " has no significant GO terms.")
    return(NULL)
  }

  ego_df$Condition <- condition_name
  ego_df
}

calc_bubble_size <- function(df) {
  df %>%
    separate(GeneRatio, into = c("GeneRatio_Num", "GeneRatio_Denom"),
             sep = "/", fill = "right", remove = FALSE) %>%
    separate(BgRatio, into = c("BgRatio_Num", "BgRatio_Denom"),
             sep = "/", fill = "right", remove = FALSE) %>%
    mutate(
      GeneRatio_Num   = as.numeric(GeneRatio_Num),
      GeneRatio_Denom = as.numeric(GeneRatio_Denom),
      BgRatio_Num     = as.numeric(BgRatio_Num),
      BgRatio_Denom   = as.numeric(BgRatio_Denom),
      BubbleSize = ifelse(!is.na(GeneRatio_Num) & !is.na(GeneRatio_Denom) & GeneRatio_Denom > 0,
                          (GeneRatio_Num / GeneRatio_Denom) * 100, NA_real_)
    )
}

# -------------------------
# 3) Run enrichment
# -------------------------
category_lengths <- sapply(gene_categories, length)
print(category_lengths)

conditions <- gene_categories
names(conditions) <- paste0(names(gene_categories), " (n = ", category_lengths, ")")

all_enrich_results <- bind_rows(lapply(names(conditions), function(nm) {
  run_enrichGO(conditions[[nm]], nm)
}))

if (nrow(all_enrich_results) == 0) {
  stop("No enrichment results across all conditions (nothing passed cutoffs).")
}

# Save FULL table with ORIGINAL Description names
write.table(all_enrich_results,
            file = file.path(OUTDIR, "GO_enrichment_results_all.tsv"),
            sep = "\t", row.names = FALSE, quote = FALSE)

# -------------------------
# 4) Pick top terms per condition (Top 15 by padj)
# -------------------------
TOP_N <- 15

selected_terms_ordered <- all_enrich_results %>%
  filter(p.adjust <= 0.05) %>%
  group_by(Condition) %>%
  arrange(p.adjust, .by_group = TRUE) %>%
  slice_head(n = TOP_N) %>%
  ungroup() %>%
  distinct(Description) %>%
  pull(Description)

expanded_data <- expand.grid(
  Condition   = names(conditions),
  Description = selected_terms_ordered,
  stringsAsFactors = FALSE
) %>%
  left_join(
    all_enrich_results %>% filter(Description %in% selected_terms_ordered),
    by = c("Condition", "Description")
  ) %>%
  calc_bubble_size()

expanded_data$Condition <- factor(expanded_data$Condition, levels = names(conditions))
expanded_data$Description <- factor(expanded_data$Description, levels = selected_terms_ordered)

# -------------------------
# 5) FIGURE 1 (Vertical) - ORIGINAL names, no line breaks
# -------------------------
p_vertical <- ggplot(expanded_data, aes(x = Condition, y = Description)) +
  geom_point(aes(size = BubbleSize, color = -log10(p.adjust)), na.rm = TRUE) +
  scale_size_continuous(name = "Gene Ratio %", range = c(1, 3)) +
  scale_color_gradient(name = "-log10(p.adjust)", low = "#B3CDE3", high = "#011F4B") +
  labs(title = paste0("Top ", TOP_N, " GO:BP terms per category"),
       x = "Condition", y = "GO Term") +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        axis.text.y = element_text(size = 8))

ggsave(file.path(OUTDIR, "20260200_GO_enrichment_bubble_plot_vertical_siEP300_effect.png"),
       plot = p_vertical, width = 7, height = 10, dpi = 300)

# -------------------------
# 6) FIGURE 2 (Horizontal) - ORIGINAL names, no line breaks
# -------------------------
p_horizontal <- ggplot(expanded_data, aes(x = Description, y = Condition)) +
  geom_point(aes(size = BubbleSize, color = -log10(p.adjust)), na.rm = TRUE) +
  scale_size_continuous(name = "Gene Ratio %", range = c(1, 5)) +
  scale_color_gradient(name = "-log10(p.adjust)", low = "#B3CDE3", high = "#011F4B") +
  labs(title = paste0("Top ", TOP_N, " GO:BP terms per category"),
       x = "GO Term", y = "Condition") +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
        axis.text.y = element_text(size = 8))

ggsave(file.path(OUTDIR, "20260200_GO_enrichment_bubble_plot_horizontal_siEP300_effect.png"),
       plot = p_horizontal, width = 12, height = 4, dpi = 300)

cat("\nSaved:\n",
    file.path(OUTDIR, "GO_enrichment_results_all.tsv"), "\n",
    file.path(OUTDIR, "20260200_GO_enrichment_bubble_plot_vertical_siEP300_effect.png"), "\n",
    file.path(OUTDIR, "20260200_GO_enrichment_bubble_plot_horizontal_siEP300_effect.png"), "\n")

# =========================
# Selected-term bubble plot (horizontal) with SHORTENED names
# Uses EXACT matches from the full (original) GO Description strings
# =========================
library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)

OUTDIR <- file.path(out_dir, "20260224_siEP300_silencing_GO")
dir.create(OUTDIR, showWarnings = FALSE, recursive = TRUE)

# ---- helper: bubble size from GeneRatio ----
calc_bubble_size <- function(df) {
  df %>%
    separate(GeneRatio, into = c("GeneRatio_Num", "GeneRatio_Denom"),
             sep = "/", fill = "right", remove = FALSE) %>%
    mutate(
      GeneRatio_Num   = as.numeric(GeneRatio_Num),
      GeneRatio_Denom = as.numeric(GeneRatio_Denom),
      BubbleSize = ifelse(!is.na(GeneRatio_Num) & !is.na(GeneRatio_Denom) & GeneRatio_Denom > 0,
                          (GeneRatio_Num / GeneRatio_Denom) * 100, NA_real_)
    )
}

# ---- your exact original terms to KEEP (must match Description in GO_enrichment_results_all.tsv) ----
keep_terms_original <- c(
  "regulation of small GTPase mediated signal transduction",
  "wound healing",
  "positive regulation of cell-matrix adhesion",
  "positive regulation of MAPK cascade",
  "regulation of actin filament organization"
)

# ---- short labels you want to DISPLAY (same order as keep_terms_original) ----
keep_terms_short <- c(
  "reg. of small GTPase mediated signal transduction",
  "wound healing",
  "positive reg.of cell-matrix adhesion",
  "positive reg.of MAPK cascade",
  "reg. of actin filament organization"
)
name_map <- setNames(keep_terms_short, keep_terms_original)

# ---- load your saved enrichment table (created by your previous script) ----
all_enrich_results <- read.delim(
  file.path(OUTDIR, "GO_enrichment_results_all.tsv"),
  stringsAsFactors = FALSE, check.names = FALSE
)

# ---- filter to selected terms only ----
sel <- all_enrich_results %>%
  filter(Description %in% keep_terms_original) %>%
  calc_bubble_size()

# If some terms are missing, say which (no silent failure)
missing <- setdiff(keep_terms_original, unique(all_enrich_results$Description))
if (length(missing) > 0) {
  message("These selected terms were NOT found in the results table:\n  ",
          paste(missing, collapse = "\n  "))
}

# ---- add display labels + factor order ----
sel$Description_short <- name_map[sel$Description]
sel$Description_short <- factor(sel$Description_short, levels = keep_terms_short)

# Keep your condition order as it appears in the table (or set explicitly if you want)
cond_levels <- unique(all_enrich_results$Condition)
sel$Condition <- factor(sel$Condition, levels = cond_levels)

# ---- plot (horizontal: terms on x, conditions on y) ----
p_selected <- ggplot(sel, aes(x = Description_short, y = Condition)) +
  geom_point(aes(size = BubbleSize, color = -log10(p.adjust)), na.rm = TRUE) +
  scale_size_continuous(name = "Gene Ratio %", range = c(2, 6)) +
  scale_color_gradient(name = "-log10(p.adjust)", low = "#B3CDE3", high = "#011F4B") +
  labs(
    title = "Selected GO:BP terms (siEP300 silencing)",
    x = "GO term",
    y = "Condition"
  ) +
  theme_classic() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 9),
    axis.text.y = element_text(size = 10)
  )

print(p_selected)

ggsave(
  file.path(OUTDIR, "20260224_GO_selected_terms_horizontal_shortnames_siEP300_silencing.png"),
  plot = p_selected,
  width = 6, height = 3, dpi = 300
)

# ---- Gene lists for LISA (Supplementary Figure S9) ----
PREFIX <- "20260221"
OUTDIR <- file.path(out_dir, "lisa_inputs_4sets")
dir.create(OUTDIR, showWarnings = FALSE, recursive = TRUE)

gene_sets_4 <- list(
  siNON_Dex_Up_Unique     = siNON_dex_up_unique,
  siEP300_Dex_Up_Unique   = siEP300_dex_up_unique,
  siNON_Dex_Down_Unique   = siNON_dex_down_unique,
  siEP300_Dex_Down_Unique = siEP300_dex_down_unique
)

# cleanup helper: keep non-empty unique symbols
clean_genes <- function(x) {
  x <- as.character(x)
  x <- x[!is.na(x)]
  x <- trimws(x)
  x <- x[nzchar(x)]
  unique(x)
}

# Save (one gene per line)
for (nm in names(gene_sets_4)) {
  g <- clean_genes(gene_sets_4[[nm]])
  g <- sort(g)
  out_path <- file.path(OUTDIR, paste0(PREFIX, "_", nm, ".txt"))
  writeLines(g, con = out_path)
  message(nm, ": ", length(g), " genes -> ", out_path)
}

PREFIX <- "20260221"
OUTDIR <- file.path(out_dir, "lisa_inputs_silencing")
dir.create(OUTDIR, showWarnings = FALSE, recursive = TRUE)

padj_cutoff <- 0.05
lfc_cutoff  <- 0.5   # set to NULL if you don't want an LFC cutoff

df <- as.data.frame(res_silencing_siEP300_vs_siNON)
df$gene <- rownames(df)

df <- df[!is.na(df$padj) & df$padj < padj_cutoff, , drop = FALSE]
if (!is.null(lfc_cutoff)) {
  df <- df[abs(df$log2FoldChange) >= lfc_cutoff, , drop = FALSE]
}

silencing_up <- df$gene[df$log2FoldChange > 0]
silencing_dn <- df$gene[df$log2FoldChange < 0]

# Clean + unique + sort
clean_genes <- function(x) sort(unique(trimws(x[nzchar(trimws(x))])))

silencing_up <- clean_genes(silencing_up)
silencing_dn <- clean_genes(silencing_dn)

writeLines(silencing_up, file.path(OUTDIR, paste0(PREFIX, "_siEP300_vs_siNON_Up.txt")))
writeLines(silencing_dn, file.path(OUTDIR, paste0(PREFIX, "_siEP300_vs_siNON_Down.txt")))

cat("Saved UP:", length(silencing_up), "\n")
cat("Saved DN:", length(silencing_dn), "\n")
res_silencing_siEP300_vs_siNON <- results(dds, name = "silencing_siEP300_vs_siNON")

df <- as.data.frame(res_silencing_siEP300_vs_siNON)
df$gene <- rownames(df)

top500_dn <- df[
  !is.na(df$padj) &
    df$padj < 0.05 &
    !is.na(df$log2FoldChange) &
    df$log2FoldChange < 0,
  c("gene","log2FoldChange","padj","baseMean"),
  drop = FALSE
  ]

# Rank: strongest DOWN first (most negative log2FC), then most significant padj
top500_dn <- top500_dn[order(top500_dn$log2FoldChange, top500_dn$padj), , drop = FALSE]
top500_dn <- head(top500_dn, 500)

# Save 1 gene per line for LISA
writeLines(top500_dn$gene,
           file.path(OUTDIR, paste0(PREFIX, "_siEP300_vs_siNON_Down_top500.txt")))

# Optional: save full table with stats
write.table(top500_dn,
            file.path(OUTDIR, paste0(PREFIX, "_siEP300_vs_siNON_Down_top500_table.tsv")),
            sep = "\t", quote = FALSE, row.names = FALSE)

cat("Saved:", nrow(top500_dn), "genes\n")
PREFIX <- "20260319"
OUTDIR <- file.path(out_dir, "lisa_inputs_shared")
dir.create(OUTDIR, showWarnings = FALSE, recursive = TRUE)

clean_genes <- function(x) {
  x <- as.character(x)
  x <- x[!is.na(x)]
  x <- trimws(x)
  x <- x[nzchar(x)]
  unique(x)
}

# full dex-response gene sets
siNON_dex_up_all     <- clean_genes(siNON_dex_up_all)
siEP300_dex_up_all   <- clean_genes(siEP300_dex_up_all)
siNON_dex_down_all   <- clean_genes(siNON_dex_down_all)
siEP300_dex_down_all <- clean_genes(siEP300_dex_down_all)

# shared sets
shared_dex_up   <- sort(intersect(siNON_dex_up_all, siEP300_dex_up_all))
shared_dex_down <- sort(intersect(siNON_dex_down_all, siEP300_dex_down_all))

gene_sets_shared <- list(
  Shared_Dex_Up   = shared_dex_up,
  Shared_Dex_Down = shared_dex_down
)

for (nm in names(gene_sets_shared)) {
  g <- gene_sets_shared[[nm]]
  out_path <- file.path(OUTDIR, paste0(PREFIX, "_", nm, ".txt"))
  writeLines(g, con = out_path)
  message(nm, ": ", length(g), " genes -> ", out_path)
}

writeLines(capture.output(sessionInfo()), file.path(out_dir, "sessionInfo.txt"))
