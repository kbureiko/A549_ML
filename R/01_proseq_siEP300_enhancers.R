# =============================================================================
# siEP300 PRO-seq, enhancer level (eRNA regions): dex response in siNON and siEP300 cells
#
#   Figure 3B                 heatmap of the differentially transcribed enhancers, split by category
#   Supplementary Figure S7A  proportional Venn (Euler) diagram of the six enhancer sets
#   Supplementary Figure S7B,C log2 fold changes of the dex-up / dex-down categories in siNON and siEP300 cells
#   BED file per category     input of the motif enrichment (Figure 3C) and the ChIP-seq signal summary (S7D),
#                             which were run outside R (HOMER, deepTools)
#
# Model: DESeq2. Dex effect in siNON cells and siEP300 effect in EtOH from
#   ~ silencing + treatment + silencing:treatment (reference levels siNON, EtOH); dex effect in siEP300 cells
#   from the same samples with one four-level group factor and siEP300.EtOH as reference.
# Sets: adjusted p < 0.05 and |log2 fold change| > 0.5. No count pre-filter.
#
# Input : <proseq_counts>/raw.eRNA_combined_A549_PROseq_siEP300.txt (HOMER count table; columns 20-27 = samples)
# Output: <knockdown>/proseq_siEP300_enhancers/
# Run from the repository root:  Rscript R/01_proseq_siEP300_enhancers.R
# =============================================================================
library(DESeq2)
library(dplyr)
library(data.table)
library(eulerr)
library(ComplexHeatmap)
library(circlize)
library(ggplot2)        # stat_summary(fun.data = mean_sdl) also needs the Hmisc package to be installed

source(if (file.exists("R/paths.R")) "R/paths.R" else "paths.R")
P <- read_paths()
out_dir <- output_dir(P, "proseq_siEP300_enhancers")

# ---- Count table ----
rawCounts_eRNA <- read.delim(file.path(P$proseq_counts, "raw.eRNA_combined_A549_PROseq_siEP300.txt"))

# Rename the first column to 'Peak ID' in each dataframe
names(rawCounts_eRNA)[1] <- "Peak_ID"
eRNA_level_counts_deseq_exons <- rawCounts_eRNA %>%
  dplyr::select(Peak_ID, 20:27)

# The genes c0lumn to row names!
rownames(eRNA_level_counts_deseq_exons) <- eRNA_level_counts_deseq_exons[, 1]

# Remove the first column
eRNA_level_counts_deseq_exons <- eRNA_level_counts_deseq_exons[, -1]
eRNA_level_counts_deseq_exons <- as.data.frame(eRNA_level_counts_deseq_exons)
original_sample_names_eRNA <- colnames(eRNA_level_counts_deseq_exons)

extract_relevant_part <- function(name) {
  # Extract the part before the first occurrence of ".Tag.Count.in.given.bp"
  sub("\\.Tag\\.Count\\.in\\.given\\.bp.*", "", name)
}

# Apply the function to extract the new column names
new_column_names_eRNA <- sapply(original_sample_names_eRNA, extract_relevant_part)

# Print the new column names
print(new_column_names_eRNA)

# ---- DESeq2: interaction design (reference siNON, EtOH) ----
# rename the columns
colnames(eRNA_level_counts_deseq_exons) <- new_column_names_eRNA

sampleData_eRNA <- data.frame(sample = colnames(eRNA_level_counts_deseq_exons))

sampleParts <- strsplit(sampleData_eRNA$sample, "_")
silencing_treatment <- lapply(sampleParts, function(x) strsplit(x[3], "\\."))

# Step 3: Extract silencing and treatment parts
sampleData_eRNA$silencing <- sapply(silencing_treatment, function(x) x[[1]][1])
sampleData_eRNA$treatment <- sapply(silencing_treatment, function(x) x[[1]][2])

sampleData_eRNA$silencing <- factor(sampleData_eRNA$silencing)
sampleData_eRNA$treatment <- factor(sampleData_eRNA$treatment)

# Relevel the factors to set the reference levels
sampleData_eRNA$silencing <- relevel(sampleData_eRNA$silencing, ref = "siNON")
sampleData_eRNA$treatment <- relevel(sampleData_eRNA$treatment, ref = "EtOH")

dds_eRNA <- DESeqDataSetFromMatrix(countData = eRNA_level_counts_deseq_exons, colData = sampleData_eRNA, design = ~silencing + treatment + silencing:treatment)
dds_eRNA <- DESeq(dds_eRNA)

resultsNames(dds_eRNA)
# [1] "Intercept"                     "silencing_siEP300_vs_siNON"    "treatment_dex_vs_EtOH"         "silencingsiEP300.treatmentdex"

res_silencing_siEP300_vs_siNON_eRNA <- results(dds_eRNA, name="silencing_siEP300_vs_siNON")
res_treatment_dex_vs_EtOH_eRNA <- results(dds_eRNA, name="treatment_dex_vs_EtOH")

# ---- DESeq2: dex effect in siEP300 cells (group design, reference siEP300.EtOH) ----
sampleData_siep300_ETOH <- data.frame(sample = colnames(eRNA_level_counts_deseq_exons))

sampleParts_siep300_ETOH  <- strsplit(sampleData_siep300_ETOH$sample, "_")

sampleData_siep300_ETOH $treatment <- sapply(sampleParts_siep300_ETOH, function(x) x[3])

sampleData_siep300_ETOH$treatment <- factor(sampleData_siep300_ETOH$treatment)
# Relevel the factors to set the reference levels
sampleData_siep300_ETOH$treatment <- relevel(sampleData_siep300_ETOH$treatment, ref = "siEP300.EtOH")

dds_siEP300EtOH_eRNA <- DESeqDataSetFromMatrix(countData = eRNA_level_counts_deseq_exons, colData = sampleData_siep300_ETOH, design = ~treatment)
dds_siEP300EtOH_eRNA <- DESeq(dds_siEP300EtOH_eRNA)

result_names_siEP300EtOH_eRNA <- resultsNames(dds_siEP300EtOH_eRNA)
# [1] "Intercept"                             "treatment_siEP300.dex_vs_siEP300.EtOH" "treatment_siNON.dex_vs_siEP300.EtOH"
# [4] "treatment_siNON.EtOH_vs_siEP300.EtOH"

res_treatment_siEP300_dex_vs_siEP300_EtOH_eRNA <- results(
  dds_siEP300EtOH_eRNA,
  name = "treatment_siEP300.dex_vs_siEP300.EtOH"
)

# ---- Enhancer sets and Venn diagram (Supplementary Figure S7A) ----
upregulated_genes_siNON_vs_siEP300 <- as.data.frame(res_silencing_siEP300_vs_siNON_eRNA) %>%
  mutate(gene = rownames(res_silencing_siEP300_vs_siNON_eRNA)) %>%
  filter(padj < 0.05 & log2FoldChange > 0.5) %>%
  pull(gene)

downregulated_genes_siNON_vs_siEP300 <- as.data.frame(res_silencing_siEP300_vs_siNON_eRNA) %>%
  mutate(gene = rownames(res_silencing_siEP300_vs_siNON_eRNA)) %>%
  filter(padj < 0.05 & log2FoldChange < -0.5) %>%
  pull(gene)

# 2. siNON dex vs veh
upregulated_genes_siNON_dex_vs_veh <- as.data.frame(res_treatment_dex_vs_EtOH_eRNA) %>%
  mutate(gene = rownames(res_treatment_dex_vs_EtOH_eRNA)) %>%
  filter(padj < 0.05 & log2FoldChange > 0.5) %>%
  pull(gene)

downregulated_genes_siNON_dex_vs_veh <- as.data.frame(res_treatment_dex_vs_EtOH_eRNA) %>%
  mutate(gene = rownames(res_treatment_dex_vs_EtOH_eRNA)) %>%
  filter(padj < 0.05 & log2FoldChange < -0.5) %>%
  pull(gene)

# 3. siEP300 dex vs siEP300 EtOH
upregulated_genes_siEP300_dex_vs_siEP300_EtOH <- as.data.frame(res_treatment_siEP300_dex_vs_siEP300_EtOH_eRNA) %>%
  mutate(gene = rownames(res_treatment_siEP300_dex_vs_siEP300_EtOH_eRNA)) %>%
  filter(padj < 0.05 & log2FoldChange > 0.5) %>%
  pull(gene)

downregulated_genes_siEP300_dex_vs_siEP300_EtOH <- as.data.frame(res_treatment_siEP300_dex_vs_siEP300_EtOH_eRNA) %>%
  mutate(gene = rownames(res_treatment_siEP300_dex_vs_siEP300_EtOH_eRNA)) %>%
  filter(padj < 0.05 & log2FoldChange < -0.5) %>%
  pull(gene)

# Prepare the data with all six categories
venn_data <- list(
  "siNON Dex Up" = upregulated_genes_siNON_dex_vs_veh,
  "siNON Dex Down" = downregulated_genes_siNON_dex_vs_veh,
  "siEP300 Dex Up" = upregulated_genes_siEP300_dex_vs_siEP300_EtOH,
  "siEP300 Dex Down" = downregulated_genes_siEP300_dex_vs_siEP300_EtOH,
  "siNON vs siEP300 Up" = upregulated_genes_siNON_vs_siEP300,
  "siNON vs siEP300 Down" = downregulated_genes_siNON_vs_siEP300
)
library(eulerr)
# Fit the Euler diagram using `eulerr`
fit <- euler(venn_data)

# Save the plot with counts as labels
png(file.path(out_dir, "20260200_proportional_venn_siNON_siEP300_dex_vs_veh_with_counts_siNON_eRNA.png"), width = 800, height = 600)

# Plot with labels showing the number of genes in each section using `quantities`
plot(fit,
     fills = c("#ea0934", "#0b21f7", "#eea9a9", "#9992f7", "#1e792a", "#59f70b"),  # Use reddish and bluish colors for up and down categories
     alpha = 0.6,
     quantities = list(type = c("counts", "percent"), font = 3, round = 2, cex = 0.8), # Display counts explicitly
     main = "Proportional Venn Diagram with Gene Counts\nUp and Down Regulated Enhancers in Multiple Comparisons")

# Close the device to save the file
dev.off()

# Save a clean plot without counts, names, or conditions
png(file.path(out_dir, "20260200_proportional_venn_siNON_siEP300_dex_vs_veh_clean_siNON_new_eRNA.png"), width = 800, height = 600)

# Plot without labels or main title
plot(fit,
     fills = c("#ea0934", "#0b21f7", "#eea9a9", "#9992f7", "#1e792a", "#59f70b"),  # Reddish and bluish colors
     alpha = 0.5,
     quantities = FALSE,  # No counts
     labels = FALSE,
     edges = list(lty = 1))      # No labels or conditions

# Close the device to save the file
dev.off()

# ---- Heatmap (Figure 3B) ----
# Step 1: Extract normalized counts from DESeq2 object
normalized_counts <- counts(dds_eRNA, normalized = TRUE)
normalized_counts <- as.data.frame(normalized_counts)

# Define gene categories with new names and criteria, adding the count to each name

# 1. siNON Dex Up excluding Down in siNON vs siEP300 and excluding Up in siEP300 Dex
siNON_dex_up_excluding_others <- setdiff(
  upregulated_genes_siNON_dex_vs_veh,
  union(downregulated_genes_siNON_vs_siEP300, upregulated_genes_siEP300_dex_vs_siEP300_EtOH)
)
name1 <- paste("siNON Dex Up (n =", length(siNON_dex_up_excluding_others), ")")

# 2. siEP300 Dex Up excluding Down in siNON vs siEP300 and excluding Up in siNON Dex
siEP300_dex_up_excluding_others <- setdiff(
  upregulated_genes_siEP300_dex_vs_siEP300_EtOH,
  union(downregulated_genes_siNON_vs_siEP300, upregulated_genes_siNON_dex_vs_veh)
)
name2 <- paste("siEP300 Dex Up (n =", length(siEP300_dex_up_excluding_others), ")")

# 3. Shared siNON Dex Up and siEP300 Dex Up
shared_dex_up <- setdiff(
  intersect(upregulated_genes_siNON_dex_vs_veh, upregulated_genes_siEP300_dex_vs_siEP300_EtOH), downregulated_genes_siNON_vs_siEP300)

name3 <- paste("Shared siNON Dex Up & siEP300 Dex Up - Excluding siNON vs siEP300 Down (n =", length(shared_dex_up), ")")

# 4. siEP300 Dex Up - siNON vs siEP300 Down
siEP300_dex_up_down_in_siNON_vs_siEP300 <-
  intersect(upregulated_genes_siEP300_dex_vs_siEP300_EtOH, downregulated_genes_siNON_vs_siEP300)

name4 <- paste("siEP300 Dex Up & siNON vs siEP300 Down (n =", length(siEP300_dex_up_down_in_siNON_vs_siEP300), ")")

# Downregulated categories with the same approach

# 6. siNON Dex Down
siNON_dex_down_excluding_others <- setdiff(
  downregulated_genes_siNON_dex_vs_veh, downregulated_genes_siEP300_dex_vs_siEP300_EtOH)

name6 <- paste("siNON Dex Down (n =", length(siNON_dex_down_excluding_others), ")")

# 7. siEP300 Dex Down - Excluding Up in siNON vs siEP300 and Down in siNON Dex
siEP300_dex_down_excluding_others <- setdiff(
  downregulated_genes_siEP300_dex_vs_siEP300_EtOH,
  downregulated_genes_siNON_dex_vs_veh
)
name7 <- paste("siEP300 Dex Down (n =", length(siEP300_dex_down_excluding_others), ")")

# 8. Shared siNON Dex Down & siEP300 Dex Down
shared_dex_down <-
  intersect(downregulated_genes_siNON_dex_vs_veh, downregulated_genes_siEP300_dex_vs_siEP300_EtOH)
name8 <- paste("Shared siNON Dex Down & siEP300 Dex Down (n =", length(shared_dex_down), ")")
# 9. siEP300 Dn
siEP300_dn <-
  setdiff(downregulated_genes_siNON_vs_siEP300, upregulated_genes_siEP300_dex_vs_siEP300_EtOH)
name9 <- paste("siEP300 Down (n =", length(siEP300_dn), ")")
# 10. siEP300 Dn
siEP300_up <-
  setdiff(upregulated_genes_siNON_vs_siEP300, upregulated_genes_siNON_dex_vs_veh)
name10 <- paste("siEP300 Up (n =", length(siEP300_up), ")")

# Combine all categories into a named list for easy reference with the updated names
gene_categories <- list(
  name1 = siNON_dex_up_excluding_others,
  name2 = siEP300_dex_up_excluding_others,
  name3 = shared_dex_up,
  name4 = siEP300_dex_up_down_in_siNON_vs_siEP300,
  name6 = siNON_dex_down_excluding_others,
  name7 = siEP300_dex_down_excluding_others,
  name8 = shared_dex_down,
  name9 = siEP300_dn,
  name10 = siEP300_up
)
names(gene_categories) <- c(
  paste("siNON Dex Up (n =", length(siNON_dex_up_excluding_others), ")"),
  paste("siEP300 Dex Up  (n =", length(siEP300_dex_up_excluding_others), ")"),
  paste("siNON Dex Up & siEP300 Dex Up (n =", length(shared_dex_up), ")"),
  paste("siEP300 Dex Up & siNON vs siEP300 Down (n =", length(siEP300_dex_up_down_in_siNON_vs_siEP300), ")"),
  paste("siNON Dex Down (n =", length(siNON_dex_down_excluding_others), ")"),
  paste("siEP300 Dex Down  (n =", length(siEP300_dex_down_excluding_others), ")"),
  paste("siNON Dex Down & siEP300 Dex Down (n =", length(shared_dex_down), ")"),
  paste("siEP300 Dn (n =", length(siEP300_dn), ")"),
  paste("siEP300 Up (n =", length(siEP300_up), ")"))

# Print the length of each category
category_lengths <- sapply(gene_categories, length)
print(category_lengths)
# Step 1: Extract all genes (enhancers) from your dataset
all_enhancers <- rownames(normalized_counts)

# Step 2: Create a data.table with one row per enhancer
enhancer_table <- data.table(enhancer = all_enhancers)

# Step 3: Add columns with descriptive names for each category
category_names <- c(
  "siNON_Dex_Up",
  "siEP300_Dex_Up",
  "Shared_Dex_Up",
  "siEP300_Dex_Up_siEP300_Dn",
  "siNON_Dex_Down",
  "siEP300_Dex_Down",
  "Shared_Dex_Down",
  "siEP300_Dn",
  "siEP300_Up"
)

# Ensure the number of categories matches the descriptive names
if (length(category_names) != length(gene_categories)) {
  stop("The number of descriptive names does not match the number of categories in gene_categories.")
}

# Add columns for each category with descriptive names
for (i in seq_along(category_names)) {
  category_name <- category_names[i]
  enhancer_table[, (category_name) := ifelse(enhancer %in% gene_categories[[i]], 1, 0)]
}

# View the resulting table
print(enhancer_table)

# Step 4: Save the table to a txt file
write.table(enhancer_table, file = file.path(out_dir, "enhancers_de.txt"), sep = "\t", row.names = FALSE, quote = FALSE)

# Regions that fall into two categories (for example dex-down in siNON cells and lower in siEP300 cells)
# would give duplicated row names in the heatmap. They are removed from all categories; with the published
# data these are 16 regions.
all_members <- unlist(gene_categories, use.names = FALSE)
genes_to_remove <- unique(all_members[duplicated(all_members)])
message("regions in more than one category (removed from the heatmap): ", length(genes_to_remove))
# Remove these genes from all categories
gene_categories <- lapply(gene_categories, function(category) {
  setdiff(category, genes_to_remove)
})

# Prepare the data for heatmap
all_genes <- unique(unlist(gene_categories))
heatmap_data <- normalized_counts[all_genes, ]
# Step 1: Ensure that the data is scaled as z-scores across rows
z_scored_heatmap_data <- t(scale(t(heatmap_data)))  # Calculate z-scores by rows
z_scored_heatmap_data[is.na(z_scored_heatmap_data)] <- 0  # Replace any NaNs with 0 for proper visualization
# Create an annotation data frame to separate rows by category with descriptive names
annotation_row <- data.frame(Category = rep(names(gene_categories), sapply(gene_categories, length)))
rownames(annotation_row) <- unlist(gene_categories)
# Ensure category_colors is correctly named for each category in annotation_row$Category
unique_categories <- unique(annotation_row$Category)

# Assign colors to each category
category_colors <- setNames(
  c("red", "blue", "purple", "orange", "green", "yellow", "cyan", "darkred", "darkblue")[1:length(unique_categories)],
  unique_categories
)
# Update the annotation_row data frame
annotation_row <- data.frame(
  Gene = unlist(gene_categories),
  Category = rep(names(gene_categories), sapply(gene_categories, length))
)
print(nrow(z_scored_heatmap_data))  # Number of rows in the heatmap matrix
print(nrow(annotation_row))        # Number of rows in the annotation data frame

# Plot the heatmap again
png(file.path(out_dir, "20241125_Differentially_Regulated_Enahncers_ComplexHeatmap_Split_Zscore_Updated_new.png"), width = 1200, height = 1200, res = 150)
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

heatmap_categories <- gene_categories        # kept for the BED export below
names(heatmap_categories) <- category_names

#---- Box plots new 2026 ----
# Define your gene categories
gene_categories <- list(
  category1 = siNON_dex_up_excluding_others,
  category2 = siEP300_dex_up_excluding_others,
  category3 = shared_dex_up
)

# Initialize an empty data frame to store the log2FC values and category
log2fc_data <- data.frame()

# Extract log2FC values for each category and bind them into a single data frame
for (category_name in names(gene_categories)) {
  genes <- gene_categories[[category_name]]

  # Get log2FC values for siNON_dex and siEP300_dex from the DESeq2 results
  siNON_dex_log2fc <- as.data.frame(res_treatment_dex_vs_EtOH_eRNA) %>%
    mutate(gene = rownames(res_treatment_dex_vs_EtOH_eRNA)) %>%
    filter(gene %in% genes) %>%
    dplyr::select(gene, log2FoldChange) %>%
    rename(log2FC = log2FoldChange) %>%
    mutate(Comparison = "siNON_dex", Category = category_name)

  siEP300_dex_log2fc <- as.data.frame(res_treatment_siEP300_dex_vs_siEP300_EtOH_eRNA) %>%
    mutate(gene = rownames(res_treatment_siEP300_dex_vs_siEP300_EtOH_eRNA)) %>%
    filter(gene %in% genes) %>%
    dplyr::select(gene, log2FoldChange) %>%
    rename(log2FC = log2FoldChange) %>%
    mutate(Comparison = "siEP300_dex", Category = category_name)
  # Combine siNON_dex, siEP300_dex, and siNON_vs_siEP300 log2FC data for this category
  category_data <- bind_rows(siNON_dex_log2fc, siEP300_dex_log2fc)
  log2fc_data <- bind_rows(log2fc_data, category_data)
}

# Update `Category` labels in the `log2fc_data` data frame
log2fc_data$Category <- factor(log2fc_data$Category, levels = names(gene_categories))

# Set the factor levels for `Comparison` to enforce the specific order
log2fc_data$Comparison <- factor(log2fc_data$Comparison, levels = c("siNON_dex", "siEP300_dex"))

# Plot the box plots with white background, mean, and standard deviation
boxplot_plot <- ggplot(log2fc_data, aes(x = Category, y = log2FC, fill = Comparison)) +
  geom_boxplot(alpha = 0.6) +  # Boxplot with some transparency
  # Add points for mean with error bars for standard deviation
  stat_summary(fun = mean, geom = "point", shape = 18, size = 3, color = "black", position = position_dodge(0.75)) +
  stat_summary(fun.data = mean_sdl, fun.args = list(mult = 1), geom = "errorbar", width = 0.2, color = "black", position = position_dodge(0.75)) +
  geom_hline(yintercept = 0, color = "black", size = 1, linetype = "dashed") +
  labs(title = "Log2 Fold Change Box Plots for Each Gene Category",
       x = "Gene Category",
       y = "Log2 Fold Change") +
  scale_fill_manual(values = c("siNON_dex" = "#ea0934", "siEP300_dex" = "#eea9a9")) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.background = element_rect(fill = "white", color = "white"),
    plot.background = element_rect(fill = "white", color = "white")
  )

# Display the plot
print(boxplot_plot)

# Save the plot as a PNG file with white background
ggsave(file.path(out_dir, "20260200_log2FC_boxplot_with_mean_sd_by_category_enhancers_up.png"), plot = boxplot_plot, width = 8, height = 8, dpi = 300)

# Define your gene categories
gene_categories <- list(
  category1 = siNON_dex_down_excluding_others,
  category2 = siEP300_dex_down_excluding_others,
  category3 = shared_dex_down
)

# Initialize an empty data frame to store the log2FC values and category
log2fc_data <- data.frame()

# Extract log2FC values for each category and bind them into a single data frame
for (category_name in names(gene_categories)) {
  genes <- gene_categories[[category_name]]

  # Get log2FC values for siNON_dex and siEP300_dex from the DESeq2 results
  siNON_dex_log2fc <- as.data.frame(res_treatment_dex_vs_EtOH_eRNA) %>%
    mutate(gene = rownames(res_treatment_dex_vs_EtOH_eRNA)) %>%
    filter(gene %in% genes) %>%
    dplyr::select(gene, log2FoldChange) %>%
    rename(log2FC = log2FoldChange) %>%
    mutate(Comparison = "siNON_dex", Category = category_name)

  siEP300_dex_log2fc <- as.data.frame(res_treatment_siEP300_dex_vs_siEP300_EtOH_eRNA) %>%
    mutate(gene = rownames(res_treatment_siEP300_dex_vs_siEP300_EtOH_eRNA)) %>%
    filter(gene %in% genes) %>%
    dplyr::select(gene, log2FoldChange) %>%
    rename(log2FC = log2FoldChange) %>%
    mutate(Comparison = "siEP300_dex", Category = category_name)

  # Combine siNON_dex, siEP300_dex, and siNON_vs_siEP300 log2FC data for this category
  category_data <- bind_rows(siNON_dex_log2fc, siEP300_dex_log2fc)
  log2fc_data <- bind_rows(log2fc_data, category_data)
}

# Update `Category` labels in the `log2fc_data` data frame
log2fc_data$Category <- factor(log2fc_data$Category, levels = names(gene_categories))

# Set the factor levels for `Comparison` to enforce the specific order
log2fc_data$Comparison <- factor(log2fc_data$Comparison, levels = c("siNON_dex", "siEP300_dex"))

# Plot the box plots with white background, mean, and standard deviation
boxplot_plot <- ggplot(log2fc_data, aes(x = Category, y = log2FC, fill = Comparison)) +
  geom_boxplot(alpha = 0.6) +  # Boxplot with some transparency
  # Add points for mean with error bars for standard deviation
  stat_summary(fun = mean, geom = "point", shape = 18, size = 3, color = "black", position = position_dodge(0.75)) +
  stat_summary(fun.data = mean_sdl, fun.args = list(mult = 1), geom = "errorbar", width = 0.2, color = "black", position = position_dodge(0.75)) +
  geom_hline(yintercept = 0, color = "black", size = 1, linetype = "dashed") +
  labs(title = "Log2 Fold Change Box Plots for Downregulated Gene Categories",
       x = "Gene Category",
       y = "Log2 Fold Change") +
  scale_fill_manual(values = c("siNON_dex" = "#0b21f7", "siEP300_dex" = "#9992f7")) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.background = element_rect(fill = "white", color = "white"),
    plot.background = element_rect(fill = "white", color = "white")
  )

# Display the plot
print(boxplot_plot)

# Save the plot as a PNG file with white background
ggsave(file.path(out_dir, "20260200_log2FC_boxplot_with_mean_sd_for_downregulated_categories_new_enahncers.png"), plot = boxplot_plot, width = 8, height = 8, dpi = 300)

# ---- BED file per heatmap category (input of the motif and ChIP-seq signal analyses) ----
for (category_name in names(heatmap_categories)) {
  genes_in_category <- heatmap_categories[[category_name]]
  category_coordinates <- rawCounts_eRNA %>%
    filter(Peak_ID %in% genes_in_category) %>%
    dplyr::select(Chr, Start, End, Peak_ID)
  if (nrow(category_coordinates) > 0) {
    write.table(category_coordinates, file = file.path(out_dir, paste0(category_name, ".bed")),
                quote = FALSE, sep = "\t", row.names = FALSE, col.names = FALSE)
    message("Saved BED file for: ", category_name)
  } else {
    message("No regions found for category: ", category_name)
  }
}

writeLines(capture.output(sessionInfo()), file.path(out_dir, "sessionInfo.txt"))

