# =============================================================================
# LISA scatter plots (Supplementary Figures S9, S11 and S13)
#
# LISA itself was run outside R on the gene lists written by R/02 and R/03 ("multi" runs with two gene sets,
# gs1 and gs2). Every panel compares the two gene sets of one run for one LISA model (ChIP-seq or motif):
# x = -log10(p) for gs1, y = -log10(p) for gs2, one point per factor.
# Highlighted: the TOP_N factors of either axis (orange), AP-1 family (green), NR3C1 (red, both models) and,
# in the ChIP-seq model, RELA, FOXA1, TEAD4, EP300 and CEBPB (red).
#
# Input : *_<model>_cauchy_combine_dedup.csv files of the LISA runs, anywhere below <knockdown>
# Output: <knockdown>/lisa_scatter_plots/
# RUNS lists the run-name prefixes and the two gene sets of each run; check it against your LISA job names.
# Run from the repository root:  Rscript R/09_lisa_scatter_plots.R
# =============================================================================
library(ggplot2)
library(ggrepel)

source(if (file.exists("R/paths.R")) "R/paths.R" else "paths.R")
P <- read_paths()
lisa_dir <- P$knockdown
outdir <- output_dir(P, "lisa_scatter_plots")

# ---- Config ----
TOP_N <- 15

AP1_TFs <- c("JUN", "JUNB", "JUND", "FOS", "FOSB", "FOSL1", "FOSL2", "Jun")
SPECIAL_RED_ALL_MODELS <- c("NR3C1")                                      # red in both models
SPECIAL_RED_CHIPSEQ <- c("RELA", "FOXA1", "TEAD4", "EP300", "CEBPB")      # red in the ChIP-seq model only

RUNS <- list(   # run-name prefix = c(gs1 label, gs2 label)
  siEP300_silencing          = c("siEP300_UP_silencing", "siEP300_DN_silencing_top500"),
  siEP300siNONdexUP_Unique   = c("siEP300_Dex_Up_Unique", "siNON_Dex_Up_Unique"),
  siEP300siNONdexDN_Unique   = c("siEP300_Dex_Down_Unique", "siNON_Dex_Down_Unique"),
  siEP300_shared_UPvsDN      = c("siEP300_Dex_Up_Shared", "siEP300_Dex_Down_Shared"),
  siFOXAsiNON_dex_UP_Unique  = c("siFOXA1_Dex_Up_Unique", "siNON_Dex_Up_Unique"),
  siFOXAsiNON_dex_DN_Unique  = c("siFOXA1_Dex_Down_Unique", "siNON_Dex_Down_Unique"),
  siFOXA1_shared_UpvsDN      = c("siFOXA1_Dex_Up_Shared", "siFOXA1_Dex_Down_Shared"),
  siTEAD4siNON_dex_UP_Unique = c("siTEAD4_Dex_Up_Unique", "siNON_Dex_Up_Unique"),
  siTEAD4siNON_dex_DN_Unique = c("siTEAD4_Dex_Down_Unique", "siNON_Dex_Down_Unique"),
  siTEAD4_shared_UpvsDN      = c("siTEAD4_Dex_Up_Shared", "siTEAD4_Dex_Down_Shared")
)

# ---- Helpers: file picking + reading ----
all_files <- list.files(lisa_dir, pattern = "_cauchy_combine_dedup\\.csv$", full.names = TRUE, recursive = TRUE)

choose_lisa_file <- function(prefix, gs = c("gs1", "gs2"), model = c("chipseq", "motif")) {
  gs <- match.arg(gs); model <- match.arg(model)
  # the run name is followed by the LISA time stamp, so "..._UP" does not match "..._UP_Unique"
  pat <- paste0("^", prefix, "_[0-9]{4}_[0-9]{2}_[0-9]{2}_.*__hg38_", gs, "\\.txt_", model,
                "_cauchy_combine_dedup\\.csv$")
  hits <- all_files[grepl(pat, basename(all_files))]
  if (length(hits) == 0) stop("No file found for prefix=", prefix, " ", gs, " ", model, "\nPattern: ", pat)
  if (length(hits) > 1) warning("Multiple matches for ", prefix, " ", gs, " ", model, "; using ", hits[1])
  hits[1]
}

read_lisa <- function(path) {
  df <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  if (!all(c("TF", "pval") %in% colnames(df))) {
    stop("Expected columns TF and pval not found in: ", basename(path),
         "\nColumns: ", paste(colnames(df), collapse = ", "))
  }
  df$TF   <- trimws(as.character(df$TF))
  df$pval <- suppressWarnings(as.numeric(df$pval))
  df <- df[!is.na(df$TF) & nzchar(df$TF) & !is.na(df$pval), , drop = FALSE]
  df
}

# ---- Plotter with model-specific AP-1 colors and extra red TFs in ChIP-seq ----
plot_lisa_scatter_one_model <- function(gs1_file, gs2_file, model_name,
                                        x_label, y_label, title_prefix, out_png) {

  gs1 <- read_lisa(gs1_file)
  gs2 <- read_lisa(gs2_file)

  gs1_p <- gs1[, c("TF", "pval")]; colnames(gs1_p)[2] <- "gs1_pval"
  gs2_p <- gs2[, c("TF", "pval")]; colnames(gs2_p)[2] <- "gs2_pval"

  dat <- merge(gs1_p, gs2_p, by = "TF", all = TRUE)
  dat$gs1_pval[is.na(dat$gs1_pval)] <- 1
  dat$gs2_pval[is.na(dat$gs2_pval)] <- 1

  dat$x <- -log10(pmax(dat$gs1_pval, 1e-300))
  dat$y <- -log10(pmax(dat$gs2_pval, 1e-300))

  dat$rank_gs1 <- rank(dat$gs1_pval, ties.method = "min")
  dat$rank_gs2 <- rank(dat$gs2_pval, ties.method = "min")
  dat$is_top_n <- (dat$rank_gs1 <= TOP_N) | (dat$rank_gs2 <= TOP_N)

  # special TF flags
  dat$is_nr3c1 <- dat$TF %in% SPECIAL_RED_ALL_MODELS
  dat$is_red_chip <- (model_name == "ChIP-seq") & (dat$TF %in% SPECIAL_RED_CHIPSEQ)
  dat$is_ap1 <- dat$TF %in% AP1_TFs

  # priority: red > ap1 > orange > default
  dat$hl <- "default"
  dat$hl[dat$is_top_n] <- "top"
  if (model_name == "ChIP-seq") {
    dat$hl[dat$is_ap1] <- "ap1_chip"
  } else {
    dat$hl[dat$is_ap1] <- "ap1_motif"
  }
  dat$hl[dat$is_nr3c1 | dat$is_red_chip] <- "special_red"
  dat$hl <- factor(dat$hl, levels = c("default", "top", "ap1_chip", "ap1_motif", "special_red"))

  # label: top_n + special + ap1
  dat$label <- ifelse(dat$is_top_n | dat$is_nr3c1 | dat$is_red_chip | dat$is_ap1, dat$TF, "")

  cols <- c(
    default     = "#2C7FB8",
    top         = "orange",
    ap1_chip    = "darkgreen",
    ap1_motif   = "green",
    special_red = "red"
  )

  p <- ggplot(dat, aes(x = x, y = y)) +
    theme_classic() +
    geom_point(aes(color = hl), shape = 19, alpha = 0.85, size = 1.7) +
    scale_color_manual(values = cols, guide = "none") +
    geom_text_repel(
      data = subset(dat, label != ""),
      aes(label = label),
      max.overlaps = 80, size = 3.5, color = "black"
    ) +
    labs(
      title = paste0(title_prefix, " | ", model_name),
      x = x_label,
      y = y_label
    )

  ggsave(out_png, p, width = 7.5, height = 6.5, dpi = 300)
  p
}

# ---- All runs, both models ----
for (prefix in names(RUNS)) {
  labels <- RUNS[[prefix]]
  for (model in c("chipseq", "motif")) {
    model_name <- if (model == "chipseq") "ChIP-seq" else "Motif"
    plot_lisa_scatter_one_model(
      gs1_file = choose_lisa_file(prefix, "gs1", model),
      gs2_file = choose_lisa_file(prefix, "gs2", model),
      model_name = model_name,
      x_label = paste0("-log10(p-value) of ", labels[1], " (gs1)"),
      y_label = paste0("-log10(p-value) of ", labels[2], " (gs2)"),
      title_prefix = prefix,
      out_png = file.path(outdir, paste0(prefix, "_", model, "_scatter_top", TOP_N, ".png"))
    )
    message("saved: ", prefix, " / ", model_name)
  }
}
cat("\nSaved all LISA plots to:\n", outdir, "\n")
