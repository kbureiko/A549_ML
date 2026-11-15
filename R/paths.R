# Local folders for the R scripts, read from the same git-ignored paths.toml that the Python code uses.
#
#   source("R/paths.R")            # from the repository root (or: source("paths.R") from inside R/)
#   P <- read_paths()
#   P$proseq_counts                # [knockdown] proseq_counts
#   P$brbseq_counts                # [knockdown] brbseq_counts
#   P$knockdown                    # <analysis>/<[layout] knockdown>, default <analysis>/knockdown
#
# paths.toml is looked for in the file named by the environment variable A549_PATHS, then in the working
# directory and its parent folders. Only the simple form used in paths.example.toml is understood:
# [section] headers and  key = "value"  lines (single or double quotes), with # comments.

read_paths <- function(file = Sys.getenv("A549_PATHS", unset = "")) {
  if (!nzchar(file)) {
    folder <- normalizePath(getwd(), winslash = "/")
    repeat {
      candidate <- file.path(folder, "paths.toml")
      if (file.exists(candidate)) {
        file <- candidate
        break
      }
      parent <- dirname(folder)
      if (identical(parent, folder)) {
        stop("paths.toml not found. Copy paths.example.toml to paths.toml and fill in your folders, ",
             "or set the environment variable A549_PATHS.")
      }
      folder <- parent
    }
  }
  if (!file.exists(file)) stop("paths file not found: ", file)

  lines <- trimws(readLines(file, warn = FALSE, encoding = "UTF-8"))
  lines <- lines[nzchar(lines) & !startsWith(lines, "#")]
  cfg <- list()
  section <- "top"
  for (ln in lines) {
    if (grepl("^\\[[A-Za-z0-9_.-]+\\]$", ln)) {
      section <- substr(ln, 2, nchar(ln) - 1)
      next
    }
    eq <- regexpr("=", ln, fixed = TRUE)
    if (eq < 2) next
    key <- trimws(substr(ln, 1, eq - 1))
    value <- trimws(substr(ln, eq + 1, nchar(ln)))
    quote <- substr(value, 1, 1)
    if (!quote %in% c("\"", "'")) next
    closing <- regexpr(quote, substring(value, 2), fixed = TRUE)
    if (closing < 1) stop("unterminated string in ", file, ": ", ln)
    value <- substr(value, 2, closing)
    if (quote == "\"") value <- gsub("\\\\", "\\", value, fixed = TRUE)
    cfg[[paste(section, key, sep = ".")]] <- value
  }

  need <- function(key) {
    value <- cfg[[key]]
    if (is.null(value) || !nzchar(value)) stop(key, " is not set in ", file, " (see paths.example.toml)")
    value
  }
  layout_knockdown <- cfg[["layout.knockdown"]]
  if (is.null(layout_knockdown)) layout_knockdown <- "knockdown"
  out <- list(
    file = file,
    analysis = need("results.analysis"),
    proseq_counts = need("knockdown.proseq_counts"),
    brbseq_counts = need("knockdown.brbseq_counts")
  )
  out$knockdown <- file.path(out$analysis, layout_knockdown)
  dir.create(out$knockdown, showWarnings = FALSE, recursive = TRUE)
  out
}

# Folder for the output of one script (created if needed).
output_dir <- function(P, name) {
  d <- file.path(P$knockdown, name)
  dir.create(d, showWarnings = FALSE, recursive = TRUE)
  d
}
