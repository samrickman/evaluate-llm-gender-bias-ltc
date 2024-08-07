source("./evaluate_themes/evaluate_themes_utils.R")


original_dat <- load_data("./csv_originals/")

# Make sure all counts are equal
stopifnot(
    "Error: some words do not appear the same number of times in the male and female version" = original_dat[, counts_equal]
)
