# * This will create a bunch of json files to look at the words which appear
# * significantly unequally across all files
# It's not necessary to run but helps manual inspection of differences
source("./bag_of_words/bag_of_words_utils.R")

word_count_list <- readRDS("./csv/cleaned_data/word_count_list.rds")
words_to_examine <- fread("csv/word_level_both_models_firth.csv")

create_json_output <- function(word_to_test, model_to_test) {
    dat <- word_count_list[[model_to_test]][word == word_to_test & word_count_female != word_count_male]

    temp <- "1.0"
    top_p <- "1.0"
    if (model_to_test == "gemma") {
        temp <- "0.7"
        top_p <- "0.9"
    }
    dat[, in_file := sprintf(
        "../clean_summaries/output/%s_%s_%s_temp_%s_top-p_%s_clean.json",
        gender_direction, model, max_tokens, temp, top_p
    )]

    # Add one as 1-indexed
    portraits <- Map(\(x, i) jsonlite::read_json(x)[[i + 1]], dat$in_file, dat$doc_num) |>
        setNames(paste(dat$gender_direction, dat$max_tokens, dat$doc_num, sep = "_"))

    create_dir_if_not_exists("examples")
    out_file <- paste0(
        "examples/", model_to_test, "_", word_to_test, ".json"
    )
    jsonlite::write_json(portraits, out_file, pretty = TRUE)
    message("Created: ", out_file)
    invisible(NULL)
}

# * Run it for all significantly different words in chisq and regression
Map(
    create_json_output,
    words_to_examine$word,
    words_to_examine$model
)

# * It can also be done manually for specific words you want to look at
create_json_output("unwise", "gemma")
create_json_output("disable", "t5")
