source("./model_sentiment_output/model_sentiment_utils.R")

# Create every possible combination of output
create_patterns_dt <- function(
    genders = c("fm", "mf"),
    models = c("llama3", "t5", "gemma", "bart", "chatgpt"),
    sizes = c("None", "300", "150", "100", "75", "50")) {
    patterns_dt <- CJ(genders, models, sizes)
    patterns_dt[, pattern := paste0(genders, "_", models, "_", sizes)]
    patterns_dt[]
}

# We can pass this a row of pattern_dt to read from the params
read_sentiment_from_pattern <- function(in_dir, subdir, row) {
    read_sentiment_output(in_dir = in_dir, subdir = subdir, pattern = row["pattern"]) |>
        bind_nested_sentiment_list() |>
        _[, `:=`(
            sent_model = model,
            gender_direction = row["genders"],
            model = row["models"],
            size = row["sizes"]
        )]
}

create_sentiment_dt_summaries <- function(
    in_dir = "../evaluate_sentiment/output_summaries",
    out_dir = "./csv/summaries_sentiment/",
    subdir = "mean") {
    patterns_dt <- create_patterns_dt()
    l <- vector(mode = "list", length = nrow(patterns_dt))
    for (i in seq_len(nrow(patterns_dt))) {
        row <- patterns_dt[i, ] |>
            as.character() |>
            setNames(names(patterns_dt))
        l[[i]] <- read_sentiment_output(
            in_dir = "evaluate_sentiment/output_summaries",
            subdir = "mean",
            pattern = row["pattern"],
            doc_type = "summaries"
        ) |>
            bind_nested_sentiment_list() |>
            _[, `:=`(
                sent_model = model,
                gender_direction = row["genders"],
                model = row["models"],
                size = row["sizes"]
            )][]
    }
    rbindlist(l)
    metrics_full <- rbindlist(l)
    # Let's model "sentiment" in the continuous case
    # (i.e. regard) as 1 - negative
    # easier to interpret if the coefficient is negative

    metrics_full[, sentiment := 1 - negative]

    # We want this as categorical probably
    metrics_full[, max_tokens := factor(size)]

    # Make everything factor
    character_cols <- c("model", "gender", "gender_direction", "sent_model")
    metrics_full[, (character_cols) := lapply(.SD, factor), .SDcols = character_cols]

    # Let us treat this as continuous if we want
    metrics_full[, size := as.integer(
        fifelse(size == "None", "600", size)
    )]

    metrics_full[, max_tokens := factor(
        max_tokens,
        c("50", "75", "100", "150", "300", "None")
    )]
    # This is the random intercept
    metrics_full[, doc_id := factor(paste0(gender_direction, doc_num))]

    sentiment_by_model <- split(metrics_full, metrics_full$sent_model)
    create_dir(out_dir)
    Map(
        \(dt, nm)  saveRDS(dt, sprintf("%s/%s.rds", out_dir, nm)),
        sentiment_by_model,
        names(sentiment_by_model)
    )
    invisible(NULL)
}

create_sentiment_dt_summaries()
