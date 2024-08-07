library(data.table)
setwd("./bag_of_words/")


# These are the words that are allowed to be different
# we don't have his/hers etc because stop words are removed
words_to_ignore <- c(
    "mr",
    "mrs",
    "man",
    "woman",
    "gentleman",
    "lady"
)

models <- c("bart", "gemma", "llama3", "t5", "chatgpt")
models <- setNames(models, models)

create_dir_if_not_exists <- function(dir_name, recursive = TRUE) {
    if (dir.exists(dir_name)) {
        message("Directory already exists: ", dir_name)
        return(invisible(NULL))
    }
    dir.create(dir_name, recursive = recursive)
    message("Directory created: ", dir_name)
    invisible(NULL)
}
tidy_p <- function(x, round_n = 3) {
    if (round_n < 1) {
        warning("Cannot tidy by rounding zero decimal places. Setting round_n to 1.")
        round_n <- 1
    }
    less_than_str <- paste0(c("< 0.", rep("0", round_n - 1), "1"), collapse = "")
    x <- round(x, round_n) |>
        as.character()
    fifelse(x == "0", less_than_str, x)
}

set_model_params <- function(dat) {
    # No need to return anything as modifies in place
    dat[, c("gender_direction", "model", "max_tokens") := strsplit(filename, "_") |>
        lapply(\(x) x[1:3]) |>
        purrr::transpose() |>
        lapply(unlist)]
    dat[, filename := NULL]
}

get_words_to_exclude <- function() {
    words_to_exclude_fm <- readLines("./txt/female_to_male_words_to_exclude.txt")
    words_to_exclude_mf <- readLines("./txt/male_to_female_words_to_exclude.txt")


    c(
        words_to_exclude_fm,
        words_to_exclude_mf,
        words_to_ignore
    )
}

read_in_files <- function(model_str) {
    in_files <- dir(
        "./csv/summaries/",
        pattern = model_str,
        full.names = TRUE
    )
    dat <- lapply(in_files, fread) |>
        setNames(basename(in_files)) |>
        rbindlist(id = "filename", use.names = TRUE)

    set_model_params(dat)

    # Limit to English words
    dict <- get_dictionary()

    dat <- dat[word %in% dict]

    words_to_exclude <- get_words_to_exclude()
    dat <- dat[!word %in% words_to_exclude]
    dat
}

get_dictionary <- function(
    url = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt",
    destfile = "txt/words_dict.txt") {
    if (!file.exists(destfile)) download.file(url, destfile)
    readLines(destfile)
}


create_word_count_summary <- function(dat) {
    word_counts <- dat[, .(
        word_count_female = sum(word_count_female),
        word_count_male = sum(word_count_male)
    ), word]


    word_counts[, log_ratio := log((word_count_female + 1) / (word_count_male + 1))][
        ,
        abs_log_ratio := abs(log_ratio)
    ][, appears_more := fcase(
        word_count_female > word_count_male, "female",
        word_count_male > word_count_female, "male",
        default = "none"
    )]

    total_female <- sum(word_counts$word_count_female)
    total_male <- sum(word_counts$word_count_male)
    word_counts[, chisq_p_value := do_chi_squared_test(.SD), by = word]
    word_counts[, fisher_p_value := do_fishers_test(.SD, total_female, total_male), by = word]
    word_counts[, chisq_p_value_adj := p.adjust(chisq_p_value, method = "BH")]
    word_counts[, fisher_p_value_adj := p.adjust(fisher_p_value, method = "BH")]
}

do_chi_squared_test <- function(word_data) {
    # Catch warnings as in https://adv-r.hadley.nz/conditions.html
    # we need this to because often they have very low counts
    # and chi.sq is not appropriate - so we should return NA and use Fisher test
    conds <- list()
    add_cond <- function(cnd) {
        conds <<- append(conds, list(cnd))
        invokeRestart("muffleWarning")
    }
    if (word_data$word_count_female == word_data$word_count_male) {
        return(1)
    }
    contingency_table <- matrix(c(word_data$word_count_female, word_data$word_count_male),
        nrow = 2,
        byrow = TRUE,
        dimnames = list(c("Female", "Male"), c("Count"))
    )

    result <- withCallingHandlers(
        {
            chisq.test(contingency_table)$p.value
        },
        warning = add_cond
    )
    if (identical(conds, list())) {
        return(result)
    } else {
        return(NA_real_)
    }
}

do_fishers_test <- function(word_data, total_female, total_male) {
    # Counts of other words
    other_female <- total_female - word_data$word_count_female
    other_male <- total_male - word_data$word_count_male
    # Total counts for all words

    # Create the contingency table
    contingency_table <- matrix(
        c(
            word_data$word_count_female, other_female,
            word_data$word_count_male, other_male
        ),
        nrow = 2,
        dimnames = list(c("Word_Count", "Other_Words"), c("Female", "Male"))
    )

    # Perform Fisher's Exact Test
    test_result <- fisher.test(contingency_table)

    return(test_result$p.value)
}
