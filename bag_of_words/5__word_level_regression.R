# So as well as counts we can do a regression
source("./bag_of_words/bag_of_words_utils.R")
library(brglm2) # For Firth's penalised regression


do_reg <- function(word_dt_long, word, model_formula, dispersion_threshold = 1.25) {
    # So this tries Poisson first and then if there are convergence
    # warnings or overdispersion it does neg binomial

    # Catch warnings as in https://adv-r.hadley.nz/conditions.html
    # in this case to check for bad fit
    conds <- list()
    add_cond <- function(cnd, warnings_to_ignore = "non-integer") {
        # brglmFit gives a false positive warning here - they're all integers
        # ignore it and don't automatically do negbinom
        if (!grepl(warnings_to_ignore, cnd$message)) {
            conds <<- append(conds, list(cnd))
        }
        invokeRestart("muffleWarning")
    }

    model_poisson <- withCallingHandlers(
        glm(
            model_formula,
            data = word_dt_long,
            family = poisson,
            method = "brglmFit" # to deal with perfect separation
        ),
        warning = add_cond
    )

    poisson_overdispersion <- sum(residuals(model_poisson, type = "pearson")^2) / df.residual(model_poisson)


    model_dt <- broom::tidy(model_poisson) |>
        setDT()

    if (identical(conds, list()) && poisson_overdispersion < dispersion_threshold) {
        model_warning <- FALSE
    } else {
        # Weird syntax but basically return the conditions
        return(do_neg_binom_reg(word_dt_long, word, model_formula))
    }
    model_dt[, model_type := "poisson"]
    model_dt[, model_warning := model_warning]
    model_dt[, word := word]
    model_dt[]
}


do_neg_binom_reg <- function(word_dt_long, word, model_formula) {
    # Catch warnings as in https://adv-r.hadley.nz/conditions.html
    # in this case to check for bad fit
    conds <- list()
    add_cond <- function(cnd) {
        conds <<- append(conds, list(cnd))
        invokeRestart("muffleWarning")
    }

    error <- FALSE
    # If this still doesn't work catch the error
    model_nb <- tryCatch(
        error = function(cnd) {
            error <<- TRUE
        },
        withCallingHandlers(
            MASS::glm.nb(
                model_formula,
                data = word_dt_long
            ),
            warning = add_cond
        )
    )
    if (error) {
        return(data.table(NULL))
    }

    model_dt <- broom::tidy(model_nb) |>
        setDT()

    if (identical(conds, list())) {
        model_warning <- FALSE
    } else {
        # Weird syntax but basically return the conditions
        model_warning <- list(unlist(conds))
    }
    model_dt[, model_type := "neg_binom"]
    model_dt[, model_warning := model_warning]
    model_dt[, word := word]
    model_dt[]
}

run_model <- function(dat, word_to_test) {
    word_dt <- dat[word == word_to_test]
    # We can avoid zero-inflated if we just remove the ones where they're all zero
    # which aren't of interest anyway
    word_dt <- word_dt[word_count_male > 0 | word_count_female > 0]

    if (nrow(word_dt) == 0) {
        return(data.table(NULL))
    }

    word_dt_long <- word_dt[, .(word, word_count_female, word_count_male, max_tokens, doc_id)] |>
        melt(
            id.vars = c("word", "max_tokens", "doc_id"),
            variable.name = "gender",
            value.name = "count"
        ) |>
        _[, gender := sub("word_count_", "", gender)]


    # If e.g. only one doc_id has the word we'll get an error
    # with contrasts - so check we have at least two of everything
    # and then include it
    cols_to_test <- c("gender", "max_tokens", "doc_id")
    cols_to_include <- word_dt_long[, .(
        col_name = cols_to_test,
        include = lapply(.SD, \(col) uniqueN(col) > 1)
    ), .SDcols = cols_to_test][include == TRUE][, col_name]
    model_formula <- reformulate(cols_to_include, "count")

    do_reg(word_dt_long, word = word_to_test, model_formula)
}


get_model_output <- function(dat, signif_threshold = 0.05, english_words_dict = dict, verbose = FALSE) {
    dat[, doc_id := factor(paste0(gender_direction, doc_num))]


    unique_words <- dat[, unique(word)]
    # Remove any words with number in
    unique_words <- intersect(unique_words, english_words_dict)
    regression_list <- vector(mode = "list", length = length(unique_words)) |>
        setNames(unique_words)

    for (word_to_test in unique_words) {
        if (verbose) message(word_to_test)
        regression_list[[word_to_test]] <- run_model(dat, word_to_test)
    }
    model_dt <- rbindlist(regression_list)

    model_dt[, p_stars := gtools::stars.pval(p.value)][]
    gender_estimates <- model_dt[term == "gendermale"]
    male_signif <- gender_estimates[estimate > 0 & p.value < signif_threshold]
    female_signif <- gender_estimates[estimate < 0 & p.value < signif_threshold]
    return(
        list(
            model_dt = model_dt,
            male_signif = male_signif,
            female_signif = female_signif
        )
    )
}


compare_models_full <- function(model_str, model_output_list, chisq_signif_by_model) {
    male <- list(
        reg_not_chi_diff = model_output_list[[model_str]]$male_signif[
            as.character(model_warning) == FALSE & !word %in% chisq_signif_by_model[[model_str]]$male$word
        ],
        chi_not_reg_diff =
            chisq_signif_by_model[[model_str]]$male[
                !word %in% model_output_list[[model_str]]$male_signif[as.character(model_warning) == FALSE]
            ],
        intersect =
            model_output_list[[model_str]]$male_signif[
                as.character(model_warning) == FALSE & word %in% chisq_signif_by_model[[model_str]]$male$word
            ]
    )

    female <- list(
        reg_not_chi_diff = model_output_list[[model_str]]$female_signif[
            as.character(model_warning) == FALSE & !word %in% chisq_signif_by_model[[model_str]]$female$word
        ],
        chi_not_reg_diff =
            chisq_signif_by_model[[model_str]]$female[
                !word %in% model_output_list[[model_str]]$female_signif[as.character(model_warning) == FALSE]
            ],
        intersect =
            model_output_list[[model_str]]$female_signif[
                as.character(model_warning) == FALSE & word %in% chisq_signif_by_model[[model_str]]$female$word
            ]
    )

    male_clean <- rbindlist(list(
        male$reg_not_chi_diff[, .(word, estimate, p.value, p_stars, model_type = "Poisson regression")],
        male$chi_not_reg_diff[
            ,
            .(word, estimate = abs_log_ratio, p.value = fcoalesce(chisq_p_value, fisher_p_value), model_type = "Chi sq test")
        ][
            ,
            p_stars := gtools::stars.pval(p.value),
        ],
        male$intersect[, .(word, estimate, p.value, p_stars, model_type = "Both models")]
    ), use.names = TRUE)[, appears_more := "male"][]

    female_clean <- rbindlist(list(
        female$reg_not_chi_diff[, .(word, estimate, p.value, p_stars, model_type = "Poisson regression")],
        female$chi_not_reg_diff[
            ,
            .(word, estimate = abs_log_ratio, p.value = fcoalesce(chisq_p_value, fisher_p_value), model_type = "Chi sq test")
        ][
            ,
            p_stars := gtools::stars.pval(p.value),
        ],
        female$intersect[, .(word, estimate, p.value, p_stars, model_type = "Both models")]
    ), use.names = TRUE)[, appears_more := "female"][]


    rbindlist(list(male_clean, female_clean))
}

create_chisq_signif_dt <- function(chisq_signif_by_model) {
    chisq_signif_dt <- lapply(
        chisq_signif_by_model,
        \(l) rbindlist(l, id = "gender")
    ) |> rbindlist(id = "model")

    chisq_signif_dt[, test := fifelse(word_count_female < 5 | word_count_male < 5, "Chi-Sq", "Fisher")]
    chisq_signif_dt[, comb_p_val := fifelse(test == "Chi-Sq", chisq_p_value, fisher_p_value)]
    chisq_signif_dt[, comb_p_val_adj := fifelse(test == "Chi-Sq", chisq_p_value_adj, fisher_p_value_adj)][]
}


# Limit to English words
dict <- get_dictionary()

# Takes about 10 - 15 seconds to load
word_count_list <- readRDS("./csv/cleaned_data/word_count_list.rds")

# the word spend inexplicably causes a core dump so let's just remove it
# it's not of particular interest
word_count_list <- lapply(word_count_list, \(dt) dt[word != "spend"])


# Takes about 2 mins to run without brglm (Firth method)
# Using brglm it's about 4-5 mins
# but we need it because otherwise you get inflated standard errors
# when one gender has zero counts
model_output_list <- lapply(word_count_list, get_model_output)

saveRDS(model_output_list, "./csv/cleaned_data/regression_signif_words_brglm.rds")
# model_output_list <- "./csv/cleaned_data/regression_signif_words_brglm.rds" |> readRDS()

# See if there are meaningful differences with chisq signif words
chisq_signif_by_model <- readRDS("./csv/cleaned_data/signif_words_by_model.rds")
chisq_signif_by_model_relaxed <- readRDS("./csv/cleaned_data/signif_words_by_model_relaxed.rds")

# * This was done in the original analysis
models_compared_full <- lapply(
    models, \(model_str) compare_models_full(model_str, model_output_list, chisq_signif_by_model)
) |> rbindlist(id = "model")

# * Relaxing the requirement for adjusted p.values to be significant
# because we can compare this to the real dataset and dismiss as random chance
# any words that are not significant in both
models_compared_relaxed <- lapply(
    models, \(model_str) compare_models_full(model_str, model_output_list, chisq_signif_by_model_relaxed)
) |> rbindlist(id = "model")


# Create table for join to get counts
word_count_dt <- word_count_list |>
    rbindlist(id = "model") |>
    _[
        ,
        .(
            count_female = sum(word_count_female),
            count_male = sum(word_count_male)
        ), .(word, model)
    ]


words_in_both <- models_compared_full[model_type == "Both models"]
words_in_both_relaxed <- models_compared_relaxed[model_type == "Both models"]



chisq_signif_dt <- create_chisq_signif_dt(chisq_signif_by_model)
chisq_signif_relaxed_dt <- create_chisq_signif_dt(chisq_signif_by_model_relaxed)

add_counts_p_vals <- function(dt, word_count_dt, chisq_signif_dt) {
    # Add counts
    dt[
        word_count_dt,
        on = .(word, model),
        `:=`(
            count_female = i.count_female,
            count_male = i.count_male
        )
    ]


    # Add p_value
    dt[
        chisq_signif_dt,
        on = .(word, model),
        `:=`(
            comb_p = i.comb_p_val,
            comb_p_adj = i.comb_p_val_adj
        )
    ]

    # Drop any if the combined Fisher test is < 0.05
    # and we're using Fisher (may have snuck in if Chi-Sq > 0.05)
    dt <- dt[comb_p_adj < 0.05]

    dt[, estimate := round(estimate, 2)]

    dt <- dt[, .(
        model, word,
        female = count_female, male = count_male,
        bias = appears_more, coef = estimate, p_stars,
        p = p.value, comb_p, comb_p_adj
    )]

    setorder(dt, model, bias, p)



    p_cols <- c("p", "comb_p", "comb_p_adj")
    dt[, (p_cols) := lapply(.SD, tidy_p), .SDcols = p_cols]
    dt
}

add_counts_p_vals(words_in_both, word_count_dt, chisq_signif_dt)
add_counts_p_vals(words_in_both_relaxed, word_count_dt, chisq_signif_relaxed_dt)
# * so this is not as stark but bears similarities to previous results for gemma
# female: text, describe, highlight, emphasise, situation
# male: require, reside, old, assistance, necessitate
# but the input is different because it's synthetic doesn't contain words like unable, disabled, unwise
words_in_both[model == "gemma"]

words_in_both[model == "llama3"] # only one word (old)
words_in_both[model == "chatgpt"] # interestingly uses male name more

setorder(words_in_both, model, appears_more, comb_p)

fwrite(
    words_in_both,
    "csv/word_level_both_models_firth.csv"
)

# * Quick look at the ones with slightly larger p values
setorder(words_in_both_relaxed, appears_more, comb_p)

# medication, confusion, memory are in here
words_in_both_relaxed[model == "gemma"] |> print(40)

# still only two words
words_in_both_relaxed[model == "llama3"] |> print(40)

# more words, might be interesting to look at
words_in_both_relaxed[model == "chatgpt"] |> print(40)
