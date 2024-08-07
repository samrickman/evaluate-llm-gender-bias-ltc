source("./model_sentiment_output/model_sentiment_utils.R")
# * Same as 5a but splits by model and runs regression separately for each model

regard <- readRDS("./csv/summaries_sentiment/regard.rds")
regard_split <- split(regard, regard$model)

run_model_lm <- function(dat) {
    lm(
        sentiment ~ gender + max_tokens + doc_id,
        data = dat
    )
}

run_model_gls <- function(dat) {
    nlme::gls(
        sentiment ~ gender + max_tokens + doc_id,
        data = dat
    )
}

run_model_robust <- function(dat) {
    MASS::rlm(
        sentiment ~ gender + max_tokens,
        data = dat
    )
}

run_lm_same_as_robust <- function(dat) {
    lm(
        sentiment ~ gender + max_tokens,
        data = dat
    )
}


get_marginal_means <- function(mod, dat) {
    out_dt <- emmeans::emmeans(mod, ~gender, data = dat) |>
        pairs() |>
        as.data.table() |>
        _[, signif := gtools::stars.pval(p.value)]

    z_or_t <- grep("ratio$", names(out_dt), value = TRUE)
    cols_to_keep <- c("estimate", z_or_t, "p.value", "signif")
    out_dt[, .SD, .SDcols = cols_to_keep]
}


# gender direction adds nothing as it's perfectly nested in doc_id
get_best_model <- function(dat) {
    message("Running models...")
    m1 <- lm(sentiment ~ doc_id, data = dat)
    m2 <- lm(sentiment ~ gender + doc_id, data = dat)
    m3 <- lm(sentiment ~ gender + doc_id, data = dat)
    m4 <- lm(sentiment ~ gender + max_tokens + doc_id, data = dat)
    m5 <- lm(sentiment ~ gender + max_tokens + doc_id, data = dat)
    m6 <- lm(sentiment ~ gender * max_tokens + doc_id, data = dat)

    # Can't do m7 as it leads to perfect collinearity (each doc_id
    # has 12 observations, across 6 lengths of max_tokens, and 2 genders)
    # m7 <- lm(sentiment ~ gender + max_tokens * doc_id, data = dat)

    model_list <- tibble::lst(m1, m2, m3, m4, m5, m6)
    for (i in seq_along(model_list)[-1]) {
        anova(model_list[[i - 1]], model_list[[i]])["Pr(>F)"][2, ]
    }
    better_than_previous_model <- sapply(seq_along(model_list)[-1], \(i) anova(model_list[[i - 1]], model_list[[i]])["Pr(>F)"][2, ] < 0.05) |>
        setNames(names(model_list[-1]))
    best_model <- names(better_than_previous_model)[max(which(better_than_previous_model == TRUE))]

    # Return which model is best performing and the model output
    list(
        best_model = best_model,
        model_output = model_list[[best_model]]
    )
}

# In the original data it's always m4 but here we get one m6
best_models <- lapply(regard_split, get_best_model)
sapply(best_models, \(x) x$best_model)
#    bart chatgpt   gemma  llama3      t5
#    "m4"    "m4"    "m6"    "m4"    "m4"


# * Linear model
model_output_lm <- lapply(regard_split, run_model_lm)

lm_dt <- lapply(
    model_output_lm,
    \(m) summary(m)$coefficients |>
        data.table(keep.rownames = "Coef")
) |>
    rbindlist(id = "model") |>
    _[!startsWith(Coef, "doc_id")][
        ,
        p_stars := gtools::stars.pval(`Pr(>|t|)`)
    ]

lm_dt <- lm_dt[
    ,
    .(
        model, Coef, Estimate, p_stars, `Std. Error`,
        t = `t value`, `Pr(>|t|)`
    )
]

create_dir("./csv/summaries_models/regard/model_output_tables/")
fwrite(lm_dt, "./csv/summaries_models/regard/model_output_tables/linear_model.csv")

saveRDS(model_output_lm, "./csv/summaries_models/regard/lm_model.rds")

# * GLS model
model_output_gls <- lapply(regard_split, run_model_gls)


gls_dt <- lapply(
    model_output_gls,
    \(m) summary(m) |>
        coefficients() |>
        data.table(keep.rownames = "Coef")
) |>
    rbindlist(id = "model") |>
    _[!startsWith(Coef, "doc_id")][
        ,
        p_stars := gtools::stars.pval(`p-value`)
    ]

gls_dt <- gls_dt[
    ,
    .(
        model, Coef,
        Estimate = Value, p_stars, `Std. Error` = `Std.Error`,
        t = `t-value`, `Pr(>|t|)` = `p-value`
    )
]

fwrite(gls_dt, "./csv/summaries_models/regard/model_output_tables/gls_dt.csv")
saveRDS(model_output_gls, "./csv/summaries_models/regard/gls_model.rds")

model_output_robust <- lapply(regard_split, run_model_robust)

saveRDS(model_output_robust, "./csv/summaries_models/regard/lm_robust_model.rds")

# Same coefficients as lmm, different t
emm_dt_lm <- Map(
    get_marginal_means, model_output_lm, regard_split
) |>
    rbindlist(id = "model")

fwrite(emm_dt_lm, "./csv/summaries_models/regard/emm_dt_lm.csv")
