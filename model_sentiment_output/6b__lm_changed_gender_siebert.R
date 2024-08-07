source("./model_sentiment_output/model_sentiment_utils.R")


# * Same as 5b but splits by model and runs regression separately for each model

siebert <- readRDS("./csv/summaries_sentiment/siebert.rds")
siebert_split <- split(siebert, siebert$model)


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

# A mix of m6 and m4 but stick with m6 which we used in original data
best_models <- lapply(siebert_split, get_best_model)
sapply(best_models, \(x) x$best_model)

# * Run models
model_output_lm <- lapply(siebert_split, run_model_lm)

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


create_dir("./csv/summaries_models/siebert/model_output_tables/")
fwrite(lm_dt, "./csv/summaries_models/siebert/model_output_tables/linear_model_siebert.csv")

model_output_gls <- lapply(siebert_split, run_model_gls)

model_output_robust <- lapply(siebert_split, run_model_robust)

# Same estimates as lmm, p value makes chatgpt a bit more significant
emm_dt_lm <- Map(
    get_marginal_means, model_output_lm, siebert_split
) |>
    rbindlist(id = "model")

#      model     estimate    t.ratio      p.value signif
#     <char>        <num>      <num>        <num> <char>
# 1:    bart  0.001315760  0.4493971 6.531713e-01
# 2: chatgpt  0.008569058  1.9769421 4.812162e-02      *
# 3:   gemma  0.037765558  9.2704680 3.050596e-20    ***
# 4:  llama3  0.001601590  0.4105576 6.814205e-01
# 5:      t5 -0.006502027 -1.0124903 3.113693e-01

fwrite(emm_dt_lm, "./csv/summaries_models/siebert/emm_dt_lm.csv")
