source("./model_sentiment_output/model_sentiment_utils.R")
library(data.table)
# * This is just a big ugly script with lots of repeated code to format the tables nicely

format_t_test <- function(dt, gender_direction = FALSE) {
    cols <- c(
        "metric" = "metric",
        "Effect size" = "effsize",
        "Pr(>|t|)" = "t_test_p",
        "Signif" = "t_signif"
    )
    if (gender_direction) {
        dt[, gender := fifelse(gender == "fm", "female", "male")]
        cols <- c(cols, "Original gender" = "gender")
    }

    dt[, .SD, .SDcols = cols] |>
        setnames(names(cols)) |>
        _[
            ,
            `Effect size` := as.character(round(`Effect size`, 2))
        ][, `Pr(>|t|)` := as.character(round(`Pr(>|t|)`, 3))][
            `Effect size` == 0, `Effect size` := "<0.01"
        ][`Pr(>|t|)` == 0, `Pr(>|t|)` := "<0.001"][]
}

t_test_mf_fm_sentences <- fread("./csv/original_models/t_test_mf_sentences.csv") |>
    format_t_test(gender_direction = TRUE)
t_test_mf_fm_docs <- fread("./csv/original_models/t_test_mf_docs.csv") |>
    format_t_test(gender_direction = TRUE)
t_test_combined_sentences <- fread("./csv/original_models/t_test_combined_sentences.csv") |>
    format_t_test(gender_direction = FALSE)
t_test_combined_docs <- fread("./csv/original_models/t_test_combined_docs.csv") |>
    format_t_test(gender_direction = FALSE)

fwrite(t_test_mf_fm_sentences, "./csv/original_models/final_output/t_test_mf_fm_sentences.csv")
fwrite(t_test_mf_fm_docs, "./csv/original_models/final_output/t_test_mf_fm_docs.csv")
fwrite(t_test_combined_sentences, "./csv/original_models/final_output/t_test_combined_sentences")
fwrite(t_test_combined_docs, "./csv/original_models/final_output/t_test_combined_docs")

# * Originals: mixed model, sentence level

models <- c("regard", "siebert", "distilbert")
model_list <- lapply(
    setNames(models, models),
    \(model) readRDS(sprintf("./csv/original_models/%s_lmer.rds", model))
)

coef_dt <- lapply(
    model_list,
    \(mod) summary(mod)$coefficients |>
        data.table(keep.rownames = "Coef") |>
        setnames(\(x) gsub("z", "t", x))
) |> rbindlist(id = "model", fill = TRUE, use.names = TRUE)

# * We ave z values for binomial and t values for linear
# which I've put in the same column - remember to say that

coef_dt[, df := NULL]

coef_dt[, Signif := gtools::stars.pval(`Pr(>|t|)`)]

fwrite(coef_dt, "./csv/original_models/lmer_models_all.csv")


# * Originals: mixed model, document level

model_list <- lapply(
    setNames(models, models),
    \(model) readRDS(sprintf("./csv/original_models/%s_lmer_mean.rds", model))
)

coef_dt <- lapply(
    model_list,
    \(mod) summary(mod)$coefficients |>
        data.table(keep.rownames = "Coef") |>
        setnames(\(x) gsub("z", "t", x))
) |> rbindlist(id = "model", fill = TRUE, use.names = TRUE)


coef_dt[, df := NULL]

coef_dt[, Signif := gtools::stars.pval(`Pr(>|t|)`)]

fwrite(coef_dt, "./csv/original_models/lmer_models_all_means.csv")

# * Primary model
regard_final_model <- readRDS("./csv/summaries_models/regard/final_lmer_model.rds")
siebert_final_model <- readRDS("./csv/summaries_models/siebert/final_lmer_model.rds")

regard_dt <- summary(regard_final_model)$coefficients |>
    data.table(keep.rownames = "Coef") |>
    _[, metric := "regard"][
        ,
        p_stars := gtools::stars.pval(`Pr(>|t|)`)
    ][
        ,
        p_vals := signif(`Pr(>|t|)`, 2)
    ][, Estimate := signif(Estimate, 2)][
        ,
        `Std. Error` := signif(`Std. Error`, 2)
    ][, `t value` := signif(`t value`, 2)]

siebert_dt <- summary(siebert_final_model)$coefficients |>
    data.table(keep.rownames = "Coef") |>
    _[, metric := "siebert"][
        ,
        p_stars := gtools::stars.pval(`Pr(>|t|)`)
    ][
        ,
        p_vals := signif(`Pr(>|t|)`, 2)
    ][, Estimate := signif(Estimate, 2)][
        ,
        `Std. Error` := signif(`Std. Error`, 2)
    ][, `t value` := signif(`t value`, 2)]


models_joined <- merge(
    regard_dt[, .(Coef, Estimate, ` ` = p_stars, `Std. Error`, t = `t value`, p_vals, metric)],
    siebert_dt[, .(Coef, Estimate, ` ` = p_stars, `Std. Error`, t = `t value`, p_vals, metric)],
    by = c("Coef"),
    suffix = c("_regard", "_siebert"),
    sort = FALSE
)

models_joined[, Coef := gsub("max_tokens", "Max tokens ", Coef)]

models_joined[, Coef := gsub("^model", "Model ", Coef)]
models_joined[, Coef := gsub(":gendermale", " : Male", Coef)]


dir.create("./csv/summaries_models/models_joined/")
fwrite(models_joined, "./csv/summaries_models/models_joined/final_models.csv")


# * Now the emmeans output
emmeans_regard <- fread("./csv/summaries_models/regard/emm_lmm.csv")
emmeans_siebert <- fread("./csv/summaries_models/siebert/emm_lmm.csv")

cols <- c("estimate", "t.ratio", "p.value")
emmeans_regard[, (cols) := lapply(.SD, signif, 2), .SDcols = cols]
emmeans_siebert[, (cols) := lapply(.SD, signif, 2), .SDcols = cols]

emmeans_joined <- merge(
    emmeans_regard[, .(Model = model, Estimate = estimate, ` ` = signif, t = t.ratio, p = p.value)],
    emmeans_siebert[, .(Model = model, Estimate = estimate, ` ` = signif, t = t.ratio, p = p.value)],
    by = "Model",
    suffix = c("_regard", "_siebert"),
    sort = FALSE
)

fwrite(emmeans_joined, "./csv/summaries_models/models_joined/emmeans.csv")


# * Robust model output
regard <- readRDS("./csv/summaries_sentiment/regard.rds")
siebert <- readRDS("./csv/summaries_sentiment/siebert.rds")

regard_final_model <- readRDS("./csv/summaries_models/regard/robust_lmer.rds")
siebert_final_model <- readRDS("./csv/summaries_models/siebert/robust_lmer.rds")

regard_dt <- summary(regard_final_model)$coefficients |>
    data.table(keep.rownames = "Coef") |>
    _[, metric := "regard"][, Estimate := signif(Estimate, 2)][
        ,
        `Std. Error` := signif(`Std. Error`, 2)
    ][, `t value` := signif(`t value`, 2)][]


siebert_dt <- summary(siebert_final_model)$coefficients |>
    data.table(keep.rownames = "Coef") |>
    _[, metric := "siebert"][, Estimate := signif(Estimate, 2)][
        ,
        `Std. Error` := signif(`Std. Error`, 2)
    ][, `t value` := signif(`t value`, 2)][]

models_joined <- merge(
    regard_dt[, .(Coef, Estimate, `Std. Error`, t = `t value`, metric)],
    siebert_dt[, .(Coef, Estimate, `Std. Error`, t = `t value`, metric)],
    by = c("Coef"),
    suffix = c("_regard", "_siebert"),
    sort = FALSE
)

models_joined[, Coef := gsub("max_tokens", "Max tokens ", Coef)]

models_joined[, Coef := gsub("^model", "Model ", Coef)]
models_joined[, Coef := gsub(":gendermale", " : Male", Coef)]

fwrite(models_joined, "./csv/summaries_models/models_joined/final_models_robust.csv")


# * Now the emmeans robust output
emmeans_regard <- fread("./csv/summaries_models/regard/robust_emm.csv")
emmeans_siebert <- fread("./csv/summaries_models/siebert/emm_robust.csv")


cols <- c("estimate", "z.ratio", "p.value")
emmeans_regard[, (cols) := lapply(.SD, signif, 2), .SDcols = cols]
emmeans_siebert[, (cols) := lapply(.SD, signif, 2), .SDcols = cols]

emmeans_joined <- merge(
    emmeans_regard[, .(Model = model, Estimate = estimate, ` ` = signif, z = z.ratio, p = p.value)],
    emmeans_siebert[, .(Model = model, Estimate = estimate, ` ` = signif, z = z.ratio, p = p.value)],
    by = "Model",
    suffix = c("_regard", "_siebert"),
    sort = FALSE
)


fwrite(emmeans_joined, "./csv/summaries_models/models_joined/emmeans_robust.csv")


# * nlme model
regard_final_model <- readRDS("./csv/summaries_models/regard/nlme_model.rds")
siebert_final_model <- readRDS("./csv/summaries_models/siebert/nlme_model.rds")

regard_dt <- summary(regard_final_model) |>
    coefficients() |>
    data.table(keep.rownames = "Coef") |>
    _[, metric := "regard"][
        ,
        p_stars := gtools::stars.pval(`p-value`)
    ][
        ,
        p_vals := signif(`p-value`, 2)
    ][, Estimate := signif(Value, 2)][
        ,
        `Std. Error` := signif(`Std.Error`, 2)
    ][, `t value` := signif(`t-value`, 2)]

siebert_dt <- summary(siebert_final_model) |>
    coefficients() |>
    data.table(keep.rownames = "Coef") |>
    _[, metric := "regard"][
        ,
        p_stars := gtools::stars.pval(`p-value`)
    ][
        ,
        p_vals := signif(`p-value`, 2)
    ][, Estimate := signif(Value, 2)][
        ,
        `Std. Error` := signif(`Std.Error`, 2)
    ][, `t value` := signif(`t-value`, 2)]



models_joined <- merge(
    regard_dt[, .(Coef, Estimate, ` ` = p_stars, `Std. Error`, t = `t-value`, p_vals, metric)],
    siebert_dt[, .(Coef, Estimate, ` ` = p_stars, `Std. Error`, t = `t-value`, p_vals, metric)],
    by = c("Coef"),
    suffix = c("_regard", "_siebert"),
    sort = FALSE
)

models_joined[, Coef := gsub("max_tokens", "Max tokens ", Coef)]

models_joined[, Coef := gsub("^model", "Model ", Coef)]
models_joined[, Coef := gsub(":gendermale", " : Male", Coef)]

fwrite(models_joined, "./csv/summaries_models/models_joined/nlme_final_models.csv")


# * Now the emmeans output
emmeans_regard <- fread("./csv/summaries_models/regard/nlme_emm.csv")
emmeans_siebert <- fread("./csv/summaries_models/siebert/emm_nlme.csv")

cols <- c("estimate", "t.ratio", "p.value")
emmeans_regard[, (cols) := lapply(.SD, signif, 2), .SDcols = cols]
emmeans_siebert[, (cols) := lapply(.SD, signif, 2), .SDcols = cols]

emmeans_joined <- merge(
    emmeans_regard[, .(Model = model, Estimate = estimate, ` ` = signif, t = t.ratio, p = p.value)],
    emmeans_siebert[, .(Model = model, Estimate = estimate, ` ` = signif, t = t.ratio, p = p.value)],
    by = "Model",
    suffix = c("_regard", "_siebert"),
    sort = FALSE
)

fwrite(emmeans_joined, "./csv/summaries_models/models_joined/nlme_final_models_emm.csv")


# * GEE

regard_gee <- readRDS("./csv/summaries_models/regard/gee_model.rds")
siebert_gee <- readRDS("./csv/summaries_models/siebert/gee_model.rds")

regard_dt <- summary(regard_gee)$coefficients |>
    data.table(keep.rownames = "Coef") |>
    _[, metric := "regard"][
        ,
        p_stars := gtools::stars.pval(`Pr(>|W|)`)
    ][
        ,
        p_vals := signif(`Pr(>|W|)`, 2)
    ][, Estimate := signif(Estimate, 2)][
        ,
        `Std. Error` := signif(`Std.err`, 2)
    ][, Wald := signif(Wald, 2)]

siebert_dt <- summary(siebert_gee)$coefficients |>
    data.table(keep.rownames = "Coef") |>
    _[, metric := "siebert"][
        ,
        p_stars := gtools::stars.pval(`Pr(>|W|)`)
    ][
        ,
        p_vals := signif(`Pr(>|W|)`, 2)
    ][, Estimate := signif(Estimate, 2)][
        ,
        `Std. Error` := signif(`Std.err`, 2)
    ][, Wald := signif(Wald, 2)]

models_joined <- merge(
    regard_dt[, .(Coef, Estimate, ` ` = p_stars, `Std. Error`, Wald, p_vals, metric)],
    siebert_dt[, .(Coef, Estimate, ` ` = p_stars, `Std. Error`, Wald, p_vals, metric)],
    by = c("Coef"),
    suffix = c("_regard", "_siebert"),
    sort = FALSE
)

models_joined[, Coef := gsub("max_tokens", "Max tokens ", Coef)]

models_joined[, Coef := gsub("^model", "Model ", Coef)]
models_joined[, Coef := gsub(":gendermale", " : Male", Coef)]


fwrite(models_joined, "./csv/summaries_models/models_joined/gee_models.csv")


# * Now the emmeans GEE
emmeans_regard <- fread("./csv/summaries_models/regard/gee_emm.csv")
emmeans_siebert <- fread("./csv/summaries_models/siebert/gee_emm.csv")


cols <- c("estimate", "z.ratio", "p.value")
emmeans_regard[, (cols) := lapply(.SD, signif, 2), .SDcols = cols]
emmeans_siebert[, (cols) := lapply(.SD, signif, 2), .SDcols = cols]

emmeans_joined <- merge(
    emmeans_regard[, .(Model = model, Estimate = estimate, ` ` = signif, z = z.ratio, p = p.value)],
    emmeans_siebert[, .(Model = model, Estimate = estimate, ` ` = signif, z = z.ratio, p = p.value)],
    by = "Model",
    suffix = c("_regard", "_siebert"),
    sort = FALSE
)

fwrite(emmeans_joined, "./csv/summaries_models/models_joined/gee_emm.csv")

# * Now emmeans lm (no need to to do full output as doing those separately for regard and siebert)

emmeans_regard <- fread("./csv/summaries_models/regard/emm_dt_lm.csv")
emmeans_siebert <- fread("./csv/summaries_models/siebert/emm_dt_lm.csv")


cols <- c("estimate", "t.ratio", "p.value")
emmeans_regard[, (cols) := lapply(.SD, signif, 2), .SDcols = cols]
emmeans_siebert[, (cols) := lapply(.SD, signif, 2), .SDcols = cols]

emmeans_joined <- merge(
    emmeans_regard[, .(Model = model, Estimate = estimate, ` ` = signif, t = t.ratio, p = p.value)],
    emmeans_siebert[, .(Model = model, Estimate = estimate, ` ` = signif, t = t.ratio, p = p.value)],
    by = "Model",
    suffix = c("_regard", "_siebert"),
    sort = FALSE
)

fwrite(emmeans_joined, "./csv/summaries_models/models_joined/lm_emm.csv")
