source("./model_sentiment_output/model_sentiment_utils.R")
regard <- readRDS("./csv/summaries_sentiment/regard.rds")

emmeans::emm_options(
    lmerTest.limit = 1e6,
    pbkrtest.limit = 1e6
)

# Manual bootstrap


run_bootstrap <- function(unique_doc_ids = unique_docs, dt = regard) {
    boot_docs <- sample(unique_doc_ids, replace = TRUE)

    # Create cases bootstrap (i.e. by doc_id)
    boot_data <- dt[doc_id %in% boot_docs, ]

    final_model <- lmerTest::lmer(
        sentiment ~ model * gender + max_tokens + (model | doc_id),
        data = boot_data, ,
        control = lme4::lmerControl(
            optimizer = "bobyqa" # for convergence
        )
    )

    contrasts_dt <- emmeans::emmeans(final_model, ~ gender | model) |>
        pairs() |>
        as.data.table() |>
        _[, signif := gtools::stars.pval(p.value)][]

    # Clean it up a little
    contrasts_dt_clean <- contrasts_dt[, .(
        model,
        estimate,
        t.ratio,
        p.value,
        signif
    )]
    list(
        boot_data = boot_data,
        final_model = final_model,
        emm = contrasts_dt_clean
    )
}

set.seed(42)
B <- 1e3 # number of iterationss

unique_docs <- regard[, unique(doc_id)]


bootstrap_emms <- pbapply::pblapply(seq_len(B), \(.) run_bootstrap(unique_docs, regard))

# Takes a few mins to save (it's 1.6GB compressed)
saveRDS(bootstrap_emms, "./csv/summaries_models/regard/bootstrap_emms.rds")

bootstrap_emms_emm <- lapply(
    bootstrap_emms,
    \(x) x$emm
) |> rbindlist(id = "id")

bootstrap_emms_means <- bootstrap_emms_emm[, .(
    estimate = mean(estimate),
    t.ratio = median(t.ratio),
    N = .N,
    N_less_01 = sum(p.value < 0.01),
    N_less_05 = sum(p.value < 0.05)
), .(model)]

#      model      estimate    t.ratio     N N_less_01 N_less_05
#     <fctr>         <num>      <num> <int>     <int>     <int>
# 1:    bart -0.0037599207 -5.9684557  1000       999      1000
# 2: chatgpt -0.0005688589 -0.9205942  1000         2        47
# 3:   gemma  0.0033443412  5.3511452  1000      1000      1000
# 4:  llama3 -0.0002552990 -0.4068158  1000         0         0
# 5:      t5 -0.0026858517 -4.2869879  1000       988      1000

fwrite(bootstrap_emms_means, "./csv/summaries_models/regard/bootstrap_emms_means.csv")


# * Then look at the bias
bootstrap_coefs <- lapply(bootstrap_emms, \(x) lme4::fixef(x$final_model)) |>
    simplify2array()

# Calculate mean of bootstrap estimates
mean_bootstrap_coefs <- rowMeans(bootstrap_coefs)

# Get coefficients from final model
final_model <- lmerTest::lmer(
    sentiment ~ model * gender + max_tokens + (model | doc_id),
    data = regard,
    control = lme4::lmerControl(
        optimizer = "bobyqa" # for convergence
    )
)

final_model_dt <- summary(final_model) |>
    coefficients() |>
    data.table(keep.rownames = "model")

final_model_dt[, mean_bootstrap_coefs := mean_bootstrap_coefs]
final_model_dt[, bias := mean_bootstrap_coefs - Estimate]
final_model_dt[, relative_bias := bias / Estimate]
final_model_dt[, standardised_bias := bias / `Std. Error`]

numeric_cols <- names(final_model_dt)[sapply(final_model_dt, is.numeric)]
final_model_dt[, (numeric_cols) := lapply(.SD, \(x) as.character(round(x, 3))), .SDcols = numeric_cols]
final_model_dt[, (numeric_cols) := lapply(.SD, \(x) fifelse(x == "0", "<0.001", x)), .SDcols = numeric_cols]

setnames(
    final_model_dt,
    c("mean_bootstrap_coefs", "bias", "relative_bias", "standardised_bias"),
    c("Estimate", "Absolute", "Relative", "Standardised"),
)
final_model_dt[, df := NULL]
fwrite(final_model_dt, "./csv/summaries_models/regard/model_output_tables/bootstrap_estimates_regard.csv")
