source("./model_sentiment_output/model_sentiment_utils.R")
siebert <- readRDS("./csv/summaries_sentiment/siebert.rds")

emmeans::emm_options(
    lmerTest.limit = 1e6,
    pbkrtest.limit = 1e6
)


set.seed(42)
B <- 1e3 # number of iterationss

unique_docs <- siebert[, unique(doc_id)]


bootstrap_emms <- pbapply::pblapply(seq_len(B), \(.) run_bootstrap(unique_docs, siebert))

# Takes a few mins to save (it's 1.6GB compressed)
saveRDS(bootstrap_emms, "./csv/summaries_models/siebert/bootstrap_emms.rds")


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


#      model     estimate    t.ratio     N N_less_01 N_less_05
#     <fctr>        <num>      <num> <int>     <int>     <int>
# 1:    bart  0.001556925  0.3028667  1000         1        28
# 2: chatgpt  0.008635507  1.4348441  1000        19       157
# 3:   gemma  0.037740055  6.3255378  1000      1000      1000
# 4:  llama3  0.001535663  0.2508609  1000         0         0
# 5:      t5 -0.006540350 -1.1190165  1000        20       105

fwrite(bootstrap_emms_means, "./csv/summaries_models/siebert/bootstrap_emms_means.csv")


# * Then look at the bias
bootstrap_coefs <- lapply(bootstrap_emms, \(x) lme4::fixef(x$final_model)) |>
    simplify2array()

# Calculate mean of bootstrap estimates
mean_bootstrap_coefs <- rowMeans(bootstrap_coefs)

# Get coefficients from final model
final_model <- lmerTest::lmer(
    sentiment ~ model * gender + max_tokens + (model | doc_id),
    data = siebert,
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
fwrite(final_model_dt, "./csv/summaries_models/siebert/model_output_tables/bootstrap_estimates_siebert.csv")
