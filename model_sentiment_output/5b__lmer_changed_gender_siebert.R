source("./model_sentiment_output/model_sentiment_utils.R")

# Continuous metric without gender_direction
siebert <- readRDS("./csv/summaries_sentiment/siebert.rds")


# Let's just work through this methodically
# null model - no slope, just intercept
null_model <- lmerTest::lmer(
    sentiment ~ (1 | doc_id),
    data = siebert,
    REML = FALSE
)

mod_model_fe <- lmerTest::lmer(
    sentiment ~ model + (1 | doc_id),
    data = siebert,
    REML = FALSE
)

anova(null_model, mod_model_fe)

mod_covars_fe <- lmerTest::lmer(
    sentiment ~ model + max_tokens + (1 | doc_id),
    data = siebert,
    REML = FALSE
)

anova(mod_model_fe, mod_covars_fe)

mod_covars_fe_model_fe <- lmerTest::lmer(
    sentiment ~ model * gender + max_tokens + (1 | doc_id),
    data = siebert,
    REML = FALSE
)

# This one doesn't converge with MLE so can't do anova
mod_covars_fe_model_re <- lmerTest::lmer(
    sentiment ~ model + max_tokens + (model | doc_id),
    data = siebert,
    REML = FALSE,
    control = lme4::lmerControl(
        optimizer = "bobyqa" # for convergence
    )
)

anova(mod_covars_fe_model_fe, mod_covars_fe_model_re)

anova(mod_covars_fe, mod_covars_fe_model_re)



# * Final model
final_model <- lmerTest::lmer(
    sentiment ~ model * gender + max_tokens + (model | doc_id),
    data = siebert,
    REML = FALSE,
    control = lme4::lmerControl(
        optimizer = "bobyqa" # for convergence
    )
)

# Final model is better than both
anova(mod_covars_fe_model_fe, final_model)
anova(mod_covars_fe_model_re, final_model)

create_dir("./csv/summaries_models/siebert/")
saveRDS(final_model, "./csv/summaries_models/siebert/final_lmer_model.rds")

emmeans::emm_options(
    lmerTest.limit = 1e5,
    pbkrtest.limit = 1e5
)
siebert_contrasts_dt <- emmeans::emmeans(final_model, ~ gender | model) |>
    pairs() |>
    as.data.table() |>
    _[, signif := gtools::stars.pval(p.value)][]

# Clean it up a little
siebert_contrasts_dt_clean <- siebert_contrasts_dt[, .(
    model,
    estimate,
    t.ratio,
    p.value,
    signif
)]

#      model     estimate    t.ratio      p.value signif
#     <fctr>        <num>      <num>        <num> <char>
# 1:    bart  0.001315760  0.2767638 7.819645e-01
# 2: chatgpt  0.008569058  1.8024601 7.148913e-02      .
# 3:   gemma  0.037765558  7.9438039 2.071101e-15    ***
# 4:  llama3  0.001601590  0.3368867 7.362061e-01
# 5:      t5 -0.006502027 -1.3676702 1.714318e-01

fwrite(siebert_contrasts_dt_clean, "./csv/summaries_models/siebert/emm_lmm.csv")


# * Do the gee model as well

gee_model <- geepack::geeglm(
    sentiment ~ model * gender + max_tokens,
    id = doc_id,
    data = siebert,
    family = gaussian(),
    corstr = "exchangeable"
)


siebert_gee_contrasts_dt <- emmeans::emmeans(gee_model, ~ gender | model) |>
    pairs() |>
    as.data.table() |>
    _[, signif := gtools::stars.pval(p.value)][]

# Clean it up a little
siebert_gee_contrasts_dt_clean <- siebert_gee_contrasts_dt[, .(
    model,
    estimate,
    z.ratio,
    p.value,
    signif
)]


saveRDS(gee_model, "./csv/summaries_models/siebert/gee_model.rds")
fwrite(siebert_gee_contrasts_dt_clean, "./csv/summaries_models/siebert/gee_emm.csv")


# * Robust model
library(robustlmm)

robust_model <- robustlmm::rlmer(
    sentiment ~ model * gender + max_tokens + (model | doc_id),
    data = siebert,
    method = "DASvar",
    verbose = 3,
    control = lme4::lmerControl(
        optimizer = "bobyqa" # for convergence
    )
)

robust_clean <- emmeans::emmeans(robust_model, ~ gender | model) |>
    pairs() |>
    as.data.table() |>
    _[, signif := gtools::stars.pval(p.value)][, .(
        model,
        estimate,
        z.ratio,
        p.value,
        signif
    )][]

#      model     estimate    z.ratio      p.value signif
#     <fctr>        <num>      <num>        <num> <char>
# 1:    bart  0.001651021  0.4429644 6.577914e-01
# 2: chatgpt  0.005873067  1.5757277 1.150886e-01
# 3:   gemma  0.032875162  8.8203148 1.141383e-18    ***
# 4:  llama3  0.001063965  0.2854589 7.752926e-01
# 5:      t5 -0.005353343 -1.4362872 1.509206e-01

fwrite(robust_clean, "./csv/summaries_models/siebert/emm_robust.csv")
saveRDS(robust_model, "./csv/summaries_models/siebert/robust_lmer.rds")


# * nlme model - allow for heteroscedasticity over max_tokens
final_model_nlme <- nlme::lme(
    fixed = sentiment ~ model * gender + max_tokens,
    random = ~ 1 | doc_id,
    data = siebert,
    weights = nlme::varIdent(form = ~ doc_id | model),
    method = "REML",
    control = nlme::lmeControl(opt = "optim", maxIter = 1000, msMaxIter = 1000)
)

nlme_emm <- emmeans::emmeans(final_model_nlme, ~ gender | model) |>
    pairs() |>
    as.data.table() |>
    _[, signif := gtools::stars.pval(p.value)][, .(
        model,
        estimate,
        t.ratio,
        p.value,
        signif
    )][]

#      model     estimate    t.ratio      p.value signif
#     <fctr>        <num>      <num>        <num> <char>
# 1:    bart  0.001315760  0.2421971 8.086299e-01
# 2: chatgpt  0.008569058  1.7024557 8.868548e-02      .
# 3:   gemma  0.037765558  8.3414080 7.810345e-17    ***
# 4:  llama3  0.001601590  0.3586895 7.198312e-01
# 5:      t5 -0.006502027 -0.8738392 3.822163e-01

fwrite(nlme_emm, "./csv/summaries_models/siebert/emm_nlme.csv")
saveRDS(final_model_nlme, "./csv/summaries_models/siebert/nlme_model.rds")
