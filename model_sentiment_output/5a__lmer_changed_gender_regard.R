source("./model_sentiment_output/model_sentiment_utils.R")

regard <- readRDS("./csv/summaries_sentiment/regard.rds")

# Let's just work through this methodically
# null model - no slope, just intercept
null_model <- lmerTest::lmer(
    sentiment ~ (1 | doc_id),
    data = regard,
    REML = FALSE
)

mod_model_fe <- lmerTest::lmer(
    sentiment ~ model + (1 | doc_id),
    data = regard,
    REML = FALSE
)

anova(null_model, mod_model_fe)

mod_covars_fe <- lmerTest::lmer(
    sentiment ~ model + max_tokens + (1 | doc_id),
    data = regard,
    REML = FALSE
)

anova(mod_model_fe, mod_covars_fe)

mod_covars_fe_model_re <- lmerTest::lmer(
    sentiment ~ model + max_tokens + (model | doc_id),
    data = regard,
    REML = FALSE
)

anova(mod_covars_fe, mod_covars_fe_model_re)

# * Final model
final_model <- lmerTest::lmer(
    sentiment ~ model * gender + max_tokens + (model | doc_id),
    data = regard,
    REML = FALSE
)

anova(mod_covars_fe_model_re, final_model)

#                        npar    AIC    BIC logLik deviance  Chisq Df Pr(>Chisq)
# mod_covars_fe_model_re   20 -87006 -86852  43523   -87046
# final_model              24 -87137 -86952  43593   -87185 139.43  4  < 2.2e-16 ***


create_dir("./csv/summaries_models/regard/")
saveRDS(final_model, "./csv/summaries_models/regard/final_lmer_model.rds")

emmeans::emm_options(
    lmerTest.limit = 1e5,
    pbkrtest.limit = 1e5
)
regard_contrasts_dt <- emmeans::emmeans(final_model, ~ gender | model) |>
    pairs() |>
    as.data.table() |>
    _[, signif := gtools::stars.pval(p.value)][]

# Clean it up a little
regard_contrasts_dt_clean <- regard_contrasts_dt[, .(
    model,
    estimate,
    t.ratio,
    p.value,
    signif
)]


fwrite(regard_contrasts_dt_clean, "./csv/summaries_models/regard/emm_lmm.csv")

#      model      estimate    t.ratio      p.value signif
#     <fctr>         <num>      <num>        <num> <char>
# 1:    bart -0.0037437859 -7.5076408 6.290156e-14    ***
# 2: chatgpt -0.0005691409 -1.1413327 2.537461e-01
# 3:   gemma  0.0033263097  6.6704506 2.621528e-11    ***
# 4:  llama3 -0.0002531010 -0.5075587 6.117688e-01
# 5:      t5 -0.0026892524 -5.3929208 7.016236e-08    ***

# * Do the gee model as well

gee_model <- geepack::geeglm(
    sentiment ~ model * gender + max_tokens,
    id = doc_id,
    data = regard,
    family = gaussian(),
    corstr = "exchangeable" # unstructured would be better but won't run
)


regard_gee_contrasts_dt <- emmeans::emmeans(gee_model, ~ gender | model) |>
    pairs() |>
    as.data.table() |>
    _[, signif := gtools::stars.pval(p.value)][]

# Clean it up a little
regard_gee_contrasts_dt_clean <- regard_gee_contrasts_dt[, .(
    model,
    estimate,
    z.ratio,
    p.value,
    signif
)]

#      model      estimate    z.ratio      p.value signif
#     <fctr>         <num>      <num>        <num> <char>
# 1:    bart -0.0037437859 -4.9049920 9.343115e-07    ***
# 2: chatgpt -0.0005691409 -0.8492563 3.957387e-01
# 3:   gemma  0.0033263097  6.1613976 7.210569e-10    ***
# 4:  llama3 -0.0002531010 -0.4610281 6.447784e-01
# 5:      t5 -0.0026892524 -3.5888180 3.321806e-04    ***

saveRDS(gee_model, "./csv/summaries_models/regard/gee_model.rds")
fwrite(regard_gee_contrasts_dt_clean, "./csv/summaries_models/regard/gee_emm.csv")

# * Try robust lmm model
# This is quite slow (about 30 mins) and also requires you to load the library

library(robustlmm)

robust_model <- robustlmm::rlmer(
    sentiment ~ model * gender + max_tokens + (model | doc_id),
    data = regard,
    method = "DASvar",
    verbose = 3
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

#      model      estimate    z.ratio      p.value signif
#     <fctr>         <num>      <num>        <num> <char>
# 1:    bart -0.0035098888 -8.8374927 9.789059e-19    ***
# 2: chatgpt -0.0001648231 -0.4150056 6.781378e-01
# 3:   gemma  0.0027794474  6.9983260 2.590387e-12    ***
# 4:  llama3 -0.0001313427 -0.3307057 7.408668e-01
# 5:      t5 -0.0029252215 -7.3653685 1.766582e-13    ***

fwrite(robust_clean, "./csv/summaries_models/regard/robust_emm.csv")
saveRDS(robust_model, "./csv/summaries_models/regard/robust_lmer.rds")

# * nlme model - allow for heteroscedasticity over max_tokens

final_model_nlme <- nlme::lme(
    fixed = sentiment ~ model * gender + max_tokens,
    random = ~ 1 | doc_id,
    data = regard,
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

#      model      estimate    t.ratio      p.value signif
#     <fctr>         <num>      <num>        <num> <char>
# 1:    bart -0.0037437859 -5.5941199 2.246387e-08    ***
# 2: chatgpt -0.0005691409 -0.9159491 3.597046e-01
# 3:   gemma  0.0033263097  6.5347013 6.526931e-11    ***
# 4:  llama3 -0.0002531010 -0.5248530 5.996912e-01
# 5:      t5 -0.0026892524 -3.8429664 1.219310e-04    ***

fwrite(nlme_emm, "./csv/summaries_models/regard/nlme_emm.csv")
saveRDS(final_model_nlme, "./csv/summaries_models/regard/nlme_model.rds")
