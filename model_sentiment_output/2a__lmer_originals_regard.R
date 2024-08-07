source("./model_sentiment_output/model_sentiment_utils.R")
regard <- readRDS("./csv/originals_sentiment/regard.rds")

# * lmer model
mod <- lmerTest::lmer(
    sentiment ~ gender + gender_direction + (1 | sent_id),
    data = regard,
    REML = FALSE
)

create_dir("./csv/original_models/")
saveRDS(mod, "./csv/original_models/regard_lmer.rds")

# * gender not significant
summary(mod)
# Fixed effects:
#                      Estimate Std. Error         df  t value Pr(>|t|)
# (Intercept)         9.459e-01  8.264e-04  7.371e+03 1144.520   <2e-16 ***
# gendermale         -9.274e-05  1.134e-04  7.302e+03   -0.818    0.414
# gender_directionmf  8.223e-04  1.072e-03  7.302e+03    0.767    0.443

# * Now by document
regard_doc <- readRDS("./csv/originals_sentiment/regard_mean.rds")

# * lmer model
mod <- lmerTest::lmer(
    sentiment ~ gender + gender_direction + (1 | doc_id),
    data = regard_doc,
    REML = FALSE
)

# * gender still not significant
summary(mod)
# Fixed effects:
#                      Estimate Std. Error         df  t value Pr(>|t|)
# (Intercept)         9.459e-01  7.998e-04  3.409e+02 1182.699   <2e-16 ***
# gendermale         -6.946e-05  5.960e-05  3.400e+02   -1.165    0.245
# gender_directionmf  7.413e-04  1.032e-03  3.400e+02    0.718    0.473

saveRDS(mod, "./csv/original_models/regard_lmer_mean.rds")
