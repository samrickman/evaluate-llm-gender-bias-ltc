source("./model_sentiment_output/model_sentiment_utils.R")

siebert <- readRDS("./csv/originals_sentiment/siebert.rds")

# * lmer model

mod <- lme4::glmer(
    pred ~ gender + gender_direction + (1 | sent_id),
    data = siebert,
    family = binomial
)

create_dir("./csv/original_models/")
saveRDS(mod, "./csv/original_models/siebert_lmer.rds")

# gender is not significant
summary(mod)
# Fixed effects:
#                     Estimate Std. Error z value Pr(>|z|)
# (Intercept)        13.070903   0.309175  42.277   <2e-16 ***
# gendermale          0.028883   0.238495   0.121    0.904
# gender_directionmf -0.004246   0.353332  -0.012    0.990

# * Now by document
siebert_doc <- readRDS("./csv/originals_sentiment/siebert_mean.rds")

# * lmer model
mod <- lmerTest::lmer(
    negative ~ gender + gender_direction + (1 | doc_id),
    data = siebert_doc,
    REML = FALSE
)
summary(mod)
# Fixed effects:
#                      Estimate Std. Error         df t value Pr(>|t|)
# (Intercept)         1.550e-01  5.303e-03  3.406e+02  29.230   <2e-16 ***
# gendermale         -9.619e-05  3.040e-04  3.400e+02  -0.316    0.752
# gender_directionmf  2.448e-03  6.844e-03  3.400e+02   0.358    0.721

saveRDS(mod, "./csv/original_models/siebert_lmer_mean.rds")
