source("./model_sentiment_output/model_sentiment_utils.R")

distilbert <- readRDS("./csv/originals_sentiment/distilbert.rds")

# * lmer model
mod <- lmerTest::lmer(
    negative ~ gender + gender_direction + (1 | sent_id),
    data = distilbert,
    REML = FALSE
)
create_dir("./csv/original_models/")
saveRDS(mod, "./csv/original_models/distilbert_lmer.rds")

# gender is is significant for distilbert
summary(mod)
# Fixed effects:
#                      Estimate Std. Error         df t value Pr(>|t|)
# (Intercept)         2.216e-01  3.219e-03  7.311e+03  68.857   <2e-16 ***
# gendermale          1.110e-02  1.606e-04  7.302e+03  69.132   <2e-16 ***
# gender_directionmf -5.238e-03  4.182e-03  7.302e+03  -1.252     0.21


# * Now by document
distilbert_doc <- readRDS("./csv/originals_sentiment/distilbert_mean.rds")

# * lmer model
mod <- lmerTest::lmer(
    sentiment ~ gender + gender_direction + (1 | doc_id),
    data = distilbert_doc,
    REML = FALSE
)
summary(mod)
# Fixed effects:
#                      Estimate Std. Error         df t value Pr(>|t|)
# (Intercept)         7.777e-01  2.841e-03  3.404e+02 273.747   <2e-16 ***
# gendermale         -1.113e-02  1.436e-04  3.400e+02 -77.464   <2e-16 ***
# gender_directionmf  4.426e-03  3.666e-03  3.400e+02   1.207    0.228

saveRDS(mod, "./csv/original_models/distilbert_lmer_mean.rds")
