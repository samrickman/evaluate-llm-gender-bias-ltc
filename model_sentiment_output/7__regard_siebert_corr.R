source("./model_sentiment_output/model_sentiment_utils.R")
regard <- readRDS("./csv/originals_sentiment/regard.rds")
siebert <- readRDS("./csv/originals_sentiment/siebert.rds")

regard_siebert <- merge(
    regard[, .(gender, sent_id, regard = positive)],
    siebert[, .(gender, sent_id, siebert = positive)],
    by = c("gender", "sent_id")
)

# It was 0.09 in originals but with synthetic data it's 0.26
# so the data does have some differences
cor.test(
    regard_siebert$regard,
    regard_siebert$siebert
)
