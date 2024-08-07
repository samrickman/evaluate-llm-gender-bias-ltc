source("./bag_of_words/bag_of_words_utils.R")


# OK so we know that the number of words is always the same excluding the
# words to exclude like mr, mrs and removing the docs identified at the start
# but let's just double-check no mistakes have been made along the way
docs_to_remove <- jsonlite::fromJSON("../model_sentiment_output/csv/docs_to_remove.json")
words_to_exclude <- get_words_to_exclude()

fm <- fread("./csv/originals/female_to_male_word_df.csv")
fm <- fm[!doc_num %in% docs_to_remove$female_to_male]
fm <- fm[!word %in% words_to_exclude]

stopifnot(
    fm[, word_count_female == word_count_male]
)


mf <- fread("./csv/originals/male_to_female_word_df.csv")
mf <- mf[!word %in% words_to_exclude]
mf <- mf[!doc_num %in% docs_to_remove$male_to_female]

stopifnot(
    mf[, word_count_female == word_count_male]
)

message(
    "Success. Words appear equally in male and female originals."
)
