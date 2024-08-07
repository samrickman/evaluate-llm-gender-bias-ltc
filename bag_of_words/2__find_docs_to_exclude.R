source("./bag_of_words/bag_of_words_utils.R")

mf_words <- fread("../bag_of_words/csv/originals/male_to_female_word_df.csv")
fm_words <- fread("../bag_of_words/csv/originals/female_to_male_word_df.csv")

mf_exclude <- mf_words[word_count_male != word_count_female & !word %in% words_to_ignore]
fm_exclude <- fm_words[word_count_male != word_count_female & !word %in% words_to_ignore]

docs_to_remove <- list(
    female_to_male  = sort(unique(fm_exclude$doc_num)),
    male_to_female  = sort(unique(mf_exclude$doc_num))
)
jsonlite::write_json(docs_to_remove, "../model_sentiment_output/csv/docs_to_remove.json")
