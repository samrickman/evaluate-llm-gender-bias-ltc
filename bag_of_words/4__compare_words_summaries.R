source("./bag_of_words/bag_of_words_utils.R")



word_count_list <- lapply(models, read_in_files)
# Summarise by word and do chisq and Fisher's exact test
word_count_summary_list <- lapply(word_count_list, create_word_count_summary)

# * Extract significant words if adjusted p < 0.05
signif_words_by_model <- lapply(word_count_summary_list, \(dat) {
    male_words <- dat[(word_count_female < word_count_male) &
        (fisher_p_value_adj < 0.05 | chisq_p_value_adj < 0.05)]
    female_words <- dat[(word_count_female > word_count_male) &
        (fisher_p_value_adj < 0.05 | chisq_p_value_adj < 0.05)]
    list(
        male = male_words[order(-abs(log_ratio))],
        female = female_words[order(-abs(log_ratio))]
    )
})

# * Relax the requirement of adjusted p-value
# This was not done in the original analysis but it makes sense for comparing
# e.g. a word like "memory" appears 1000 times for men and 902 times for women
# in the gemma model
# on its own this could be random chance (p = 0.025)
# but it also appeared more for men in the original data
# so it's more meaningful when it happens twice
signif_words_by_model_relaxed <- lapply(word_count_summary_list, \(dat) {
    male_words <- dat[(word_count_female < word_count_male) &
        (fisher_p_value < 0.05 | chisq_p_value < 0.05)]
    female_words <- dat[(word_count_female > word_count_male) &
        (fisher_p_value < 0.05 | fisher_p_value < 0.05)]
    list(
        male = male_words[order(-abs(log_ratio))],
        female = female_words[order(-abs(log_ratio))]
    )
})


create_dir_if_not_exists("csv/cleaned_data/")
saveRDS(word_count_list, "./csv/cleaned_data/word_count_list.rds")
saveRDS(word_count_summary_list, "./csv/cleaned_data/word_count_summary_list.rds")
saveRDS(signif_words_by_model, "./csv/cleaned_data/signif_words_by_model.rds")
saveRDS(signif_words_by_model_relaxed, "./csv/cleaned_data/signif_words_by_model_relaxed.rds")
