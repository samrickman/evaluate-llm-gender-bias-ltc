source("./check_hallucination/check_hallucination_utils.R")

original_dat <- load_data("./csv_originals/", doc_type = "originals")
summaries_dat <- load_data("./csv_summaries/", doc_type = "summaries")


# Make sure all counts are equal - we know they are
stopifnot(
    original_dat[, counts_equal]
)


# Join them
summaries_dat[
    original_dat,
    on = .(doc_id, term),
    `:=`(
        female_count_original = i.female_count,
        male_count_original = i.male_count
    )
][]

# Create a binary table of whether it is mentioned
compare_dat <- summaries_dat[, .(
    doc_id,
    term,
    term_type,
    gender_direction,
    model,
    max_tokens,
    in_original = (female_count_original > 0), # female_count == male_count so either would be fine
    in_female_summary = female_count > 0,
    in_male_summary = male_count > 0
)]


# Only t5 here
compare_dat[in_female_summary & !in_original, .N, .(model)]
#     model     N
#    <char> <int>
# 1:     t5    19

compare_dat[in_male_summary & !in_original, .N, .(model)]
#      model     N
#     <char> <int>
# 1: chatgpt     2
# 2:      t5     9


# wait -
# OK so just look at difference
female_possible_halluc <- compare_dat[in_female_summary & !in_original, .(female_n = .N), .(model, term, doc_id, max_tokens)]
male_possible_halluc <- compare_dat[in_male_summary & !in_original, .(male_n = .N), .(model, term, doc_id, max_tokens)]

# T5 hallucinates dementia a lot which is consistent with real data
# ChatGPT has one real hallucination - mf25 is just rephrasing "heart failure"
# but renal in mf7 is real
possible_halluc <- merge(
    female_possible_halluc,
    male_possible_halluc,
    by = c("model", "term", "doc_id", "max_tokens"),
    all = TRUE
)

# In the real data I went through them all but here is just an example of
# how to filter them - they're all T5 anyway and we know from the real
# data it hallucinates quite a bit
halluc <- possible_halluc[!(term == "heart disease" & model == "chatgpt")]

dir.create("csv/")
fwrite(halluc, "./csv/hallucinations.csv")

halluc[, .(
    female_n = sum(female_n, na.rm = TRUE),
    male_n = sum(male_n, na.rm = TRUE)
), .(
    model
)]

# * This is really very few hallucinations
# given over 1m checks (550800 rows, each row contains
# a male and female check)

#     model female_n male_n
#     <char>    <int>  <int>
# 1: chatgpt        0      1
# 2:      t5       19      9
