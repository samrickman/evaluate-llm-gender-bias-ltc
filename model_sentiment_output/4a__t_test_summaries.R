source("./model_sentiment_output/model_sentiment_utils.R")

create_null_dt <- function() {
    # If there's an error because not enough levels (e.g. all preds are negative)
    col_names <- c(
        "t_test_p", "mcnemar_p", "corr", "corr_p", "diff", "effsize",
        "magnitude", "col", "t_signif", "mcnemar_signif", "corr_signif",
        "metric"
    )
    data.table(matrix(rep(NA, length(col_names)), nrow = 1)) |>
        setnames(col_names)
}


generate_metrics_summaries <- function(dat, col_to_order_by = "t_test_p") {
    # Generate list of lists of sentiment pairs, male and female
    # for each set of parameter e.g. lama3_mf_None
    dat_split <- with(
        dat,
        split(
            dat,
            list(model, gender_direction, max_tokens),
            sep = "_"
        ) |>
            lapply(\(dt) split(dt, dt$gender))
    )

    metrics_results <- Map(
        \(dt, nm) tryCatch(
            do_stat_tests_all(dt, nm),
            error = \(cnd) create_null_dt()
        ),
        dat_split,
        names(dat_split)
    ) |>
        rbindlist() |>
        _[
            ,
            c("model", "gender_direction", "max_tokens") := tstrsplit(metric, "_")
        ][, metric := NULL][, col := NULL] # specify metric and col is same per metric (negative or pred)

    # order by either t_test or mcnemar_p value
    setorderv(metrics_results, col_to_order_by)
}


regard <- readRDS("./csv/summaries_sentiment/regard.rds")
siebert <- readRDS("./csv/summaries_sentiment/siebert.rds")

# Don't need the mcnemar test as document level is continuous
metrics_regard <- generate_metrics_summaries(regard)[, `:=`(
    mcnemar_p = NULL,
    mcnemar_signif = NULL
)][]

# * ok so there's an effect here - again mostly in gemma, not in chatgpt
metrics_regard[t_signif != " "] |> head(10)
#     t_test_p       corr       corr_p         diff    effsize magnitude t_signif corr_signif  model gender_direction max_tokens
#        <num>      <num>        <num>        <num>      <num>     <ord>   <char>      <char> <char>           <char>     <char>
#  1: 1.31e-07 0.33811357 7.575531e-07 -0.006938677 -0.3830265     small      ***         ***  gemma               mf       None
#  2: 1.09e-05 0.34555084 3.784791e-05 -0.006357001 -0.3918430     small      ***         ***  gemma               fm        300
#  3: 1.64e-05 0.85114283 2.514305e-39  0.005992791  0.3832578     small      ***         ***   bart               fm         50
#  4: 7.12e-05 0.57186940 4.115464e-19  0.005779666  0.2839600     small      ***         ***     t5               mf        150
#  5: 7.62e-05 0.73470180 2.438073e-24  0.006170406  0.3499788     small      ***         ***   bart               fm         75
#  6: 1.43e-04 0.75178456 5.318586e-26  0.005744030  0.3357255     small      ***         ***   bart               fm        100
#  7: 1.43e-04 0.75178456 5.318586e-26  0.005744030  0.3357255     small      ***         ***   bart               fm        150
#  8: 1.43e-04 0.75178456 5.318586e-26  0.005744030  0.3357255     small      ***         ***   bart               fm        300
#  9: 1.43e-04 0.75178456 5.318586e-26  0.005744030  0.3357255     small      ***         ***   bart               fm       None
# 10: 1.17e-03 0.08005899 3.541806e-01 -0.007357672 -0.2844137     small       **              gemma               fm        100

metrics_siebert <- generate_metrics_summaries(siebert)[, `:=`(
    mcnemar_p = NULL,
    mcnemar_signif = NULL
)][]

metrics_siebert[!is.na(t_signif)][order(-abs(effsize))]

#     t_test_p        corr       corr_p        diff    effsize  magnitude t_signif corr_signif  model gender_direction max_tokens
#        <num>       <num>        <num>       <num>      <num>      <ord>   <char>      <char> <char>           <char>     <char>
#  1: 3.98e-09 0.071365291 3.104249e-01 -0.14480711 -0.4308034      small      ***              gemma               mf         50
#  2: 5.07e-05 0.263575105 1.395025e-04 -0.03615170 -0.2899226      small      ***         ***  gemma               mf       None
#  3: 1.72e-03 0.007319529 9.326010e-01 -0.07083652 -0.2742612      small       **              gemma               fm         75
#  4: 2.00e-03 0.286848938 7.097927e-04 -0.03400772 -0.2702368      small       **         ***  gemma               fm        300
#  5: 2.75e-03 0.189274887 2.731964e-02 -0.05366357 -0.2615446      small       **           *  gemma               fm        100
#  6: 4.45e-03 0.212980391 1.279364e-02 -0.08376671 -0.2480456      small       **           *  gemma               fm         50
#  7: 1.45e-02 0.452567572 3.173552e-08  0.05824741  0.2123443      small        *         ***     t5               fm         75
#  8: 5.94e-03 0.604174228 1.102179e-21  0.04770744  0.1946636 negligible       **         ***     t5               mf         75
#  9: 4.05e-02 0.681043379 7.321357e-20 -0.03905283 -0.1773338 negligible        *         ***     t5               fm         50
# 10: 5.97e-02 0.662454234 1.588352e-18  0.03128685  0.1628604 negligible        .         ***     t5               fm        150
# 11: 6.18e-02 0.759645156 8.227747e-27 -0.02310562 -0.1614970 negligible        .         ***   bart               fm        100
# 12: 6.18e-02 0.759645156 8.227747e-27 -0.02310562 -0.1614970 negligible        .         ***   bart               fm        150
# 13: 6.18e-02 0.759645156 8.227747e-27 -0.02310562 -0.1614970 negligible        .         ***   bart               fm        300
# 14: 6.18e-02 0.759645156 8.227747e-27 -0.02310562 -0.1614970 negligible        .         ***   bart               fm       None
# 15: 1.08e-01 0.592417095 3.030448e-14  0.02920311  0.1387893 negligible                  ***     t5               fm        300
# 16: 9.31e-02 0.427436203 1.816520e-10 -0.02738159 -0.1181371 negligible        .         ***     t5               mf       None
# 17: 2.01e-01 0.760263677 7.082503e-27 -0.01577432 -0.1101285 negligible                  ***   bart               fm         75

create_dir("./csv/")
fwrite(metrics_siebert, "./csv/summaries_sentiment/siebert_mcnemar.csv")
fwrite(metrics_regard, "./csv/summaries_sentiment/regard_t_test.csv")
