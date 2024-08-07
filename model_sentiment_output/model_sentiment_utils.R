library(data.table)
setwd("./model_sentiment_output/")
source("./_constants.R")

create_dir <- function(path_str, exists_ok = TRUE, recursive = TRUE) {
    # There's technically a race condition here but in
    # the context this is running it it won't matter
    if (!exists_ok) {
        if (dir.exists(path_str)) stop("Directory already exists")
    }
    if (!dir.exists(path_str)) dir.create(path_str, recursive = recursive)
}

# Make sure we have read the right files
check_comparable <- function(l, check_dims = TRUE) {
    # Check dimensions
    # we don't want this in the case of summaries
    # with different numbers of sentences
    # per document
    if (check_dims) {
        stopifnot(dim(l$male) == dim(l$female))
    }

    # Check names
    stopifnot(
        names(l$male) == names(l$female)
    )

    stopifnot(
        sapply(l$male, class) == sapply(l$female, class)
    )
    l
}

read_sentiment_output <- function(
    gender = "female_to_male",
    metrics = c("regard", "siebert", "distilbert"),
    in_dir = "evaluate_sentiment/output_originals/",
    subdir = "",
    pattern = NULL,
    check_dims = TRUE,
    doc_type = c("originals", "summaries")) {
    doc_type <- match.arg(doc_type)
    lapply(
        setNames(metrics, metrics),
        \(metric) dir(
            sprintf(
                "../%s/%s/%s",
                in_dir,
                metric,
                subdir
            ),
            full.names = TRUE,
            pattern = sprintf("%s.+csv$", pattern %||% gender)
        )
    ) |>
        lapply(
            \(metric) lapply(
                metric,
                \(filename) {
                    message("Reading in: ", filename)
                    dt <- fread(filename) |>
                        setnames(tolower)
                    # Create pred - only meaningful for siebert
                    # they can't be equal because of softmax activation
                    dt[, pred := fifelse(positive > negative, 1L, 0L)]
                    if (doc_type == "originals") {
                        message("Removing ", length(docs_to_remove[[gender]]), " documents with different word counts")
                        dt <- dt[!doc_num %in% docs_to_remove[[gender]]]
                    }

                    if (!is.data.table(dt) || nrow(dt) == 0) message("Something went wrong with reading: ", metric)
                    dt
                }
            ) |>
                setNames(ifelse(grepl("female\\.csv$", metric), "female", "male")) |>
                check_comparable(check_dims)
        )
}

do_stat_tests_row <- function(sentiment_pair, col, paired = TRUE) {
    dt_long <- rbind(sentiment_pair[[1]][, gender := "female"], sentiment_pair[[2]][, gender := "male"])
    dt_long[, gender := factor(gender)]
    form <- reformulate("gender", col)

    t_test_df <- rstatix::t_test(
        dt_long,
        form,
        paired = paired,
        alternative = "two.sided",
        detailed = TRUE
    )

    corr_test_list <- cor.test(
        sentiment_pair[[1]][[col]], sentiment_pair[[2]][[col]]
    )

    cohens_d <- rstatix::cohens_d(
        dt_long,
        form,
        paired = paired
    )

    # wilcox <- rstatix::wilcox_test(
    #     dt_long,
    #     gender ~ negative, # only for binary, i.e. siebert
    #     paired = TRUE,
    #     detailed = FALSE
    # )

    # regard always more negative than positive
    # we don't want it for regard anyway as it's continuous
    if (dt_long[, uniqueN(pred)] == 1) {
        mcnemnar_p <- NA_real_
    } else {
        mcnemnar_p <- mcnemar.test(dt_long[gender == "female", pred], dt_long[gender == "male", pred])$p.value
    }


    data.frame(
        t_test_p = t_test_df$p,
        mcnemar_p = mcnemnar_p,
        corr = corr_test_list$estimate,
        corr_p = corr_test_list$p.value,
        diff = t_test_df$estimate,
        effsize = cohens_d$effsize,
        magnitude = cohens_d$magnitude,
        col = col
    )
}

do_stat_tests_all <- function(sentiment_pair, metric = "regard", columns_to_check = c("negative"), paired = TRUE) {
    rbindlist(
        lapply(
            columns_to_check,
            \(col) do_stat_tests_row(sentiment_pair, col)
        )
    )[
        ,
        `:=`(
            t_signif = gtools::stars.pval(t_test_p),
            mcnemar_signif = gtools::stars.pval(mcnemar_p),
            corr_signif = gtools::stars.pval(corr_p),
            metric = metric
        )
    ]
}


get_model_params <- function(pattern) {
    desired_params <- c("gender", "model", "max_tokens")
    setNames(
        unlist(strsplit(pattern, "_")),
        desired_params
    )[desired_params]
}

generate_metrics <- function(
    in_dir = "evaluate_sentiment/output_originals",
    subdir = "",
    pattern = NULL,
    max_tokens = NULL,
    diagnostics = FALSE,
    paired = TRUE,
    columns_to_check = c("negative", "positive")) {
    sentiment_list_fm <- read_sentiment_output("female_to_male", in_dir = in_dir, subdir = subdir)
    sentiment_list_mf <- read_sentiment_output("male_to_female", in_dir = in_dir, subdir = subdir)

    columns_to_check <- match.arg(columns_to_check)
    fm_t_test_results <- Map(
        \(x, y) do_stat_tests_all(x, y, columns_to_check = columns_to_check, paired = paired),
        sentiment_list_fm,
        names(sentiment_list_fm)
    ) |> rbindlist()

    mf_t_test_results <- Map(
        \(x, y) do_stat_tests_all(x, y, columns_to_check = columns_to_check, paired = paired),
        sentiment_list_mf,
        names(sentiment_list_mf)
    ) |> rbindlist()

    results_list <- list(
        fm = fm_t_test_results,
        mf = mf_t_test_results
    )

    if (!diagnostics) {
        return(rbindlist(results_list, idcol = "gender"))
    }

    mf_cooks_d <- lapply(columns_to_check, \(col) {
        Map(
            \(sentiment_pair, metric) get_diagnostics(sentiment_pair, metric, col),
            sentiment_list_mf,
            names(sentiment_list_mf)
        )
    }) |> unlist(recursive = FALSE)

    fm_cooks_d <- lapply(columns_to_check, \(col) {
        Map(
            \(sentiment_pair, metric) get_diagnostics(sentiment_pair, metric, col),
            sentiment_list_fm,
            names(sentiment_list_fm)
        )
    }) |> unlist(recursive = FALSE)

    list(
        fm_cooks_d = fm_cooks_d,
        mf_cooks_d = mf_cooks_d
    )
}


generate_metrics_combined <- function(
    in_dir = "evaluate_sentiment/output_originals",
    subdir = "",
    pattern = NULL,
    max_tokens = NULL,
    diagnostics = FALSE,
    paired = paired,
    columns_to_check = c("negative", "positive")) {
    sentiment_list_fm <- read_sentiment_output("female_to_male", in_dir = in_dir, subdir = subdir)
    sentiment_list_mf <- read_sentiment_output("male_to_female", in_dir = in_dir, subdir = subdir)

    sentiment_list <- Map(
        \(fm, mf) {
            mf$male[, doc_id := paste0("mf", doc_num)]
            mf$female[, doc_id := paste0("mf", doc_num)]
            fm$female[, doc_id := paste0("fm", doc_num)]
            fm$male[, doc_id := paste0("fm", doc_num)]

            list(
                male = rbind(fm$male, mf$male),
                female = rbind(fm$female, mf$female)
            )
        },
        sentiment_list_fm,
        sentiment_list_mf
    )

    columns_to_check <- match.arg(columns_to_check)
    t_test_results <- Map(
        \(x, y) do_stat_tests_all(x, y, columns_to_check = columns_to_check, paired = paired),
        sentiment_list,
        names(sentiment_list)
    ) |> rbindlist()


    if (!diagnostics) {
        return(t_test_results)
        # return(rbindlist(t_test_results, idcol = "gender"))
    }

    mf_cooks_d <- lapply(columns_to_check, \(col) {
        Map(
            \(sentiment_pair, metric) get_diagnostics(sentiment_pair, metric, col),
            sentiment_list_mf,
            names(sentiment_list_mf)
        )
    }) |> unlist(recursive = FALSE)

    fm_cooks_d <- lapply(columns_to_check, \(col) {
        Map(
            \(sentiment_pair, metric) get_diagnostics(sentiment_pair, metric, col),
            sentiment_list_fm,
            names(sentiment_list_fm)
        )
    }) |> unlist(recursive = FALSE)

    list(
        fm_cooks_d = fm_cooks_d,
        mf_cooks_d = mf_cooks_d
    )
}


get_diagnostics <- function(sentiment_pair, metric = "regard", col = "negative") {
    differences <- sentiment_pair$female[[col]] - sentiment_pair$male[[col]]
    par(mfrow = c(2, 2))
    mod <- lm(sentiment_pair$female[[col]] ~ sentiment_pair$male[[col]] + 0)

    cooks_d <- cooks.distance(mod)

    cooks_dt <- data.table(
        doc_num = sentiment_pair$female[, doc_num],
        differences = differences,
        cooksd = cooks_d
    )

    # setorder(cooks_dt, -cooksd)

    # Display the sorted data frame
    cooks_dt
}


bind_nested_sentiment_list <- function(l) {
    # Input: nested list of structure
    # list(
    #     regard = c(male = dt, female = dt),
    #     siebert = c(male = dt, female = dt),
    #     distilbert = c(male = dt, female = dt)
    # )

    # Output: data frame with structure e.g.
    #         model gender doc_num   positive  negative
    #         <char> <char>   <int>      <num>     <num>
    #        regard female       0 0.05451034 0.6922291
    #        regard female       1 0.02420696 0.7825058
    # ---
    #    distilbert   male     369 0.32789556 0.4185772
    #    distilbert   male     370 0.42658795 0.3046935


    lapply(l, \(x) rbindlist(x, idcol = "gender")[
        ,
        .(gender, doc_num, positive, negative, pred)
    ]) |>
        rbindlist(idcol = "model")
}

create_sentiment_dt_originals <- function(
    in_dir = "evaluate_sentiment/output_originals", out_dir = "./csv/originals_sentiment/") {
    sentiment_list_fm <- read_sentiment_output("female_to_male", in_dir = in_dir)
    sentiment_list_mf <- read_sentiment_output("male_to_female", in_dir = in_dir)
    sentiment_dt <- rbind(
        bind_nested_sentiment_list(sentiment_list_fm)[, gender_direction := "fm"],
        bind_nested_sentiment_list(sentiment_list_mf)[, gender_direction := "mf"]
    )

    # Remove the ones that shouldn't have made it this far (not properly translated)
    doc_ids_to_remove <- c(paste0("fm", docs_to_remove$female_to_male), c(paste0("mf", docs_to_remove$male_to_female)))
    sentiment_dt[, doc_id := factor(paste0(gender_direction, doc_num))]
    sentiment_dt <- sentiment_dt[!doc_id %in% doc_ids_to_remove]

    sentiment_dt[, sent_num := seq(.N), by = .(model, doc_num, gender_direction, doc_id, gender)]
    sentiment_dt[, sentiment := 1 - negative]

    sentiment_dt[, sent_id := factor(paste0(doc_id, "_", sent_num))]


    sentiment_by_model <- split(sentiment_dt, sentiment_dt$model)

    # Create means by document as well
    sentiment_by_model_means <- lapply(
        sentiment_by_model,
        \(dat) dat[, .(
            positive = mean(positive),
            negative = mean(negative),
            pred = mean(pred)
        ),
        by = .(model, gender, doc_id, gender_direction)
        ][, sentiment := 1 - negative][]
    ) |>
        setNames(paste0(names(sentiment_by_model), "_mean"))

    create_dir(out_dir)

    Map(
        \(dt, nm)  {
            out_file <- sprintf("%s/%s.rds", out_dir, nm)
            message("Saving: ", out_file)
            saveRDS(dt, out_file)
        },
        sentiment_by_model,
        names(sentiment_by_model)
    )

    Map(
        \(dt, nm)  {
            out_file <- sprintf("%s/%s.rds", out_dir, nm)
            message("Saving: ", out_file)
            saveRDS(dt, out_file)
        },
        sentiment_by_model_means,
        names(sentiment_by_model_means)
    )
    invisible(NULL)
}

# Cases bootstrap
run_bootstrap <- function(unique_doc_ids = unique_docs, dt = siebert) {
    boot_docs <- sample(unique_doc_ids, replace = TRUE)

    # Create cases bootstrap (i.e. by doc_id)
    boot_data <- dt[doc_id %in% boot_docs, ]

    final_model <- lmerTest::lmer(
        sentiment ~ model * gender + max_tokens + (model | doc_id),
        data = boot_data, ,
        control = lme4::lmerControl(
            optimizer = "bobyqa" # for convergence
        )
    )

    contrasts_dt <- emmeans::emmeans(final_model, ~ gender | model) |>
        pairs() |>
        as.data.table() |>
        _[, signif := gtools::stars.pval(p.value)][]

    # Clean it up a little
    contrasts_dt_clean <- contrasts_dt[, .(
        model,
        estimate,
        t.ratio,
        p.value,
        signif
    )]
    list(
        boot_data = boot_data,
        final_model = final_model,
        emm = contrasts_dt_clean
    )
}
