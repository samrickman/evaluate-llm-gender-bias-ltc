library(data.table)
setwd("./evaluate_themes/")

load_data <- function(in_dir, type = c("originals", "summaries")) {
    type <- match.arg(type)



    in_files <- dir(
        in_dir,
        pattern = "\\.csv$",
        full.names = TRUE
    )
    file_list <- lapply(
        setNames(in_files, basename(in_files)),
        fread
    )

    if (type == "originals") dat <- rbindlist(file_list, use.names = TRUE)
    if (type == "summaries") {
        dat <- rbindlist(file_list, use.names = TRUE, id = "filename")
        set_model_params(dat)
    }
    dat[, doc_id := paste0(fifelse(original_gender == "female", "fm", "mf"), doc_num)]
}


get_model_params_list_for_dt <- function(n, in_file) {
    get_model_params <- function(in_file) {
        desired_params <- c("gender_direction", "model", "max_tokens")
        setNames(
            unlist(strsplit(basename(in_file), "_")),
            desired_params
        )[desired_params]
    }
    params_full <- rep(get_model_params(in_file), n)
    split(
        params_full,
        names(params_full)
    )
}

set_model_params <- function(dat) {
    # No need to return anything as modifies in place
    dat[, c("gender_direction", "model", "max_tokens") := strsplit(filename, "_") |>
        lapply(\(x) x[1:3]) |>
        purrr::transpose() |>
        lapply(unlist)]
    dat[, filename := NULL]
}

do_chi_squared_test <- function(term_type_totals) {
    # Catch warnings as in https://adv-r.hadley.nz/conditions.html
    # we need this to because often they have very low counts
    # and chi.sq is not appropriate - so we should return NA and use Fisher test
    conds <- list()
    add_cond <- function(cnd) {
        conds <<- append(conds, list(cnd))
        invokeRestart("muffleWarning")
    }
    if (term_type_totals$female == term_type_totals$male) {
        return(1)
    }
    contingency_table <- matrix(c(term_type_totals$female, term_type_totals$male),
        nrow = 2,
        byrow = TRUE,
        dimnames = list(c("Female", "Male"), c("Count"))
    )

    result <- withCallingHandlers(
        {
            chisq.test(contingency_table)$p.value
        },
        warning = add_cond
    )
    if (identical(conds, list())) {
        return(result)
    } else {
        return(NA_real_)
    }
}

do_fishers_test <- function(term_data, total_female, total_male) {
    # Counts of other words
    other_female <- total_female - term_data$female
    other_male <- total_male - term_data$male
    # Total counts for all words

    # Create the contingency table
    contingency_table <- matrix(
        c(
            term_data$female, other_female,
            term_data$male, other_male
        ),
        nrow = 2,
        dimnames = list(c("Word_Count", "Other_Words"), c("Female", "Male"))
    )

    # Perform Fisher's Exact Test
    test_result <- fisher.test(contingency_table)

    return(test_result$p.value)
}
