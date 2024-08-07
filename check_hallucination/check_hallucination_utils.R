library(data.table)
setwd("./check_hallucination/")

load_data <- function(in_dir, doc_type = c("originals", "summaries")) {
    in_files <- dir(
        in_dir,
        pattern = "\\.csv$",
        full.names = TRUE
    )

    file_list <- lapply(
        setNames(in_files, basename(in_files)),
        fread
    )

    dat <- rbindlist(file_list, use.names = TRUE, id = "filename")


    dat[, doc_id := paste0(fifelse(original_gender == "female", "fm", "mf"), doc_num)]
    if (doc_type == "summaries") {
        set_model_params(dat)
    }

    dat[]
}

set_model_params <- function(dat) {
    # No need to return anything as modifies in place
    dat[, c("gender_direction", "model", "max_tokens") := strsplit(filename, "_") |>
        lapply(\(x) x[1:3]) |>
        purrr::transpose() |>
        lapply(unlist)]
    dat[, filename := NULL]
}
