source("./evaluate_themes/evaluate_themes_utils.R")


dat <- load_data("./csv_summaries/", type = "summaries")

term_type_totals_by_model <- dat[
    ,
    .(female = sum(female_count), male = sum(male_count)),
    .(term_type, model)
]

term_totals_by_model <- dat[
    ,
    .(female = sum(female_count), male = sum(male_count)),
    .(term, term_type, model)
]


term_type_totals_by_model[, chisq_p_value := do_chi_squared_test(.SD), .(term_type, model)][
    ,
    chisq_p_stars := gtools::stars.pval(chisq_p_value)
][
    ,
    chisq_p_value_adj := p.adjust(chisq_p_value, method = "BH")
][
    ,
    chisq_p_value_adj := round(chisq_p_value_adj, 3)
][, adj_p_stars := gtools::stars.pval(chisq_p_value_adj)]


# * So we get physical health with gemma again
term_type_totals_by_model[chisq_p_value_adj < 0.05]

# * still gemma physical health -
#              term_type  model female  male chisq_p_value chisq_p_stars chisq_p_value_adj adj_p_stars
#                 <char> <char>  <int> <int>         <num>        <char>             <num>      <char>
# 1:     physical_health  gemma  13269 14180  3.827364e-08           ***             0.000         ***
# 2:       mental_health  gemma   9160  9667  2.198650e-04           ***             0.002          **
# 3: physical_appearance     t5    407   320  1.252480e-03            **             0.008          **

term_type_totals_by_model[chisq_p_value < 0.05]
#              term_type   model female  male chisq_p_value chisq_p_stars chisq_p_value_adj adj_p_stars
#                 <char>  <char>  <int> <int>         <num>        <char>             <num>      <char>
# 1: subjective_language    bart   2414  2592  1.187655e-02             *             0.059           .
# 2: subjective_language chatgpt   5589  5375  4.097756e-02             *             0.164
# 3:     physical_health   gemma  13269 14180  3.827364e-08           ***             0.000         ***
# 4:       mental_health   gemma   9160  9667  2.198650e-04           ***             0.002          **
# 5: physical_appearance      t5    407   320  1.252480e-03            **             0.008          **

totals_for_paper <- copy(term_type_totals_by_model)

totals_for_paper[, term_type := tools::toTitleCase(term_type) |>
    sub("_", " ", x = _)]


setnames(
    totals_for_paper,
    c(
        "term_type", "female", "male", "chisq_p_value", "chisq_p_stars", "chisq_p_value_adj", "adj_p_stars"
    ),
    c(
        "Term type", "Count (female)", "Count (male)", "Chi-sq p-value", " ", "Adj. p-value (BH)", "  "
    )
)

if (!dir.exists("./csv_output/")) dir.create("./csv_output/")
fwrite(totals_for_paper, "./csv_output/chisq_themes.csv")
