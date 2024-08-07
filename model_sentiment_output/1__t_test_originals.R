source("./model_sentiment_output/model_sentiment_utils.R")

docs_to_remove <- jsonlite::fromJSON("./csv/docs_to_remove.json") # 61mf, 83fm

# * t-test sentence level
metrics_split_mf_fm_sentences <- generate_metrics(in_dir = "evaluate_sentiment/output_originals")
#    gender      t_test_p    mcnemar_p      corr corr_p          diff      effsize  magnitude      col t_signif mcnemar_signif corr_signif     metric
#    <char>         <num>        <num>     <num>  <num>         <num>        <num>      <ord>   <char>   <char>         <char>      <char>     <char>
# 1:     fm  5.390000e-01           NA 0.9784750      0 -1.082395e-04 -0.011257697 negligible negative                                 ***     regard
# 2:     fm  5.420000e-01 1.000000e+00 0.9983480      0  2.294793e-04  0.011165058 negligible negative                                 ***    siebert
# 3:     fm 1.799881e-320 5.412161e-11 0.9974774      0 -1.098019e-02 -0.797031121   moderate negative      ***            ***         *** distilbert
# 4:     mf  5.800000e-01           NA 0.9775395      0 -8.205864e-05 -0.008416775 negligible negative                                 ***     regard
# 5:     mf  8.140000e-01 1.000000e+00 0.9952662      0  1.242655e-04  0.003572245 negligible negative                                 ***    siebert
# 6:     mf  0.000000e+00 4.519809e-13 0.9974061      0 -1.119084e-02 -0.817174527      large negative      ***            ***         *** distilbert

# * t-test document level
metrics_split_mf_fm_docs <- generate_metrics(in_dir = "evaluate_sentiment/output_originals", subdir = "mean")
#    gender  t_test_p mcnemar_p      corr        corr_p          diff      effsize  magnitude      col t_signif mcnemar_signif corr_signif     metric
#    <char>     <num>     <num>     <num>         <num>         <num>        <num>      <ord>   <char>   <char>         <char>      <char>     <char>
# 1:     fm  3.56e-01        NA 0.9951759 5.339184e-137 -7.821826e-05 -0.079477604 negligible negative                                 ***     regard
# 2:     fm  4.58e-01        NA 0.9977324 6.266570e-159  2.663303e-04  0.063872071 negligible negative                                 ***    siebert
# 3:     fm  3.83e-83        NA 0.9967715 1.159177e-148 -1.097619e-02 -3.857940300      large negative      ***                        *** distilbert
# 4:     mf  4.40e-01        NA 0.9914259 1.659505e-180 -6.361950e-05 -0.054177999 negligible negative                                 ***     regard
# 5:     mf  9.69e-01        NA 0.9947911 2.703551e-202 -1.723501e-05 -0.002690154 negligible negative                                 ***    siebert
# 6:     mf 3.98e-136        NA 0.9976908 6.483312e-238 -1.122602e-02 -4.458658140      large negative      ***                        *** distilbert

# * sentences
metrics_combined_sentences <- generate_metrics_combined(in_dir = "evaluate_sentiment/output_originals")

#    t_test_p    mcnemar_p      corr corr_p          diff      effsize  magnitude      col t_signif mcnemar_signif corr_signif     metric
#       <num>        <num>     <num>  <num>         <num>        <num>      <ord>   <char>   <char>         <char>      <char>     <char>
# 1:    0.414           NA 0.9779231      0  9.273967e-05  0.009566673 negligible negative                                 ***     regard
# 2:    0.632 1.000000e+00 0.9965228      0 -1.671896e-04 -0.005608449 negligible negative                                 ***    siebert
# 3:    0.000 6.161082e-23 0.9974306      0  1.110490e-02  0.808956667      large negative      ***            ***         *** distilbert

# * t-test sentence level combine gender direction
metrics_combined_docs <- generate_metrics_combined(in_dir = "evaluate_sentiment/output_originals", subdir = "mean")
#     t_test_p mcnemar_p      corr        corr_p          diff     effsize  magnitude      col t_signif mcnemar_signif corr_signif     metric
#        <num>     <num>     <num>         <num>         <num>       <num>      <ord>   <char>   <char>         <char>      <char>     <char>
# 1:  2.45e-01        NA 0.9931285 5.294182e-317  0.0000694590  0.06310603 negligible negative                                 ***     regard
# 2:  7.52e-01        NA 0.9959167  0.000000e+00 -0.0000961911 -0.01713270 negligible negative                                 ***    siebert
# 3: 1.87e-217        NA 0.9973328  0.000000e+00  0.0111260888  4.19490221      large negative      ***                        *** distilbert

fwrite(metrics_split_mf_fm_sentences, "./csv/original_models/t_test_mf_sentences.csv")
fwrite(metrics_split_mf_fm_docs, "./csv/original_models/t_test_mf_docs.csv")
fwrite(metrics_combined_sentences, "./csv/original_models/t_test_combined_sentences.csv")
fwrite(metrics_combined_docs, "./csv/original_models/t_test_combined_docs.csv")
