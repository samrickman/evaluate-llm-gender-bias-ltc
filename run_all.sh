#!/bin/bash

if [[ "$VIRTUAL_ENV" == "" ]]; then 
    echo "Activating virtual env"
    source .venv/bin/activate
fi

# * First ensure we can log in to Hugging Face to access the gated models

# First make sure there is a Hugging Face access token
if [[ -z "$HF_TOKEN" ]]; then
    echo "Error: HF_TOKEN must be set. This is required for access to the gated Gemma and Llama 3 models. \
    Please see the README for how to get an access token and pass it to the container."
    exit 1
fi

# Check if already logged in 
huggingface_username=$(huggingface-cli whoami 2>&1)

# Log in to HuggingFace if necessary (should always be necessary in Docker container)
if echo "$huggingface_username" | grep -q "Not logged in"; then
    echo "Logging in to Hugging Face..."
    huggingface-cli login --token $HF_TOKEN
else
    echo "Already logged in to Hugging Face."
fi

#* Check for the --delete-all-output argument
if [[ "$1" == "--delete-all-output" ]]; then
    echo "Deleting existing output..."
    python ./change_gender/0__delete_existing_output.py
fi

# * Create gender-swapped originals
echo "Creating gender-swapped portraits"
in_dir="./raw_data/"
out_dir="./gender_swapped_portraits/"
python ./change_gender/change_gender.py --in_dir=$in_dir --out_dir=$out_dir --original_gender="male"
python ./change_gender/change_gender.py --in_dir=$in_dir --out_dir=$out_dir --original_gender="female"

# * Clean and split into sentences
in_dir="./gender_swapped_portraits/"
out_dir="./gender_swapped_portraits/clean"
python ./clean_generated_text/clean_text.py --in_dir="./gender_swapped_portraits/" --out_dir=$out_dir

# * Count words originals - find docs to exclude
python ./bag_of_words/1__count_words.py --originals
# Exclude docs where m/f summaries have different counts of words
# i.e. not properly translated from male to female
Rscript ./bag_of_words/2__find_docs_to_exclude.R

# * Evaluate sentiment of originals
# Takes about 10 mins to run
echo "Evaluating sentiment"

in_dir=$out_dir
out_dir="evaluate_sentiment/output_originals/"

python ./evaluate_sentiment/1__evaluate_regard.py --in_dir=$in_dir --out_dir=$out_dir
python ./evaluate_sentiment/2__siebert.py --in_dir=$in_dir --out_dir=$out_dir
python ./evaluate_sentiment/3__distilbert.py --in_dir=$in_dir --out_dir=$out_dir
python ./evaluate_sentiment/4__create_mean_sentiment_by_doc.py --in_dir=$in_dir --out_dir=$out_dir

# * Compare sentiment metrics across originals
# * This finds documents to remove as changed gender version has different words to original
# then runs t test on sentiment
Rscript ./model_sentiment_output/1__t_test_originals.R
# Run the lmer models - again regard and siebert are fine, distilbert is not a good metric
# as it finds sentiment-based differences in the originals
Rscript ./model_sentiment_output/2__lmer_originals_create_data.R
Rscript ./model_sentiment_output/2a__lmer_originals_regard.R
Rscript ./model_sentiment_output/2b__lmer_originals_siebert.R
Rscript ./model_sentiment_output/2c__lmer_originals_distilbert.R


# * Remove the ones with unequal words in originals
# (no point wasting time making summaries of documents that aren't comparable)
echo "Removing texts with different words in the male/female versions"
python ./clean_generated_text/remove_docs.py

# * Create the ChatGPT summaries
# Only do this step if the OpenAI key is set
 
if [[ "$OPENAI_API_KEY" != "" ]]; then 
    python ./generate_summaries/chatgpt/1__create_request_jsonl_files.py
    python ./generate_summaries/chatgpt/2__make_batch_requests.py
fi


# * Create summaries
# This can take several days or more to run - depends on length of input
echo "Creating summaries"
declare -a genders=(mf fm)
declare -a output_token_limits=(None 300 150 100 75 50)

in_dir="./gender_swapped_portraits/clean/minimal/"
out_dir="./generate_summaries/output/"
for gender in "${genders[@]}"
do
    for output_token_limit in "${output_token_limits[@]}"
    do
        python ./generate_summaries/1__llama3.py --model_name="llama3" --output_token_limit=$output_token_limit --gender=$gender --temperature=0.6 --top_p=0.9 --portraits_in_dir=$in_dir --out_dir=$out_dir
        python ./generate_summaries/2__gemma.py --model_name="gemma" --output_token_limit=$output_token_limit --gender=$gender --temperature=0.7 --top_p=0.9 --portraits_in_dir=$in_dir  --out_dir=$out_dir
        python ./generate_summaries/3__bart_large.py --model_name="bart" --output_token_limit=$output_token_limit --gender=$gender --gender=$gender --temperature=1.0 --top_p=1.0 --portraits_in_dir=$in_dir --out_dir=$out_dir
        python ./generate_summaries/4__t5.py --model_name="t5" --output_token_limit=$output_token_limit --gender=$gender --gender=$gender --temperature=1.0 --top_p=1.0 --portraits_in_dir=$in_dir --out_dir=$out_dir
    done
done

# * Retrieve the ChatGPT summaries (which will be done much more quickly than ones running locally)
if [[ "$OPENAI_API_KEY" != "" ]]; then 
    python ./generate_summaries/chatgpt/3__check_batches.py
    python ./generate_summaries/chatgpt/4__retrieve_batch_results.py
    python ./generate_summaries/chatgpt/5__extract_responses.py
fi

# * Clean the generated summaries (remove eos tokens, extra spaces etc.)
in_dir=$out_dir 
out_dir="./generate_summaries/clean_output/"
echo "Cleaning output"
python ./clean_generated_text/clean_text.py --in_dir=$in_dir --out_dir=$out_dir

# * Evaluate sentiment of the summaries
echo "Evaluating sentiment (summaries)"

in_dir=$out_dir
out_dir="evaluate_sentiment/output_summaries/"

python ./evaluate_sentiment/1__evaluate_regard.py --in_dir=$in_dir --out_dir=$out_dir
python ./evaluate_sentiment/2__siebert.py --in_dir=$in_dir --out_dir=$out_dir
python ./evaluate_sentiment/3__distilbert.py --in_dir=$in_dir --out_dir=$out_dir
python ./evaluate_sentiment/4__create_mean_sentiment_by_doc.py --in_dir=$in_dir --out_dir=$out_dir

#* Run regression on summaries sentiment
Rscript ./model_sentiment_output/3__create_sentiment_data_summaries.R
Rscript ./model_sentiment_output/4a__t_test_summaries.R
Rscript ./model_sentiment_output/5a__lmer_changed_gender_regard.R
Rscript ./model_sentiment_output/5b__lmer_changed_gender_siebert.R
Rscript ./model_sentiment_output/6a__lm_changed_gender_regard.R
Rscript ./model_sentiment_output/6b__lm_changed_gender_siebert.R
Rscript ./model_sentiment_output/7__regard_siebert_corr.R
# These last two takes a few hours each - they are the only results not included
# in the docker container as they are several GB 
Rscript ./model_sentiment_output/8a__bootstrap_regard.R
Rscript ./model_sentiment_output/8b__bootstrap_siebert.R
# Formats the regression tables nicely
Rscript ./model_sentiment_output/9__create_output_tables.R


# * Count the themes

# So this uses llama3 and gemma to automatically extract themes from docs
input_keys=("original" "result")
topic_types=("health" "physical_appearance" "subjective_language")
scripts=("./evaluate_themes/1__extract_topics_llama3.py" "./evaluate_themes/2__extract_topics_gemma.py")
output_dirs=("./evaluate_themes/output_llama3" "./evaluate_themes/output_gemma")

for i in "${!scripts[@]}"; do
  script="${scripts[$i]}"
  output_dir="${output_dirs[$i]}"

  # Loop over each topic type
  for topic in "${topic_types[@]}"; do

    # Loop over each input key
    for key in "${input_keys[@]}"; do
      python ./"$script" --input_key="$key" --topic_type="$topic" --in_dir="./gender_swapped_portraits/clean/minimal/" --out_dir="$output_dir"
    done
  done
done


# This this produces a list of themes in ./evaluate_themes/themes_output
# However these are not perfect and should be manually edited.
# The terms to keep should be placed in text files in ./evaluate_themes/txt/
python ./evaluate_themes/3__extract_topic_terms.py

# Once the text files in ./evaluate_themes/txt/ are created
# or if using the ones already there
# This raises an error if any themes in originals appear a different number of times
# We already know all words are the same so this shouldn't happen but it's just to confirm
python ./evaluate_themes/4__count_terms_list.py --in_dir="./gender_swapped_portraits/clean/minimal/" --out_dir="./evaluate_themes/csv_originals"

# This counts themes in summaries. Takes about 25 mins with real data.
python ./evaluate_themes/4__count_terms_list.py --in_dir=$in_dir --out_dir="./evaluate_themes/csv_summaries"
Rscript ./evaluate_themes/5__compare_themes_originals.R
Rscript ./evaluate_themes/6_compare_themes_summaries.R


# * Count the words
python ./bag_of_words/1__count_words.py --summaries

# Confirm again words are all the same in male/female originals
Rscript ./bag_of_words/3__compare_words_originals.R
# Run Chi-Sq test on words in summaries
Rscript ./bag_of_words/4__compare_words_summaries.R
# Run regression on words in summaries
Rscript ./bag_of_words/5__word_level_regression.R


# * Check hallucination
# i.e. Are differences hallucination or omission?
# Answer: omission. There's very little hallucination. 

# Takes about 2 mins for the originals
python ./check_hallucination/1__count_terms_list.py --in_dir="./gender_swapped_portraits/clean/minimal/" --out_dir="./check_hallucination/csv_originals"

# Summaries takes about 15 mins
python ./check_hallucination/1__count_terms_list.py --in_dir="./generate_summaries/clean_output/" --out_dir="./check_hallucination/csv_summaries"
Rscript ./check_hallucination/2__compare_terms_originals_summaries.R