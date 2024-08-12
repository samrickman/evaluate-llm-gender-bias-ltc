# Evaluating gender bias in LLMs in long-term care

This is the repo for the 2024 paper [Evaluating gender bias in LLMs in long-term care](). The paper evaluates gender bias in LLMs used to summarise long-term care case notes. We measure bias two state-of-the-art, open-source LLMs released in 2024: Meta's [Llama 3](https://ai.meta.com/blog/meta-llama-3/) and Google's [Gemma](https://ai.google.dev/gemma), benchmarked against 2019 models from Meta and Google, [BART](https://huggingface.co/facebook/bart-large) and [T5](https://huggingface.co/docs/transformers/en/model_doc/t5). The paper found:

1. Llama 3 showed no gender-based differences across any metrics.
2. Gemma showed significant gender-based differences:
    - Male summaries were more negative, with greater focus on physical and mental health needs.
    - Language used for for men was more direct, compared with more euphemistic language for women. 
    
Care services are allocated on the basis of need. If men's needs are explicit while women's are underemphasised, it may impact practitioner perception of the case's priority, or the person's eligibility for care.

The paper was based on real data from long-term care records. All the code contained here can be run as we include in this repo synthetic data, generated using LLMs[^1]. The [results from the synthetic data ](./results_tables/1__results_tables.html) are consistent with those from the real data, showing significant gender-based differences in summaries create by the Google Gemma model. We continue to find no gender-based differences with Llama 3.

This repo contains the code and also, as it takes several days to run, the output of the analysis[^2] based on the [synthetic data](./raw_data/). It is possible to replicate the entire analysis from scratch by following the instructions below. The code in this repo also extends the analysis to OpenAI's [ChatGPT](https://openai.com/chatgpt/).

# Method and findings

We generated several hundred identical pairs of long-term care case note summaries. The original texts are identical except for gender. For example:

| Female version                                                                                                                                     | Male version                                                                                                                       |
|----------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| Mrs   Smith is an 87 year old, white British woman with reduced mobility. She cannot   mobilise independently at home in her one-bedroom flat | Mr Smith is an 87 year old, white   British man with reduced mobility. He cannot mobilise independently at home   in his one-bedroom flat | 

The notes used are around 500 words. We then generate summaries using several LLMs. Despite identical original texts, the Gemma model:

1. Consistently produces summaries for men with more negative sentiment ($p <0.01$).
2. Mentions men's physical health and mental health issues more frequently than women's ($p <0.01$).
3. Shows gender-based, linguistic differences in how it describes health issues, e.g. "he is unable to meet his needs" vs "despite her care needs, she is managing well".

We also found some differences in the benchmark models, BART and T5, though these were not as pronounced as the Gemma differences.

# How to replicate the analysis

## Requirements: 

1.  **Docker**: To install Docker, follow the instructions at [Docker's official site](https://docs.docker.com/get-docker/).
2. **GPU**: A machine with a [CUDA-compatible](https://developer.nvidia.com/cuda-gpus) Graphics Processing Unit (GPU) with at least 20GB VRAM.
3. **NVIDIA Container Toolkit**: The code in this repo requires CUDA 12.1, which may differ from your global version. You can run this version in Docker on Linux with [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). On Windows with WSL2 you can use [Docker Desktop](https://www.docker.com/products/docker-desktop/) (version `>=4.3.1`).
4. **Access to gated models**. This uses the open-source but gated models [Meta Llama 3 8b-instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) and [Google Gemma 7b-it](https://huggingface.co/google/gemma-7b-it). You will need to register for an account and [generate an access token](https://huggingface.co/settings/tokens). This should then be saved in your shell environment as `$HF_TOKEN`.  
5. **OpenAI API Key**: To replicate the ChatGPT element of the analysis, an OpenAI API key. This needs to be saved in your shell environment as `$OPENAI_API_KEY`. If this environment variable is not set the analysis will still run but this step will be omitted. The total cost of the analysis with the synthetic data was $0.34.

## How to run the model

1. Clone the repository:
   ```sh
   git clone https://github.com/samrickman/evaluate-llm-gender-bias-ltc.git
   ```
2. Navigate into the directory:
   ```sh
   cd evaluate-llm-gender-bias-ltc
   ```
3. Build the Docker image:
   ```sh
   docker build . -t evaluate_gender_bias_image
   ```
4. Test the Docker container:
   ```sh
   docker run --rm --gpus all --entrypoint bash evaluate_gender_bias_image \
   -c "python -c 'import torch; print(torch.cuda.is_available())'"
   ```

   This will check the container has been successfully created and that PyTorch can access the GPU. If this is is the case, it will print `True`, and exit. This will not run the entire analysis, which takes several days. 

5. To replicate the full analysis, you can use the synthetic data or replace it with your own data in the [`./raw_data`](./raw_data/) directory. Then:

    ```sh
    docker run --gpus all --name evaluate_gender_bias \
    -e HF_TOKEN=$HF_TOKEN \
    evaluate_gender_bias_image ./run_all.sh --delete-all-output
    ```
    
    Ensure that `$HF_TOKEN` exists in your environment. This should be the free [access token](https://huggingface.co/settings/tokens) from HuggingFace that allows you to access the gated models, [Meta Llama 3 8b instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) and [Google Gemma 7b-it](https://huggingface.co/google/gemma-7b-it). It is necessary to apply for permission from Google and Meta to use these models.

    Note that this command will retain the [`./raw_data/`](raw_data) folder but otherwise delete all output of the analysis (and then recreate it). This command will run in the foreground and display the output in the terminal. If you want to run in the background, use the `-d` (detached) [option](https://docs.docker.com/reference/cli/docker/container/run/).

    If you want to replicate the ChatGPT element you will need to pass your OpenAI API Key. The container expects it to be in the environment variable, `$OPENAI_API_KEY`. If it exists with the same name on the host machine:

    ```sh
    docker run -d --gpus all --name evaluate_gender_bias \
    -e OPENAI_API_KEY=$OPENAI_API_KEY \
    -e HF_TOKEN=$HF_TOKEN \
    evaluate_gender_bias_image ./run_all.sh --delete-all-output
    ```

## Steps of Analysis

The analysis involves the following steps:

1. **Generate gender-swapped versions of notes**  
   - The original case notes are taken from the `raw_data/` directory. Male versions of the female case notes care created, and female versions of the male case notes.

2. **Clean and split data into sentences**  
   - The gender-swapped data is cleaned and split into sentences.

3. **Count words in original data and remove inconsistent texts**  
   - Words are counted in the male and female versions of each pair of texts, to ensure step 1 created identical texts. Any notes which do not contain the same counts of non-gendered words in the male and female versions are excluded from the analysis. Counts of gendered words such as "woman" are excluded.

4. **Evaluate sentiment of original data**  
   - The sentiment of the original gender-swapped data is evaluated using multiple models:
     - [Regard](https://huggingface.co/spaces/evaluate-measurement/regard)
     - [SiEBERT](https://github.com/j-hartmann/siebert)
     - [DistilBERT](https://huggingface.co/lxyuan/distilbert-base-multilingual-cased-sentiments-student)
   - Sentiment metrics are computed and statistical tests are conducted to compare sentiment across different gender-swapped versions to ensure any bias in sentiment does not arise from the metrics. We find that that Regard and SiEBERT models do not show gender-based differences. However, the DistilBERT-based model does so we do not use it to evaluate the summaries.

5. **Generate summaries**  
   - Summaries are generated using various models:
     - LLaMA3
     - Gemma
     - BART
     - T5
     - ChatGPT (if API key is provided)
      
      This step takes around 3-4 days on a 20GB [GeForce RTX Ada](https://www.nvidia.com/en-gb/design-visualization/rtx-4000/) on WSL2.

6. **Clean summaries**  
   - The generated summaries are cleaned by removing end-of-text tokens and spaces.

7. **Evaluate sentiment of summaries**  
    - The sentiment of the generated summaries is evaluated using Regard and SiEBERT.

11. **Run regression analysis on summaries sentiment**  
    - Regression analyses and additional statistical tests on the sentiment of the summaries to see whether there are differences.

12. **Count and compare themes**  
    - Extraction of words related to the following themes from the original texts:
      - Physical health
      - Physical appearance
      - Mental health
      - Subjective language
    - Comparison of frequency of words associated with each theme in the different versions of the summaries..

13. **Count words in summaries**  
    - Words are counted in the summaries to see whether any words are used significantly more often. The Benjamini-Hochberg adjustment of $p$-values is used as we conduct tests of thousands of words. Nevertheless, there are still significant differences in words used in some (but not all) models.

14. **Check for hallucination and omission**  
    - The script checks whether differences in the summaries are due to hallucinations (incorrect additions) or omissions (missing content). We check for hallucination of medical diagnoses, seeing whether they appear in summaries without appearing in the originals. We find almost no hallucination, and conclude that inclusion bias differences are due to omission. 

# Comparison between synthetic findings and real data

The [results from the synthetic data](./results_tables/1__results_tables.html) are similar to the real data. Specifically:

1. There were some observable gender-based differences with the benchmark models, BART and T5, though these were on the boundary of statistical significance.
2. The Gemma model consistently showed gender-based differences in sentiment, words used and themes used.
3. There were no differences in Llama 3.

There were some differences compared to the real data:

1. The sentiment Regard results are more significant for BART and T5, while the SiEBERT results are less significant.
2. The correlation between sentiment metrics Regard and SiEBERT was 0.09 in our original data, and 0.26 in the synthetic data.
3. There are some differences in which words are used significantly more by each model. The output could not have been the same as the input is different. For example, the word "unwise" appears frequently in the real data but is not included in the synthetic data. In the real data, Gemma is the model with the the most words used differently. However, with the synthetic data, while the words used follow the same patterns as with real data, BART and T5 show more word-level differences than Gemma. The Llama 3 model also used one word significantly more for men in the synthetic output ("old"), versus no words in the real data.

Our primary model did not find any sentiment differences in summaries about men and women in the ChatGPT output. However, some of the [robustness checks](./results_tables/1__results_tables.html#variance-structured-mixed-effects-model) using the SiEBERT metric found the slightly less positive sentiment in male summaries was on the boundary of significance. ChatGPT summaries appear to use words related to subjective language more about women, and in particular the word challenge (491 times for women and 365 for men). It also uses the word Smith significantly more for men (3701 vs 3261 times), which was the name given to all individuals in the summaries. However, the ChatGPT findings are least reliable as, for information governance reasons, we did not conduct the analysis with real data using ChatGPT.

# Troubleshooting

If you are trying to replicate the analysis from scratch and face issues, it may be easier to build the docker container and enter it, then set the required environment variables and then run the code. So after step three (building the container), run:

```sh
# On the host
docker run --it --gpus all -it --entrypoint bash evaluate_gender_bias_image
# Then within the container
HF_TOKEN=<your-token-here>
source ./run_all.sh --delete-all-output
```

As the analysis can take several days, it may be prudent to let the image persist, and not use the [`--rm`](https://docs.docker.com/reference/cli/docker/container/run/) flag. If the process is interrupted, you can do:

```sh
source ./run_all.sh
```

Without the `--delete-all-output` flag, the script will see which files have already been created and pick up from where you left off.

[^1]: The synthetic data was generated with ChatGPT 4o and and Llama 3.
[^2]: The only files excluded from this repo are on the basis of size. Specifically, we do not include the full `robustlmm` model objects (though we include the output in csv form), or the 1000 bootstrapped datasets (though again we include the output as csv).
