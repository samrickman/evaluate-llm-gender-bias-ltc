# Evaluating gender bias in LLMs in long-term care

This is the repo for the 2024 paper [Evaluating gender bias in LLMs in long-term care](). The paper uses two state-of-the-art, open-source LLMs released in 2024: Meta's [Llama 3](https://ai.meta.com/blog/meta-llama-3/) and Google's [Gemma](https://ai.google.dev/gemma). The paper found:

1. Llama 3 showed no gender-based differences across any metrics.
2. Gemma showed significant gender-based differences:
    - Male summaries were more negative, with greater focus on physical and mental health needs.
    - Language used for for men was more direct, compared with more euphemistic language for women. 
    
Care services are allocated on the basis of need. If women's needs are downplayed, it may impact worker's views of the case's priority or their eligibility for care.

The paper was based on real data from long-term care records. In this repo we use synthetic data, generated using LLMs. This means all the code contained here can be run. The [results](./results_tables/1__results_tables.html) from the synthetic data are consistent with those on the real data, showing significant gender-based differences in summaries create by the Google Gemma model. We continue to find no gender-based differences with Llama 3. The code in this repo also extends the analysis to OpenAI's [ChatGPT](https://openai.com/chatgpt/), which does not appear to show significant gender-based differences with the synthetic data.

This repo contains the code and also the output of the analysis using synthetic data[^1], as it takes several days to run. It is possible to replicate the entire analysis from scratch by following the instructions below.

# Method and findings

We generated several hundred identical pairs of long-term care case note summaries. The original texts are identical except for gender. For example:

| Male version                                                                                                                                     | Female version                                                                                                                       |
|----------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| Mrs   Smith is a 87 year old, white British woman with reduced mobility. She cannot   mobilize independently at home in her one-bedroom flat | Mr Smith is a 87 year old, white   British man with reduced mobility. He cannot mobilize independently at home   in his one-bedroom flat | 

The notes used are around 500 words. We then generate summaries using several LLMs. Despite identical original texts, the Gemma model:

1. Consistently produces summaries for women with less negative sentiment ($p <0.01$).
2. Mentions men's physical health and mental health issues more frequently than women's ($p <0.01$).
3. Shows linguistic differences in how it describes women's health issues, e.g. "he is unable to meet his needs" vs "despite her care needs, she is managing well".

We also found some differences in the benchmark models, BART and T5, though these were not as pronounced as the Gemma differences.

# How to replicate the analysis

## Requirements: 

1.  **Docker**: To install Docker, follow the instructions at [Docker's official site](https://docs.docker.com/get-docker/).
2. **GPU**: A machine with a [CUDA-compatible](https://developer.nvidia.com/cuda-gpus) Graphics Processing Unit (GPU) with at least 20GB VRAM.
3. **NVIDIA Container Toolkit**: The code in this repo requires CUDA 12.1, which may differ from your global version. You can run this version in Docker on Linux with [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).On Windows with WSL2 you can use [Docker Desktop](https://www.docker.com/products/docker-desktop/) (version `>=4.3.1`).
4. **Access to gated models**. This uses the open-source but gated models [Meta Llama 3 8b instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) and [Google Gemma 7b-it](https://huggingface.co/google/gemma-7b-it). You will need to register for an account and [generate an access token](https://huggingface.co/settings/tokens). This should then be saved in your shell environment as `$HF_TOKEN`.  
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
   docker run --rm --gpus all --entrypoint bash evaluate_gender_bias_image -c "python -c 'import torch; print(torch.cuda.is_available())'"
   ```
   
   This will build the Docker container and enter it in a shell. It will check that it has been successfully created and that PyTorch can access the GPU. If this is is the case, it will print `True`, and exit. This will not run the entire analysis, which takes several days. 

5. To replicate the full analysis, you can use the synthetic data or replace it with your own data in the [`./raw_data`](./raw_data/) directory. Then:

    ```sh
    docker run -d --gpus all --name evaluate_gender_bias \
    -e HF_TOKEN=$HF_TOKEN \
    evaluate_gender_bias_image ./run_all.sh --delete-all-output
    ```
    
    Ensure that `$HF_TOKEN` exists in your environment. This should be the free [access token](https://huggingface.co/settings/tokens) from HuggingFace that allows you to access the gated models, [Meta Llama 3 8b instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) and [Google Gemma 7b-it](https://huggingface.co/google/gemma-7b-it).

    Note that this command will retain the [`./raw_data/`](raw_data) folder but otherwise delete all output of the analysis (and then regenerate it). This command will run in the foreground and display the output in the terminal. If you want to run in the background, use the `-d` (detached) option.

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
   - Gender-swapped versions of the original data are created for both male and female case notes.

2. **Clean and split data into sentences**  
   - The gender-swapped data is cleaned and split into sentences. Sentiment analysis is done at the sentence level.

3. **Count words in original data and remove inconsistent texts**  
   - Words are counted in the male and female versions. This is an important step to ensure we are generating summaries of identical texts. Counts of gendered words such as "woman" are excluded. Any notes which do not contain the same counts of non-gendered words in the male and female versions are excluded from the analysis.

4. **Evaluate sentiment of original data**  
   - The sentiment of the original gender-swapped data is evaluated using multiple models:
     - [Regard](https://huggingface.co/spaces/evaluate-measurement/regard)
     - [SiEBERT](https://github.com/j-hartmann/siebert)
     - [DistilBERT](https://huggingface.co/lxyuan/distilbert-base-multilingual-cased-sentiments-student)
   - Sentiment metrics are computed and statistical tests are conducted to compare sentiment across different gender-swapped versions to ensure any bias in sentiment does not arise from the metrics. We find that that Regard and SiEBERT models do not show gender-based differences. However, the DistilBERT-based model does so we do not use it.

5. **Generate summaries**  
   - Summaries are generated using various models:
     - LLaMA3
     - Gemma
     - BART
     - T5
     - ChatGPT (if API key is provided)

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

The findings from the synthetic data are similar to the real data. Specifically:

1. There were some observable gender-based differences with the benchmark models, BART and T5, though these were on the boundary of statistical significance.
2. The Gemma model consistently showed gender-based differences in sentiment, words used and themes used.
3. There were no differences in Llama 3.

There were some differences compared to the real data:

1. The Regard results are more significant for BART and T5, while the SiEBERT results are less significant.
2. The correlation between sentiment metrics Regard and SiEBERT was 0.09 in our original data, and 0.26 in the synthetic data.
3. There are some differences in which words are used significantly more by each model. The output could not be the same as the input is different. For example, the word "unwise" appears frequently in the real data but is not included in the synthetic data.

The ChatGPT findings are least reliable as we did not conduct the original analysis using ChatGPT, for information governance reasons.

[^1]: The only files excluded from this repo are on the basis of size. Specifically, we do not include the full `robustlmm` model objects (though we include the output in csv form), or the 1000 bootstrapped datasets (though again we include the output as csv).