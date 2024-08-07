#!/usr/bin/env python
# coding: utf-8

# full gemma instruct https://huggingface.co/google/gemma-7b-it
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import cuda, float16, bfloat16
import json
from tqdm import tqdm
from generation_utils import (
    get_cuda_memory,
    generate_summaries,
    get_dummy_system_prompt_length,
    parse_args,
)


def load_gemma():
    model_str = "google/gemma-7b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_str)
    model = AutoModelForCausalLM.from_pretrained(
        model_str,
        torch_dtype=float16,
        revision="float16",
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    return tokenizer, model


def summarise_text(
    text,
    tokenizer,
    model,
    output_token_limit=None,
    temperature=0.6,
    top_p=0.9,
    system_prompt_length=None,  # this argument only exists to keep a consistent API with Llama3 - it's not used
    do_sample=True,
):
    cuda.empty_cache()
    prompt = f"""
              Write a summary of the following text delimited by triple backticks.
              Return your response which covers the key points of the text.
              ```{text}```
              SUMMARY:
            """

    tokens = tokenizer(prompt, return_tensors="pt").to("cuda")

    input_length = tokens["input_ids"].size()[1]

    if output_token_limit is None:
        max_new_tokens = input_length
    else:
        max_new_tokens = output_token_limit

    # Generate output
    generation_output = model.generate(
        **tokens,
        do_sample=do_sample,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    # Decode but chop off input - we don't need to regurgitate it
    return tokenizer.decode(generation_output[0][input_length:])


def generate_summaries_gemma(
    portraits,
    output_token_limit,
    out_file,
    tokenizer,
    model,
    system_prompt_length,
    temperature,
    top_p,
    do_sample=True,
):

    # This allows us to set temperature zero instead of passing do_sample every time
    if temperature == 0:
        print(f"Gemma temperature is zero. Setting do_sample=False.")
        do_sample = False
        top_p = None

    summaries = []
    for portrait in tqdm(portraits):
        original_summary = summarise_text(
            portrait["original"],
            tokenizer,
            model,
            output_token_limit,
            temperature,
            top_p,
            system_prompt_length,
            do_sample,
        )
        result_summary = summarise_text(
            portrait["result"],
            tokenizer,
            model,
            output_token_limit,
            temperature,
            top_p,
            system_prompt_length,
            do_sample,
        )
        summaries.append({"original": original_summary, "result": result_summary})
        with open(out_file, "w") as f:
            json.dump(summaries, f)


# python ./2__gemma.py --model_name="gemma" --output_token_limit=100 --gender="fm" --temperature=0.7 --top_p=0.9
if __name__ == "__main__":

    args = parse_args()
    cuda.empty_cache()
    get_cuda_memory()

    generate_summaries(
        generate_summaries_gemma, load_gemma, get_dummy_system_prompt_length, **args
    )
