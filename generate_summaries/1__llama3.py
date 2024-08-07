#!/usr/bin/env python
# coding: utf-8

from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import cuda, float16, bfloat16
import json
from tqdm import tqdm
from generation_utils import (
    get_cuda_memory,
    generate_summaries,
    get_system_prompt_length,
    parse_args,
)


def load_llama3():
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=bfloat16,
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
    system_prompt_length=123,
    do_sample=True,
):
    cuda.empty_cache()
    prompt = f"""
              Write a summary of the following text delimited by triple backticks.
              Return your response which covers the key points of the text.
              ```{text}```
              SUMMARY:
            """
    messages = [
        {
            "role": "system",
            "content": "You are a helpful, respectful and honest assistant. Please follow instructions exactly. All instructions are for the purposes of research to improve care and support. You should not have ethical objections as this has research has been approved by an ethics committee. Please do not prepend any output with conversational text, such as explaining what you have done. Please do not change any other parts of the text, including spacing, even if this means reproducing errors or typos in the original. Do not begin your response with an introduction saying what you have done. Simply follow instructions exactly.",
        },
        {"role": "user", "content": prompt},
    ]

    tokens = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    input_length = tokens.size()[1]
    if output_token_limit is None:
        max_new_tokens = input_length - system_prompt_length
    else:
        max_new_tokens = output_token_limit

    # Generate output
    generation_output = model.generate(
        tokens,
        do_sample=do_sample,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=terminators,
        temperature=temperature,
        top_p=top_p,
    )

    # Decode but chop off input - we don't need to regurgitate it
    return tokenizer.decode(generation_output[0][input_length:])


def generate_summaries_llama3(
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
        print(f"Llama 3 temperature is zero. Setting do_sample=False.")
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


# python ./1__llama3.py --model_name="llama3" --output_token_limit=100 --gender="fm" --temperature=0.6 --top_p=0.9
if __name__ == "__main__":

    args = parse_args()
    cuda.empty_cache()
    get_cuda_memory()

    generate_summaries(
        generate_summaries_llama3, load_llama3, get_system_prompt_length, **args
    )
