from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch import cuda, float16
import json
from tqdm import tqdm
from generation_utils import (
    get_cuda_memory,
    get_system_prompt_length,
    load_portraits,
    parse_args,
)
from typing import Callable
import pathlib
import argparse


def load_t5(model_str="t5-base"):
    tokenizer = T5Tokenizer.from_pretrained(model_str)
    model = T5ForConditionalGeneration.from_pretrained(
        model_str, torch_dtype=float16
    ).to("cuda")
    return tokenizer, model


def summarise_text(tokenizer, model, text_list, output_token_limit=None):
    cuda.empty_cache()

    if output_token_limit is None:
        max_new_tokens = 1_000  # nothing is longer than this
    else:
        max_new_tokens = output_token_limit

    text_to_summarise = ["summarize: " + note for note in text_list]
    inputs = tokenizer.batch_encode_plus(
        text_to_summarise,
        truncation=True,
        max_length=max_new_tokens,
        return_tensors="pt",
        pad_to_max_length=True,
    )
    outputs = model.generate(
        inputs["input_ids"].to("cuda"), max_length=max_new_tokens, early_stopping=True
    )

    return [
        tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for g in outputs
    ]


def generate_summaries_t5(
    gender: str,
    model_name: str,
    output_token_limit: int,
    temperature: float,
    top_p: float,
    out_dir: str = "./generate_summaries/output",
    chunk_size: int = 5,
    portraits_in_dir: str = "./raw_data/",
):
    """
    Temperature and top_p are only for the file name as they do not seem to affect output.
    Above are the defaults.
    """

    out_file = pathlib.Path(
        f"{out_dir}/{gender}_{model_name}_{output_token_limit}_temp_{temperature}_top-p_{top_p}.json"
    )
    pathlib.Path(out_file.parent).mkdir(exist_ok=True)

    # Don't overwrite if already exists
    if out_file.is_file():
        print(f"File already exists: {out_file}. Skipping.")
        return

    portraits = load_portraits(gender, portraits_in_dir)
    tokenizer, model = load_t5()
    original_txt = [portrait["original"] for portrait in portraits]
    result_txt = [portrait["result"] for portrait in portraits]

    # split into chunks so we don't get OOM error
    # also seems to make it faster
    def divide_chunks(l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n]

    original_txt_chunked = divide_chunks(original_txt, chunk_size)
    result_txt_chunked = divide_chunks(result_txt, chunk_size)

    original_summaries = []
    for chunk in original_txt_chunked:
        original_summaries.append(
            summarise_text(tokenizer, model, chunk, output_token_limit)
        )
    original_summaries = [item for sublist in original_summaries for item in sublist]

    result_summaries = []
    for chunk in result_txt_chunked:
        result_summaries.append(
            summarise_text(tokenizer, model, chunk, output_token_limit)
        )
    result_summaries = [item for sublist in result_summaries for item in sublist]

    summaries = []
    for original_summary, result_summary in zip(original_summaries, result_summaries):
        summaries.append({"original": original_summary, "result": result_summary})
        with open(out_file, "w") as f:
            json.dump(summaries, f)


def main():
    generate_summaries_t5(**parse_args())


# python ./4__t5.py --model_name="t5" --output_token_limit=50 --gender="fm" --temperature=1.0 --top_p=1.0
if __name__ == "__main__":
    main()
