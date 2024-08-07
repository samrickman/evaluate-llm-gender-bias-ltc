from transformers import pipeline
from torch import cuda
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


def load_bart(model_id="facebook/bart-large-cnn"):
    model = pipeline("summarization", model=model_id, device=0)
    return model


def summarise_text(summarizer, text_list, output_token_limit=None, min_length=21):
    cuda.empty_cache()

    """
    We can pass this a list. There's no point passing temperature or top_p as it doesn't
    change the output at all.
    """

    if output_token_limit is None:
        max_new_tokens = 1e3  # nothing is longer than this
    else:
        max_new_tokens = output_token_limit

    summaries = summarizer(
        text_list, max_length=max_new_tokens, min_length=min_length, do_sample=False
    )
    return summaries


def generate_summaries_bart(
    gender: str,
    model_name: str,
    output_token_limit: int,
    temperature: float,
    top_p: float,
    out_dir: str = "./generate_summaries/output",
    portraits_in_dir: str = "./raw_data/",
):
    """
    Temperature and top_p are only for the file name as they do not seem to affect output.
    Above are the defaults.

    https://huggingface.co/transformers/v3.0.2/model_doc/bart.html#transformers.BartForConditionalGeneration.generate
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
    summarizer = load_bart()
    original_txt = [portrait["original"] for portrait in portraits]
    result_txt = [portrait["result"] for portrait in portraits]

    original_summaries = summarise_text(
        summarizer, original_txt, output_token_limit=output_token_limit
    )
    result_summaries = summarise_text(
        summarizer, result_txt, output_token_limit=output_token_limit
    )

    summaries = []
    for original_summary, result_summary in zip(original_summaries, result_summaries):
        summaries.append(
            {
                "original": original_summary["summary_text"],
                "result": result_summary["summary_text"],
            }
        )
    with open(out_file, "w") as f:
        json.dump(summaries, f)


def main():
    generate_summaries_bart(**parse_args())


# python ./3__bart_large.py --model_name="bart" --output_token_limit=50 --gender="fm" --temperature=1.0 --top_p=1.0
if __name__ == "__main__":
    main()
