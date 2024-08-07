import json
import re
from typing import Callable, Any
import pathlib
from spacy.lang.en import English
import argparse


def replace_eos_token(summaries: list[dict], model: str) -> None:
    """
    Removes end of string token (and any backticks that have crept in).
    No need to return anything as it mutates the list in place.
    """

    eot_token_dict = {"llama3": "<|eot_id|>", "gemma": "<eos>"}
    for summary in summaries:
        summary["original"] = (
            summary["original"].replace(eot_token_dict[model], "").replace("`", "")
        )
        summary["result"] = (
            summary["result"].replace(eot_token_dict[model], "").replace("`", "")
        )


def remove_summary_prefixes_llama3(summaries: list[dict], *args) -> None:
    """
    This removes the lines which says, 'Here is a summary of the text'.
    The original/result pairs don't always both include it and we don't
    want it to affect the sentiment analysis result.
    Only an issue for Llama3.
    """

    def remove_summary_prefix(txt):
        txt_lines = txt.split("\n")
        if "summar" in txt_lines[0].lower():
            return "\n".join(txt_lines[1:]).strip()
        else:
            return txt

    for summary in summaries:
        summary["original"] = remove_summary_prefix(summary["original"])
        summary["result"] = remove_summary_prefix(summary["result"])


def squish_spaces(summaries: list[dict], *args) -> None:
    """
    This replaces all instances of new lines and spaces with one spaces.
    Sometimes there can be \n\n\n which can mess up the splitting into sentences.
    """

    def str_squish(txt):
        return re.sub(r"\s+", " ", txt)

    for summary in summaries:
        summary["original"] = str_squish(summary["original"])
        summary["result"] = str_squish(summary["result"])


def delete_blank_summaries(summaries: list[dict], *args) -> None:
    """
    Occasionally a summary can be completely blank (or just new lines which we then remove).
    This can cause problems downstream so we should identify and delete these ones.
    """

    def is_blank(summary):
        blank_summaries_exist = (
            len(summary["original"]) == 0 or len(summary["result"]) == 0
        )
        if blank_summaries_exist:
            print(f"There are blank summaries in this file.")
        return blank_summaries_exist

    summaries[:] = [summary for summary in summaries if not is_blank(summary)]


def split_into_sentences(summaries: list[dict], *args) -> None:
    """
    This splits the summaries into list of sentences. This is for
    sentiment analysis, so we can analyse them sentence by sentence.
    """
    nlp = English()
    nlp.add_pipe("sentencizer")

    def get_sentences(txt: str, nlp: English = nlp) -> list[str]:
        return [sent.text for sent in nlp(txt).sents]

    for summary in summaries:
        summary["original_sentences"] = get_sentences(summary["original"])
        summary["result_sentences"] = get_sentences(summary["result"])


def get_model_params(file_path: pathlib.PosixPath) -> dict[str, str]:
    """
    This extracts the model parameters from the filename.
    e.g.

    fm_gemma_None_temp_0.7_top-p_0.9.json ->
    {
         'gender': 'fm',
         'model': 'gemma',
         'max_tokens': 'None',
         'temp': '0.7',
         'top_p': '0.9'
     }
    """

    # Originals
    if "gender_swapped_portraits" in file_path.parts:
        return {"model": "originals"}

    s = str(file_path)
    gender, model, max_tokens, temp, top_p = re.findall(
        "(fm|mf)_(\\w+)_(\\d+|None)_temp_(.+)_top-p_(.+)\\.json$", s
    )[0]
    return {
        "gender": gender,
        "model": model,
        "max_tokens": max_tokens,
        "temp": temp,
        "top_p": top_p,
    }


def map_funcs(func_list: list[Callable], summaries: list[dict], model: str) -> None:
    """
    Apply the desired functions to the model to clean the data.
    These all mutate the list in place so do not return anything.
    """

    [f(summaries, model) for f in func_list]


def clean_output(
    in_dir: str = "../generate_summaries/output/", out_dir: str = "output"
) -> None:
    """
    This applies the cleaning functions to the json and writes them out.
    They do not all need the same cleaning. The functions they need are
    defined in funcs_dict.

    Returns None as it writes to file.
    """

    funcs_dict = {
        "gemma": [
            replace_eos_token,
            squish_spaces,
            split_into_sentences,
            delete_blank_summaries,
        ],
        "llama3": [
            replace_eos_token,
            remove_summary_prefixes_llama3,
            squish_spaces,
            split_into_sentences,
            delete_blank_summaries,
        ],
        "bart": [squish_spaces, split_into_sentences, delete_blank_summaries],
        "t5": [squish_spaces, split_into_sentences, delete_blank_summaries],
        "chatgpt": [squish_spaces, split_into_sentences, delete_blank_summaries],
        "originals": [squish_spaces, split_into_sentences],
    }

    files = pathlib.Path(in_dir).glob("*.json")
    for in_file in files:
        print(f"Cleaning: {in_file}")
        params = get_model_params(in_file)

        print(f"Cleaning file with functions associated with: {params['model']}")

        with open(in_file, "r") as f:
            summaries = json.load(f)

        map_funcs(funcs_dict[params["model"]], summaries, params["model"])
        out_file = pathlib.Path(f"{out_dir}/{in_file.stem}_clean.json")
        pathlib.Path(out_file.parent).mkdir(exist_ok=True)
        with open(out_file, "w") as f:
            json.dump(summaries, f)

        print(f"Saved: {out_file}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--in_dir", default="../generate_summaries/output/")
    parser.add_argument("--out_dir", default="output")
    args = parser.parse_args()

    print(
        f"""
    Cleaning text:
    
    in_dir: {args.in_dir}
    out_dir: {args.out_dir}
    """
    )

    return args.in_dir, args.out_dir


if __name__ == "__main__":
    """
    We can run this directly for the summaries or pass it other arguments
    for the original portraits.
    """
    in_dir, out_dir = parse_args()
    clean_output(in_dir, out_dir)
