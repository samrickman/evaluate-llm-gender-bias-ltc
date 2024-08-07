from evaluate import load, EvaluationModule
import json
import pandas as pd
from typing import Callable, Any
import pathlib
import re
from sentiment_utils import (
    glob_files_in_dir,
    create_outfile_path,
    parse_args,
)


def calculate_regard(
    summaries: list[dict], regard: EvaluationModule
) -> tuple[list, list]:
    originals = [summary["original_sentences"] for summary in summaries]
    results = [summary["result_sentences"] for summary in summaries]

    original_regard = [regard.compute(data=sentence) for sentence in originals]
    result_regard = [regard.compute(data=sentence) for sentence in results]
    return original_regard, result_regard


def create_sentence_df(sent_regard_list: list, doc_num: int) -> pd.DataFrame:
    df_list = [
        pd.DataFrame.from_dict(d).transpose().drop("label")
        for d in sent_regard_list["regard"]
    ]
    df = pd.concat(df_list)
    df.columns = ["positive", "other", "neutral", "negative"]
    df["doc_num"] = doc_num
    return df


def write_regard_df(
    in_file: pathlib.Path, regard: EvaluationModule, out_dir: str
) -> None:

    with open(in_file) as f:
        summaries = json.load(f)

    out_file_original, out_file_result = create_outfile_path(in_file, "regard", out_dir)

    if out_file_original.is_file() and out_file_result.is_file():
        print(f"Regard already calculated for {in_file.name}. Skipping.")
        return

    original_regard, result_regard = calculate_regard(summaries, regard)

    original_df_list = [
        create_sentence_df(sent_regard_list, i)
        for i, sent_regard_list in enumerate(original_regard)
    ]
    original_df = pd.concat(original_df_list)
    result_df_list = [
        create_sentence_df(sent_regard_list, i)
        for i, sent_regard_list in enumerate(result_regard)
    ]
    result_df = pd.concat(result_df_list)

    original_df.to_csv(out_file_original, index=False)
    print(f"Created: {out_file_original}")
    result_df.to_csv(out_file_result, index=False)
    print(f"Created: {out_file_result}")


def evaluate_regard(in_dir: str, out_dir: str):

    # Takes a couple of seconds - only do it once
    regard = load("regard", module_type="measurement")

    in_files = glob_files_in_dir(in_dir)
    for in_file in in_files:
        print(f"Reading in: {in_file}")
        write_regard_df(in_file, regard, out_dir)


if __name__ == "__main__":
    in_dir, out_dir = parse_args()
    evaluate_regard(in_dir, out_dir)
