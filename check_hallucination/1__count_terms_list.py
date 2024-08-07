import json
from spacy.lang.en import English
import pandas as pd
import pathlib
import argparse
from constants import terms_files
from string import punctuation


def get_original_result_gender(file_path: pathlib.Path) -> tuple[str, str]:

    # Originals
    if file_path.name.startswith("male_to_female"):
        return "male", "female"
    if file_path.name.startswith("female_to_male"):
        return "female", "male"

    # Summaries
    if file_path.name.startswith("mf"):
        return "male", "female"
    if file_path.name.startswith("fm"):
        return "female", "male"

    raise ValueError(
        "Expected file name to start with 'fm', 'mf', 'female_to_male' or 'male_to_female'"
    )


def clean_portrait(portrait: dict[str, str]) -> dict[str, str]:
    """
    The originals have an "original_clean" key.
    The summaries just have "original".
    This gets them into the same format so we can
    use the same functions.
    """

    if "original_clean" in portrait:
        original = portrait["original_clean"].translate(
            str.maketrans(punctuation, " " * len(punctuation))
        )
    else:
        original = portrait["original"].translate(
            str.maketrans(punctuation, " " * len(punctuation))
        )

    return {
        "original": original.lower(),
        "result": portrait["result"]
        .translate(str.maketrans(punctuation, " " * len(punctuation)))
        .lower(),
    }


def count_terms_in_portrait(
    portrait: dict[str, str],
    doc_num: int,
    term: str,
    nlp: English,
    original_gender: str,
    result_gender: str,
):

    portrait_clean = clean_portrait(portrait)
    original_tokens = nlp(portrait_clean["original"])
    result_tokens = nlp(portrait_clean["result"])

    # So we can have mult-word terms
    # spacy will handle the out of bounds references for us thankfully
    term_nlp = nlp(term)
    term_length = len(term_nlp)

    original_count = 0
    result_count = 0

    for i, _ in enumerate(original_tokens):
        # .startswith() so we can supply "incontinen"
        # as lemma doesn't work for incontinent/incontinence
        # similarly "pressure sore" will include "pressure sores"
        # theoretically could be a false positive but looking
        # at the terms it seems unlikely
        if original_tokens[i : (i + term_length)].text.startswith(term):
            original_count += 1

    for i, tok in enumerate(result_tokens):
        if result_tokens[i : (i + term_length)].text.startswith(term):
            result_count += 1

    return {
        "doc_num": doc_num,
        "term": term,
        f"{original_gender}_count": original_count,
        f"{result_gender}_count": result_count,
        "counts_equal": original_count == result_count,
        "original_gender": original_gender,
    }


def count_term_all_portraits(
    portraits: list[dict],
    term: str,
    nlp: English,
    original_gender: str,
    result_gender: str,
):
    term_list = []
    for doc_num, portrait in enumerate(portraits):
        term_list.append(
            count_terms_in_portrait(
                portrait, doc_num, term, nlp, original_gender, result_gender
            )
        )

    return term_list


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--in_dir")
    parser.add_argument("--out_dir")
    args = parser.parse_args()

    print(
        f"""
    Counting terms:
    
    in_dir: {args.in_dir}
    out_dir: {args.out_dir}
    """
    )

    return args.in_dir, args.out_dir


def generate_output_from_terms_list(
    in_file: pathlib.Path,
    nlp: English,
    term_type: str,
    terms_list: list[str],
    original_gender: str,
    result_gender: str,
):
    original_gender, result_gender = get_original_result_gender(in_file)
    with open(in_file, "r") as f:
        portraits = json.load(f)

    term_counts = [
        count_term_all_portraits(portraits, term, nlp, original_gender, result_gender)
        for term in terms_list
    ]

    df = pd.DataFrame([item for sublist in term_counts for item in sublist])

    df["term_type"] = term_type

    return df


def generate_output_all_terms(
    in_file: pathlib.Path,
    out_file: pathlib.Path,
    terms_dict: dict,
    nlp: English,
    original_gender: str,
    result_gender: str,
) -> None:
    """
    Creates a data frame with the counts of output of all terms.
    Output file name is based on the input file name so no need to parse it for params.
    """
    nlp = English()

    df_list = [
        generate_output_from_terms_list(
            in_file=in_file,
            nlp=nlp,
            term_type=term_type,
            terms_list=terms_list,
            original_gender=original_gender,
            result_gender=result_gender,
        )
        for term_type, terms_list in terms_dict.items()
    ]
    df = pd.concat(df_list)
    df.to_csv(out_file, index=False)
    print(f"Created: {out_file}")


def open_file(in_file):
    with open(in_file, "r") as f:
        terms_list = [line.replace("\n", "") for line in f.readlines()]
    return terms_list


def count_all_terms_all_files():
    nlp = English()
    in_dir, out_dir = parse_args()

    terms_dict = {k: open_file(v) for k, v in terms_files.items()}

    in_files = [f for f in pathlib.Path(in_dir).glob("*.json")]
    for in_file in in_files:
        out_file = pathlib.Path(f"{out_dir}/{in_file.stem}_term_counts.csv")
        pathlib.Path.mkdir(out_file.parent, exist_ok=True)
        if out_file.is_file():
            print(f"File already exists. Skipping: {out_file}")
            continue
        original_gender, result_gender = get_original_result_gender(in_file)
        generate_output_all_terms(
            in_file=in_file,
            out_file=out_file,
            terms_dict=terms_dict,
            nlp=nlp,
            original_gender=original_gender,
            result_gender=result_gender,
        )


# python ./1__count_terms_list.py --in_dir="../generate_summaries/raw_data" --out_dir="csv_originals"
if __name__ == "__main__":
    count_all_terms_all_files()
