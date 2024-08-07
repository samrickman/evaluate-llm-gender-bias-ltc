import json
import pathlib
import argparse


def read_json(path):
    with open(path) as f:
        return json.load(f)


def load_portraits(gender: str, portraits_in_dir) -> dict[str, str]:
    if gender == "fm":
        portraits = read_json(f"{portraits_in_dir}/female_to_male_clean.json")
    elif gender == "mf":
        portraits = read_json(f"{portraits_in_dir}/male_to_female_clean.json")
    else:
        raise ValueError("The gender parameter must be either 'fm' or 'mf'")

    return portraits


def load_summaries(in_file: pathlib.Path) -> dict[str, str]:

    portraits = read_json(in_file)

    return portraits


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--in_dir")
    parser.add_argument("--out_dir")
    parser.add_argument("--doc_type")
    args = parser.parse_args()

    print(
        f"""
    Counting words:
    
    in_dir: {args.in_dir}
    out_dir: {args.out_dir}
    type: {args.type}
    """
    )

    return args.in_dir, args.out_dir, args.doc_type
