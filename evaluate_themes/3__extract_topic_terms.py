import json
import pandas as pd
from collections import Counter
import pathlib


def get_unique_info(in_file: pathlib.Path, key: str) -> set[str]:
    with open(in_file, "r") as f:
        info_dict = json.load(f)
    all_info = set(
        [item for sublist in [d.get(key, "") for d in info_dict] for item in sublist]
    )
    return all_info


def write_unique_terms(
    pattern: str, key: str, models: list[str] = ["llama3", "gemma"]
) -> None:

    # Get files to read in
    in_dirs = [f"./evaluate_themes/output_{model}/parsed/" for model in models]

    in_files = [list(pathlib.Path(in_dir).glob(f"*{pattern}*")) for in_dir in in_dirs]

    in_files = [item for sublist in in_files for item in sublist]

    info_set_list = []
    print("Reading in files: ")
    for file in in_files:
        print(file)
        info_set = get_unique_info(file, key)
        info_set_list.append(info_set)
    info_set_list = [item for sublist in info_set_list for item in sublist]
    info_set = set(info_set_list)
    info_set_count = Counter(info_set_list).most_common()

    out_file_list = pathlib.Path(f"./evaluate_themes/themes_output/{key}_full.txt")
    out_file_count = pathlib.Path(f"./evaluate_themes/themes_output/{key}_count.json")
    pathlib.Path.mkdir(out_file_list.parent, exist_ok=True)

    with open(out_file_list, "w") as f:
        for line in info_set:
            f.write(f"{line}\n")

    with open(out_file_count, "w") as file:
        json.dump(info_set_count, file)

    print(f"File created: {out_file_list}. Length: {len(info_set)}")


def extract_topic_terms():
    args = {
        "mental_health_details": "health",
        "physical_health_details": "health",
        "subjective_language_info": "subjective_language",
        "appearance_info": "physical_appearance",
    }

    for pattern, filename in args.items():
        write_unique_terms(filename, pattern)


if __name__ == "__main__":
    extract_topic_terms()
