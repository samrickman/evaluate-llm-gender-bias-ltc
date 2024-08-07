import json
from pathlib import Path


def remove_docs():
    """
    The purpose of this is to remove documents identified as having different counts of words
    (excluding a few like man/woman, mr/mrs).

    This ensures that all summaries are identical. We do this before summarisation as there is
    no point summarising documents we can't compare. This saves a lot of time.
    """

    # * Open the portraits
    with open("./model_sentiment_output/csv/docs_to_remove.json") as f:
        docs_to_remove = json.load(f)
    with open("./gender_swapped_portraits/clean/female_to_male_clean.json") as f:
        female_to_male = json.load(f)
    with open("./gender_swapped_portraits/clean/male_to_female_clean.json") as f:
        male_to_female = json.load(f)

    # * Remove the ones identified previously for removal
    male_to_female_minimal = [
        portrait
        for portrait in male_to_female
        if not portrait["DocumentID"] in docs_to_remove["male_to_female"]
    ]
    female_to_male_minimal = [
        portrait
        for portrait in female_to_male
        if not portrait["DocumentID"] in docs_to_remove["female_to_male"]
    ]

    out_file_mf = Path(
        "./gender_swapped_portraits/clean/minimal/male_to_female_clean.json"
    )
    out_file_fm = Path(
        "./gender_swapped_portraits/clean/minimal/female_to_male_clean.json"
    )
    Path.mkdir(out_file_mf.parent, exist_ok=True)
    with open(out_file_mf, "w") as f:
        json.dump(male_to_female_minimal, f)
    with open(out_file_fm, "w") as f:
        json.dump(female_to_male_minimal, f)


if __name__ == "__main__":
    remove_docs()
