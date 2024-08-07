from sklearn.feature_extraction.text import CountVectorizer
import json
import numpy as np
import pandas as pd
import spacy
from spacy.lang.en import English
import pathlib
from portraits_utils import load_portraits, load_summaries
import argparse


def pre_process_text(text: str, nlp: English) -> pd.DataFrame:
    """
    Function to lemmatise text so we can compare words.
    A few words are not lemmatised properly so are manually
    added in lemma_dict.
    """

    lemma_dict = {
        "agitated": "agitate",
        "annoy": "annoying",
        "assessed": "assess",
        "attemp": "attempt",
        "befriender": "befriend",
        "breath": "breathe",
        "cancelled": "cancel",
        "challenging": "challenge",
        "circumstances": "circumstance",
        "closer": "close",
        "decided": "decide",
        "diabete": "diabetes",
        "discus": "discuss",
        "dishevel": "dishevelled",
        "distance": "distances",
        "difficulty": "difficult",
        "discus": "discuss",
        "drinking": "drink",
        "eating": "eat",
        "expressive": "express",
        "falls": "fall",
        "fed": "feed",
        "finances": "finance",
        "groomed": "groom",
        "grooming": "groom",
        "haircut": "hair",
        "impaired": "impair",
        "indoor": "indoors",
        "mobile": "mobilise",
        "mobilises": "mobilise",
        "moving": "move",
        "need": "needs",
        "outdoor": "outdoors",
        "prescribed": "prescribe",
        "prescriber": "prescribe",
        "prevailing": "prevail",
        "procession": "processions",
        "recomendations": "recomend",
        "recomendation": "recomend",
        "relationships": "relationship",
        "resistance": "resist",
        "resistant": "resist",
        "risks": "risk",
        "safety": "safe",
        "screaming": "scream",
        "service": "services",
        "sever": "severe",
        "shakes": "shake",
        "showering": "shower",
        "sheltered": "shelter",
        "shopping": "shop",
        "standing": "stand",
        "state": "states",
        "tablets": "tablet",
        "temporarily": "temporary",
        "traveling": "travel",
        "travelling": "travel",
        "toile": "toilet",
        "undressing": "undress",
        "wandering": "wander",
        "washing": "wash",
        "youngest": "young",
    }
    doc = nlp(text)
    lemmas = []
    for token in doc:
        if not token.is_stop and not token.is_punct:
            lemma = token.lemma_
            if lemma in lemma_dict:
                lemmas.append(lemma_dict[lemma])
            else:
                lemmas.append(lemma)
    return " ".join(lemmas)


def clean_portrait(portrait: dict[str, str], nlp: English) -> dict[str, str]:
    """
    The originals have an "original_clean" key.
    The summaries just have "original".
    This gets them into the same format so we can
    use the same functions.
    """

    if "original_clean" in portrait:
        original = portrait["original_clean"]
    else:
        original = portrait["original"]

    return {
        "original": pre_process_text(original.lower(), nlp),
        "result": pre_process_text(portrait["result"].lower(), nlp),
    }


def get_word_counts(doc_list: list[str], gender: str) -> pd.DataFrame:
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(doc_list)
    word_counts = X.toarray()  # from sparse array to numpy array
    words = vectorizer.get_feature_names_out()
    df = pd.DataFrame(word_counts, columns=words)
    df["doc_num"] = df.index

    df_long = df.melt(
        id_vars=["doc_num"], var_name="word", value_name=f"word_count_{gender}"
    )
    return df_long


def create_word_df(
    gender: str, originals_or_summaries: str, summaries_in_file: pathlib.Path = None
):
    nlp = spacy.load("en_core_web_lg")

    if originals_or_summaries == "originals":
        portraits = load_portraits(gender, "./gender_swapped_portraits/clean/")
    elif originals_or_summaries == "summaries":
        portraits = load_summaries(summaries_in_file)
    else:
        raise ValueError(
            "The originals_or_summaries parameter must be either 'originals' or 'summaries'."
        )

    clean_portraits = [clean_portrait(portrait, nlp) for portrait in portraits]
    originals = [portrait["original"] for portrait in clean_portraits]
    results = [portrait["result"] for portrait in clean_portraits]

    if gender == "fm":
        original_gender = "female"
        result_gender = "male"
    elif gender == "mf":
        original_gender = "male"
        result_gender = "female"
    else:
        raise ValueError("The gender parameter must be either 'fm' or 'mf'")

    original_df = get_word_counts(originals, original_gender)
    result_df = get_word_counts(results, result_gender)

    word_df = pd.merge(original_df, result_df, on=["doc_num", "word"], how="outer")

    # Replace NaN with 0 - these are words that do not appear at all in the corpus so don't join (e.g. grandfather)
    word_df.loc[:, ["word_count_female", "word_count_male"]] = word_df[
        ["word_count_female", "word_count_male"]
    ].fillna(0)

    # List any words that do not appear equally in originals and then make sure we do not count
    # those words in the summaries
    words_to_exclude_file = pathlib.Path(
        f"./bag_of_words/txt/{original_gender}_to_{result_gender}_words_to_exclude.txt"
    )

    if originals_or_summaries == "originals":
        # Not fair or useful to count words in summaries that are unequal in originals e.g. woman
        words_to_exclude = list(
            word_df[word_df["word_count_female"] != word_df["word_count_male"]][
                "word"
            ].unique()
        )

        pathlib.Path.mkdir(words_to_exclude_file.parent, exist_ok=True)
        print(f"Creating words to exclude file: {words_to_exclude_file}")
        with open(words_to_exclude_file, "w") as f:
            for word in words_to_exclude:
                f.write(f"{word}\n")

        word_df_out_file = pathlib.Path(
            f"./bag_of_words/csv/originals/{original_gender}_to_{result_gender}_word_df.csv"
        )

    if originals_or_summaries == "summaries":
        with open(words_to_exclude_file, "r") as f:
            words_to_exclude = [line.replace("\n", "") for line in f.readlines()]
        word_df = word_df[~word_df["word"].isin(words_to_exclude)]

        word_df_out_file = pathlib.Path(
            f"./bag_of_words/csv/summaries/{summaries_in_file.stem}_word_df.csv"
        )

    pathlib.Path.mkdir(word_df_out_file.parent, exist_ok=True, parents=True)

    word_df.to_csv(word_df_out_file, index=False)

    return word_df


def create_word_df_originals():
    create_word_df("mf", "originals")
    create_word_df("fm", "originals")


def parse_args() -> str:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--originals", action="store_true", help="Count words in original texts"
    )
    parser.add_argument(
        "--summaries", action="store_true", help="Count words in summaries"
    )

    # Parse the arguments
    args = parser.parse_args()

    # Implement the logic based on arguments
    if args.originals:
        print("Counting words in original texts...")
        return "originals"

    if args.summaries:
        print("Counting words in summaries...")
        return "summaries"

    if not args.originals and not args.summaries:
        raise ValueError(
            "Please specify `python ./bag_of_words/1__count_words.py --originals` or `--summaries`."
        )


def main():

    doc_type = parse_args()
    if doc_type == "originals":
        print("Generating word counts for originals...")
        create_word_df_originals()
        print("Done.")
        exit(0)

    print("Generating word counts for summaries...")
    summary_files = pathlib.Path("./generate_summaries/clean_output/").glob("*.json")
    for summary_in_file in summary_files:
        gender = summary_in_file.name[0:2]  # they all start with mf or fm
        print(f"Generating: {summary_in_file}")
        create_word_df(
            gender,
            originals_or_summaries="summaries",
            summaries_in_file=summary_in_file,
        )
    print("Done. All word counts generated.")


if __name__ == "__main__":
    main()
