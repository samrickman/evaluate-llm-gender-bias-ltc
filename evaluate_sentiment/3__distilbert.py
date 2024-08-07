import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
import json
from sentiment_utils import glob_files_in_dir, create_outfile_path, parse_args
import pathlib


def load_distilbert_model(
    model_name: str = "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
):
    """
    In this case we need to return a Trainer as well to use trainer.predict()
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(0)
    trainer = Trainer(model=model)
    return tokenizer, model, trainer


class SimpleDataset:
    """
    Adapted from siebert docs.
    See e.g. https://colab.research.google.com/github/chrdistilbert/sentiment-roberta-large-english/blob/main/sentiment_roberta_prediction_example.ipynb
    """

    def __init__(self, tokenized_texts):
        self.tokenized_texts = tokenized_texts

    def __len__(self):
        return len(self.tokenized_texts["input_ids"])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tokenized_texts.items()}


def get_distilbert_results(
    l: list[dict],
    doc_num: int,
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    trainer: Trainer,
):
    # Tokenize texts and create prediction data set
    tokenized_texts = tokenizer(l, truncation=True, padding=True)
    pred_dataset = SimpleDataset(tokenized_texts)
    predictions = trainer.predict(pred_dataset)
    # Transform predictions to labels
    preds = predictions.predictions.argmax(-1)
    labels = [model.config.id2label[pred] for pred in preds]
    scores = np.exp(predictions[0]) / np.exp(predictions[0]).sum(-1, keepdims=True)
    # Create DataFrame with texts, predictions, labels, and scores
    outdf = pd.DataFrame(list(zip(l, preds, labels)), columns=["text", "pred", "label"])
    outdf[list(model.config.id2label.values())] = scores
    outdf["doc_num"] = doc_num

    return outdf


def write_distilbert_df(
    in_file: pathlib.Path,
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    trainer: Trainer,
    out_dir: str = "output",
) -> None:
    with open(in_file) as f:
        summaries = json.load(f)

    out_file_original, out_file_result = create_outfile_path(
        in_file, "distilbert", out_dir
    )

    if out_file_original.is_file() and out_file_result.is_file():
        print(f"Distilbert sentiment already calculated for {in_file.name}. Skipping.")
        return

    originals = [summary["original_sentences"] for summary in summaries]
    results = [summary["result_sentences"] for summary in summaries]

    out_df_list_originals = [
        get_distilbert_results(l, i, tokenizer, model, trainer)
        for i, l in enumerate(originals)
    ]
    out_df_originals = pd.concat(out_df_list_originals)

    out_df_list_results = [
        get_distilbert_results(l, i, tokenizer, model, trainer)
        for i, l in enumerate(results)
    ]
    out_df_results = pd.concat(out_df_list_results)

    out_df_originals.to_csv(out_file_original, index=False)
    print(f"Created: {out_file_original}")
    out_df_results.to_csv(out_file_result, index=False)
    print(f"Created: {out_file_result}")


def evaluate_distilbert(
    in_dir: str = "../clean_summaries/output/", out_dir: str = "output"
):

    tokenizer, model, trainer = load_distilbert_model()
    in_files = glob_files_in_dir(in_dir)
    for in_file in in_files:
        print(f"Reading in: {in_file}")
        write_distilbert_df(in_file, tokenizer, model, trainer, out_dir)


if __name__ == "__main__":
    in_dir, out_dir = parse_args()
    evaluate_distilbert(in_dir, out_dir)
