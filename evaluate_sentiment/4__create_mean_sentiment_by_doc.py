import pandas as pd
from sentiment_utils import glob_files_in_dir, parse_args
import pathlib


def create_mean_by_doc(metric: str, in_dir: str = f"./output/") -> None:
    """
    We can't compare sentences like-for-like with male and female summaries as they do not
    necessarily correspond. This calculates the mean by sentence for each document.
    It is not the only way to do it. We could also do min, max etc. It's possible
    that we will see more differences if we do that as the mean may be homogenising them.
    """

    sentiment_files = glob_files_in_dir(f"{in_dir}/{metric}", glob_pattern="*.csv")

    def create_regard_mean(in_file: pathlib.Path, out_file: pathlib.Path) -> None:
        df = pd.read_csv(in_file)
        out_file.parent.mkdir(exist_ok=True)
        df.groupby("doc_num").mean().to_csv(out_file)

    def create_siebert_distilbert_mean(
        in_file: pathlib.Path, out_file: pathlib.Path
    ) -> None:
        df = pd.read_csv(in_file)
        out_file.parent.mkdir(exist_ok=True)
        df.drop(["text", "label"], axis=1).groupby("doc_num").mean().to_csv(out_file)

    mean_funcs_dict = {
        "regard": create_regard_mean,
        "siebert": create_siebert_distilbert_mean,
        "distilbert": create_siebert_distilbert_mean,
    }

    for sentiment_file in sentiment_files:

        out_file = pathlib.Path(f"./{in_dir}/{metric}/mean/{sentiment_file.name}")

        mean_funcs_dict[metric](sentiment_file, out_file)
        print(f"File created: {out_file.parent}/{out_file.name}")


def create_mean_all_metrics():

    # The out_dir of the previous files is the in_dir of these files
    # as we're summarising them
    _, in_dir = parse_args()
    metrics = ["regard", "siebert", "distilbert"]

    for metric in metrics:
        create_mean_by_doc(metric, in_dir)


if __name__ == "__main__":
    create_mean_all_metrics()
