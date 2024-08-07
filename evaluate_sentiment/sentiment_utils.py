import pathlib
import re
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--in_dir")
    parser.add_argument("--out_dir")
    args = parser.parse_args()

    print(
        f"""
    Evaluating sentiment:
    
    in_dir: {args.in_dir}
    out_dir: {args.out_dir}
    """
    )

    return args.in_dir, args.out_dir


def get_model_params(file_path: pathlib.PosixPath) -> dict[str, str]:
    """
    This extracts the model parameters from the filename.
    e.g.

    fm_gemma_None_temp_0.7_top-p_0.9_clean.json ->
    {
         'gender': 'fm',
         'model': 'gemma',
         'max_tokens': 'None',
         'temp': '0.7',
         'top_p': '0.9'
     }
    """

    s = str(file_path)
    gender, model, max_tokens, temp, top_p = re.findall(
        "(fm|mf)_(\\w+)_(\\d+|None)_temp_(.+)_top-p_(.+)_clean\\.json$", s
    )[0]
    return {
        "gender": gender,
        "model": model,
        "max_tokens": max_tokens,
        "temp": temp,
        "top_p": top_p,
    }


def glob_files_in_dir(
    in_dir: str = "../clean_summaries/output/", glob_pattern="*.json"
) -> pathlib.Path:
    return pathlib.Path(in_dir).glob(glob_pattern)


def get_gender(in_file: pathlib.PosixPath) -> tuple[str, str]:
    """
    Get the gender from the file name.
    Two different cases, one for originals and one for the summaries.
    """

    # Originals
    if "gender_swapped_portraits" in in_file.parts:

        s = str(in_file)
        if "male_to_female" in s:
            original_gender = "male"
            result_gender = "female"
            return original_gender, result_gender
        elif "female_to_male" in s:
            original_gender = "female"
            result_gender = "male"
            return original_gender, result_gender
        else:
            raise ValueError("File name is not in expected format")

    # Summaries
    else:

        params = get_model_params(in_file)

        if params["gender"] == "mf":
            original_gender = "male"
            result_gender = "female"

        if params["gender"] == "fm":
            original_gender = "female"
            result_gender = "male"

        return original_gender, result_gender


def create_outfile_path(
    in_file: pathlib.Path,
    metric_name: str,
    out_dir: str,
) -> tuple[pathlib.Path, pathlib.Path]:

    original_gender, result_gender = get_gender(in_file)

    out_file_original = pathlib.Path(
        f"{out_dir}/{metric_name}/{in_file.stem}_{original_gender}.csv"
    )
    out_file_result = pathlib.Path(
        f"{out_dir}/{metric_name}/{in_file.stem}_{result_gender}.csv"
    )
    pathlib.Path(out_file_original.parent).mkdir(exist_ok=True, parents=True)
    return out_file_original, out_file_result
