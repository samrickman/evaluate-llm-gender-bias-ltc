import json
import pathlib


def get_text(response):
    return response["response"]["body"]["choices"][0]["message"]["content"]


def get_filename_params(response_list: list):

    # They should all be the same per file but just double-check we have only one set of params
    params = [response["custom_id"].split("-") for response in response_list]
    gender_direction, doc_num, gender, max_tokens = map(set, list(zip(*params)))

    if not (len(gender_direction) == 1 & len(gender) == 1 & len(max_tokens) == 1):
        raise ValueError(
            "The API appears to have returned more than one response in this file"
        )

    # Return the values not the sets
    gender_direction = gender_direction.pop()
    gender = gender.pop()

    # i.e. fm and female, mf and male
    if gender_direction[0] == gender[0]:
        response_type = "original"
    else:
        response_type = "result"

    return gender_direction, response_type, max_tokens.pop()


def get_content(in_file: pathlib.Path):
    print(in_file)
    with open(in_file) as f:
        response_list = [json.loads(line) for line in f.readlines()]
    content_list = [get_text(response) for response in response_list]

    try:
        gender_direction, response_type, max_tokens = get_filename_params(response_list)
    except ValueError:
        print("This file could not be parsed. It may have returned an error.")
        return None, None

    content_key = f"{gender_direction}_{max_tokens}_{response_type}"

    return content_key, content_list


def write_json_output(content_dict: dict[str, str]):
    unique_keys = set(
        [
            key.replace("_original", "").replace("_result", "")
            for key in content_dict.keys()
        ]
    )

    while len(unique_keys) > 0:
        unique_key = unique_keys.pop()
        originals = content_dict[f"{unique_key}_original"]
        results = content_dict[f"{unique_key}_result"]
        summaries = []
        for original_summary, result_summary in zip(originals, results):
            summaries.append({"original": original_summary, "result": result_summary})

        # https://platform.openai.com/docs/api-reference/making-requests#chat/create-temperature
        # looks like default temp is 0.7 though surprisingly hard to tell for sure
        original_gender, max_tokens = unique_key.split("_")
        out_file = f"./generate_summaries/output/{original_gender}_chatgpt_{max_tokens}_temp_0.7_top-p_0.9.json"

        with open(out_file, "w") as f:
            json.dump(summaries, f)
        print(f"File created: {out_file}. Length: {len(summaries)}")


def extract_responses(in_dir: str = "./generate_summaries/chatgpt/responses/"):
    responses_dir = pathlib.Path(in_dir)
    responses_files = list(responses_dir.glob("*.jsonl"))

    content_dict = {}
    for response_file in responses_files:
        content_key, content_list = get_content(response_file)
        if content_key in content_dict:
            raise ValueError(
                "These parameters already exist in the dict. Check the files to avoid overwriting."
            )

        if content_key is not None:
            content_dict[content_key] = content_list

    write_json_output(content_dict)


if __name__ == "__main__":
    extract_responses()
