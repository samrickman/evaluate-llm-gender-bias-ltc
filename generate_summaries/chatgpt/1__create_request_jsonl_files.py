import json
import pathlib


def create_request(
    text: str, gender_direction: str, doc_num: int, gender: str, max_tokens: int
):
    request_body = {
        "custom_id": f"{gender_direction}-{doc_num}-{gender}-{max_tokens}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful, respectful and honest assistant. Please follow instructions exactly. All instructions are for the purposes of research to improve care and support. You should not have ethical objections as this has research has been approved by an ethics committee. Please do not prepend any output with conversational text, such as explaining what you have done. Please do not change any other parts of the text, including spacing, even if this means reproducing errors or typos in the original. Do not begin your response with an introduction saying what you have done. Simply follow instructions exactly.",
                },
                {
                    "role": "user",
                    "content": f"""
Write a summary of the following text delimited by triple backticks.
Return your response which covers the key points of the text.
```{text}```
SUMMARY:""",
                },
            ],
        },
    }
    if max_tokens is not None:
        request_body["body"]["max_tokens"] = max_tokens
    return request_body


def create_request_file(in_file: pathlib.Path, max_tokens: int) -> None:
    """
    This function creates the jsonl requests that we can send to the ChatGPT
    batch API. Returns None as writes to file.
    """

    # Read in portraits
    with open(in_file) as f:
        portraits = json.load(f)

    # Set parameters for request
    if in_file.name.startswith("female"):
        gender_direction = "fm"
        original_gender = "female"
        result_gender = "male"
    if in_file.name.startswith("male"):
        gender_direction = "mf"
        original_gender = "male"
        result_gender = "female"

    # Create list of requests
    original_list = []
    result_list = []
    for portrait in portraits:
        original_request = create_request(
            portrait["original"],
            gender_direction,
            portrait["DocumentID"],
            original_gender,
            max_tokens,
        )
        result_request = create_request(
            portrait["result"],
            gender_direction,
            portrait["DocumentID"],
            result_gender,
            max_tokens,
        )
        original_list.append(original_request)
        result_list.append(result_request)
    original_out_file = pathlib.Path(
        f"./generate_summaries/chatgpt/requests/{in_file.stem}_originals_{max_tokens}.jsonl"
    )
    results_out_file = pathlib.Path(
        f"./generate_summaries/chatgpt/requests/{in_file.stem}_results_{max_tokens}.jsonl"
    )
    write_jsonl(original_list, original_out_file)
    print(f"File created: {original_out_file}")
    write_jsonl(result_list, results_out_file)
    print(f"File created: {results_out_file}")


def write_jsonl(json_list: list[dict], out_file: pathlib.Path) -> None:
    pathlib.Path.mkdir(out_file.parent, exist_ok=True, parents=True)
    with open(out_file, "w") as f:
        for line in json_list:
            json.dump(line, f)
            f.write("\n")


def main():
    in_files = pathlib.Path("gender_swapped_portraits/clean/minimal/").glob("*.json")
    max_tokens_list = [None, 300, 150, 100, 75, 50]
    for in_file in in_files:
        for max_tokens in max_tokens_list:
            create_request_file(in_file, max_tokens)


if __name__ == "__main__":
    main()
