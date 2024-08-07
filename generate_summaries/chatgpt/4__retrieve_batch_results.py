from openai import OpenAI
import json
import pathlib


def download_result(output_file_id: str) -> None:

    file_response = client.files.content(output_file_id).text
    out_file = pathlib.Path(
        f"./generate_summaries/chatgpt/responses/{output_file_id}.jsonl"
    )
    pathlib.Path.mkdir(out_file.parent, exist_ok=True, parents=True)

    with open(out_file, "w") as f:
        f.write(file_response)


client = OpenAI()

# This file will only exist when they're all done
with open("./generate_summaries/chatgpt/requests/completed_ids.json") as f:
    request_ids = json.load(f)

for output_file_id in request_ids:
    download_result(output_file_id)
