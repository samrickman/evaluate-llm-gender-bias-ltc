from openai import OpenAI
import json
import pathlib


def make_batch_request(in_file: pathlib.Path, client: OpenAI) -> None:
    batch_input_file = client.files.create(file=open(in_file, "rb"), purpose="batch")

    batch_obj = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "nightly eval job"},
    )
    print(f"Request made: {in_file.stem}    {batch_input_file.id}")


def main():
    client = OpenAI()
    in_files = pathlib.Path(f"./generate_summaries/chatgpt/requests/").glob("*.jsonl")
    for in_file in in_files:
        make_batch_request(in_file, client)

    batches = client.batches.list()

    # Print out the details of all the batches
    for batch in batches:
        print(batch)


if __name__ == "__main__":
    main()
