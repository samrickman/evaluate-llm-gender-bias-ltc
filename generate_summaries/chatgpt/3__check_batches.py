from openai import OpenAI
import json
import pathlib


def get_all_completed():
    client = OpenAI()

    # List all batches
    batches = client.batches.list()

    # Print out the details of each batch
    completed_batches = []
    for batch in batches:
        if batch.status == "completed":
            completed_batches.append(batch.output_file_id)
        else:
            print(batch)

    # Compare to the number of in files
    request_files = list(
        pathlib.Path(f"./generate_summaries/chatgpt/requests/").glob("*.jsonl")
    )
    n_requests = len(request_files)
    n_completed = len(completed_batches)
    print(f"Number of requests made: {n_requests}")
    print(f"Number of batches completed: {n_completed}")

    if n_requests <= n_completed:
        out_file = pathlib.Path(
            "./generate_summaries/chatgpt/requests/completed_ids.json"
        )
        pathlib.Path.mkdir(out_file.parent, exist_ok=True, parents=True)
        with open(out_file, "w") as f:
            json.dump(completed_batches, f)


if __name__ == "__main__":
    get_all_completed()
