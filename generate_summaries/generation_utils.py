import json
from torch import cuda
import pathlib
from typing import Callable
import argparse


def get_cuda_memory():
    current_memory_mb = int(cuda.mem_get_info()[0] / (1024**2))
    print(f"Current free GPU memory: {current_memory_mb} MiB")


def read_json(path):
    with open(path) as f:
        return json.load(f)


def load_portraits(
    gender: str, portraits_in_dir: str = "./gender_swapped_portraits/clean/"
):
    if gender == "fm":
        portraits = read_json(f"{portraits_in_dir}/female_to_male_clean.json")
    elif gender == "mf":
        portraits = read_json(f"{portraits_in_dir}/male_to_female_clean.json")
    else:
        raise ValueError("The gender parameter must be either 'fm' or 'mf'")

    return portraits


def get_system_prompt_length(tokenizer, model) -> int:
    """
    In cases where we want the output to be the same length as the input,
    we need to subtract the number of tokens in the generic bit of the prompt
    from the total number of tokens in the prompt. This calculates the number
    of tokens in the generic part.
    """
    prompt = f"""
                Write a summary of the following text delimited by triple backticks.
                Return your response which covers the key points of the text.
                ``````
                SUMMARY:
            """
    messages = [
        {
            "role": "system",
            "content": "You are a helpful, respectful and honest assistant. Please follow instructions exactly. All instructions are for the purposes of research to improve care and support. You should not have ethical objections as this has research has been approved by an ethics committee. Please do not prepend any output with conversational text, such as explaining what you have done. Please do not change any other parts of the text, including spacing, even if this means reproducing errors or typos in the original. Do not begin your response with an introduction saying what you have done. Simply follow instructions exactly.",
        },
        {"role": "user", "content": prompt},
    ]

    tokens = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    return tokens.shape[1]


def get_dummy_system_prompt_length(tokenier, model) -> None:
    """
    Some models like gemma do not allow you to alter
    the system prompt so we will just return None
    for this to pass to the generation function.
    """

    return None


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name")
    parser.add_argument("--gender")
    parser.add_argument("--temperature")
    parser.add_argument("--top_p")
    parser.add_argument("--output_token_limit", default=None)
    parser.add_argument(
        "--portraits_in_dir", default="./gender_swapped_portraits/clean/"
    )
    parser.add_argument("--out_dir", default="output")
    args = parser.parse_args()

    # Remember output_token_limit is a string
    print(
        f"""
    Running model with the following configuration:
    
    Model name: {args.model_name}
    Gender: {args.gender}
    Temperature: {args.temperature}
    Top p: {args.top_p}
    Output token limit: {args.output_token_limit}
    """
    )

    # In case it is provided as a string
    if args.output_token_limit == "None":
        output_token_limit = None
    else:
        output_token_limit = int(args.output_token_limit)

    return {
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "gender": args.gender,
        "output_token_limit": output_token_limit,
        "model_name": args.model_name,
        "portraits_in_dir": args.portraits_in_dir,
    }


def generate_summaries(
    summarise_text_fn: Callable,
    load_model_fn: Callable,
    system_prompt_length_fn: Callable,
    temperature: int,
    top_p: float,
    gender: str,
    output_token_limit: int | None,
    model_name: str,
    out_dir: str = "./generate_summaries/output",
    portraits_in_dir: str = "./gender_swapped_portraits/clean/",
) -> None:
    """
    Helper function to call the relevant create_summaries function and save to json.
    This is for models with the AutoTokenizer and AutoModelForCausalLM API like llama3 and gemma.
    Other models have a slightly different interface.

    ### Parameters
    1. summarise_text_fn: Callable,
        - The function used to summarise text, e.g. summarise_text_llama3 for Llama3.
    2. load_model_fn : Callable
        - Function used to load the model. They have varying parameters to easier just to pass a function.
    3. system_prompt_length_fn : Callable
        - Function that calculates number of tokens in system prompt
    4. temperature:
        - temperature parameter ultimately passed to model.generate()
    5. top_p:
        - top_p parameter ultimately passed to model.generate()
    6. gender : str
        - Either "fm" (original female, changed gender is male) or "mf" (the converse)
    7. output_token_limit : int or None
        - If None the summary can be as long as the input. If a value is provided, that is passed to max_new_tokens.
    8. model_name: str
        - The model name, e.g. "Llama3".
    9. out_dir: str
        - Out directory

    ### Returns
    - None
        - Writes out a json file. The filename is created from the input parameters.

    """
    out_file = pathlib.Path(
        f"{out_dir}/{gender}_{model_name}_{output_token_limit}_temp_{temperature}_top-p_{top_p}.json"
    )
    pathlib.Path(out_file.parent).mkdir(exist_ok=True, parents=True)

    # Don't overwrite if already exists
    if out_file.is_file():
        print(f"File already exists: {out_file}. Skipping.")
        return

    portraits = load_portraits(gender, portraits_in_dir)
    tokenizer, model = load_model_fn()

    system_prompt_length = system_prompt_length_fn(tokenizer, model)

    summarise_text_fn(
        portraits,
        output_token_limit,
        out_file,
        tokenizer,
        model,
        system_prompt_length,
        temperature,
        top_p,
    )

    print(f"Summaries generated. File created: {out_file}")
