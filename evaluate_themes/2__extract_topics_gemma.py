from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import cuda, float16
import json
from json.decoder import JSONDecodeError
from tqdm import tqdm
import pathlib
from constants import system_prompts, system_prompt_keys
import argparse
import re


def load_gemma():
    model_str = "google/gemma-7b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_str)
    model = AutoModelForCausalLM.from_pretrained(
        model_str,
        torch_dtype=float16,
        revision="float16",
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    return tokenizer, model


def extract_topic(
    text,
    tokenizer,
    model,
    system_prompt,
    max_new_tokens=400,
    temperature=None,
    top_p=None,
):
    cuda.empty_cache()
    prompt = f"""
              {system_prompt}
              Input: {text}
              Output:
            """

    tokens = tokenizer(prompt, return_tensors="pt").to("cuda")

    input_length = tokens["input_ids"].size()[1]

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    input_length = tokens["input_ids"].size()[1]

    # Generate output
    generation_output = model.generate(
        **tokens,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=terminators,
        temperature=temperature,
        top_p=top_p,
    )
    # Decode but chop off input - we don't need to regurgitate it
    return tokenizer.decode(generation_output[0][input_length:])


def parse_json_string(s: str, i: int, expected_keys: set[str]):
    # Remove new lines and extract smallest string betwen {}
    # (sometimes it repeats it)
    s = s.replace("\n", "")
    s = re.sub("^.+(\\{.*?\\}).+$", "\\1", s)
    json_str = s[s.find("{") : s.find("}") + 1]

    json_out = json.loads(json_str)

    # Sometimes it creates a field called mental_health_info
    # rather than mental_health_details
    if "mental_health_info" in json_out:
        json_out["mental_health_details"] = json_out["mental_health_info"]
        del json_out["mental_health_info"]

    # This doesn't seem to happen but for completeness
    if "physical_health_info" in json_out:
        json_out["physical_health_details"] = json_out["physical_health_info"]
        del json_out["physical_health_info"]

    for key in expected_keys.difference(set(json_out.keys())):
        json_out[key] = "Missing"
    return {"doc_num": i, **json_out}


def extract_topics_all(
    in_file: pathlib.Path,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    input_key: str,  # e.g. "original_clean", "result"
    output_keys: str,
    system_prompt: str,
    topic_type: str = "health",
    out_dir: str = "output",
) -> None:
    """
    This applies the extract_topic() function to all the portraits/summaries.

    Returns None as it writes to file.
    """

    out_file = pathlib.Path(f"{out_dir}/{in_file.stem}_{topic_type}_{input_key}.json")
    out_file_clean = pathlib.Path(
        f"{out_dir}/parsed/{in_file.stem}_{topic_type}_{input_key}.json"
    )

    pathlib.Path(out_file.parent).mkdir(exist_ok=True)
    pathlib.Path(out_file_clean.parent).mkdir(exist_ok=True)
    if out_file.is_file():
        print(f"File already exists: {out_file}. Skipping.")
        return

    with open(in_file, "r") as f:
        portraits = json.load(f)

    results_list = []
    results_list_clean = []
    for i, portrait in enumerate(tqdm(portraits)):
        result = extract_topic(portrait[input_key], tokenizer, model, system_prompt)
        results_list.append(result)
        try:
            results_list_clean.append(parse_json_string(result, i, output_keys))
        except JSONDecodeError:
            results_list_clean.append({"doc_num": i, "status": "error"})

        # Do this on every iteration so we can keep an eye on it
        with open(out_file, "w") as f:
            json.dump(results_list, f)
        with open(out_file_clean, "w") as f:
            json.dump(results_list_clean, f)


# python ./1__extract_topics.py --input_key="original_clean" --topic_type="health" --in_dir="../generate_summaries/raw_data" --out_dir="output_originals"
def main(
    input_key: str,
    topic_type: str,
    system_prompt_dict: dict[str] = system_prompts,
    output_keys_dict: dict[set[str]] = system_prompt_keys,
    in_dir: str = "../generate_summaries/raw_data",
    out_dir: str = "output_originals",
) -> None:

    tokenizer, model = load_gemma()

    files = pathlib.Path(in_dir).glob("*.json")
    for in_file in files:
        print(f"Identifying topics in: {in_file}")
        extract_topics_all(
            in_file,
            tokenizer,
            model,
            input_key,
            output_keys_dict[topic_type],
            system_prompt_dict[topic_type],
            topic_type,
            out_dir,
        )


# python ./1__extract_topics.py --input_key="original_clean" --topic_type="health" --in_dir="../generate_summaries/raw_data" --out_dir="output_originals"
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--in_dir")
    parser.add_argument("--out_dir")
    parser.add_argument("--input_key")
    parser.add_argument("--topic_type")
    args = parser.parse_args()

    print(
        f"""
    Extracting topics:
    
    input_key: {args.input_key}
    topic_type: {args.topic_type}
    in_dir: {args.in_dir}
    out_dir: {args.out_dir}
    """
    )

    return args.input_key, args.topic_type, args.in_dir, args.out_dir


if __name__ == "__main__":
    input_key, topic_type, in_dir, out_dir = parse_args()
    main(
        input_key=input_key,
        topic_type=topic_type,
        system_prompt_dict=system_prompts,
        output_keys_dict=system_prompt_keys,
        in_dir=in_dir,
        out_dir=out_dir,
    )
