from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import cuda, float16
import pathlib
import json
import argparse
from constants import user_prompt_dict


def load_llama3() -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=float16,
        device_map="auto",
    )
    return tokenizer, model


def load_portraits(in_dir: str, original_gender: str) -> list[dict]:
    in_file = pathlib.Path(f"{in_dir}/{original_gender}_portraits.json")

    with open(in_file, "r") as f:
        portraits = json.load(f)

    return portraits


def change_gender(
    text: str,
    original_gender: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    system_prompt_length: int = 123,
):

    user_prompt = user_prompt_dict[original_gender]
    cuda.empty_cache()
    prompt = f"""
            {user_prompt}
            ```{text}```
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

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    input_length = tokens.size()[1]
    max_new_tokens = input_length - system_prompt_length

    # Generate output
    generation_output = model.generate(
        tokens,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=terminators,
        # as do_sample = False unset temp and top_p
        top_p=None,
        temperature=None,
    )

    # Decode but chop off input - we don't need to regurgitate it
    return tokenizer.decode(generation_output[0][input_length:])


def create_results(
    portraits: list[dict],
    original_gender: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
):
    results_list = []
    for portrait in tqdm(portraits):
        results_list.append(
            change_gender(portrait["text"], original_gender, tokenizer, model)
        )
    return results_list


def create_output(portraits: list[dict], results: list[dict]) -> list[dict]:
    results_clean = [
        txt.replace("`", "").replace("<|eot_id|>", "").strip() for txt in results
    ]
    clean_output_list = []
    for portrait, result in zip(portraits, results_clean):
        clean_output_list.append(
            {
                "DocumentID": portrait["DocumentID"],
                "original": portrait["text"],
                "result": result,
            }
        )
    return clean_output_list


def write_output(output: list[dict], out_dir: str, original_gender: str) -> None:
    if original_gender == "male":
        out_gender = "female"
    if original_gender == "female":
        out_gender = "male"
    out_file = pathlib.Path(f"{out_dir}/{original_gender}_to_{out_gender}.json")
    pathlib.Path(out_file.parent).mkdir(exist_ok=True, parents=True)
    with open(out_file, "w") as f:
        json.dump(output, f)
    print(f"Gender-swapped portraits created: {out_file}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--in_dir")
    parser.add_argument("--out_dir")
    parser.add_argument("--original_gender")
    args = parser.parse_args()

    print(
        f"""
    Swapping gender:
    
    in_dir: {args.in_dir}
    out_dir: {args.out_dir}
    original gender: {args.original_gender}
    """
    )

    return args.in_dir, args.out_dir, args.original_gender


def main():
    in_dir, out_dir, original_gender = parse_args()
    tokenizer, model = load_llama3()
    portraits = load_portraits(in_dir=in_dir, original_gender=original_gender)
    results = create_results(portraits, original_gender, tokenizer, model)
    output = create_output(portraits, results)
    write_output(output, out_dir, original_gender)


if __name__ == "__main__":
    main()
