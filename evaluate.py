import re
import json
import random
import numpy as np
from tqdm import tqdm
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def addition_api(input_query: str):
    input_query = input_query.strip("()")
    try:
        num_1, _, num_2 = input_query.partition(",")
    except ValueError:
        return input_query
    if num_1.isdigit() and num_2.isdigit():
        return str(int(num_1) + int(num_2)) 
    else:
        return input_query


def generate(text, model, tokenizer, device, max_len=100):

    def get_next_sequence(seq):
        outputs = model(seq, labels=seq)
        loss, logits = outputs[:2]
        softmax_logits = torch.softmax(logits[0, -1], dim=0)
        next_token_id = np.argmax(softmax_logits.to("cpu").numpy())
        seq = torch.cat([seq, torch.ones((1, 1)).long().to(device) * next_token_id], dim=1)
        next_token = tokenizer.convert_ids_to_tokens(int(next_token_id))
        return seq, (next_token_id, next_token)

    answer_ids = []

    model.eval()
    with torch.no_grad():

        seq = torch.tensor(tokenizer.encode(text)).unsqueeze(0).to(device)

        i = 0
        next_token = ""

        while i < max_len and next_token != "<|endoftext|>":
            seq, (next_token_id, next_token) = get_next_sequence(seq)
            answer_ids.append(next_token_id)
            i += 1

            # if start of api call is encountered
            if next_token == "Ġ[":
                answer_ids.pop()
                seq, (next_token_id, next_token) = get_next_sequence(seq)
                i += 1 
                if next_token == "Sum":
                    input_query_ids = []
                    while next_token != "Ġ->" and i < max_len:
                        seq, (next_token_id, next_token) = get_next_sequence(seq)
                        input_query_ids.append(next_token_id)
                        i += 1             
                    input_query = tokenizer.decode(input_query_ids[:-1])
                    result = addition_api(input_query)
                    answer_ids.extend(tokenizer.encode(result))
                    seq, (next_token_id, next_token) = get_next_sequence(
                        torch.cat([seq, torch.ones((1, 1)).long().to(device) * tokenizer.convert_tokens_to_ids("Ġ]")], dim=1)
                    )
                else:
                    continue
    return tokenizer.decode(answer_ids, skip_special_tokens=True)  


def sum_is_correct(model, tokenizer, device, num_1, num_2, prompt="The output of addition of <NUM1> and <NUM2> is"):
    prompt = re.sub("<NUM1>", str(num_1), prompt)
    prompt = re.sub("<NUM2>", str(num_2), prompt)
    answer = generate(prompt, model, tokenizer, device)
    result = re.findall("\d+", answer)
    if result:
        if int(result[0]) == num_1 + num_2:
            return True
        else:
            return False 
    else:
        return False


def evaluate_with_prompts(model, tokenizer, device, output_file, num_1_range=(1e+9, 1e+19), num_2_range=(1e+9, 1e+19), num_examples=500):

    with open("./data/prompts_inference.txt", "r", encoding="utf-8") as f:
        prompts = f.readlines()
    prompts = [prompt.strip() for prompt in prompts]

    prompts_results = {}

    for prompt in prompts:
        accuracy = 0
        for i in tqdm(range(num_examples)):
            num_1 = random.randint(num_1_range[0], num_1_range[1])
            num_2 = random.randint(num_2_range[0], num_2_range[1])
            if sum_is_correct(
              model=model, tokenizer=tokenizer, device=device, num_1=num_1, num_2=num_2, prompt=prompt
            ):
                accuracy += 1
        prompts_results[prompt] = round(accuracy / num_examples, 2)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(prompts_results, f, indent=2)
    
    return prompts_results

def evaluate_on_random_numbers(model, tokenizer, device, num_examples=100):
    accuracy = 0
    for i in tqdm(range(num_examples)):
        num_1 = random.randint(1e+9, 1e+19)
        num_2 = random.randint(1e+9, 1e+19)
        if sum_is_correct(
            model=model, tokenizer=tokenizer, device=device, num_1=num_1, num_2=num_2
        ):
            accuracy += 1 
    print(f"Accuracy on {num_examples} pairs of random numbers:\t{round(accuracy / num_examples, 2)}")


if __name__ == "__main__":

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_NAME = "gpt2"

    SEED = 42
    random.seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    if DEVICE.type == "cpu":
        model.load_state_dict(torch.load("./model/state_dict_model.pth", map_location=torch.device("cpu")))
    else:
        model.load_state_dict(torch.load("./model/state_dict_model.pth"))
        
    # evaluate with prompts on random numbers from 1e+9 to 1e+19
    # evaluate_with_prompts(
    #     model=model,
    #     tokenizer=tokenizer,
    #     device=DEVICE,
    #     output_file="./results/results_average.json"
    # )

    # evaluate with prompts on random numbers from 1 to 1e+9
    # evaluate_with_prompts(
    #     model=model,
    #     tokenizer=tokenizer,
    #     device=DEVICE,
    #     output_file="./results/results_small.json"
    # )

    # count accuracy score on `num_examples` pairs of random numbers from 1e+9 to 1e+19
    # default `num_examples` is set to 100
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_examples', type=int, default=100) 
    args = parser.parse_args()
    num_examples = args.num_examples
    evaluate_on_random_numbers(model=model, tokenizer=tokenizer, device=DEVICE, num_examples=num_examples)