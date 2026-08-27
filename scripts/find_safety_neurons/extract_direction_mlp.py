import os
import argparse
import torch
import json
import numpy as np

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import os as _os, sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root != _os.path.dirname(_root) and not _os.path.exists(_os.path.join(_root, "utils.py")):
    _root = _os.path.dirname(_root)
_sys.path.insert(0, _root)
from utils import MODEL_DICT, filter_response, load_conversation_template
from transformers.utils import logging
logging.set_verbosity_error()

np.random.seed(20)
torch.manual_seed(20)
torch.cuda.manual_seed_all(20)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="llama3")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

response_path = f"responses/{args.model}.json"
with open(response_path, "r") as f:
    response_data = json.load(f)

harmful_prompts = [entry["Harmful Prompt"] for entry in response_data]
neutral_prompts = [entry["Neutral Prompt"] for entry in response_data]
harmful = [entry["Harmful Response"] for entry in response_data]
neutral = [entry["Neutral Response"] for entry in response_data]

# Filter out responses that models refused to generate under the neutral prompt
harmful_prompts, neutral_prompts, harmful_response, neutral_response = filter_response(harmful_prompts, neutral_prompts, harmful, neutral)
print("#Filtered Harmful Prompts", len(harmful_prompts))
print("#Filtered Neutral Prompts", len(neutral_prompts))
print("#Filtered Harmful Response", len(harmful_response))
print("#Filtered Neutral Response", len(neutral_response))

print(f"Loading model {args.model}...")
model_path = MODEL_DICT[args.model]
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
model.generation_config.do_sample = True
tokenizer.pad_token = tokenizer.eos_token

conv_template = load_conversation_template(args.model)

harmful_embeddings = []
neutral_embeddings = []

for harmful_prompt, harmful_response in tqdm(zip(harmful_prompts, harmful_response), total=len(harmful_prompts)):

    conv_template.messages = []
    conv_template.append_message(conv_template.roles[0], f"{harmful_prompt}")
    conv_template.append_message(conv_template.roles[1], None)
    
    toks = tokenizer(conv_template.get_prompt()).input_ids
    start = len(toks)

    conv_template.update_last_message(f"{harmful_response}")
    toks = tokenizer(conv_template.get_prompt()).input_ids
    
    if "qwen" in args.model:
        end = len(toks) - 2
    elif "llama3" in args.model:
        end = len(toks) - 1
    else:
        end = len(toks)

    toks = tokenizer(conv_template.get_prompt(), return_tensors="pt")

    with torch.no_grad():
        residual_outputs = model(
            input_ids=toks['input_ids'].to(device),
            attention_mask=toks['attention_mask'].to(device),
            output_hidden_states=True
        ).hidden_states[1:]

    harmful_embeddings.append(
        torch.stack(residual_outputs, dim=0)[:, :, start-1:end-1, :].mean(dim=2).cpu()
    )

for neutral_prompt, neutral_response in tqdm(zip(neutral_prompts, neutral_response), total=len(neutral_prompts)):
    
    conv_template.messages = []
    conv_template.append_message(conv_template.roles[0], f"{neutral_prompt}")
    conv_template.append_message(conv_template.roles[1], None)
    
    toks = tokenizer(conv_template.get_prompt()).input_ids
    start = len(toks)
    
    conv_template.update_last_message(f"{neutral_response}")
    toks = tokenizer(conv_template.get_prompt()).input_ids
    
    if "qwen" in args.model:
        end = len(toks) - 2
    elif "llama3" in args.model or "phi3" in args.model:
        end = len(toks) - 1
    else:
        end = len(toks)
    
    toks = tokenizer(conv_template.get_prompt(), return_tensors="pt")

    with torch.no_grad():
        residual_outputs = model(
            input_ids=toks['input_ids'].to(device),
            attention_mask=toks['attention_mask'].to(device),
            output_hidden_states=True
        ).hidden_states[1:]

    neutral_embeddings.append(
        torch.stack(residual_outputs, dim=0)[:, :, start-1: end-1, :].mean(dim=2).cpu()
    )

mean_harmful_embeddings = torch.stack(harmful_embeddings, dim=0).mean(dim=0)
mean_neutral_embeddings = torch.stack(neutral_embeddings, dim=0).mean(dim=0)

safety_directions = mean_harmful_embeddings - mean_neutral_embeddings
os.makedirs("directions", exist_ok=True)
torch.save(safety_directions, f"directions/{args.model}_safety_direction_mlp.pt")