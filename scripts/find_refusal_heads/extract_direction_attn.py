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

json_file_path = f"responses/{args.model}.json"
with open(json_file_path, "r") as f:
    response_data = json.load(f)

harmful_prompts = [entry["Harmful Prompt"] for entry in response_data]
neutral_prompts = [entry["Neutral Prompt"] for entry in response_data]
harmful = [entry["Harmful Response"] for entry in response_data]
neutral = [entry["Neutral Response"] for entry in response_data]

# Filter out responses that models refused to generate under the neutral prompt
filtered_harmful_prompt, filtered_neutral_prompt, filtered_harmful, filtered_neutral = filter_response(harmful_prompts, neutral_prompts, harmful, neutral)
print("#Filtered Harmful Prompts", len(filtered_harmful_prompt))
print("#Filtered Neutral Prompts", len(filtered_neutral_prompt))
print("#Filtered Harmful", len(filtered_harmful))
print("#Filtered Neutral", len(filtered_neutral))

residual_outputs = []
def capture_residual_hook():
    def hook_fn(module, input, output):
        residual_outputs.append(input[0].detach()[0, :, :])
    return hook_fn

print(f"Loading model {args.model}...")
model_path = MODEL_DICT[args.model]
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
model.generation_config.do_sample = True
tokenizer.pad_token = tokenizer.eos_token

for layer in model.model.layers:
    layer.post_attention_layernorm.register_forward_hook(capture_residual_hook())

conv_template = load_conversation_template(args.model)

harmful_embeddings = []
neutral_embeddings = []

for neutral_prompt, harmful_prompt, neutral_response, harmful_response in tqdm(zip(filtered_neutral_prompt, filtered_harmful_prompt, filtered_neutral, filtered_harmful), total=len(filtered_neutral_prompt)):
    
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
    _ = model(input_ids=toks['input_ids'].to(device), attention_mask=toks['attention_mask'].to(device))
    neutral_embeddings.append(torch.stack(residual_outputs, dim=0)[:, start-1: end-1, :].mean(dim=1).cpu())
    residual_outputs = []

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
    _ = model(input_ids=toks['input_ids'].to(device), attention_mask=toks['attention_mask'].to(device))
    harmful_embeddings.append(torch.stack(residual_outputs, dim=0)[:, start-1: end-1, :].mean(dim=1).cpu())
    residual_outputs = []


mean_harmful_embedding = torch.stack(harmful_embeddings, dim=0).mean(dim=0)
mean_neutral_embedding = torch.stack(neutral_embeddings, dim=0).mean(dim=0)

refusal_direction = mean_harmful_embedding - mean_neutral_embedding
os.makedirs("directions", exist_ok=True)
torch.save(refusal_direction, f"directions/{args.model}_refusal_direction_attn.pt")