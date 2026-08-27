import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
logging.set_verbosity_error()

import os as _os, sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root != _os.path.dirname(_root) and not _os.path.exists(_os.path.join(_root, "utils.py")):
    _root = _os.path.dirname(_root)
_sys.path.insert(0, _root)
from utils import load_conversation_template, MODEL_DICT, top_k_neurons, filter_response

np.random.seed(20)
torch.manual_seed(20)
torch.cuda.manual_seed_all(20)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="llama3")
parser.add_argument("--percent", type=float, default=1.0)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = MODEL_DICT[args.model]
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

safety_direction = torch.load(f"directions/{args.model}_safety_direction_mlp.pt").to(device)
safety_direction = safety_direction / torch.norm(safety_direction, dim=-1, keepdim=True)

weight_contribution = []
for i, layer in enumerate(model.model.layers):
    down_proj = layer.mlp.down_proj.weight.detach().clone().permute(1, 0).contiguous()
    c = down_proj @ safety_direction[i].unsqueeze(-1)
    weight_contribution.append(c)

weight_contribution = torch.stack(weight_contribution, dim=0)

with open(f"responses/{args.model}.json", "r") as f:
    response_data = json.load(f)

harmful_prompts = [entry["Harmful Prompt"] for entry in response_data]
neutral_prompts = [entry["Neutral Prompt"] for entry in response_data]
harmful = [entry["Harmful Response"] for entry in response_data]
neutral = [entry["Neutral Response"] for entry in response_data]

filtered_harmful_prompt, _, filtered_harmful, _ = filter_response(harmful_prompts, neutral_prompts, harmful, neutral)

num_layers = model.config.num_hidden_layers
inter_size = model.config.intermediate_size
hidden_size = model.config.hidden_size

conv_template = load_conversation_template(args.model)
conv_template.append_message(conv_template.roles[0], None)
sys_prompt_len = len(tokenizer(conv_template.get_prompt()).input_ids)

mlp_activations = []
avg_contribution = torch.zeros((num_layers, inter_size)).to(device)

def capture_mlp_activation_hook(layer_id, sys_len):
    def hook_fn(module, input, output):
        neur_out = input[0].detach()[0, sys_len:, :]
        mlp_activations.append(neur_out * weight_contribution[layer_id].squeeze(-1))
    return hook_fn

for i, layer in enumerate(model.model.layers):
    layer.mlp.down_proj.register_forward_hook(capture_mlp_activation_hook(i, sys_prompt_len))

conv_template = load_conversation_template(args.model)
for harmfu_prompt, harmful_response in tqdm(zip(filtered_harmful_prompt, filtered_harmful), total=len(filtered_harmful)):
    
    conv_template.messages = []
    conv_template.append_message(conv_template.roles[0], f"{harmfu_prompt}")
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
    all_contributions = torch.stack(mlp_activations, dim=0)[:, start-1-sys_prompt_len: end-1-sys_prompt_len, :].mean(dim=1)
    
    mlp_activations = []

    with torch.no_grad():
        avg_contribution += all_contributions

total_neurons = num_layers * inter_size
avg_contribution = avg_contribution.detach().cpu().numpy() / len(filtered_harmful)
top_k_diff = top_k_neurons(avg_contribution, k=int(total_neurons * (args.percent / 100)))

print(len(top_k_diff))
# print([d[1] for d in top_k_diff])

mlp_neurons = [
    {"layer": layer, "neuron": neuron, "score": float(value)}
    for value, (layer, neuron) in top_k_diff
]

out_dir = "safety_neurons"
os.makedirs(out_dir, exist_ok=True)

with open(os.path.join(out_dir, f"{args.model}_{args.percent}.json"), "w") as f:
    json.dump(mlp_neurons, f, indent=2)

print(f"Saved top MLP neurons to {os.path.join(out_dir, f'{args.model}_{args.percent}.json')}") 