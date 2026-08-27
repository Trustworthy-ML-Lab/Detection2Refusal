import gc
import os
import argparse
import torch
import json

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import os as _os, sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root != _os.path.dirname(_root) and not _os.path.exists(_os.path.join(_root, "utils.py")):
    _root = _os.path.dirname(_root)
_sys.path.insert(0, _root)
from utils import MODEL_DICT, top_k_heads, filter_response, load_conversation_template

np.random.seed(20)
torch.manual_seed(20)
torch.cuda.manual_seed_all(20)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="llama3")
parser.add_argument("--percent", type=float, default=3.0, help="Percentage of heads to consider")
args = parser.parse_args()

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model_path = MODEL_DICT[args.model]
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
model.generation_config.do_sample = True
tokenizer.pad_token = tokenizer.eos_token

refusal_direction = torch.load(f"directions/{args.model}_refusal_direction_attn.pt").to(device)
print(refusal_direction.size())
refusal_direction = refusal_direction / torch.norm(refusal_direction, dim=-1, keepdim=True)

with open(f"responses/{args.model}.json", "r") as f:
    response_data = json.load(f)

harmful_prompts = [entry["Harmful Prompt"] for entry in response_data]
neutral_prompts = [entry["Neutral Prompt"] for entry in response_data]
harmful = [entry["Harmful Response"] for entry in response_data]
neutral = [entry["Neutral Response"] for entry in response_data]

filtered_harmful_prompt, _, filtered_harmful, _ = filter_response(harmful_prompts, neutral_prompts, harmful, neutral)

num_layers = model.config.num_hidden_layers
hidden_size = model.config.hidden_size
num_heads = model.config.num_attention_heads
head_dim = hidden_size // num_heads

attn_contribution = []

def capture_attn_contribution_hook():
    def hook_fn(module, input, output):
        total_head_dim = input[0].size(-1) 
        current_head_dim = total_head_dim // num_heads

        attn_out = input[0].detach()[0, :, :]
        attn_out = attn_out.reshape(attn_out.size(0), num_heads, current_head_dim) #(seq_len, num_heads, head_dim)
        o_proj = module.weight.detach().clone()
        o_proj = o_proj.reshape(hidden_size, num_heads, current_head_dim).permute(1, 2, 0).contiguous() #(num_heads, head_dim, hidden_size)
        attn_contribution.append(torch.einsum("snk,nkh->snh", attn_out, o_proj)) #(seq_len, num_heads, hidden_size)
    return hook_fn

for layer in model.model.layers:
    layer.self_attn.o_proj.register_forward_hook(capture_attn_contribution_hook())

conv_template = load_conversation_template(args.model)
avg_contribution = torch.zeros((num_layers, num_heads))

for harmful_prompt, harmful_response in tqdm(zip(filtered_harmful_prompt, filtered_harmful), total=len(filtered_harmful)):

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
    all_head_contributions = torch.stack(attn_contribution, dim=0)[:, start-1: end-1, :, :].mean(dim=1)

    refusal_dir_exp = refusal_direction.unsqueeze(1)
    dot_products = torch.einsum('lhd,ld->lh', all_head_contributions, refusal_direction)
    avg_contribution = np.add(avg_contribution, dot_products.cpu().numpy())

    attn_contribution = []
    

avg_contribution = np.asarray(avg_contribution) / len(filtered_harmful)
top_k_contributions = top_k_heads(avg_contribution, int(num_layers * num_heads * (args.percent / 100)))
print(top_k_contributions)
print([c[1] for c in top_k_contributions])

os.makedirs("refusal_heads", exist_ok=True)
with open(f"refusal_heads/{args.model}_{args.percent}.json", "w") as f:
    json.dump([{"layer": c[1][0], "head": c[1][1], "value": c[0].item()} for c in top_k_contributions], f, indent=2)

max_abs_value = np.abs(avg_contribution[:, :]).max()

os.makedirs("heat_map", exist_ok=True)
plt.figure(figsize=(12, 10))
plt.imshow(avg_contribution[:, :], cmap='coolwarm', aspect='auto', vmin=-max_abs_value, vmax=max_abs_value)
plt.colorbar(label='Average refusal contribution')
plt.title(f'Heatmap of Average refusal contribution')
plt.xlabel('Heads')
plt.ylabel('Layers')
plt.xticks(ticks=np.arange(num_heads), labels=[f'H{i}' for i in range(num_heads)], fontsize=6, rotation=45)
plt.yticks(ticks=np.arange(num_layers), labels=[f'L{i}' for i in range(num_layers)], fontsize=8)
plt.tight_layout()
plt.savefig(f'heat_map/{args.model}_refusal_attn.png', dpi=300)
plt.close()