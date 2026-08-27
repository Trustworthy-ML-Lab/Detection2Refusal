import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer
import os as _os, sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root != _os.path.dirname(_root) and not _os.path.exists(_os.path.join(_root, "utils.py")):
    _root = _os.path.dirname(_root)
_sys.path.insert(0, _root)
from utils import load_conversation_template, MODEL_DICT, top_k_heads
from transformers.utils import logging
logging.set_verbosity_error() 

np.random.seed(20)
torch.manual_seed(20)
torch.cuda.manual_seed_all(20)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="llama3")
parser.add_argument("--percent", type=float, default=3.0)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = MODEL_DICT[args.model]
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device).eval()
target_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
target_tokenizer.pad_token = target_tokenizer.eos_token

num_layers = model.config.num_hidden_layers
num_heads = model.config.num_attention_heads
head_dim = model.config.hidden_size // num_heads
num_total_heads = num_layers * num_heads

# Each harmful-neutral prompt differs in only one word
probing_data = pd.read_csv("../../data/filtered_harmful_neutral_prompts.csv")
harmful_prompts = probing_data['Harmful Prompt'].tolist()
neutral_prompts = probing_data['Neutral Prompt'].tolist()

print("target model:", args.model)
conv_template = load_conversation_template(args.model)

# pre-allocate on GPU
all_diff = torch.zeros(num_layers, num_heads, device=device)

for h, n in zip(harmful_prompts, neutral_prompts):
    conv_template.messages = []
    conv_template.append_message(conv_template.roles[0], f"{h}")
    conv_template.append_message(conv_template.roles[1], None)
    with torch.no_grad():
        toks_h = target_tokenizer(conv_template.get_prompt(), return_tensors="pt")
        h_attn_weights = model(input_ids=toks_h['input_ids'].to(device), attention_mask=toks_h['attention_mask'].to(device), output_attentions=True).attentions                

    conv_template.messages = []
    conv_template.append_message(conv_template.roles[0], f"{n}")
    conv_template.append_message(conv_template.roles[1], None)
    with torch.no_grad():
        toks_n = target_tokenizer(conv_template.get_prompt(), return_tensors="pt")
        n_attn_weights = model(input_ids=toks_n['input_ids'].to(device), attention_mask=toks_n['attention_mask'].to(device), output_attentions=True).attentions                

    # stack into tensors of shape [num_layers, batch=1, heads, seq_len, seq_len]
    h_stack = torch.stack(h_attn_weights, dim=0)
    n_stack = torch.stack(n_attn_weights, dim=0)

    # pick out the last query position: result shape [num_layers, heads, seq_len]
    h_last = h_stack[:, 0, :, -1, :]
    n_last = n_stack[:, 0, :, -1, :]

    # find positions where the two token sequences differ
    diff_pos = (toks_h['input_ids'][0] != toks_n['input_ids'][0]).nonzero(as_tuple=True)[0].to(device)

    # gather only at those positions and average over them
    # h_sel/n_sel: [num_layers, heads, num_diff_positions]
    h_sel = h_last.index_select(-1, diff_pos)
    n_sel = n_last.index_select(-1, diff_pos)

    # compute per-layer/head mean difference: shape [num_layers, heads]
    avg_diff = h_sel.mean(dim=-1) - n_sel.mean(dim=-1)
    
    all_diff += avg_diff


all_diff /= len(harmful_prompts)
all_diff = all_diff.cpu().numpy()

top_k_diff = top_k_heads(all_diff, k=int(num_total_heads * (args.percent / 100)))
print(top_k_diff)
print([d[1] for d in top_k_diff])

detection_heads = [
    {"layer": layer, "head": head, "score": float(value)}
    for value, (layer, head) in top_k_diff
]

out_dir = "detection_heads"
os.makedirs(out_dir, exist_ok=True)
with open(f"{out_dir}/{args.model}_{args.percent}.json", "w") as f:
    json.dump(detection_heads, f, indent=2)

print(f"Saved {len(detection_heads)} detection heads to detection_heads/{args.model}_{args.percent}.json")

all_diff = np.clip(all_diff, 0, None)  # Sets negative values to 0

max_value = all_diff.max()
print(f"Max value: {max_value}")

os.makedirs("heat_map", exist_ok=True)

plt.figure(figsize=(12, 10))  # Adjusting figure size for 32 heads
plt.imshow(all_diff, cmap='Reds', aspect='auto', vmin=0, vmax=max_value)  # Use only warm colors
plt.colorbar(label='Average attn difference')
plt.title('Heatmap of Average attn difference (Layers vs Heads)')
plt.xlabel('Heads')
plt.ylabel('Layers')

plt.xticks(ticks=np.arange(num_heads), labels=[f'H{i}' for i in range(num_heads)], fontsize=6, rotation=45)
plt.yticks(ticks=np.arange(num_layers), labels=[f'L{i}' for i in range(num_layers)], fontsize=8)

plt.tight_layout()
plt.savefig(f'heat_map/{args.model}_{args.percent}_avg_attn_diff_heatmap.png', dpi=300)
plt.close()