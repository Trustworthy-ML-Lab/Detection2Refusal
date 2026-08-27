"""
Zero out detection heads at varying percentages and measure
average refusal contribution of top-3% refusal heads.

For each model, zero out detection heads at 1/3/5/7/10%, then run a
forward pass on harmful prompts and compute how much the top-3% refusal
heads contribute to the refusal direction. Reports one number per
(model, zero-out %) pair.

Usage:
    python experiments/refusal_contribution.py --model llama2
    python experiments/refusal_contribution.py --model llama2 --det_percents 1.0 3.0 5.0 7.0 10.0
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import os as _os, sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root != _os.path.dirname(_root) and not _os.path.exists(_os.path.join(_root, "utils.py")):
    _root = _os.path.dirname(_root)
_sys.path.insert(0, _root)
from utils import MODEL_DICT, load_conversation_template, filter_response

np.random.seed(20)
torch.manual_seed(20)
torch.cuda.manual_seed_all(20)

RESULTS_DIR = "results/refusal_contribution"
REFUSAL_HEAD_PERCENT = 3.0


def load_detection_heads(model_name, percent):
    path = f"scripts/find_detection_heads/detection_heads/{model_name}_{percent}.json"
    with open(path, "r") as f:
        det_list = json.load(f)
    return [(int(d["layer"]), int(d["head"])) for d in det_list]


def load_refusal_heads(model_name, percent):
    path = f"scripts/find_refusal_heads/refusal_heads/{model_name}_{percent}.json"
    with open(path, "r") as f:
        ref_list = json.load(f)
    return [(int(d["layer"]), int(d["head"])) for d in ref_list]


def run_experiment(model_name, det_percent, device):
    """Zero out detection heads at det_percent, measure refusal contribution."""
    model_path = MODEL_DICT[model_name]

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )
    tokenizer.pad_token = tokenizer.eos_token

    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_dim = getattr(model.config, "head_dim", hidden_size // num_heads)

    # Load refusal direction
    refusal_direction = torch.load(
        f"scripts/find_refusal_heads/directions/{model_name}_refusal_direction_attn.pt"
    ).to(device)
    refusal_direction = refusal_direction / torch.norm(refusal_direction, dim=-1, keepdim=True)

    # Load prompts
    with open(f"scripts/find_refusal_heads/responses/{model_name}.json", "r") as f:
        response_data = json.load(f)

    harmful_prompts = [entry["Harmful Prompt"] for entry in response_data]
    neutral_prompts = [entry["Neutral Prompt"] for entry in response_data]
    harmful = [entry["Harmful Response"] for entry in response_data]
    neutral = [entry["Neutral Response"] for entry in response_data]

    filtered_harmful_prompt, _, filtered_harmful, _ = filter_response(
        harmful_prompts, neutral_prompts, harmful, neutral
    )

    # Zero out detection heads (factor=0)
    if det_percent > 0:
        detection_heads = load_detection_heads(model_name, det_percent)
        for layer_idx, head_idx in detection_heads:
            start = head_idx * head_dim
            end = start + head_dim
            W = model.model.layers[layer_idx].self_attn.o_proj.weight.detach().clone()
            W[:, start:end] *= 0.0
            model.model.layers[layer_idx].self_attn.o_proj.weight = nn.Parameter(W)
        print(f"  Zeroed out {len(detection_heads)} detection heads ({det_percent}%)")
    else:
        print(f"  No detection heads zeroed (baseline)")

    # Load refusal heads (top 3%) to know which heads to measure
    refusal_heads = load_refusal_heads(model_name, REFUSAL_HEAD_PERCENT)
    refusal_head_set = set(refusal_heads)

    # Register hooks to capture attention contributions
    attn_contribution = []

    def capture_attn_contribution_hook():
        def hook_fn(module, input, output):
            attn_out = input[0].detach()[0, :, :]
            total_head_dim = attn_out.size(-1)
            current_head_dim = total_head_dim // num_heads
            attn_out = attn_out.reshape(attn_out.size(0), num_heads, current_head_dim)
            o_proj = module.weight.detach().clone()
            o_proj = o_proj.reshape(hidden_size, num_heads, current_head_dim).permute(1, 2, 0).contiguous()
            attn_contribution.append(torch.einsum("snk,nkh->snh", attn_out, o_proj))
        return hook_fn

    hooks = []
    for layer in model.model.layers:
        h = layer.self_attn.o_proj.register_forward_hook(capture_attn_contribution_hook())
        hooks.append(h)

    conv_template = load_conversation_template(model_name)

    # Compute average contribution of refusal heads to refusal direction
    total_refusal_contribution = 0.0
    n_samples = 0

    for harmful_prompt, harmful_response in tqdm(
        zip(filtered_harmful_prompt, filtered_harmful),
        total=len(filtered_harmful),
        desc=f"  {model_name} det_zero={det_percent}%"
    ):
        conv_template.messages = []
        conv_template.append_message(conv_template.roles[0], f"{harmful_prompt}")
        conv_template.append_message(conv_template.roles[1], None)

        toks = tokenizer(conv_template.get_prompt()).input_ids
        start_pos = len(toks)

        conv_template.update_last_message(f"{harmful_response}")
        toks = tokenizer(conv_template.get_prompt()).input_ids

        if "qwen" in model_name:
            end_pos = len(toks) - 2
        elif "llama3" in model_name:
            end_pos = len(toks) - 1
        else:
            end_pos = len(toks)

        toks_tensor = tokenizer(conv_template.get_prompt(), return_tensors="pt")
        with torch.no_grad():
            _ = model(
                input_ids=toks_tensor['input_ids'].to(device),
                attention_mask=toks_tensor['attention_mask'].to(device)
            )

        # Stack contributions: (num_layers, seq_len, num_heads, hidden_size)
        all_head_contributions = torch.stack(attn_contribution, dim=0)
        # Average over response token positions
        all_head_contributions = all_head_contributions[:, start_pos-1:end_pos-1, :, :].mean(dim=1)

        # Dot product with refusal direction: (num_layers, num_heads)
        dot_products = torch.einsum('lhd,ld->lh', all_head_contributions, refusal_direction)

        # Average contribution of refusal heads only
        refusal_contributions = []
        for (l, h) in refusal_heads:
            refusal_contributions.append(dot_products[l, h].item())

        total_refusal_contribution += np.mean(refusal_contributions)
        n_samples += 1
        attn_contribution = []

    for h in hooks:
        h.remove()

    avg_refusal_contrib = total_refusal_contribution / n_samples
    return avg_refusal_contrib


def main():
    parser = argparse.ArgumentParser(description="Exp1: Refusal Contribution vs Detection Head Zero-out")
    parser.add_argument("--model", type=str, default="llama2")
    parser.add_argument("--det_percents", type=float, nargs="+",
                        default=[0.0, 1.0, 3.0, 5.0, 7.0, 10.0])
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = []
    for det_pct in args.det_percents:
        print(f"\n{'='*60}")
        print(f"Model: {args.model} | Zero-out detection heads: {det_pct}%")
        print(f"{'='*60}")

        avg_contrib = run_experiment(args.model, det_pct, device)
        print(f"  Avg refusal contribution (top {REFUSAL_HEAD_PERCENT}% refusal heads): {avg_contrib:.6f}")

        results.append({
            "model": args.model,
            "det_zero_percent": det_pct,
            "avg_refusal_contribution": round(avg_contrib, 6),
        })

        # Free GPU memory between runs
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    df = pd.DataFrame(results)
    save_path = f"{RESULTS_DIR}/refusal_contribution_{args.model}.csv"
    df.to_csv(save_path, index=False)

    print(f"\n{'='*60}")
    print(f"RESULTS — Refusal Contribution vs Detection Head Zero-out")
    print(f"{'='*60}")
    print(df.to_string(index=False))
    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()
