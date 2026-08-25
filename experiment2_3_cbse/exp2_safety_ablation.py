"""
Exp 2: Full 4x4x4 = 64 config safety ablation.
Zero out detection heads, refusal heads, and safety neurons at
{0%, 1%, 3%, 5%} independently, measuring template-based safety
against GCG attacks.

Runs entirely in-memory: loads model once, modifies weights per config,
generates with HF, restores weights. No disk I/O per config.

Usage:
    python exp2_safety_ablation.py --model llama2
    python exp2_safety_ablation.py --model llama2 --n_data 50
"""

import argparse
import gc
import itertools
import json
import os

import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from utils import (
    MODEL_DICT,
    test_prefixes,
    load_conversation_template,
    merge_csv,
)

RESULTS_DIR = "emnlp_results/exp2_safety_ablation"
PERCENTS = [0.0, 1.0, 3.0, 5.0]


def load_heads(path):
    with open(path, "r") as f:
        return [(int(d["layer"]), int(d["head"])) for d in json.load(f)]


def load_neurons(path):
    with open(path, "r") as f:
        return [(int(m["layer"]), int(m["neuron"])) for m in json.load(f)]


def get_components(model_name, det_pct, ref_pct, neuron_pct):
    det_heads = []
    ref_heads = []
    neurons = []
    if det_pct > 0:
        det_heads = load_heads(
            f"find_detection_heads/detection_heads/{model_name}_{det_pct}.json"
        )
    if ref_pct > 0:
        ref_heads = load_heads(
            f"find_refusal_heads/refusal_heads/{model_name}_{ref_pct}.json"
        )
    if neuron_pct > 0:
        neurons = load_neurons(
            f"find_safety_neurons/safety_neurons/{model_name}_{neuron_pct}.json"
        )
    return det_heads, ref_heads, neurons


def apply_ablation(model, det_heads, ref_heads, neurons, head_dim):
    """Zero out specified heads and neurons. Returns backup for restoration."""
    backup = {}

    all_heads_to_zero = set()
    for l, h in det_heads:
        all_heads_to_zero.add((l, h))
    for l, h in ref_heads:
        all_heads_to_zero.add((l, h))

    for layer, head in all_heads_to_zero:
        key = ("head", layer, head)
        start = head * head_dim
        end = start + head_dim
        W = model.model.layers[layer].self_attn.o_proj.weight
        backup[key] = W[:, start:end].detach().clone()
        W.data[:, start:end] = 0.0

    for layer, neuron in neurons:
        key = ("neuron", layer, neuron)
        W = model.model.layers[layer].mlp.down_proj.weight
        backup[key] = W[:, neuron].detach().clone()
        W.data[:, neuron] = 0.0

    return backup


def restore_weights(model, backup, head_dim):
    """Restore weights from backup."""
    for key, saved in backup.items():
        if key[0] == "head":
            _, layer, head = key
            start = head * head_dim
            end = start + head_dim
            model.model.layers[layer].self_attn.o_proj.weight.data[:, start:end] = saved
        else:
            _, layer, neuron = key
            model.model.layers[layer].mlp.down_proj.weight.data[:, neuron] = saved


def evaluate_safety(model, tokenizer, model_name, prompts_text, gen_kwargs, batch_size=16):
    """Generate responses in batches and compute template safety."""
    responses = []
    model.eval()

    for i in tqdm(range(0, len(prompts_text), batch_size), desc="  Generating", leave=False):
        batch = prompts_text[i:i + batch_size]
        toks = tokenizer(
            batch, return_tensors="pt", padding=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **toks,
                pad_token_id=tokenizer.eos_token_id,
                **gen_kwargs,
            )

        for j, output in enumerate(outputs):
            input_len = toks["attention_mask"][j].sum().item()
            new_tokens = output[input_len:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            responses.append(response)

    safe_count = sum(
        1 for r in responses
        if any(pref in r for pref in test_prefixes)
    )
    return safe_count / len(responses)


def get_config_label(det_pct, ref_pct, neuron_pct):
    return f"d{det_pct:.0f}_r{ref_pct:.0f}_n{neuron_pct:.0f}"


def main():
    parser = argparse.ArgumentParser(description="Exp2: 64-config Safety Ablation")
    parser.add_argument("--model", type=str, default="llama2")
    parser.add_argument("--n_data", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing results CSV")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_path = f"{RESULTS_DIR}/safety_ablation_{args.model}.csv"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model once
    model_path = MODEL_DICT[args.model]
    print(f"Loading {args.model} from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )
    tokenizer.pad_token = tokenizer.eos_token

    num_heads = model.config.num_attention_heads
    hidden_size = model.config.hidden_size
    head_dim = getattr(model.config, "head_dim", hidden_size // num_heads)

    # Build generation kwargs from model's config
    gen_kwargs = {"max_new_tokens": 100, "do_sample": True}
    try:
        from transformers import GenerationConfig
        gen_cfg = GenerationConfig.from_pretrained(model_path)
        if hasattr(gen_cfg, "temperature") and gen_cfg.temperature is not None:
            gen_kwargs["temperature"] = gen_cfg.temperature
        if hasattr(gen_cfg, "top_p") and gen_cfg.top_p is not None:
            gen_kwargs["top_p"] = gen_cfg.top_p
        if hasattr(gen_cfg, "top_k") and gen_cfg.top_k is not None and gen_cfg.top_k > 0:
            gen_kwargs["top_k"] = gen_cfg.top_k
        if hasattr(gen_cfg, "repetition_penalty") and gen_cfg.repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = gen_cfg.repetition_penalty
    except Exception:
        gen_kwargs["temperature"] = 0.6
        gen_kwargs["top_p"] = 0.95
    print(f"Generation config: {gen_kwargs}")

    # Load GCG prompts
    gcg_path = f"data/gcg_results/{args.model}/1.0_1.0/"
    gcg_df = merge_csv(gcg_path)
    gcg_goals = gcg_df["prompt"].tolist()[:args.n_data]
    gcg_advs = gcg_df["adv"].tolist()[:args.n_data]
    print(f"Loaded {len(gcg_goals)} GCG prompts")

    # Format prompts as text strings (batched tokenization happens in evaluate_safety)
    tokenizer.padding_side = "left"
    conv = load_conversation_template(args.model)
    prompts_text = []
    for goal, adv in zip(gcg_goals, gcg_advs):
        conv.messages = []
        conv.append_message(conv.roles[0], f"{goal} {adv}")
        conv.append_message(conv.roles[1], None)
        prompts_text.append(conv.get_prompt())

    # Check for existing results to resume
    completed = set()
    results = []
    if args.resume and os.path.exists(result_path):
        existing_df = pd.read_csv(result_path)
        for _, row in existing_df.iterrows():
            completed.add(get_config_label(row["det_percent"], row["ref_percent"], row["neuron_percent"]))
        results = existing_df.to_dict("records")
        print(f"Resuming: {len(completed)} configs already completed")

    # Run all 64 configs
    configs = list(itertools.product(PERCENTS, PERCENTS, PERCENTS))
    print(f"Total configs: {len(configs)}")

    for i, (det_pct, ref_pct, neuron_pct) in enumerate(configs):
        label = get_config_label(det_pct, ref_pct, neuron_pct)

        if label in completed:
            print(f"[{i+1}/{len(configs)}] SKIP {label}")
            continue

        print(f"\n[{i+1}/{len(configs)}] {args.model} | det={det_pct}% ref={ref_pct}% neuron={neuron_pct}%")

        if det_pct == 0 and ref_pct == 0 and neuron_pct == 0:
            # Baseline: no modification
            safety = evaluate_safety(model, tokenizer, args.model, prompts_text, gen_kwargs, args.batch_size)
        else:
            det_heads, ref_heads, neurons = get_components(
                args.model, det_pct, ref_pct, neuron_pct
            )
            backup = apply_ablation(model, det_heads, ref_heads, neurons, head_dim)
            safety = evaluate_safety(model, tokenizer, args.model, prompts_text, gen_kwargs, args.batch_size)
            restore_weights(model, backup, head_dim)
            del backup

        print(f"  Template safety: {safety:.1%}")

        results.append({
            "model": args.model,
            "det_percent": det_pct,
            "ref_percent": ref_pct,
            "neuron_percent": neuron_pct,
            "label": label,
            "template_safety": round(safety, 4),
        })

        # Save after each config
        df = pd.DataFrame(results)
        df.to_csv(result_path, index=False)

    # Final summary
    df = pd.DataFrame(results)
    print(f"\n{'='*60}")
    print(f"SUMMARY — 64-config Safety Ablation ({args.model})")
    print(f"{'='*60}")
    print(df.to_string(index=False))
    print(f"\nSaved to {result_path}")


if __name__ == "__main__":
    main()
