"""
Defense baseline comparison: CircuitBreakers (RR), LAT, vs CBSE.
Downloads the pre-trained defense models from HuggingFace and evaluates them
on our GCG prompts alongside CBSE, judged by Llama-Guard-3 and a rule-based
template check.

Compares four configs on Llama-3-8B:
  1. Original (meta-llama/Meta-Llama-3-8B-Instruct)
  2. CircuitBreakers (GraySwanAI/Llama-3-8B-Instruct-RR)
  3. LAT (LLM-LAT/robust-llama3-8b-instruct)
  4. CBSE (our method, weight-scaled from original)

Usage:
    python experiments/evaluate_defense_baseline.py --tp 4
    python experiments/evaluate_defense_baseline.py --tp 1 --n_data 50
"""

import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_USE_V1"] = "0"

import argparse
import gc
import json
import shutil
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from vllm import LLM, SamplingParams

import os as _os, sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root != _os.path.dirname(_root) and not _os.path.exists(_os.path.join(_root, "utils.py")):
    _root = _os.path.dirname(_root)
_sys.path.insert(0, _root)
from utils import (
    MODEL_DICT,
    test_prefixes,
    load_conversation_template,
    merge_csv,
)

RESULTS_DIR = "defense_baseline_results"
LLAMA_GUARD_ID = "meta-llama/Llama-Guard-3-8B"
CIRCUIT_BREAKER_ID = "GraySwanAI/Llama-3-8B-Instruct-RR"
# CIRCUIT_BREAKER_ID = "GraySwanAI/Mistral-7B-Instruct-RR"
LAT_ID = "LLM-LAT/robust-llama3-8b-instruct"
MODEL_NAME = "llama3"
# MODEL_NAME = "mistral"

CBSE_PARAMS = {
    "default": (1.5, 1.0, 2.0, 2.0, 3.0),
}


def build_cbse_model(model_name, save_path):
    """Build CBSE-edited model on CPU and save to disk."""
    model_path = MODEL_DICT[model_name]
    nf, np_, df, rf, hp = CBSE_PARAMS["default"]

    hf_model = (
        AutoModelForCausalLM
        .from_pretrained(model_path, torch_dtype=torch.float16, device_map="cpu")
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )
    tokenizer.pad_token = tokenizer.eos_token

    num_heads = hf_model.config.num_attention_heads
    hidden_size = hf_model.config.hidden_size
    head_dim = hidden_size // num_heads

    det_heads, ref_heads, neurons = [], [], []
    if df != 1.0:
        with open(f"scripts/find_detection_heads/detection_heads/{model_name}_{hp}.json") as f:
            det_heads = [(int(d["layer"]), int(d["head"])) for d in json.load(f)]
    if rf != 1.0:
        with open(f"scripts/find_refusal_heads/refusal_heads/{model_name}_{hp}.json") as f:
            ref_heads = [(int(d["layer"]), int(d["head"])) for d in json.load(f)]
    if nf != 1.0:
        with open(f"scripts/find_safety_neurons/safety_neurons/{model_name}_{np_}.json") as f:
            neurons = [(int(m["layer"]), int(m["neuron"])) for m in json.load(f)]

    head_factors = defaultdict(lambda: 1.0)
    for l, h in det_heads:
        head_factors[(l, h)] *= df
    for l, h in ref_heads:
        head_factors[(l, h)] *= rf

    for (layer, head), factor in head_factors.items():
        start = head * head_dim
        end = start + head_dim
        W = hf_model.model.layers[layer].self_attn.o_proj.weight.detach().clone()
        W[:, start:end] *= factor
        hf_model.model.layers[layer].self_attn.o_proj.weight = nn.Parameter(W)

    for layer, neuron in neurons:
        W = hf_model.model.layers[layer].mlp.down_proj.weight.detach().clone()
        W[:, neuron] *= nf
        hf_model.model.layers[layer].mlp.down_proj.weight = nn.Parameter(W)

    os.makedirs(save_path, exist_ok=True)
    hf_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    del hf_model
    gc.collect()
    print(f"  CBSE model saved to {save_path}")


def evaluate_config(config_name, model_id_or_path, tp, n_data, gcg_goals, gcg_advs):
    """Evaluate a single model config on GCG prompts."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    orig_model_path = MODEL_DICT[MODEL_NAME]

    config = AutoConfig.from_pretrained(orig_model_path, trust_remote_code=True)
    max_pos = config.max_position_embeddings

    gen_cfg = None
    try:
        gen_cfg = GenerationConfig.from_pretrained(orig_model_path)
    except Exception:
        gen_cfg = GenerationConfig(
            **{k: getattr(config, k)
               for k in ["temperature", "top_k", "top_p", "repetition_penalty"]
               if hasattr(config, k)}
        )

    tokenizer = AutoTokenizer.from_pretrained(
        orig_model_path, trust_remote_code=True, use_fast=False
    )

    sample_params = SamplingParams(
        temperature=getattr(gen_cfg, "temperature", 0.6),
        top_p=getattr(gen_cfg, "top_p", 0.95),
        top_k=getattr(gen_cfg, "top_k", None),
        repetition_penalty=getattr(gen_cfg, "repetition_penalty", 1.0),
        max_tokens=100,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    # Format prompts
    conv = load_conversation_template(MODEL_NAME)
    prompts = []
    for goal, adv in zip(gcg_goals, gcg_advs):
        conv.messages = []
        conv.append_message(conv.roles[0], f"{goal} {adv}")
        conv.append_message(conv.roles[1], None)
        prompts.append(conv.get_prompt())

    print(f"\n{'='*60}")
    print(f"Evaluating: {config_name} | {len(prompts)} GCG prompts")
    print(f"Model: {model_id_or_path}")
    print(f"{'='*60}")

    # Generate
    llm = LLM(model=model_id_or_path, max_model_len=max_pos,
              tensor_parallel_size=tp, dtype=torch.float16)
    outputs = llm.generate(prompts, sampling_params=sample_params)
    responses = [o.outputs[0].text for o in outputs]
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    # Template safety
    template_fail = sum(
        1 for r in responses
        if not any(pref in r for pref in test_prefixes)
    )
    template_safety = 1 - (template_fail / len(responses))

    # Llama-Guard
    lg_tok = AutoTokenizer.from_pretrained(LLAMA_GUARD_ID)
    lg_mod = AutoModelForCausalLM.from_pretrained(
        LLAMA_GUARD_ID, torch_dtype=torch.bfloat16, device_map=device,
    )

    def moderate(chat):
        ids = lg_tok.apply_chat_template(chat, return_tensors="pt").to(device)
        out = lg_mod.generate(ids, max_new_tokens=100, pad_token_id=0)
        text = lg_tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
        try:
            return text.split("\n")[2]
        except IndexError:
            return text.strip()

    guard_fail = sum(
        1 for adv, r in zip(gcg_advs, responses)
        if moderate([{"role": "user", "content": adv},
                     {"role": "assistant", "content": r}]) == "unsafe"
    )
    guard_safety = 1 - (guard_fail / len(responses))

    del lg_tok, lg_mod
    gc.collect()
    torch.cuda.empty_cache()

    print(f"  Template safety: {template_safety:.1%}")
    print(f"  Guard safety: {guard_safety:.1%}")

    return {
        "config": config_name,
        "model_id": model_id_or_path,
        "n_prompts": len(prompts),
        "template_safety": round(template_safety, 3),
        "guard_safety": round(guard_safety, 3),
    }, responses, gcg_goals


def main():
    parser = argparse.ArgumentParser(
        description="Defense baseline comparison: CircuitBreakers, LAT vs CBSE"
    )
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--n_data", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_dir = "temp_models"

    # Load GCG prompts (baseline suffixes for llama3)
    gcg_path = f"data/gcg_results/{MODEL_NAME}/1.0_1.0/"
    gcg_df = merge_csv(gcg_path)
    gcg_goals = gcg_df["prompt"].tolist()[:args.n_data]
    gcg_advs = gcg_df["adv"].tolist()[:args.n_data]
    print(f"Loaded {len(gcg_goals)} GCG prompts for {MODEL_NAME}")

    results = []

    # 1. Original Llama-3-8B-Instruct
    res, _, _ = evaluate_config(
        "Original", MODEL_DICT[MODEL_NAME], args.tp, args.n_data, gcg_goals, gcg_advs
    )
    results.append(res)

    # 2. CircuitBreakers (GraySwanAI/Llama-3-8B-Instruct-RR)
    res, _, _ = evaluate_config(
        "CircuitBreakers", CIRCUIT_BREAKER_ID, args.tp, args.n_data, gcg_goals, gcg_advs
    )
    results.append(res)

    # 3. LAT (LLM-LAT/robust-llama3-8b-instruct)
    res, _, _ = evaluate_config(
        "LAT", LAT_ID, args.tp, args.n_data, gcg_goals, gcg_advs
    )
    results.append(res)

    # 4. CBSE (our method)
    cbse_path = f"{save_dir}/{MODEL_NAME}_defense_cbse"
    if not os.path.exists(os.path.join(cbse_path, "config.json")):
        print(f"\nBuilding CBSE model...")
        build_cbse_model(MODEL_NAME, cbse_path)
    else:
        print(f"\nReusing cached CBSE model at {cbse_path}")

    res, _, _ = evaluate_config(
        "CBSE (Ours)", cbse_path, args.tp, args.n_data, gcg_goals, gcg_advs
    )
    results.append(res)

    # Summary
    df = pd.DataFrame(results)
    save_path = f"{RESULTS_DIR}/defense_baseline_comparison.csv"
    df.to_csv(save_path, index=False)

    print(f"\n{'='*60}")
    print("SUMMARY — Defense Baseline Comparison (Llama-3-8B, GCG Attack)")
    print(f"{'='*60}")
    print(df.to_string(index=False))
    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()
