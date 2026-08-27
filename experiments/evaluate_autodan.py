"""
AutoDAN attack: evaluate CBSE against pre-generated AutoDAN jailbreak prompts.

Uses pre-generated AutoDAN attack prompts (from prior AutoDAN-HGA runs).
The AutoDAN prompt is a complete jailbreak instruction that wraps the harmful
goal, so we combine goal + AutoDAN suffix using the conversation template.

Pipeline:
  1. Load AutoDAN responses JSON (original model responses already available)
  2. Apply CBSE to model, re-generate responses against same AutoDAN prompts
  3. Evaluate with template matching + Llama-Guard
  4. Compare original vs CBSE ASR

Usage:
    python experiments/evaluate_autodan.py --model llama2 --autodan_path data/autodan/llama2_responses.json --tp 4
"""

import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_USE_V1"] = "0"

import argparse
import gc
import json
import os
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

import os as _os, sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root != _os.path.dirname(_root) and not _os.path.exists(_os.path.join(_root, "utils.py")):
    _root = _os.path.dirname(_root)
_sys.path.insert(0, _root)
from utils import MODEL_DICT, test_prefixes, load_conversation_template

RESULTS_DIR = "autodan_results"
LLAMA_GUARD_ID = "meta-llama/Llama-Guard-3-8B"

CBSE_PARAMS = {
    "default": (1.5, 1.0, 2.0, 2.0, 3.0),
}


def load_autodan_data(path):
    """Load AutoDAN responses JSON. Extract goals, targets, and adv prompts."""
    with open(path) as f:
        data = json.load(f)

    goals, targets, adv_prompts, orig_responses = [], [], [], []
    for key in sorted(data.keys(), key=int):
        entry = data[key][0]  # Each key has a list of 1 entry
        goals.append(entry["goal"])
        targets.append(entry["target"])
        adv_prompts.append(entry["prompt"])
        orig_responses.append(entry["response"])

    return goals, targets, adv_prompts, orig_responses


def apply_cbse(model_name, model):
    """Apply CBSE weight scaling intervention to a loaded HF model."""
    num_heads = model.config.num_attention_heads
    hidden_size = model.config.hidden_size
    head_dim = getattr(model.config, "head_dim", hidden_size // num_heads)

    nf, np_, df, rf, hp = CBSE_PARAMS["default"]

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
        W = model.model.layers[layer].self_attn.o_proj.weight.detach().clone()
        W[:, start:end] *= factor
        model.model.layers[layer].self_attn.o_proj.weight = nn.Parameter(W)

    for layer, neuron in neurons:
        W = model.model.layers[layer].mlp.down_proj.weight.detach().clone()
        W[:, neuron] *= nf
        model.model.layers[layer].mlp.down_proj.weight = nn.Parameter(W)

    print(f"  Applied CBSE: {len(det_heads)} det heads (x{df}), "
          f"{len(ref_heads)} ref heads (x{rf}), {len(neurons)} neurons (x{nf})")


def format_autodan_prompts(model_name, tokenizer, goals, targets, adv_prompts):
    """Format AutoDAN prompts using conversation template.

    AutoDAN's adv_string replaces [REPLACE] with the goal, then wraps in chat template.
    We replicate the autodan_SuffixManager behavior: goal is the instruction,
    adv_string is prepended/appended as the jailbreak wrapper.
    """
    conv = load_conversation_template(model_name)
    formatted = []
    for goal, target, adv in zip(goals, targets, adv_prompts):
        conv.messages = []
        # AutoDAN prompt already contains the reformulated instruction
        # The goal is embedded via [REPLACE] in the adv string
        full_prompt = f"{adv} {goal} {target}"
        conv.append_message(conv.roles[0], full_prompt)
        conv.append_message(conv.roles[1], None)
        formatted.append(conv.get_prompt())
    return formatted


def evaluate_model(model_name, autodan_path, tp=4):
    """Evaluate original + CBSE against AutoDAN prompts."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = MODEL_DICT[model_name]

    goals, targets, adv_prompts, orig_responses = load_autodan_data(autodan_path)
    n_prompts = len(goals)
    print(f"Loaded {n_prompts} AutoDAN prompts from {autodan_path}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = []

    # ---- Evaluate original model (from saved responses) ----
    print(f"\n{'='*60}")
    print(f"{model_name} | original | {n_prompts} prompts (from saved responses)")
    print(f"{'='*60}")

    template_jailbroken = sum(
        1 for r in orig_responses
        if not any(pref in r for pref in test_prefixes)
    )
    template_asr = template_jailbroken / n_prompts
    print(f"  Template ASR: {template_jailbroken}/{n_prompts} = {template_asr:.1%}")

    # Llama-Guard on original responses
    print("  Loading Llama-Guard-3-8B for original responses...")
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

    guard_unsafe_orig = 0
    for goal, resp in tqdm(zip(goals, orig_responses), total=n_prompts,
                           desc=f"Llama-Guard {model_name}/original"):
        verdict = moderate([
            {"role": "user", "content": goal},
            {"role": "assistant", "content": resp},
        ])
        if verdict == "unsafe":
            guard_unsafe_orig += 1

    guard_asr_orig = guard_unsafe_orig / n_prompts
    print(f"  Guard ASR: {guard_unsafe_orig}/{n_prompts} = {guard_asr_orig:.1%}")

    del lg_tok, lg_mod
    gc.collect()
    torch.cuda.empty_cache()

    results.append({
        "model": model_name,
        "config": "original",
        "attack": "autodan",
        "n_prompts": n_prompts,
        "template_jailbroken": template_jailbroken,
        "template_asr": round(template_asr, 3),
        "guard_unsafe": guard_unsafe_orig,
        "guard_asr": round(guard_asr_orig, 3),
    })

    # ---- Generate CBSE responses ----
    print(f"\n{'='*60}")
    print(f"{model_name} | CBSE | {n_prompts} prompts")
    print(f"{'='*60}")

    from vllm import LLM, SamplingParams

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    max_pos = config.max_position_embeddings

    gen_cfg = None
    try:
        gen_cfg = GenerationConfig.from_pretrained(model_path)
    except Exception:
        gen_cfg = GenerationConfig(
            **{k: getattr(config, k)
               for k in ["temperature", "top_k", "top_p", "repetition_penalty"]
               if hasattr(config, k)}
        )

    sample_params = SamplingParams(
        temperature=getattr(gen_cfg, "temperature", 0.6),
        top_p=getattr(gen_cfg, "top_p", 0.95),
        top_k=getattr(gen_cfg, "top_k", None),
        repetition_penalty=getattr(gen_cfg, "repetition_penalty", 1.0),
        max_tokens=200,
    )

    # Apply CBSE (load on CPU to avoid CUDA fork issues with vLLM tp>1)
    save_dir = "temp_models"
    temp_path = f"{save_dir}/{model_name}_autodan_cbse"

    if not os.path.exists(os.path.join(temp_path, "config.json")):
        print(f"  Building CBSE model at {temp_path}...")
        hf_model = (
            AutoModelForCausalLM
            .from_pretrained(model_path, torch_dtype=torch.float16, device_map="cpu")
            .eval()
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )
        tokenizer.pad_token = tokenizer.eos_token
        apply_cbse(model_name, hf_model)
        os.makedirs(temp_path, exist_ok=True)
        hf_model.save_pretrained(temp_path)
        tokenizer.save_pretrained(temp_path)
        del hf_model
        gc.collect()
    else:
        print(f"  Reusing cached CBSE model at {temp_path}")

    llm = LLM(model=temp_path, max_model_len=max_pos,
              tensor_parallel_size=tp, dtype=torch.float16)

    # Format prompts
    formatted = format_autodan_prompts(model_name, tokenizer, goals, targets, adv_prompts)

    outputs = llm.generate(formatted, sampling_params=sample_params)
    cbse_responses = [o.outputs[0].text for o in outputs]

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)

    # Template check on CBSE responses
    cbse_jailbroken = sum(
        1 for r in cbse_responses
        if not any(pref in r for pref in test_prefixes)
    )
    cbse_template_asr = cbse_jailbroken / n_prompts
    print(f"  Template ASR: {cbse_jailbroken}/{n_prompts} = {cbse_template_asr:.1%}")

    # Llama-Guard on CBSE responses
    print("  Loading Llama-Guard-3-8B for CBSE responses...")
    lg_tok = AutoTokenizer.from_pretrained(LLAMA_GUARD_ID)
    lg_mod = AutoModelForCausalLM.from_pretrained(
        LLAMA_GUARD_ID, torch_dtype=torch.bfloat16, device_map=device,
    )

    guard_unsafe_cbse = 0
    for goal, resp in tqdm(zip(goals, cbse_responses), total=n_prompts,
                           desc=f"Llama-Guard {model_name}/cbse"):
        verdict = moderate([
            {"role": "user", "content": goal},
            {"role": "assistant", "content": resp},
        ])
        if verdict == "unsafe":
            guard_unsafe_cbse += 1

    cbse_guard_asr = guard_unsafe_cbse / n_prompts
    print(f"  Guard ASR: {guard_unsafe_cbse}/{n_prompts} = {cbse_guard_asr:.1%}")

    del lg_tok, lg_mod
    gc.collect()
    torch.cuda.empty_cache()

    results.append({
        "model": model_name,
        "config": "cbse",
        "attack": "autodan",
        "n_prompts": n_prompts,
        "template_jailbroken": cbse_jailbroken,
        "template_asr": round(cbse_template_asr, 3),
        "guard_unsafe": guard_unsafe_cbse,
        "guard_asr": round(cbse_guard_asr, 3),
    })

    # Save per-prompt results
    df_out = pd.DataFrame({
        "goal": goals,
        "adv_prompt": adv_prompts,
        "original_response": orig_responses,
        "cbse_response": cbse_responses,
    })
    df_out.to_csv(f"{RESULTS_DIR}/{model_name}_autodan_responses.csv", index=False)

    return results


def main():
    parser = argparse.ArgumentParser(description="AutoDAN attack evaluation")
    parser.add_argument("--model", type=str, default="llama2")
    parser.add_argument("--autodan_path", type=str, required=True,
                        help="Path to AutoDAN responses JSON (from prior AutoDAN-HGA run)")
    parser.add_argument("--tp", type=int, default=4,
                        help="Tensor parallel size for vLLM (default: 4)")
    args = parser.parse_args()

    results = evaluate_model(args.model, args.autodan_path, tp=args.tp)

    if results:
        df = pd.DataFrame(results)
        save_path = f"{RESULTS_DIR}/autodan_summary_{args.model}.csv"
        df.to_csv(save_path, index=False)
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(df.to_string(index=False))
        print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()
