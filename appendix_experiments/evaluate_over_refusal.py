"""
E2: Over-Refusal Evaluation using OR-Bench-80k.
Tests whether CBSE-edited models excessively refuse benign prompts.
Randomly samples 1k prompts from the 80k pool (seed=42).

Single-stage pipeline (GPU required):
  1. Generate responses from original + CBSE models
  2. Judge with template matching + Llama-Guard (same as paper)

Usage:
    # Default: llama2, 1k prompts
    python evaluate_over_refusal.py --model llama2

    # All models
    python evaluate_over_refusal.py --model all

    # Custom sample size
    python evaluate_over_refusal.py --model llama2 --max_prompts 500
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
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from vllm import LLM, SamplingParams

from utils import MODEL_DICT, test_prefixes, load_conversation_template

# --------------- Config ---------------
RESULTS_DIR = "over_refusal_results"
OR_BENCH_CONFIG = "or-bench-80k"
OR_BENCH_DATASET = "bench-llm/or-bench"
DEFAULT_MAX_PROMPTS = 1000
DEFAULT_SEED = 42
LLAMA_GUARD_ID = "meta-llama/Llama-Guard-3-8B"

CBSE_PARAMS = {
    # (neuron_factor, neuron_percent, detection_factor, refusal_factor, head_percent)
    "default": (1.5, 1.0, 2.0, 2.0, 3.0),
}

PAPER_MODELS = ["llama3", "llama2", "mistral", "guanaco", "qwen2.5", "qwen3"]


def load_or_bench(max_prompts: int = DEFAULT_MAX_PROMPTS, seed: int = DEFAULT_SEED):
    """Load OR-Bench prompts, randomly sample to max_prompts."""
    ds = load_dataset(OR_BENCH_DATASET, OR_BENCH_CONFIG, split="train")
    all_prompts = ds["prompt"]
    all_categories = ds["category"]
    print(f"Total prompts in {OR_BENCH_CONFIG}: {len(all_prompts)}")

    if len(all_prompts) > max_prompts:
        rng = np.random.RandomState(seed)
        indices = sorted(rng.choice(len(all_prompts), size=max_prompts, replace=False))
        prompts = [all_prompts[i] for i in indices]
        categories = [all_categories[i] for i in indices]
        print(f"Sampled {len(prompts)} prompts (seed={seed})")
    else:
        prompts = list(all_prompts)
        categories = list(all_categories)

    return prompts, categories


def apply_cbse(model_name, model, device):
    """Apply CBSE weight scaling intervention to a loaded HF model."""
    num_heads = model.config.num_attention_heads
    hidden_size = model.config.hidden_size
    head_dim = getattr(model.config, "head_dim", hidden_size // num_heads)

    nf, np_, df, rf, hp = CBSE_PARAMS["default"]

    det_heads, ref_heads, neurons = [], [], []
    if df != 1.0:
        with open(f"find_detection_heads/detection_heads/{model_name}_{hp}.json") as f:
            det_heads = [(int(d["layer"]), int(d["head"])) for d in json.load(f)]
    if rf != 1.0:
        with open(f"find_refusal_heads/refusal_heads/{model_name}_{hp}.json") as f:
            ref_heads = [(int(d["layer"]), int(d["head"])) for d in json.load(f)]
    if nf != 1.0:
        with open(f"find_safety_neurons/safety_neurons/{model_name}_{np_}.json") as f:
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


def evaluate_model(model_name: str, max_prompts: int = DEFAULT_MAX_PROMPTS,
                   seed: int = DEFAULT_SEED, tp: int = 4):
    """Generate responses and evaluate over-refusal for original + CBSE."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = MODEL_DICT[model_name]

    prompts_text, categories = load_or_bench(max_prompts=max_prompts, seed=seed)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Format prompts
    conv = load_conversation_template(model_name)
    formatted_prompts = []
    for q in prompts_text:
        conv.messages = []
        conv.append_message(conv.roles[0], q)
        conv.append_message(conv.roles[1], None)
        formatted_prompts.append(conv.get_prompt())

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

    results = []

    for config_name in ["original", "cbse"]:
        print(f"\n{'='*60}")
        print(f"{model_name} | {config_name} | {len(formatted_prompts)} prompts")
        print(f"{'='*60}")

        if config_name == "cbse":
            save_dir = "/opt/dlami/nvme/temp_models" if os.path.isdir("/opt/dlami/nvme") else "temp_models"
            temp_path = f"{save_dir}/{model_name}_orbench_cbse"

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
                apply_cbse(model_name, hf_model, "cpu")
                os.makedirs(temp_path, exist_ok=True)
                hf_model.save_pretrained(temp_path)
                tokenizer.save_pretrained(temp_path)
                del hf_model
                gc.collect()
            else:
                print(f"  Reusing cached CBSE model at {temp_path}")

            llm = LLM(model=temp_path, max_model_len=max_pos,
                       tensor_parallel_size=tp, dtype=torch.float16)
        else:
            llm = LLM(model=model_path, max_model_len=max_pos,
                       tensor_parallel_size=tp, dtype=torch.float16)

        # Generate
        outputs = llm.generate(formatted_prompts, sampling_params=sample_params)
        responses = [o.outputs[0].text for o in outputs]

        del llm
        gc.collect()
        torch.cuda.empty_cache()
        if config_name == "cbse" and os.path.exists(f"temp_models/{model_name}_orbench_cbse"):
            shutil.rmtree(f"temp_models/{model_name}_orbench_cbse")

        # Save raw responses
        save_path = f"{RESULTS_DIR}/{model_name}_{config_name}_orbench.csv"
        df_out = pd.DataFrame({
            "prompt": list(prompts_text),
            "category": list(categories),
            "response": responses,
        })
        df_out.to_csv(save_path, index=False)
        print(f"Saved {len(responses)} responses to {save_path}")

        # ---- Template-based refusal check ----
        # For over-refusal: a response IS a refusal if it matches test_prefixes
        template_refusal = sum(
            1 for r in responses
            if any(pref in r for pref in test_prefixes)
        )
        template_refusal_rate = template_refusal / len(responses)
        print(f"  Template refusal rate: {template_refusal}/{len(responses)} "
              f"= {template_refusal_rate:.1%}")

        # ---- Llama-Guard check ----
        print("  Loading Llama-Guard-3-8B...")
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

        # For over-refusal on benign prompts:
        # Llama-Guard says "unsafe" = model refused appropriately (false positive unlikely here)
        # But these are BENIGN prompts, so "unsafe" means Llama-Guard itself flags the prompt
        # We check if model's response is flagged as unsafe by guard
        guard_unsafe = 0
        for prompt, resp in tqdm(
            zip(prompts_text, responses), total=len(responses),
            desc=f"Llama-Guard {model_name}/{config_name}"
        ):
            verdict = moderate([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": resp},
            ])
            if verdict == "unsafe":
                guard_unsafe += 1

        guard_safe_rate = 1 - (guard_unsafe / len(responses))
        print(f"  Llama-Guard safe rate: {guard_safe_rate:.1%} "
              f"(unsafe={guard_unsafe}/{len(responses)})")

        del lg_tok, lg_mod
        gc.collect()
        torch.cuda.empty_cache()

        results.append({
            "model": model_name,
            "config": config_name,
            "n_prompts": len(responses),
            "template_refusal": template_refusal,
            "template_refusal_rate": round(template_refusal_rate, 3),
            "guard_unsafe": guard_unsafe,
            "guard_safe_rate": round(guard_safe_rate, 3),
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="E2: Over-Refusal Evaluation (OR-Bench)"
    )
    parser.add_argument("--model", type=str, default="llama2",
                        help="Model name or 'all' for all paper models")
    parser.add_argument("--max_prompts", type=int, default=DEFAULT_MAX_PROMPTS,
                        help="Max prompts to sample from OR-Bench-80k (default: 1000)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed for prompt sampling (default: 42)")
    parser.add_argument("--tp", type=int, default=4,
                        help="Tensor parallel size for vLLM (default: 4)")
    args = parser.parse_args()

    models = PAPER_MODELS if args.model == "all" else [args.model]

    all_results = []
    for m in models:
        all_results.extend(
            evaluate_model(m, max_prompts=args.max_prompts, seed=args.seed, tp=args.tp)
        )

    if all_results:
        df = pd.DataFrame(all_results)
        save_path = f"{RESULTS_DIR}/over_refusal_summary.csv"
        file_exists = os.path.isfile(save_path)
        df.to_csv(save_path, mode='a', index=False, header=not file_exists)
        # df.to_csv(save_path, index=False)
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(df.to_string(index=False))
        print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()
