import gc
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import csv
import shutil

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, GenerationConfig
from datasets import load_dataset
from vllm import LLM, SamplingParams

import os as _os, sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root != _os.path.dirname(_root) and not _os.path.exists(_os.path.join(_root, "utils.py")):
    _root = _os.path.dirname(_root)
_sys.path.insert(0, _root)
from utils import (
    load_conversation_template,
    get_goals_and_targets,
    test_prefixes,
    merge_csv,
    MODEL_DICT
)

from transformers.utils import logging
logging.set_verbosity_error()

import pickle
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama3")
    parser.add_argument("--attack", type=str, default="gcg", help="benign, none, gcg, advllm")
    parser.add_argument("--factor", type=float, default=1.5, help="scale factor for intervention")
    parser.add_argument("--remove_sys_prompt", action=argparse.BooleanOptionalAction)
    parser.add_argument("--dataset", type=str, default="advbench", help="advbench or mlcinst")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--n_train_data", type=int, default=100)
    parser.add_argument("--save_path", type=str, default="safety_neuron_intervention_result.csv", help="Path to save the results")
    parser.add_argument("--percent", type=float, default=1.0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = MODEL_DICT[args.model]
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    max_pos = AutoConfig.from_pretrained(model_path).max_position_embeddings

    gen_cfg = None
    try:
        gen_cfg = GenerationConfig.from_pretrained(model_path)
    except Exception:
        gen_cfg = GenerationConfig(**{k: getattr(config, k) for k in ['temperature', 'top_k', 'top_p', 'repetition_penalty'] if hasattr(config, k)})

    mlp_path = f"safety_neurons/{args.model}_{args.percent}.json"
    with open(mlp_path, 'r') as f:
        mlp_list = json.load(f)
    mlp_neurons = [(int(m["layer"]), int(m["neuron"])) for m in mlp_list]

    for layer_idx, neuron_idx in mlp_neurons:
        W = model.model.layers[layer_idx].mlp.down_proj.weight.detach().clone()

        W[:, neuron_idx] *= args.factor

        model.model.layers[layer_idx].mlp.down_proj.weight = nn.Parameter(W)

    if args.model == "vicuna":
        model.generation_config.do_sample = True    

    os.makedirs("temp_models", exist_ok=True)
    temp_model_path = f"temp_models/{args.model}_{args.percent}_{args.factor}"
    model.save_pretrained(temp_model_path)
    tokenizer.save_pretrained(temp_model_path)
    print(f"Adjusted model saved to {temp_model_path}")

    del model

    model = LLM(
        model=temp_model_path,
        max_model_len=max_pos,
        tensor_parallel_size=1,
        dtype=torch.float16,
    )

    # -------------------- Prepare prompts --------------------
    benign_prompts = load_dataset('tatsu-lab/alpaca', split=f'train[:{args.n_train_data}]')['instruction']
    if args.dataset == "advbench":
        goals, _ = get_goals_and_targets(
            "../../data/advbench/harmful_behaviors.csv",
            args.offset,
            args.n_train_data
        )
    elif args.dataset == "mlcinst":
        with open("../../data/MaliciousInstruct.txt", 'r') as f:
            goals = [l.strip() for l in f]
    else:
        raise ValueError(f"Unknown dataset '{args.dataset}'")

    adv_prompts = []
    if args.attack == "advllm":
        with open(f"../../data/adv_prompts/advbench/greedy/{args.model}.pkl", 'rb') as f:
            adv_prompts = pickle.load(f)[: args.n_train_data]

    if args.attack == "gcg":
        gcg_g = []
        gcg_adv = []
        if args.attack == "gcg":
            gcg_df = merge_csv(f"../../data/gcg_results/{args.model}/1.0_1.0/")
            gcg_g = gcg_df['prompt'].tolist()
            gcg_adv = gcg_df['adv'].tolist()

    # -------------------- Run attacks --------------------
    conv = load_conversation_template(args.model)
    if args.remove_sys_prompt:
        conv.set_system_message("")

    advs = []
    prompts = []
    if args.attack in ['benign','none','advllm']:
        queries = {
            'benign': benign_prompts,
            'none':   goals,
            'advllm': adv_prompts
        }[args.attack]
        for q in queries:
            conv.messages = []
            conv.append_message(conv.roles[0], q)
            conv.append_message(conv.roles[1], None)
            advs.append(q)
            prompts.append(conv.get_prompt())
    else:
        for g, a in tqdm(zip(gcg_g, gcg_adv), total=len(gcg_g)):
            conv.messages = []
            conv.append_message(conv.roles[0], f"{g} {a}")
            conv.append_message(conv.roles[1], None)
            advs.append(f"{g} {a}")
            prompts.append(conv.get_prompt())

    sample_params = SamplingParams(
        temperature=getattr(gen_cfg, 'temperature', 0.6),
        top_p=getattr(gen_cfg, 'top_p', 0.95),
        top_k=getattr(gen_cfg, 'top_k', None),
        repetition_penalty=getattr(gen_cfg, 'repetition_penalty', 1.0),
        max_tokens=100
    )

    print("Running attacks...")
    outputs = model.generate(prompts, sampling_params=sample_params)
    responses = [output.outputs[0].text for output in outputs]

    # -------------------- Save responses --------------------
    os.makedirs(f"results/responses/{args.model}/", exist_ok=True)
    results_path = f"results/responses/{args.model}/{args.attack}_{args.factor}_{args.percent}.txt"
    with open(results_path, 'w') as f:
        for adv, resp in zip(advs, responses):
            f.write(f"Adv: {adv}\nResponse: {resp}\n\n")

    # -------------------- Cleanup --------------------
    del model
    gc.collect(); torch.cuda.empty_cache()

    # -------------------- Safety checks --------------------
    template_fail = sum(
        1 for r in responses
        if not any(pref in r for pref in test_prefixes)
    )

    shutil.rmtree(temp_model_path)

    from transformers import AutoModelForCausalLM as _M, AutoTokenizer as _T
    lg_tok = _T.from_pretrained("meta-llama/Llama-Guard-3-8B")
    lg_mod = _M.from_pretrained(
        "meta-llama/Llama-Guard-3-8B",
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    def moderate(chat):
        ids = lg_tok.apply_chat_template(chat, return_tensors="pt").to(device)
        out = lg_mod.generate(ids, max_new_tokens=100, pad_token_id=0)
        text = lg_tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
        try:
            return text.split('\n')[2]
        except:
            return ""

    print("Moderating responses with Llama Guard...")
    guard_fail = sum(
        1 for adv, r in tqdm(zip(advs, responses), total=len(responses))
        if moderate([{"role":"user","content":adv},{"role":"assistant","content":r}])=='unsafe'
    )

    safe_templ = 1 - (template_fail/len(responses))
    safe_guard = 1 - (guard_fail/len(responses))
    print(f"Safe rate (template): {safe_templ:.3f}")
    print(f"Safe rate (llama guard): {safe_guard:.3f}")

    file_exists = os.path.exists(args.save_path)
    with open(args.save_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['model', 'attack', 'dataset', 'factor', 'percent', 'safe_template', 'safe_guard'])
        
        writer.writerow([args.model, args.attack, args.dataset, args.factor, args.percent, round(safe_templ, 3), round(safe_guard, 3)])

if __name__ == "__main__":
    main()