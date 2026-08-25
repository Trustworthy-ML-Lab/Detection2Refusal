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

from utils import (
    load_conversation_template,
    get_goals_and_targets,
    merge_csv,
    test_prefixes,
    MODEL_DICT,
    calculate_perplexity
)

from transformers.utils import logging
logging.set_verbosity_error()

import pickle
import pandas as pd

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------- Argument parsing --------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama3")
    parser.add_argument("--attack", type=str, default="advllm", help="benign, none, gcg, advllm")
    parser.add_argument("--remove_sys_prompt", action=argparse.BooleanOptionalAction)
    parser.add_argument("--dataset", type=str, default="advbench", help="advbench or mlcinst")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--n_train_data", type=int, default=100)
    parser.add_argument("--percent", type=float, default=3.0)
    parser.add_argument(
        "--factor",
        type=float,
        default=3.0,
        help="scaling factor mode"
    )
    parser.add_argument("--save_path", type=str, default="detection_head_intervention_result.csv", help="Path to save the results")
    args = parser.parse_args()

    # -------------------- Load model & tokenizer --------------------
    model_path = MODEL_DICT[args.model]
    model = (
        AutoModelForCausalLM
        .from_pretrained(model_path, torch_dtype=torch.float16)
        .to(device)
        .eval()
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )
    tokenizer.pad_token = tokenizer.eos_token
    max_pos = AutoConfig.from_pretrained(model_path).max_position_embeddings
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    gen_cfg = None
    try:
        gen_cfg = GenerationConfig.from_pretrained(model_path)
    except Exception:
        gen_cfg = GenerationConfig(
            **{k: getattr(config, k) for k in ['temperature', 'top_k', 'top_p', 'repetition_penalty'] if hasattr(config, k)}
        )

    num_heads = model.config.num_attention_heads
    hidden_size = model.config.hidden_size
    head_dim = hidden_size // num_heads

    # -------------------- Load detection heads --------------------
    detect_path = f"detection_heads/{args.model}_{args.percent}.json"
    with open(detect_path, "r") as f:
        det_list = json.load(f)
    heads = [(int(d["layer"]), int(d["head"])) for d in det_list]

    # -------------------- Apply intervention --------------------
    for layer_idx, head_idx in heads:
        start = head_idx * head_dim
        end = start + head_dim

        W = model.model.layers[layer_idx].self_attn.o_proj.weight.detach().clone()

        W[:, start:end] *= args.factor

        # write back
        model.model.layers[layer_idx].self_attn.o_proj.weight = nn.Parameter(W)

    if args.model == "vicuna":
        model.generation_config.do_sample = True

    temp_model_path = f"temp_models/{args.model}_{args.attack}_{args.factor}"
    os.makedirs("temp_models", exist_ok=True)
    model.save_pretrained(temp_model_path)
    tokenizer.save_pretrained(temp_model_path)

    print(f"Model saved to {temp_model_path}")

    del model

    # -------------------- Load model for vLLM ----------------
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
            "../data/advbench/harmful_behaviors.csv",
            args.offset,
            args.n_train_data
        )
    elif args.dataset == "mlcinst":
        with open("../data/MaliciousInstruct.txt", 'r') as f:
            goals = [l.strip() for l in f]
    else:
        raise ValueError(f"Unknown dataset '{args.dataset}'")

    adv_prompts = []
    if args.attack == "advllm":
        with open(f"../data/adv_prompts/advbench/greedy/{args.model}.pkl", 'rb') as f:
            adv_prompts = pickle.load(f)[: args.n_train_data]
    elif args.attack == "advllm_gbs":
        with open(f"../data/adv_prompts/advbench/group_beam_search_50/{args.model}.pkl", 'rb') as f:
            adv_prompts = pickle.load(f)[: args.n_train_data]

    gcg_g = []
    gcg_adv = []
    if args.attack == "gcg":
        gcg_df = merge_csv(f"../data/gcg_results/{args.model}/1.0_1.0/")
        gcg_g = gcg_df['prompt'].tolist()
        gcg_adv = gcg_df['adv'].tolist()

    # -------------------- Run attacks --------------------
    conv = load_conversation_template(args.model)
    if args.remove_sys_prompt:
        conv.set_system_message("")

    advs = []
    prompts = []
    gbs_prompts = []
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
    elif args.attack == 'advllm_gbs':
        for q in tqdm(adv_prompts):
            for sub_q in q:
                conv.messages = []
                conv.append_message(conv.roles[0], sub_q)
                conv.append_message(conv.roles[1], None)
                advs.append(sub_q)
                prompts.append(conv.get_prompt())

            gbs_prompts.append(prompts)
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

    # -------------------- Save results --------------------
    os.makedirs("results", exist_ok=True)
    with open(f"results/{args.model}_{args.attack}_{args.factor}_{args.percent}.txt", 'w') as f:
        for adv, resp in zip(advs, responses):
            f.write(f"Adv: {adv}\nResponse: {resp}\n\n")

    # -------------------- Cleanup --------------------
    del model
    gc.collect(); torch.cuda.empty_cache()
    shutil.rmtree(temp_model_path)

    # -------------------- Safety checks --------------------
    template_fail = sum(
        1 for r in responses
        if not any(pref in r for pref in test_prefixes)
    )

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
        return text.split('\n')[2]

    guard_fail = sum(
        1 for adv, r in zip(advs, responses)
        if moderate([{"role":"user","content":adv},{"role":"assistant","content":r}])=='unsafe'
    )

    del lg_tok, lg_mod
    gc.collect()
    torch.cuda.empty_cache()

    # perplexity = calculate_perplexity(responses, device=device)

    safe_templ = 1 - (template_fail/len(responses))
    safe_guard = 1 - (guard_fail/len(responses))
    print(f"Safe rate (template): {safe_templ:.3f}")
    print(f"Safe rate (llama guard): {safe_guard:.3f}")
    # print(f"Perplexity: {perplexity:.3f}")

    file_exists = os.path.exists(args.save_path)
    with open(args.save_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['model', 'attack', 'dataset', 'factor', 'percent','safe_template', 'safe_guard'])
        
        writer.writerow([args.model, args.attack, args.dataset, args.factor, args.percent, round(safe_templ, 3), round(safe_guard, 3)])

if __name__ == "__main__":
    main()