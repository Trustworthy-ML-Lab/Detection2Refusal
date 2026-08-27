"""
Probe robustness: refusal-direction and component-selection stability.
Tests whether the refusal direction and downstream component selection
are robust to subsampling the probing data.

Approach:
  1. Load 51 response pairs, filter (~37 after filtering)
  2. Create N bootstrap subsets (80% each, different seeds)
  3. For each subset: recompute refusal direction
  4. Measure: cosine similarity between all direction pairs (per layer)
  5. For each subset: re-identify refusal heads and safety neurons
  6. Measure: Jaccard overlap of selected component sets

GPU REQUIRED for forward passes (direction extraction + head/neuron scoring).

Usage:
    # Full pipeline (default: llama2, 5 subsets)
    python experiments/evaluate_probe_robustness.py --model llama2
    python experiments/evaluate_probe_robustness.py --model llama3 --n_subsets 10 --subsample_ratio 0.8
"""

import argparse
import gc
import itertools
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

import os as _os, sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root != _os.path.dirname(_root) and not _os.path.exists(_os.path.join(_root, "utils.py")):
    _root = _os.path.dirname(_root)
_sys.path.insert(0, _root)
from utils import MODEL_DICT, filter_response, load_conversation_template

RESULTS_DIR = "probe_robustness_results"
RESPONSE_DIR = "scripts/find_refusal_heads/responses"
HEAD_PERCENT = 3.0   # Top 3% of heads (same as paper)
NEURON_PERCENT = 1.0  # Top 1% of neurons (same as paper)


def top_k_heads(matrix, k, descending=True):
    """Select top-k heads from a (layers, heads) score matrix."""
    flat = [(float(matrix[i, j]), (i, j))
            for i in range(matrix.shape[0]) for j in range(matrix.shape[1])]
    flat.sort(key=lambda x: x[0], reverse=descending)
    return flat[:k]


def top_k_neurons(matrix, k, descending=True):
    """Select top-k neurons from a (layers, neurons) score matrix."""
    flat = [(float(matrix[i, j]), (i, j))
            for i in range(matrix.shape[0]) for j in range(matrix.shape[1])]
    flat.sort(key=lambda x: x[0], reverse=descending)
    return flat[:k]


def jaccard(set_a, set_b):
    """Jaccard similarity between two sets of (layer, idx) tuples."""
    a, b = set(set_a), set(set_b)
    if len(a | b) == 0:
        return 1.0
    return len(a & b) / len(a | b)


def get_token_end(model_name, toks_len):
    """Model-specific end token offset."""
    if "qwen" in model_name:
        return toks_len - 2
    elif "llama3" in model_name or "phi3" in model_name:
        return toks_len - 1
    return toks_len


# ===================== Direction Extraction =====================

def extract_refusal_direction_attn(model, tokenizer, model_name, conv_template,
                                   harmful_prompts, neutral_prompts,
                                   harmful_responses, neutral_responses, device):
    """Extract refusal direction from post_attention_layernorm residuals."""
    residual_outputs = []

    def capture_residual_hook():
        def hook_fn(module, input, output):
            residual_outputs.append(input[0].detach()[0, :, :])
        return hook_fn

    hooks = []
    for layer in model.model.layers:
        h = layer.post_attention_layernorm.register_forward_hook(capture_residual_hook())
        hooks.append(h)

    harmful_embeddings = []
    neutral_embeddings = []

    for n_prompt, h_prompt, n_resp, h_resp in zip(
        neutral_prompts, harmful_prompts, neutral_responses, harmful_responses
    ):
        # Neutral
        conv_template.messages = []
        conv_template.append_message(conv_template.roles[0], n_prompt)
        conv_template.append_message(conv_template.roles[1], None)
        toks = tokenizer(conv_template.get_prompt()).input_ids
        start = len(toks)
        conv_template.update_last_message(n_resp)
        toks = tokenizer(conv_template.get_prompt()).input_ids
        end = get_token_end(model_name, len(toks))
        toks = tokenizer(conv_template.get_prompt(), return_tensors="pt")
        _ = model(input_ids=toks["input_ids"].to(device),
                  attention_mask=toks["attention_mask"].to(device))
        neutral_embeddings.append(
            torch.stack(residual_outputs, dim=0)[:, start - 1:end - 1, :].mean(dim=1).cpu()
        )
        residual_outputs = []

        # Harmful
        conv_template.messages = []
        conv_template.append_message(conv_template.roles[0], h_prompt)
        conv_template.append_message(conv_template.roles[1], None)
        toks = tokenizer(conv_template.get_prompt()).input_ids
        start = len(toks)
        conv_template.update_last_message(h_resp)
        toks = tokenizer(conv_template.get_prompt()).input_ids
        end = get_token_end(model_name, len(toks))
        toks = tokenizer(conv_template.get_prompt(), return_tensors="pt")
        _ = model(input_ids=toks["input_ids"].to(device),
                  attention_mask=toks["attention_mask"].to(device))
        harmful_embeddings.append(
            torch.stack(residual_outputs, dim=0)[:, start - 1:end - 1, :].mean(dim=1).cpu()
        )
        residual_outputs = []

    for h in hooks:
        h.remove()

    mean_harmful = torch.stack(harmful_embeddings, dim=0).mean(dim=0)
    mean_neutral = torch.stack(neutral_embeddings, dim=0).mean(dim=0)
    return mean_harmful - mean_neutral  # (num_layers, hidden_size)


def extract_safety_direction_mlp(model, tokenizer, model_name, conv_template,
                                 harmful_prompts, neutral_prompts,
                                 harmful_responses, neutral_responses, device):
    """Extract safety direction from hidden states for MLP pathway."""
    harmful_embeddings = []
    neutral_embeddings = []

    for h_prompt, h_resp in zip(harmful_prompts, harmful_responses):
        conv_template.messages = []
        conv_template.append_message(conv_template.roles[0], h_prompt)
        conv_template.append_message(conv_template.roles[1], None)
        toks = tokenizer(conv_template.get_prompt()).input_ids
        start = len(toks)
        conv_template.update_last_message(h_resp)
        toks = tokenizer(conv_template.get_prompt()).input_ids
        end = get_token_end(model_name, len(toks))
        toks = tokenizer(conv_template.get_prompt(), return_tensors="pt")
        with torch.no_grad():
            hs = model(
                input_ids=toks["input_ids"].to(device),
                attention_mask=toks["attention_mask"].to(device),
                output_hidden_states=True,
            ).hidden_states[1:]
        harmful_embeddings.append(
            torch.stack(hs, dim=0)[:, :, start - 1:end - 1, :].mean(dim=2).cpu()
        )

    for n_prompt, n_resp in zip(neutral_prompts, neutral_responses):
        conv_template.messages = []
        conv_template.append_message(conv_template.roles[0], n_prompt)
        conv_template.append_message(conv_template.roles[1], None)
        toks = tokenizer(conv_template.get_prompt()).input_ids
        start = len(toks)
        conv_template.update_last_message(n_resp)
        toks = tokenizer(conv_template.get_prompt()).input_ids
        end = get_token_end(model_name, len(toks))
        toks = tokenizer(conv_template.get_prompt(), return_tensors="pt")
        with torch.no_grad():
            hs = model(
                input_ids=toks["input_ids"].to(device),
                attention_mask=toks["attention_mask"].to(device),
                output_hidden_states=True,
            ).hidden_states[1:]
        neutral_embeddings.append(
            torch.stack(hs, dim=0)[:, :, start - 1:end - 1, :].mean(dim=2).cpu()
        )

    mean_harmful = torch.stack(harmful_embeddings, dim=0).mean(dim=0)
    mean_neutral = torch.stack(neutral_embeddings, dim=0).mean(dim=0)
    return mean_harmful - mean_neutral  # (num_layers, hidden_size)


# ===================== Head / Neuron Scoring =====================

def score_refusal_heads(model, tokenizer, model_name, conv_template,
                        refusal_direction, harmful_prompts, harmful_responses, device):
    """Score attention heads by contribution along refusal direction."""
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    hidden_size = model.config.hidden_size

    rd = refusal_direction.to(device)
    rd = rd / torch.norm(rd, dim=-1, keepdim=True)

    attn_contribution = []

    def capture_attn_hook():
        def hook_fn(module, input, output):
            attn_out = input[0].detach()[0, :, :]
            attn_out = attn_out.reshape(attn_out.size(0), num_heads, -1)
            o_proj = module.weight.detach().clone()
            o_proj = o_proj.reshape(hidden_size, num_heads, -1).permute(1, 2, 0).contiguous()
            attn_contribution.append(torch.einsum("snk,nkh->snh", attn_out, o_proj))
        return hook_fn

    hooks = []
    for layer in model.model.layers:
        h = layer.self_attn.o_proj.register_forward_hook(capture_attn_hook())
        hooks.append(h)

    avg = torch.zeros((num_layers, num_heads))
    for h_prompt, h_resp in zip(harmful_prompts, harmful_responses):
        conv_template.messages = []
        conv_template.append_message(conv_template.roles[0], h_prompt)
        conv_template.append_message(conv_template.roles[1], None)
        toks = tokenizer(conv_template.get_prompt()).input_ids
        start = len(toks)
        conv_template.update_last_message(h_resp)
        toks = tokenizer(conv_template.get_prompt()).input_ids
        end = get_token_end(model_name, len(toks))
        toks = tokenizer(conv_template.get_prompt(), return_tensors="pt")
        _ = model(input_ids=toks["input_ids"].to(device),
                  attention_mask=toks["attention_mask"].to(device))
        all_hc = torch.stack(attn_contribution, dim=0)[:, start - 1:end - 1, :, :].mean(dim=1)
        dp = torch.einsum("lhd,ld->lh", all_hc, rd)
        avg += dp.cpu().numpy()
        attn_contribution = []

    for h in hooks:
        h.remove()

    avg = avg.numpy() / len(harmful_prompts)
    k = int(num_layers * num_heads * (HEAD_PERCENT / 100))
    selected = top_k_heads(avg, k)
    return set(c[1] for c in selected), avg


def score_safety_neurons(model, tokenizer, model_name, conv_template,
                         safety_direction, harmful_prompts, harmful_responses, device):
    """Score MLP neurons by contribution along safety direction."""
    num_layers = model.config.num_hidden_layers
    inter_size = model.config.intermediate_size

    sd = safety_direction.to(device)
    sd = sd / torch.norm(sd, dim=-1, keepdim=True)

    # Pre-compute weight contributions
    weight_contrib = []
    for i, layer in enumerate(model.model.layers):
        dp = layer.mlp.down_proj.weight.detach().clone().permute(1, 0).contiguous()
        weight_contrib.append(dp @ sd[i].unsqueeze(-1))
    weight_contrib = torch.stack(weight_contrib, dim=0)

    # Get sys prompt length
    conv_template_tmp = load_conversation_template(model_name)
    conv_template_tmp.append_message(conv_template_tmp.roles[0], None)
    sys_len = len(tokenizer(conv_template_tmp.get_prompt()).input_ids)

    mlp_acts = []

    def capture_mlp_hook(layer_id):
        def hook_fn(module, input, output):
            neur_out = input[0].detach()[0, sys_len:, :]
            mlp_acts.append(neur_out * weight_contrib[layer_id].squeeze(-1))
        return hook_fn

    hooks = []
    for i, layer in enumerate(model.model.layers):
        h = layer.mlp.down_proj.register_forward_hook(capture_mlp_hook(i))
        hooks.append(h)

    avg = torch.zeros((num_layers, inter_size)).to(device)
    for h_prompt, h_resp in zip(harmful_prompts, harmful_responses):
        conv_template.messages = []
        conv_template.append_message(conv_template.roles[0], h_prompt)
        conv_template.append_message(conv_template.roles[1], None)
        toks = tokenizer(conv_template.get_prompt()).input_ids
        start = len(toks)
        conv_template.update_last_message(h_resp)
        toks = tokenizer(conv_template.get_prompt()).input_ids
        end = get_token_end(model_name, len(toks))
        toks = tokenizer(conv_template.get_prompt(), return_tensors="pt")
        _ = model(input_ids=toks["input_ids"].to(device),
                  attention_mask=toks["attention_mask"].to(device))
        all_c = torch.stack(mlp_acts, dim=0)[:, start - 1 - sys_len:end - 1 - sys_len, :].mean(dim=1)
        mlp_acts = []
        with torch.no_grad():
            avg += all_c

    for h in hooks:
        h.remove()

    avg = avg.detach().cpu().numpy() / len(harmful_prompts)
    k = int(num_layers * inter_size * (NEURON_PERCENT / 100))
    selected = top_k_neurons(avg, k)
    return set(c[1] for c in selected), avg


# ===================== Main Pipeline =====================

def main():
    parser = argparse.ArgumentParser(description="Probe robustness test")
    parser.add_argument("--model", type=str, default="llama2")
    parser.add_argument("--n_subsets", type=int, default=5,
                        help="Number of bootstrap subsets (default: 5)")
    parser.add_argument("--subsample_ratio", type=float, default=0.8,
                        help="Fraction of pairs to keep per subset (default: 0.8)")
    parser.add_argument("--skip_neurons", action="store_true",
                        help="Skip neuron scoring (faster, heads + direction only)")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Load response data ---
    resp_path = os.path.join(RESPONSE_DIR, f"{args.model}.json")
    with open(resp_path) as f:
        data = json.load(f)

    h_prompts = [e["Harmful Prompt"] for e in data]
    n_prompts = [e["Neutral Prompt"] for e in data]
    h_resps = [e["Harmful Response"] for e in data]
    n_resps = [e["Neutral Response"] for e in data]

    fh_prompts, fn_prompts, fh_resps, fn_resps = filter_response(
        h_prompts, n_prompts, h_resps, n_resps
    )
    n_filtered = len(fh_prompts)
    print(f"Filtered pairs: {n_filtered} / {len(data)}")

    # --- Load model ---
    print(f"Loading model {args.model}...")
    model_path = MODEL_DICT[args.model]
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )
    tokenizer.pad_token = tokenizer.eos_token

    # --- Bootstrap subsets ---
    subset_size = int(n_filtered * args.subsample_ratio)
    rng = np.random.RandomState(42)

    attn_directions = []  # refusal directions (attn pathway)
    mlp_directions = []   # safety directions (mlp pathway)
    head_sets = []
    neuron_sets = []

    for i in range(args.n_subsets):
        print(f"\n{'='*60}")
        print(f"Subset {i+1}/{args.n_subsets} (seed offset={i})")
        print(f"{'='*60}")

        indices = sorted(rng.choice(n_filtered, size=subset_size, replace=False))
        sub_hp = [fh_prompts[j] for j in indices]
        sub_np = [fn_prompts[j] for j in indices]
        sub_hr = [fh_resps[j] for j in indices]
        sub_nr = [fn_resps[j] for j in indices]
        print(f"  Using {len(indices)} / {n_filtered} pairs")

        # 1. Extract refusal direction (attn pathway)
        conv = load_conversation_template(args.model)
        print("  Extracting attn refusal direction...")
        rd_attn = extract_refusal_direction_attn(
            model, tokenizer, args.model, conv,
            sub_hp, sub_np, sub_hr, sub_nr, device
        )
        attn_directions.append(rd_attn)

        # 2. Score refusal heads
        conv = load_conversation_template(args.model)
        print("  Scoring refusal heads...")
        heads_set, _ = score_refusal_heads(
            model, tokenizer, args.model, conv,
            rd_attn, sub_hp, sub_hr, device
        )
        head_sets.append(heads_set)
        print(f"  Selected {len(heads_set)} refusal heads")

        # 3. Extract safety direction + score neurons (MLP pathway)
        if not args.skip_neurons:
            conv = load_conversation_template(args.model)
            print("  Extracting MLP safety direction...")
            rd_mlp = extract_safety_direction_mlp(
                model, tokenizer, args.model, conv,
                sub_hp, sub_np, sub_hr, sub_nr, device
            )
            mlp_directions.append(rd_mlp)

            conv = load_conversation_template(args.model)
            print("  Scoring safety neurons...")
            neurons_set, _ = score_safety_neurons(
                model, tokenizer, args.model, conv,
                rd_mlp, sub_hp, sub_hr, device
            )
            neuron_sets.append(neurons_set)
            print(f"  Selected {len(neurons_set)} safety neurons")

    # --- Compute Metrics ---
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")

    num_layers = model.config.num_hidden_layers

    # 1. Cosine similarity of refusal directions (attn)
    print("\n--- Attn Refusal Direction: Cosine Similarity ---")
    cos_sims_attn = []
    for (i, d1), (j, d2) in itertools.combinations(enumerate(attn_directions), 2):
        # Per-layer cosine similarity, then average
        cos_per_layer = torch.nn.functional.cosine_similarity(d1, d2, dim=-1)  # (num_layers,)
        mean_cos = cos_per_layer.mean().item()
        cos_sims_attn.append(mean_cos)
        print(f"  Subset {i+1} vs {j+1}: mean cosine = {mean_cos:.4f}, "
              f"min layer = {cos_per_layer.min().item():.4f}")

    mean_cos_attn = np.mean(cos_sims_attn)
    print(f"  >> Mean pairwise cosine: {mean_cos_attn:.4f}")

    # 2. Cosine similarity of safety directions (MLP)
    if mlp_directions:
        print("\n--- MLP Safety Direction: Cosine Similarity ---")
        cos_sims_mlp = []
        for (i, d1), (j, d2) in itertools.combinations(enumerate(mlp_directions), 2):
            cos_per_layer = torch.nn.functional.cosine_similarity(
                d1.squeeze(1), d2.squeeze(1), dim=-1
            )
            mean_cos = cos_per_layer.mean().item()
            cos_sims_mlp.append(mean_cos)
            print(f"  Subset {i+1} vs {j+1}: mean cosine = {mean_cos:.4f}")
        mean_cos_mlp = np.mean(cos_sims_mlp)
        print(f"  >> Mean pairwise cosine: {mean_cos_mlp:.4f}")

    # 3. Jaccard overlap of refusal heads
    print("\n--- Refusal Heads: Jaccard Overlap ---")
    jaccards_heads = []
    for (i, s1), (j, s2) in itertools.combinations(enumerate(head_sets), 2):
        j_val = jaccard(s1, s2)
        jaccards_heads.append(j_val)
        print(f"  Subset {i+1} vs {j+1}: Jaccard = {j_val:.4f}")
    mean_j_heads = np.mean(jaccards_heads)
    print(f"  >> Mean pairwise Jaccard: {mean_j_heads:.4f}")

    # Core heads (appear in ALL subsets)
    if head_sets:
        core_heads = head_sets[0]
        for s in head_sets[1:]:
            core_heads = core_heads & s
        print(f"  >> Core heads (in all {args.n_subsets} subsets): {len(core_heads)} / {len(head_sets[0])}")
        print(f"     {sorted(core_heads)}")

    # 4. Jaccard overlap of safety neurons
    if neuron_sets:
        print("\n--- Safety Neurons: Jaccard Overlap ---")
        jaccards_neurons = []
        for (i, s1), (j, s2) in itertools.combinations(enumerate(neuron_sets), 2):
            j_val = jaccard(s1, s2)
            jaccards_neurons.append(j_val)
            print(f"  Subset {i+1} vs {j+1}: Jaccard = {j_val:.4f}")
        mean_j_neurons = np.mean(jaccards_neurons)
        print(f"  >> Mean pairwise Jaccard: {mean_j_neurons:.4f}")

        core_neurons = neuron_sets[0]
        for s in neuron_sets[1:]:
            core_neurons = core_neurons & s
        print(f"  >> Core neurons (in all {args.n_subsets} subsets): {len(core_neurons)} / {len(neuron_sets[0])}")

    # --- Save results ---
    results = {
        "model": args.model,
        "n_subsets": args.n_subsets,
        "subsample_ratio": args.subsample_ratio,
        "n_filtered_pairs": n_filtered,
        "subset_size": subset_size,
        "head_percent": HEAD_PERCENT,
        "neuron_percent": NEURON_PERCENT,
        "attn_direction_cosine": {
            "mean": float(mean_cos_attn),
            "all_pairs": [float(c) for c in cos_sims_attn],
        },
        "refusal_heads_jaccard": {
            "mean": float(mean_j_heads),
            "all_pairs": [float(j) for j in jaccards_heads],
        },
        "core_heads": sorted(list(core_heads)) if head_sets else [],
        "n_selected_heads": len(head_sets[0]) if head_sets else 0,
    }

    if mlp_directions:
        results["mlp_direction_cosine"] = {
            "mean": float(mean_cos_mlp),
            "all_pairs": [float(c) for c in cos_sims_mlp],
        }
    if neuron_sets:
        results["safety_neurons_jaccard"] = {
            "mean": float(mean_j_neurons),
            "all_pairs": [float(j) for j in jaccards_neurons],
        }
        results["core_neurons_count"] = len(core_neurons)
        results["n_selected_neurons"] = len(neuron_sets[0])

    save_path = os.path.join(RESULTS_DIR, f"probe_robustness_{args.model}.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {save_path}")

    # Summary line for easy parsing
    neuron_line = ""
    if neuron_sets:
        neuron_line = f" | MLP cos={mean_cos_mlp:.3f} | Neuron Jaccard={mean_j_neurons:.3f}"
    print(f"\nSUMMARY: {args.model} | Attn cos={mean_cos_attn:.3f} | Head Jaccard={mean_j_heads:.3f}{neuron_line}")


if __name__ == "__main__":
    main()
