"""
N2: Activation patching — directed detection -> refusal pathway.

Addresses a8az-W1: the Fig. 3 ablation shows "removing detection heads weakens
refusal heads", but not whether detection heads DIRECTLY drive refusal heads or
whether both merely share an upstream cause. This experiment swaps the detection
heads' output between a harmful and its paired neutral prompt (everything else
held fixed) and measures the change in the refusal heads' contribution to the
refusal direction. A bidirectional effect (harmful<-neutral lowers it,
neutral<-harmful raises it) is direct causal evidence of a detection -> refusal
pathway, ruling out a shared upstream cause.

Design (clean/controlled):
  - Paired minimal-lexical prompts (harmful vs neutral), restricted to pairs whose
    templated (prompt+response) tokenizations have IDENTICAL length so positions
    align 1-to-1 for patching.
  - The SAME harmful response is teacher-forced in every condition, so only the
    prompt token (and the patched detection output) varies.
  - Metric = mean over refusal heads and response positions of
    <head output-write, refusal_direction>  (same quantity as
    exp1_refusal_contribution.py).
  - Patch = a forward_pre_hook on each detection layer's o_proj that replaces ONLY
    that detection head's slice of the attention output with the cached value from
    the other run.

Four conditions:
  C_harm         : harmful prompt, no patch
  C_harm<-neut   : harmful prompt, detection heads patched from the neutral run
  C_neut         : neutral prompt, no patch
  C_neut<-harm   : neutral prompt, detection heads patched from the harmful run

Effect size (fraction of the harmful-neutral gap explained by detection heads):
  forward  = (C_harm - C_harm<-neut) / (C_harm - C_neut)
  backward = (C_neut<-harm - C_neut) / (C_harm - C_neut)

Usage:
    python patch_detection_to_refusal.py --model llama2
"""

import os
import json
import argparse
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import MODEL_DICT, load_conversation_template, filter_response

np.random.seed(20)
torch.manual_seed(20)
torch.cuda.manual_seed_all(20)

RESULTS_DIR = "emnlp_results/patch_detection_refusal"
DET_PERCENT = 3.0
REF_PERCENT = 3.0


def load_heads(path):
    with open(path) as f:
        return [(int(d["layer"]), int(d["head"])) for d in json.load(f)]


def token_len(tokenizer, conv, prompt, response):
    conv.messages = []
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    start = len(tokenizer(conv.get_prompt()).input_ids)
    conv.update_last_message(response)
    full = len(tokenizer(conv.get_prompt()).input_ids)
    return start, full


def build_ids(tokenizer, conv, prompt, response, device):
    conv.messages = []
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    start = len(tokenizer(conv.get_prompt()).input_ids)
    conv.update_last_message(response)
    toks = tokenizer(conv.get_prompt(), return_tensors="pt")
    return toks, start, len(tokenizer(conv.get_prompt()).input_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="llama2")
    ap.add_argument("--det_percent", type=float, default=DET_PERCENT)
    ap.add_argument("--ref_percent", type=float, default=REF_PERCENT)
    ap.add_argument("--source", choices=["probing", "responses"], default="probing",
                    help="probing = all 51 pairs w/ fixed refusal response; "
                         "responses = pairs from response file w/ per-pair response "
                         "(filtered, ~17)")
    ap.add_argument("--fixed_response", type=str,
                    default="I'm sorry, but I cannot help with that request.")
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_path = MODEL_DICT[args.model]
    model = (AutoModelForCausalLM
             .from_pretrained(model_path, torch_dtype=torch.float16).to(device).eval())
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    num_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_dim = getattr(model.config, "head_dim", hidden // num_heads)

    refusal_direction = torch.load(
        f"find_refusal_heads/directions/{args.model}_refusal_direction_attn.pt").to(device)
    refusal_direction = refusal_direction / torch.norm(refusal_direction, dim=-1, keepdim=True)

    det_heads = load_heads(f"find_detection_heads/detection_heads/{args.model}_{args.det_percent}.json")
    ref_heads = load_heads(f"find_refusal_heads/refusal_heads/{args.model}_{args.ref_percent}.json")
    det_by_layer = defaultdict(list)
    for l, h in det_heads:
        det_by_layer[l].append(h)
    det_layers = sorted(det_by_layer)
    print(f"{len(det_heads)} detection heads over {len(det_layers)} layers; "
          f"{len(ref_heads)} refusal heads")

    if args.source == "probing":
        import pandas as pd
        df = pd.read_csv("data/filtered_harmful_neutral_prompts.csv")
        fhp = df["Harmful Prompt"].tolist()
        fnp = df["Neutral Prompt"].tolist()
        fhr = [args.fixed_response] * len(fhp)   # same teacher-forced response for all
        print(f"Source: probing set — {len(fhp)} pairs, fixed response "
              f"{args.fixed_response!r}")
    else:
        with open(f"find_refusal_heads/responses/{args.model}.json") as f:
            data = json.load(f)
        hp = [e["Harmful Prompt"] for e in data]
        npr = [e["Neutral Prompt"] for e in data]
        hr = [e["Harmful Response"] for e in data]
        nr = [e["Neutral Response"] for e in data]
        fhp, fnp, fhr, _ = filter_response(hp, npr, hr, nr)
        print(f"Source: response file — {len(fhp)} pairs after filter_response")

    conv = load_conversation_template(args.model)

    # keep only pairs whose (prompt+response) tokenize to identical length
    pairs = []
    for h_prompt, n_prompt, response in zip(fhp, fnp, fhr):
        hs, hf = token_len(tokenizer, conv, h_prompt, response)
        ns, nf = token_len(tokenizer, conv, n_prompt, response)
        if hf == nf and hs == ns:
            pairs.append((h_prompt, n_prompt, response, hs, hf))
    print(f"Length-matched usable pairs: {len(pairs)}/{len(fhp)}")

    # ---------- hooks ----------
    def cache_oproj_inputs(toks):
        """Run a forward pass, return {layer: o_proj input [1,seq,hidden]} for det layers."""
        cache = {}
        hooks = []
        def mk(l):
            def hook(mod, inp, out):
                cache[l] = inp[0].detach().clone()
            return hook
        for l in det_layers:
            hooks.append(model.model.layers[l].self_attn.o_proj.register_forward_hook(mk(l)))
        with torch.no_grad():
            model(input_ids=toks["input_ids"].to(device),
                  attention_mask=toks["attention_mask"].to(device))
        for hk in hooks:
            hk.remove()
        return cache

    def measure(toks, start, end, patch_cache):
        """Refusal-head contribution to refusal dir, optionally patching detection
        head slices with patch_cache."""
        contrib = []
        hooks = []
        if patch_cache is not None:
            def mk_pre(l):
                heads = det_by_layer[l]
                src = patch_cache[l]
                def pre(mod, inp):
                    x = inp[0].clone()
                    for h in heads:
                        s, e = h * head_dim, (h + 1) * head_dim
                        x[:, :, s:e] = src[:, :, s:e]
                    return (x,) + tuple(inp[1:])
                return pre
            for l in det_layers:
                hooks.append(model.model.layers[l].self_attn.o_proj
                             .register_forward_pre_hook(mk_pre(l)))

        def mk_cap():
            def hook(mod, inp, out):
                a = inp[0].detach()[0]                       # [seq, hidden]
                chd = a.size(-1) // num_heads
                a = a.reshape(a.size(0), num_heads, chd)     # [seq, heads, chd]
                w = mod.weight.detach().reshape(hidden, num_heads, chd).permute(1, 2, 0).contiguous()
                contrib.append(torch.einsum("snk,nkh->snh", a, w))  # [seq, heads, hidden]
            return hook
        for layer in model.model.layers:
            hooks.append(layer.self_attn.o_proj.register_forward_hook(mk_cap()))

        with torch.no_grad():
            model(input_ids=toks["input_ids"].to(device),
                  attention_mask=toks["attention_mask"].to(device))
        for hk in hooks:
            hk.remove()

        allc = torch.stack(contrib, dim=0)[:, start - 1:end - 1, :, :].mean(dim=1)  # [L, H, hid]
        dots = torch.einsum("lhd,ld->lh", allc, refusal_direction)                  # [L, H]
        return float(np.mean([dots[l, h].item() for (l, h) in ref_heads]))

    # ---------- run 4 conditions per pair ----------
    C = {"harm": [], "harm<-neut": [], "neut": [], "neut<-harm": []}
    for h_prompt, n_prompt, response, _, _ in tqdm(pairs, desc="patching"):
        h_toks, hs, he = build_ids(tokenizer, conv, h_prompt, response, device)
        n_toks, ns, ne = build_ids(tokenizer, conv, n_prompt, response, device)

        neut_cache = cache_oproj_inputs(n_toks)   # detection outputs on neutral run
        harm_cache = cache_oproj_inputs(h_toks)   # detection outputs on harmful run

        C["harm"].append(measure(h_toks, hs, he, patch_cache=None))
        C["harm<-neut"].append(measure(h_toks, hs, he, patch_cache=neut_cache))
        C["neut"].append(measure(n_toks, ns, ne, patch_cache=None))
        C["neut<-harm"].append(measure(n_toks, ns, ne, patch_cache=harm_cache))

    mean = {k: float(np.mean(v)) for k, v in C.items()}
    gap = mean["harm"] - mean["neut"]
    fwd = (mean["harm"] - mean["harm<-neut"]) / gap if gap else float("nan")
    bwd = (mean["neut<-harm"] - mean["neut"]) / gap if gap else float("nan")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = {
        "model": args.model, "n_pairs": len(pairs),
        "C_harm": round(mean["harm"], 5),
        "C_harm_from_neut": round(mean["harm<-neut"], 5),
        "C_neut": round(mean["neut"], 5),
        "C_neut_from_harm": round(mean["neut<-harm"], 5),
        "gap_harm_minus_neut": round(gap, 5),
        "forward_effect_frac": round(fwd, 3),
        "backward_effect_frac": round(bwd, 3),
    }
    with open(f"{RESULTS_DIR}/patch_{args.model}.json", "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 60)
    print(f"N2 Activation Patching — {args.model}  (n={len(pairs)} pairs)")
    print("=" * 60)
    print(f"  C_harm (unpatched)            : {mean['harm']:.4f}")
    print(f"  C_harm  <- detection(neutral) : {mean['harm<-neut']:.4f}   (expect DOWN)")
    print(f"  C_neut (unpatched)            : {mean['neut']:.4f}")
    print(f"  C_neut  <- detection(harmful) : {mean['neut<-harm']:.4f}   (expect UP)")
    print(f"  harmful-neutral gap           : {gap:.4f}")
    print(f"  forward  effect (h<-n / gap)  : {fwd:.1%}")
    print(f"  backward effect (n<-h / gap)  : {bwd:.1%}")
    print(f"\nSaved to {RESULTS_DIR}/patch_{args.model}.json")


if __name__ == "__main__":
    main()
