"""
N4: Component specificity — is the circuit harm-specific or a generic
difficulty/novelty detector? (Reviewer a8az-W3)

a8az-W3: the components matter for safety, but they may simply respond to any
unusual/difficult input. Test whether they stay quiet on hard-but-benign prompts.

Method: for each input we run a forward pass (prompt only) and measure the circuit's
refusal signal = the refusal heads' contribution to the refusal direction at the
last prompt token (the same quantity as N2/N3). We also report the detection heads'
contribution. If the signal is high on harmful inputs but low on hard/ambiguous
BENIGN inputs, the circuit tracks harm, not difficulty.

Input types:
  harmful          : AdvBench harmful behaviors
  hard_math        : GSM8K (difficult but benign)
  hard_knowledge   : MMLU hard subjects (college math, pro law, ... ; benign)
  ambiguous_benign : OR-Bench (looks harmful, is safe)
  easy_benign      : neutral probing prompts (floor)

Usage:
    python component_specificity.py --model llama2
    python component_specificity.py --model llama2 --n 20   # smoke test
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import MODEL_DICT, load_conversation_template

np.random.seed(20)
torch.manual_seed(20)

RESULTS_DIR = "emnlp_results/component_specificity"


def load_heads(path):
    with open(path) as f:
        return [(int(d["layer"]), int(d["head"])) for d in json.load(f)]


def load_inputs(n, seed=20):
    """Return {type: [prompts]}."""
    rng = np.random.RandomState(seed)
    out = {}

    # harmful — AdvBench
    harm = pd.read_csv("data/advbench/harmful_behaviors.csv")["goal"].tolist()
    out["harmful"] = list(rng.choice(harm, size=min(n, len(harm)), replace=False))

    # easy benign — neutral probing prompts
    neut = pd.read_csv("data/filtered_harmful_neutral_prompts.csv")["Neutral Prompt"].tolist()
    out["easy_benign"] = list(rng.choice(neut, size=min(n, len(neut)), replace=False))

    from datasets import load_dataset

    # hard benign — GSM8K
    try:
        g = load_dataset("openai/gsm8k", "main", split="test")
        qs = g["question"]
        out["hard_math_gsm8k"] = [qs[i] for i in rng.choice(len(qs), size=min(n, len(qs)), replace=False)]
    except Exception as e:
        print(f"[warn] GSM8K unavailable: {e}")

    # hard benign — MMLU hard subjects
    try:
        subs = ["college_mathematics", "professional_law", "abstract_algebra", "formal_logic"]
        mq = []
        for s in subs:
            d = load_dataset("cais/mmlu", s, split="test")
            mq += list(d["question"])
        out["hard_knowledge_mmlu"] = [mq[i] for i in rng.choice(len(mq), size=min(n, len(mq)), replace=False)]
    except Exception as e:
        print(f"[warn] MMLU unavailable: {e}")

    # ambiguous benign — OR-Bench (looks harmful, is safe)
    try:
        ob = load_dataset("bench-llm/or-bench", "or-bench-80k", split="train")
        op = ob["prompt"]
        out["ambiguous_benign_orbench"] = [op[i] for i in rng.choice(len(op), size=min(n, len(op)), replace=False)]
    except Exception as e:
        print(f"[warn] OR-Bench unavailable: {e}")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="llama2")
    ap.add_argument("--det_percent", type=float, default=3.0)
    ap.add_argument("--ref_percent", type=float, default=3.0)
    ap.add_argument("--n", type=int, default=80, help="prompts per input type")
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_path = MODEL_DICT[args.model]
    model = (AutoModelForCausalLM
             .from_pretrained(model_path, torch_dtype=torch.float16).to(device).eval())
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    hidden = model.config.hidden_size
    num_heads = model.config.num_attention_heads

    refusal_direction = torch.load(
        f"find_refusal_heads/directions/{args.model}_refusal_direction_attn.pt").to(device)
    refusal_direction = refusal_direction / torch.norm(refusal_direction, dim=-1, keepdim=True)

    det_heads = load_heads(f"find_detection_heads/detection_heads/{args.model}_{args.det_percent}.json")
    ref_heads = load_heads(f"find_refusal_heads/refusal_heads/{args.model}_{args.ref_percent}.json")

    conv = load_conversation_template(args.model)

    def signal(prompt):
        conv.messages = []
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        toks = tokenizer(conv.get_prompt(), return_tensors="pt", truncation=True, max_length=512)
        contrib = []
        hooks = []
        def mk():
            def hook(mod, inp, out):
                a = inp[0].detach()[0]
                chd = a.size(-1) // num_heads
                a = a.reshape(a.size(0), num_heads, chd)
                w = mod.weight.detach().reshape(hidden, num_heads, chd).permute(1, 2, 0).contiguous()
                contrib.append(torch.einsum("snk,nkh->snh", a, w))
            return hook
        for layer in model.model.layers:
            hooks.append(layer.self_attn.o_proj.register_forward_hook(mk()))
        with torch.no_grad():
            model(input_ids=toks["input_ids"].to(device),
                  attention_mask=toks["attention_mask"].to(device))
        for hk in hooks:
            hk.remove()
        last = torch.stack(contrib, dim=0)[:, -1, :, :]            # [L, H, hid] at last prompt token
        dots = torch.einsum("lhd,ld->lh", last, refusal_direction)  # [L, H]
        ref = float(np.mean([dots[l, h].item() for (l, h) in ref_heads]))
        det = float(np.mean([dots[l, h].item() for (l, h) in det_heads]))
        return ref, det

    inputs = load_inputs(args.n)
    print("input types:", {k: len(v) for k, v in inputs.items()})

    rows = []
    for typ, prompts in inputs.items():
        refs, dets = [], []
        for p in tqdm(prompts, desc=typ, leave=False):
            r, d = signal(str(p))
            refs.append(r); dets.append(d)
        rows.append({"input_type": typ, "n": len(prompts),
                     "refusal_signal": round(float(np.mean(refs)), 4),
                     "detection_signal": round(float(np.mean(dets)), 4)})

    df = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(f"{RESULTS_DIR}/specificity_{args.model}.csv", index=False)

    # order for display
    order = ["harmful", "ambiguous_benign_orbench", "hard_math_gsm8k",
             "hard_knowledge_mmlu", "easy_benign"]
    df["__o"] = df["input_type"].apply(lambda t: order.index(t) if t in order else 99)
    df = df.sort_values("__o").drop(columns="__o")

    print("\n" + "=" * 66)
    print(f"N4 Component Specificity — {args.model}")
    print("=" * 66)
    print(df.to_string(index=False))
    harm = df[df.input_type == "harmful"]["refusal_signal"].iloc[0]
    print(f"\nHarmful refusal signal = {harm:.4f}")
    for _, r in df[df.input_type != "harmful"].iterrows():
        print(f"  {r['input_type']:28s}: {r['refusal_signal']:.4f}  "
              f"({100*r['refusal_signal']/harm:.0f}% of harmful)")
    print(f"\nSaved to {RESULTS_DIR}/specificity_{args.model}.csv")


if __name__ == "__main__":
    main()
