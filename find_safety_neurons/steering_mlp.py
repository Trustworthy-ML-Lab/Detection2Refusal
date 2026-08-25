import gc
import os
import argparse
import torch
import torch.nn as nn
import csv
import pickle
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, GenerationConfig
from datasets import load_dataset
from utils import MODEL_DICT, test_prefixes, load_conversation_template, merge_csv, get_goals_and_targets

np.random.seed(20)
torch.manual_seed(20)
torch.cuda.manual_seed_all(20)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="llama3")
parser.add_argument("--direction_weight", type=float, default=0.00)
parser.add_argument("--attack", type=str, default="advllm", help="benign, none, gcg, advllm")
parser.add_argument("--remove_sys_prompt", action=argparse.BooleanOptionalAction)
parser.add_argument("--n_train_data", type=int, default=100)
parser.add_argument("--dataset", type=str, default="advbench", help="advbench or mlcinst")
parser.add_argument("--offset", type=int, default=0)
args = parser.parse_args()

model_path = MODEL_DICT[args.model]
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
direction = torch.load(f"directions/{args.model}_safety_direction_mlp.pt").to(device)

def adjust_residual_hook(layer_idx):
    def hook_fn(module, input, output):
        return output + args.direction_weight * direction[layer_idx]
    return hook_fn

print("add mlp hook")
for i, layer in enumerate(model.model.layers):
    layer.mlp.register_forward_hook(adjust_residual_hook(i))


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

with open(f"../data/adv_prompts/advbench/greedy/{args.model}.pkl", 'rb') as f:
    adv_prompts = pickle.load(f)[: args.n_train_data]

gcg_df = merge_csv(f"../data/gcg_results/{args.model}/original/")
gcg_g = gcg_df['prompt'].tolist()
gcg_adv = gcg_df['adv'].tolist()

# -------------------- Run attacks --------------------
conv = load_conversation_template(args.model)
if args.remove_sys_prompt:
    conv.set_system_message("")

advs = []
responses = []
print("Running attacks...")

if args.attack in ['benign','none','advllm']:
    queries = {
        'benign': benign_prompts,
        'none':   goals,
        'advllm': adv_prompts
    }[args.attack]
    for q in tqdm(queries):
        conv.messages = []
        conv.append_message(conv.roles[0], q)
        conv.append_message(conv.roles[1], None)
        toks = tokenizer(conv.get_prompt(), return_tensors="pt").to(device)
        out_ids = model.generate(
            input_ids=toks.input_ids,
            attention_mask=toks.attention_mask,
            max_new_tokens=50
        )[0]
        out = tokenizer.decode(out_ids[len(toks.input_ids[0]):])
        print(out)
        advs.append(q)
        responses.append(out)
else:
    for g, a in tqdm(zip(gcg_g, gcg_adv), total=len(gcg_g)):
        conv.messages = []
        conv.append_message(conv.roles[0], f"{g} {a}")
        conv.append_message(conv.roles[1], None)
        toks = tokenizer(conv.get_prompt(), return_tensors="pt").to(device)
        out_ids = model.generate(
            input_ids=toks.input_ids,
            attention_mask=toks.attention_mask,
            max_new_tokens=100
        )[0]
        out = tokenizer.decode(out_ids[len(toks.input_ids[0]):])
        print(out)
        advs.append(f"{g} {a}")
        responses.append(out)

# -------------------- Cleanup --------------------
del model, out_ids
gc.collect(); torch.cuda.empty_cache()

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
    device_map=device
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

safe_templ = 1 - (template_fail/len(responses))
safe_guard = 1 - (guard_fail/len(responses))
print(f"Safe rate (template): {safe_templ:.3f}")
print(f"Safe rate (llama guard): {safe_guard:.3f}")

os.makedirs("results/eval", exist_ok=True)
file_exists = os.path.exists(f"results/eval/{args.model}.csv")
with open(f"results/eval/{args.model}.csv", 'a') as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(['attack', 'direction_weight', 'safe_template', 'safe_guard'])
    writer.writerow([args.attack, args.direction_weight, f"{safe_templ:.3f}", f"{safe_guard:.3f}"])

os.makedirs(f"results/responses/{args.model}", exist_ok=True)
with open(f"results/responses/{args.model}/{args.attack}_{args.direction_weight}.txt", 'a') as f:
    for adv, resp in zip(advs, responses):
        f.write(f"Adv: {adv}\nResponse: {resp}\n\n")