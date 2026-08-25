import os
import json
import argparse
import torch

import numpy as np
import pandas as pd

from transformers import AutoConfig, GenerationConfig, AutoTokenizer
from vllm import LLM, SamplingParams
from utils import load_conversation_template, MODEL_DICT

np.random.seed(20)
torch.manual_seed(20)
torch.cuda.manual_seed_all(20)
device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="llama3")
parser.add_argument("--tensor_parallel_size", default=1, help="Tensor parallel size for vLLM")
args = parser.parse_args()

# Each harmful-neutral prompt differs in only one word
data = pd.read_csv("../data/filtered_harmful_neutral_prompts.csv")
harmful_prompt = data['Harmful Prompt'].tolist()
neutral_prompt = data['Neutral Prompt'].tolist()
conv = load_conversation_template(args.model)

model_path = MODEL_DICT[args.model]
max_pos = AutoConfig.from_pretrained(model_path).max_position_embeddings
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

print("target model:", args.model)

llm = LLM(
    model=model_path,
    tensor_parallel_size=args.tensor_parallel_size,
    max_model_len=max_pos
)

config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
gen_cfg = GenerationConfig.from_pretrained(model_path)

sampling_params = SamplingParams(
    temperature=getattr(gen_cfg, 'temperature', 0.6),
    top_p=getattr(gen_cfg, 'top_p', 0.95),
    top_k=getattr(gen_cfg, 'top_k', None),
    repetition_penalty=getattr(gen_cfg, 'repetition_penalty', 1.0),
    max_tokens=100,
    stop_token_ids=[tokenizer.eos_token_id]
)

harmful_prompts = []
neutral_prompts = []

for neutral in neutral_prompt:
    conv.messages = []
    conv.append_message(conv.roles[0], neutral)
    conv.append_message(conv.roles[1], None)
    neutral_prompts.append(conv.get_prompt())

for harmful in harmful_prompt:
    conv.messages = []
    conv.append_message(conv.roles[0], harmful)
    conv.append_message(conv.roles[1], None)
    harmful_prompts.append(conv.get_prompt())

print(f"Running batch inference on {len(harmful_prompts)} examples with model {args.model}")
outputs = llm.generate(harmful_prompts, sampling_params)
harmful_output = [output.outputs[0].text for output in outputs]

outputs = llm.generate(neutral_prompts, sampling_params)
neutral_output = [output.outputs[0].text for output in outputs]

response_data = []
for h, h_o, n, n_o in zip(harmful_prompt, harmful_output, neutral_prompt, neutral_output):
    response_data.append({
        "Harmful Prompt": h,
        "Harmful Response": h_o,
        "Neutral Prompt": n,
        "Neutral Response": n_o,
    })

os.makedirs("responses", exist_ok=True)
output_path = f"responses/{args.model}.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(response_data, f, indent=4)
print(f"Save to {output_path}")
    