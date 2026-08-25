# From Detection to Refusal: Safer LLMs via Circuit-Guided Weight Scaling

Code for the EMNLP 2026 Findings paper. We characterize a three-stage **safety
circuit** in LLMs — Harmful Detection Heads, Safety Neurons, and Refusal Heads —
and use it to build **CBSE (Circuit-Based Safety Editing)**, a training-free
weight-scaling intervention that improves adversarial safety while largely
preserving utility.

Project page: https://lilywenglab.github.io/Detection2Refusal/ ·
Paper: TBD

This repository contains only the code paths that produce a table or figure in
the paper (main text or appendix). Every script below is annotated with the
exact `\label{}` it feeds.

## Repository layout

```
utils.py                          # MODEL_DICT, conversation templates, shared helpers
requirement.txt

find_detection_heads/             # Identify Harmful Detection Heads
find_refusal_heads/               # Identify Refusal Heads + refusal direction
find_safety_neurons/              # Identify Safety Neurons + safety direction

experiment1_causal_validation/    # Experiment I (main text): ablation + activation patching
experiment2_3_cbse/               # Experiment II/III (main text): CBSE intervention + evaluation
appendix_experiments/             # Appendix experiments (probe robustness, PAIR/AutoDAN,
                                   #   defense baselines, component specificity, multi-seed, ...)

data/                             # Probing pairs, AdvBench, MaliciousInstruct, AutoDAN, GCG suffixes
```

Six architectures are used throughout: `llama2`, `llama3`, `mistral`,
`guanaco`, `qwen2.5`, `qwen3` (see `MODEL_DICT` in `utils.py` for the exact
HuggingFace repo IDs). All commands below take `--model <name>` from this set
unless noted otherwise.

**Working directory.** Every script under `experiment1_causal_validation/`,
`experiment2_3_cbse/`, and `appendix_experiments/` loads its inputs (e.g.
`find_detection_heads/detection_heads/...`, `data/...`) as paths relative to
the **repository root**, not its own folder — run them from the repo root as
`python experiment1_causal_validation/<script>.py ...`, do not `cd` into the
subfolder first. The three `find_*/` circuit-identification folders are the
one exception: their scripts resolve paths relative to their *own* folder, so
those are run with `cd find_detection_heads && python find_detection_heads.py ...`.

## Setup

```bash
pip install -r requirement.txt
# also needed, not pinned in requirement.txt:
pip install protobuf
```

Scripts that only need a forward pass (circuit identification, Experiment I,
`component_specificity.py`, `evaluate_probe_robustness.py`) run on a single
GPU with 🤗 `transformers`. Scripts that generate text at scale (Experiment
II/III, PAIR/AutoDAN, defense baselines, over-refusal, multi-seed) use `vllm`
and expose a `--tp` (tensor-parallel size) flag.

## Step 0 — Identify the circuit

The circuit is identified from 51 paired harmful/neutral prompts
(`data/filtered_harmful_neutral_prompts.csv`) that differ by a single word.
Precomputed components for all six models at the percentages used in the
paper are already included under `*/detection_heads/`, `*/refusal_heads/`,
`*/safety_neurons/`, `*/directions/`, and `*/responses/` — you only need to
re-run these if you want to regenerate from scratch or probe a new model.

```bash
cd find_detection_heads && python find_detection_heads.py --model llama2 --percent 3.0
cd find_refusal_heads   && python find_refusal_heads.py   --model llama2 --percent 3.0
cd find_safety_neurons  && python find_safety_mlp.py      --model llama2 --percent 1.0
```

This is the pipeline behind **Figure 1** (`fig:safety_circuit`, Method §3.1).

## Experiment I — Causal Validation (`Experiment.tex`)

**Targeted component ablation** (`fig:head_ablation`, `fig:neuron_ablation`,
`fig:circuit_ablation_combined`): zero out an increasing fraction of detection
heads / safety neurons and measure the resulting drop in refusal-head
contribution, against a random-component control.

```bash
python experiment1_causal_validation/exp1_refusal_contribution.py         --model llama2 --det_percents 0 1 3 5 7 10
python experiment1_causal_validation/exp1_refusal_contribution_neurons.py --model llama2 --neuron_percents 0 1 3 5
python experiment1_causal_validation/exp1_refusal_contribution_random.py         --model llama2 --det_percents 0 1 3 5 7 10 --n_seeds 10
python experiment1_causal_validation/exp1_refusal_contribution_random_neurons.py --model llama2 --neuron_percents 0 1 3 5 --n_seeds 10

# Each pair above writes CSVs into emnlp_results/exp1_refusal_contribution{,_neurons}/
# (created relative to the repo root); the matching draw script must be run
# from that directory to find them and produce Figure fig:head_ablation /
# fig:neuron_ablation:
(cd emnlp_results/exp1_refusal_contribution        && python ../../experiment1_causal_validation/draw_heads.py)
(cd emnlp_results/exp1_refusal_contribution_neurons && python ../../experiment1_causal_validation/draw_neurons.py)
```

**Activation patching** (`tab:activation_patching`): swap detection heads'
output between paired harmful/neutral runs to test the detection→refusal
pathway directly.

```bash
python experiment1_causal_validation/patch_detection_to_refusal.py --model llama2
```

## Experiment II/III — CBSE Intervention (`Experiment.tex`)

**Circuit necessity / knockout** (`tab:circuit_removal_results`, Appendix C
`sec:circuit_removal`): zero out 0/1/3/5% of the circuit and measure the
resulting collapse in GCG safety rate.

```bash
python experiment2_3_cbse/exp2_safety_ablation.py --model llama3 --n_data 100 --resume
```

**Main safety-rate table** (`tab:safety_rates`, Pure/GCG/ADV-LLM ×
Detection/Refusal/Safety/combinations), **GCG regenerated against the edited
model** (`tab:gcg_regenerate`), and the two appendix variants that reuse the
same evaluation engine — **Safe-Template** (`tab:safety_rates_template`) and
**Malicious Instruction Dataset** (`tab:guard_mlcinst`, `tab:template_mlcinst`,
via `--dataset mlcinst`):

```bash
# Pure Harmful Prompt / GCG / ADV-LLM, jointly reinforcing all three component types
python experiment2_3_cbse/evaluate_neurons_heads.py --model llama3 --attack none  --neuron_factor 1.5 --detection_factor 2.0 --refusal_factor 2.0
python experiment2_3_cbse/evaluate_neurons_heads.py --model llama3 --attack gcg   --neuron_factor 1.5 --detection_factor 2.0 --refusal_factor 2.0
python experiment2_3_cbse/evaluate_neurons_heads.py --model llama3 --attack advllm --neuron_factor 1.5 --detection_factor 2.0 --refusal_factor 2.0

# GCG regenerated against the persistently-edited model (Table: tab:gcg_regenerate)
python experiment2_3_cbse/evaluate_neurons_heads.py --model llama3 --attack gcg --intervened \
    --neuron_factor 1.5 --detection_factor 2.0 --refusal_factor 2.0

# Malicious Instruction Dataset variant (Appendix H)
python experiment2_3_cbse/evaluate_neurons_heads.py --model llama3 --attack advllm --dataset mlcinst \
    --neuron_factor 1.5 --detection_factor 2.0 --refusal_factor 2.0
```

Each run reports both `safe_guard` (Llama-Guard — the main-text number) and
`safe_template` (rule-based Safe-Template — the appendix number) in the same
row, so a single sweep populates all four tables above.

**Refusal calibration / over-refusal** (`tab:over_refusal_results`,
`sec:over_refusal`), on the OR-Bench-80k borderline-benign subset:

```bash
python appendix_experiments/evaluate_over_refusal.py --model llama2 --max_prompts 1000 --tp 4
```

Utility (**`tab:model_accuracy`**: MMLU/HellaSwag/PIQA/WinoGrande) and
**`tab:perplexity`** (WikiText) are *not* produced by a script in this repo —
they are evaluated with the external
[`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness)
(5-shot, default generation config) against the model checkpoint that
`evaluate_neurons_heads.py --intervened` persists to disk (`save_pretrained`).

## Appendix Experiments

| Table / Figure | Script | Notes |
|---|---|---|
| `tab:probe_robustness_results` (App. A) | `evaluate_probe_robustness.py` | Bootstrap stability of the circuit over 80%-subsampled probing pairs |
| `tab:multiseed` (App. A) | `evaluate_multiseed.py` | 5-seed GCG mean±std |
| `tab:pair_attack_results` (App. B) | `evaluate_pair.py` | PAIR conversational attack |
| `tab:autodan_attack_results` (App. B) | `evaluate_autodan.py --autodan_path data/autodan/<model>_responses.json` | AutoDAN-HGA |
| `tab:component_specificity` (App. D) | `component_specificity.py` | Detection/refusal signal on harmful vs. hard/ambiguous-benign inputs |
| `tab:comp_circuitbreakers`, `tab:comp_lat` (App. E) | `evaluate_defense_baseline.py --claude_judge` | vs. CircuitBreakers / LAT |
| `tab:comp_trainfree` (App. E) | `evaluate_trainfree_baselines.py` | vs. SmoothLLM / Activation-Steering |
| `tab:1p_heads`/`3p_heads`/`5p_heads`/`head_ablation_mmlu` (App. F) | `evaluate_heads.py` | Head-only $\alpha_{det}/\alpha_{ref}$ grid at 1/3/5% |
| `tab:neuron_ablation`/`neuron_ablation_mmlu` (App. F) | `evaluate_neurons_heads.py` | Neuron $\alpha_{saf}$ grid at fixed head config |
| `tab:claude_judge_results` (App. G) | `evaluate_claude_judge.py` | Independent Claude judge on saved GCG responses |

`tab:human_eval` (App. G) is not included — it is a manual annotation process
with no reusable code, not a script-driven experiment.

Example commands:

```bash
python appendix_experiments/evaluate_probe_robustness.py --model llama2 --n_subsets 5
python appendix_experiments/evaluate_pair.py --model llama2 --n_goals 50 --max_iters 20
python appendix_experiments/evaluate_autodan.py --model llama2 --autodan_path data/autodan/llama2_responses.json --tp 4
python appendix_experiments/component_specificity.py --model llama2 --n 80
python appendix_experiments/evaluate_defense_baseline.py --tp 4 --claude_judge
python appendix_experiments/evaluate_trainfree_baselines.py --model llama2 --attack gcg
python appendix_experiments/evaluate_claude_judge.py --mode gcg_all
```

## Data

- `data/filtered_harmful_neutral_prompts.csv` — 51 paired harmful/neutral
  probing prompts, used to identify every circuit component.
- `data/advbench/harmful_behaviors.csv` — AdvBench harmful behaviors.
- `data/MaliciousInstruct.txt` — Malicious Instruction Dataset (Huang et al.).
- `data/autodan/<model>_responses.json` — pre-generated AutoDAN-HGA prompts.
- `data/gcg_results/<model>/<config>/adv_suffix_*.csv` — GCG suffixes per
  model and intervention configuration.

## Citation

```bibtex
@inproceedings{chu2026detection,
  title={From Detection to Refusal: Safer LLMs via Circuit-Guided Weight Scaling},
  author={Chu, Kuan-Lin and Sun, Chung-En and Weng, Tsui-Wei},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2026},
  year={2026}
}
```
