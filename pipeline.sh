#!/usr/bin/env bash
#
# End-to-end CBSE pipeline: identify the safety circuit for one model, apply
# Circuit-Based Safety Editing, and evaluate the edited model under attack.
#
#   ./pipeline.sh                          # llama3, GCG attack (defaults)
#   ./pipeline.sh --model mistral          # another model
#   ./pipeline.sh --attack advllm          # another attack: none | gcg | advllm
#   ./pipeline.sh --n-data 25 --skip-circuit   # quick re-eval on cached circuit
#
# Requirements: 1 GPU (>= 24 GB), and HuggingFace access to the target model
# and to meta-llama/Llama-Guard-3-8B (run `huggingface-cli login` first).

set -euo pipefail
cd "$(dirname "$0")"

MODEL=llama3
ATTACK=gcg
N_DATA=100
TP=1
SKIP_CIRCUIT=0

# CBSE hyper-parameters (paper defaults; see the paper's Table of safety rates).
HEAD_PERCENT=3.0
NEURON_PERCENT=1.0
NEURON_FACTOR=1.5
DETECTION_FACTOR=2.0
REFUSAL_FACTOR=2.0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)        MODEL="$2"; shift 2 ;;
        --attack)       ATTACK="$2"; shift 2 ;;
        --n-data)       N_DATA="$2"; shift 2 ;;
        --tp)           TP="$2"; shift 2 ;;
        --skip-circuit) SKIP_CIRCUIT=1; shift ;;
        -h|--help)      sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

case "$ATTACK" in none|gcg|advllm) ;; *) echo "--attack must be none|gcg|advllm" >&2; exit 1 ;; esac

echo "=============================================================="
echo " model=$MODEL  attack=$ATTACK  n_data=$N_DATA  tp=$TP"
echo "=============================================================="

DET_JSON="scripts/find_detection_heads/detection_heads/${MODEL}_${HEAD_PERCENT}.json"
REF_JSON="scripts/find_refusal_heads/refusal_heads/${MODEL}_${HEAD_PERCENT}.json"
NEU_JSON="scripts/find_safety_neurons/safety_neurons/${MODEL}_${NEURON_PERCENT}.json"

if [[ "$SKIP_CIRCUIT" -eq 1 && -f "$DET_JSON" && -f "$REF_JSON" && -f "$NEU_JSON" ]]; then
    echo ">>> Step 0 skipped (circuit components already present)"
else
    echo ">>> Step 0a: Harmful Detection Heads"
    ( cd scripts/find_detection_heads
      python find_detection_heads.py   --model "$MODEL" --percent "$HEAD_PERCENT" )

    echo ">>> Step 0b: Refusal Heads"
    ( cd scripts/find_refusal_heads
      python response_generate.py     --model "$MODEL" --tensor_parallel_size "$TP"
      python extract_direction_attn.py --model "$MODEL"
      python find_refusal_heads.py     --model "$MODEL" --percent "$HEAD_PERCENT" )

    echo ">>> Step 0c: Safety Neurons"
    ( cd scripts/find_safety_neurons
      python response_generate.py     --model "$MODEL" --tensor_parallel_size "$TP"
      python extract_direction_mlp.py  --model "$MODEL"
      python find_safety_mlp.py        --model "$MODEL" --percent "$NEURON_PERCENT" )
fi

echo ">>> Step 1: Apply CBSE and evaluate under attack=$ATTACK"
mkdir -p results
python experiments/evaluate_neurons_heads.py \
    --model "$MODEL" --attack "$ATTACK" --n_train_data "$N_DATA" \
    --head_percent "$HEAD_PERCENT" --neuron_percent "$NEURON_PERCENT" \
    --neuron_factor "$NEURON_FACTOR" \
    --detection_factor "$DETECTION_FACTOR" --refusal_factor "$REFUSAL_FACTOR" \
    --save_path results/cbse_safety_rates.csv

echo "=============================================================="
echo " Done. Safety rates appended to results/cbse_safety_rates.csv"
echo "=============================================================="
if command -v column >/dev/null 2>&1; then
    column -t -s, results/cbse_safety_rates.csv
else
    cat results/cbse_safety_rates.csv
fi
