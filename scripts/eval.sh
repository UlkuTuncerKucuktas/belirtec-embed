#!/usr/bin/env bash
# Usage: bash scripts/eval.sh <model_path> [gpu] [output_dir]
#   bash scripts/eval.sh ./runs/champion/final 3 results
set -euo pipefail
MODEL="${1:?model path required}"
GPU="${2:-0}"
OUT="${3:-results}"
python scripts/eval.py --model "$MODEL" --gpu "$GPU" --output-dir "$OUT"
