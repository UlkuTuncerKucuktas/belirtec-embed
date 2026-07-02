#!/usr/bin/env bash
# Usage: bash scripts/train.sh [experiment] [gpu]
#   bash scripts/train.sh champion 0
#   bash scripts/train.sh gist 1
#   bash scripts/train.sh staged 2
set -euo pipefail
EXP="${1:-champion}"
GPU="${2:-0}"
python scripts/train.py --experiment "$EXP" --gpu "$GPU"
