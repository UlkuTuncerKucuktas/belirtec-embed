#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

STAGE="${1:-all}"
PUSH=0
[ "${2:-}" = "--push" ] && PUSH=1
[ "${1:-}" = "--push" ] && { STAGE="all"; PUSH=1; }

run_filter()   { echo "=== filter legal (leakage gate) ==="; python scripts/filter_legal.py; }
run_generate() { echo "=== generate ==="; python scripts/generate.py; }
run_mine()     { echo "=== mine hard negatives ==="
  python scripts/mine.py --in_file data/raw/legal.jsonl     --out_file data/mined/legal.jsonl
  python scripts/mine.py --in_file data/raw/retrieval.jsonl --out_file data/mined/retrieval.jsonl; }
run_score()    { echo "=== forgetting score (legal) ==="
  python scripts/score.py --candidate data/mined/legal.jsonl --out_file data/scored/legal.jsonl \
    --preserve data/mined/retrieval.jsonl data/raw/sts.jsonl; }
run_push()     { echo "=== push to HF ==="; python scripts/push.py; }

case "$STAGE" in
  filter)   run_filter ;;
  generate) run_generate ;;
  mine)     run_mine ;;
  score)    run_score ;;
  all)      run_filter; run_generate; run_mine; run_score; [ "$PUSH" = 1 ] && run_push || true ;;
  *) echo "usage: bash scripts/data.sh [filter|generate|mine|score|all] [--push]"; exit 1 ;;
esac
echo "=== stage '$STAGE' complete ==="
