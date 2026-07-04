#!/usr/bin/env bash
# run_matrix.sh — run all experiments across N GPUs, one per GPU, launching the
# next queued experiment on a GPU as soon as its current run finishes.
#
# Usage:
#   bash run_matrix.sh                 # all experiments on GPUs 0-3
#   bash run_matrix.sh 0,1             # only GPUs 0 and 1
#
# Re-run safe: experiments whose runs/<name>/final exists are skipped, so a
# partial failure resumes cleanly on re-run.

set -u
cd "$(dirname "$0")"

GPUS="${1:-${NGPU_LIST:-0,1,2,3}}"
IFS=',' read -r -a GPU_ARR <<< "$GPUS"

# queue: longest first (gist ~7h, staged ~4h) so GPUs stay saturated; short after
QUEUE=(
  lora_allsynth_gist
  lora_allsynth_msmarco_gist
  lora_allsynth_nolegal_msmarco_gist
  lora_staged_safe2hard_mnrl
  fullft_allsynth_mnrl
  fullft_realonly_mnrl
  fullft_singlemodel_mnrl
  fullft_multimodel_sizematch_mnrl
  fullft_nolegal_mnrl
  lora_allsynth_mnrl
  lora_allsynth_safe_mnrl
  lora_allsynth_hard_mnrl
)

POLL="${POLL:-30}"
mkdir -p runs logs
LOGDIR="logs/matrix_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$LOGDIR"
echo "[matrix] GPUs: ${GPU_ARR[*]} | queued: ${#QUEUE[@]} | poll ${POLL}s | dispatch: $LOGDIR/dispatch.log"

declare -A gpu_pid gpu_exp
QIDX=0

take_next() {
  REPLY=""
  while [ "$QIDX" -lt "${#QUEUE[@]}" ]; do
    local e="${QUEUE[$QIDX]}"
    QIDX=$((QIDX+1))
    if [ -e "runs/$e/final" ]; then
      echo "[skip] $e already has runs/$e/final"
      continue
    fi
    REPLY="$e"; return 0
  done
  return 0
}

start_on() {
  local gpu="$1"
  take_next
  local exp="$REPLY"
  if [ -z "$exp" ]; then return 1; fi
  echo "[launch] gpu$gpu <- $exp  ($(date +%H:%M:%S))"
  nohup python scripts/train_and_eval.py --experiment "$exp" --gpu "$gpu" \
      > "runs/$exp.log" 2>&1 &
  gpu_pid[$gpu]=$!
  gpu_exp[$gpu]="$exp"
  echo "$(date +%H:%M:%S) START gpu$gpu pid ${gpu_pid[$gpu]} $exp" >> "$LOGDIR/dispatch.log"
  return 0
}

for gpu in "${GPU_ARR[@]}"; do
  start_on "$gpu" || break
done

while :; do
  any_running=0
  for gpu in "${GPU_ARR[@]}"; do
    pid="${gpu_pid[$gpu]:-}"
    [ -z "$pid" ] && continue
    if kill -0 "$pid" 2>/dev/null; then
      any_running=1
    else
      finished="${gpu_exp[$gpu]}"
      wait "$pid" 2>/dev/null; rc=$?
      tag="OK"; [ "$rc" -ne 0 ] && tag="EXIT=$rc"
      nanct=$(grep -c "Input contains NaN\|grad_norm': nan" "runs/$finished.log" 2>/dev/null || echo 0)
      done_flag=$([ -e "runs/$finished/final" ] && echo "final-ok" || echo "NO-FINAL")
      echo "[done] gpu$gpu  $finished  ($tag, nan=$nanct, $done_flag)  $(date +%H:%M:%S)"
      echo "$(date +%H:%M:%S) DONE gpu$gpu $finished $tag nan=$nanct $done_flag" >> "$LOGDIR/dispatch.log"
      gpu_pid[$gpu]=""; gpu_exp[$gpu]=""
      if start_on "$gpu"; then any_running=1; fi
    fi
  done
  [ "$any_running" -eq 0 ] && break
  sleep "$POLL"
done

echo "[matrix] ALL DONE. dispatch log: $LOGDIR/dispatch.log"
echo "[matrix] results: bash results_table.sh"
