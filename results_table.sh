#!/usr/bin/env bash
# results_table.sh — pull final eval row from every experiment's log into one table.
cd "$(dirname "$0")"
EXPS=(
  fullft_realonly_mnrl fullft_allsynth_mnrl fullft_singlemodel_mnrl
  fullft_multimodel_sizematch_mnrl fullft_nolegal_mnrl
  lora_allsynth_mnrl lora_allsynth_safe_mnrl lora_allsynth_hard_mnrl
  lora_staged_safe2hard_mnrl lora_allsynth_gist
  lora_allsynth_nolegal_msmarco_gist lora_allsynth_msmarco_gist
)
python3 - "${EXPS[@]}" <<'PY'
import re, os, sys
exps = sys.argv[1:]
hdr = f"{'experiment':<38}{'Legal':>7}{'MTEB':>7}{'Contr':>7}{'Regul':>7}{'Case':>7}{'Class':>7}{'Clust':>7}{'PairC':>7}{'Retr':>7}{'STS':>7}  status"
print(hdr); print("-"*len(hdr))
# capture the model row: "<path>/final <11 floats>" — tolerate trailing param col / spacing
row_re = re.compile(r'/final\s+((?:[\d.]+\s+){10}[\d.]+)')
for e in exps:
    log = f"runs/{e}.log"
    if not os.path.exists(log):
        print(f"{e:<38}{'—':>7}{'':>63}  no log"); continue
    txt = open(log, errors="ignore").read()
    nan = len(re.findall(r"Input contains NaN|grad_norm': nan", txt))
    m = row_re.findall(txt)
    if not m:
        prog = "running/none" if nan == 0 else f"NaN×{nan}"
        print(f"{e:<38}{'—':>7}{'':>63}  {prog}"); continue
    nums = [float(x) for x in m[-1].split()]
    # columns: Contr Regul Case Legal MeanTask MeanTaskType Class Clust Pair Retr STS
    contr,regul,case,legal,mtask,mtype,cls,clust,pair,retr,sts = nums[:11]
    flag = "OK" if nan == 0 else f"NaN×{nan}!"
    print(f"{e:<38}{legal:>7.2f}{mtype:>7.2f}{contr:>7.2f}{regul:>7.2f}{case:>7.2f}"
          f"{cls:>7.2f}{clust:>7.2f}{pair:>7.2f}{retr:>7.2f}{sts:>7.2f}  {flag}")
print("\nLegal=Score(Legal)  MTEB=Mean(TaskType)  |  '—' = not finished")
PY
