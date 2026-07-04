#!/usr/bin/env bash
# results_table.sh — pull the final eval row from every experiment's log into
# one aligned table. Run any time; shows done experiments, marks missing ones.
cd "$(dirname "$0")"

EXPS=(
  fullft_realonly_mnrl
  fullft_allsynth_mnrl
  fullft_singlemodel_mnrl
  fullft_multimodel_sizematch_mnrl
  fullft_nolegal_mnrl
  lora_allsynth_mnrl
  lora_allsynth_safe_mnrl
  lora_allsynth_hard_mnrl
  lora_staged_safe2hard_mnrl
  lora_allsynth_gist
  lora_allsynth_nolegal_msmarco_gist
  lora_allsynth_msmarco_gist
)

python3 - "${EXPS[@]}" <<'PY'
import re, os, sys
exps = sys.argv[1:]
hdr = f"{'experiment':<38}{'Legal':>7}{'MTEB':>7}{'Contr':>7}{'Regul':>7}{'Case':>7}{'Class':>7}{'Clust':>7}{'PairC':>7}{'Retr':>7}{'STS':>7}  status"
print(hdr); print("-"*len(hdr))
for e in exps:
    log = f"runs/{e}.log"
    if not os.path.exists(log):
        print(f"{e:<38}{'—':>7}{'':>63}  no log"); continue
    txt = open(log, errors="ignore").read()
    nan = len(re.findall(r"Input contains NaN|grad_norm': nan", txt))
    rows = re.findall(
        r'/final\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+'
        r'([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', txt)
    if not rows:
        prog = "running" if nan == 0 else f"NaN×{nan}"
        print(f"{e:<38}{'—':>7}{'':>63}  {prog}"); continue
    contr,regul,case,legal,mtask,mtype,cls,clust,pair,retr,sts = map(float, rows[-1])
    flag = "OK" if nan == 0 else f"NaN×{nan}!"
    print(f"{e:<38}{legal:>7.2f}{mtype:>7.2f}{contr:>7.2f}{regul:>7.2f}{case:>7.2f}"
          f"{cls:>7.2f}{clust:>7.2f}{pair:>7.2f}{retr:>7.2f}{sts:>7.2f}  {flag}")
print()
print("Legal=Score(Legal)  MTEB=Mean(TaskType)  |  '—' = not finished")
PY
