# belirtec-embed

Turkish embedding models fine-tuned from BAAI/bge-m3 on grounded synthetic data,
targeting Turkish legal retrieval (MTEB-Turkish / Mizan leaderboard).

- **Model**: [UlkuTuncerKucuktas/bge-m3-vistalab](https://huggingface.co/UlkuTuncerKucuktas/bge-m3-vistalab) — 54.51 Legal / 64.50 MTEB
- **Dataset**: [UlkuTuncerKucuktas/VistalabBelirtecSyntheticDataset](https://huggingface.co/datasets/UlkuTuncerKucuktas/VistalabBelirtecSyntheticDataset)

## Setup

```bash
conda create -n belirtec python=3.11 -y && conda activate belirtec
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .
```

## Usage

```bash
# train an experiment
python scripts/train_and_eval.py --experiment lora_allsynth_gist --gpu 0

# run the full matrix across GPUs (queue runner)
bash run_matrix.sh 0,1,2,3

# extract results table
bash results_table.sh

# publish / verify a model
python scripts/push_model.py --model-dir runs/<exp>/final --repo user/name
python scripts/pull_and_eval.py --repo user/name --gpu 0
```

## Experiment Results

Full 12-experiment matrix on the pinned stack (torch 2.5.1 / transformers 4.49 / sentence-transformers 4.1, bge-m3 base, LoRA r32 unless noted). Legal = Score(Legal) (Contracts/Regulation/Caselaw avg); MTEB = Mean(TaskType), calibrated vs the Mizan leaderboard.

| experiment | data | loss | adapt | stages | Legal | MTEB |
|---|---|---|---|---|---|---|
| lora_allsynth_msmarco_gist | all synth + legal + msmarco | gist | LoRA | 1 | **54.51** | **64.50** |
| lora_allsynth_gist | all synth + legal | gist | LoRA | 1 | 54.27 | 63.83 |
| lora_allsynth_mnrl | all synth + legal | mnrl | LoRA | 1 | 53.37 | 63.25 |
| lora_staged_safe2hard_mnrl | staged safe→hard | mnrl | LoRA | 2 | 53.29 | 63.82 |
| lora_allsynth_hard_mnrl | all synth + legal (risky tail) | mnrl | LoRA | 1 | 52.76 | 63.05 |
| lora_allsynth_safe_mnrl | all synth + legal (safe subset) | mnrl | LoRA | 1 | 52.49 | 62.26 |
| lora_allsynth_nolegal_msmarco_gist | all synth − legal + msmarco | gist | LoRA | 1 | 51.34 | 64.15 |
| fullft_allsynth_mnrl | all synth + legal | mnrl | full-FT | 1 | 51.33 | 61.54 |
| fullft_multimodel_sizematch_mnrl | all synth, 4 models (vol-matched) | mnrl | full-FT | 1 | 51.24 | 62.19 |
| fullft_singlemodel_mnrl | all synth, 1 model | mnrl | full-FT | 1 | 48.07 | 62.00 |
| fullft_nolegal_mnrl | all synth − legal | mnrl | full-FT | 1 | 47.50 | 62.26 |
| fullft_realonly_mnrl | real NLI+STS only | mnrl | full-FT | 1 | 46.35 | 61.86 |

### LoRA Rank Sensitivity

We swept LoRA rank r ∈ {8, 16, 32, 64} (α = 2r) under the GIST recipe (`lora_r8_gist` / `lora_r16_gist` / `lora_r64_gist`; r32 = `lora_allsynth_gist` above).

| rank | trainable params | Legal | MTEB |
|---|---|---|---|
| 8 | 0.62% (3.56M) | 54.83 | 64.06 |
| 16 | 1.24% (7.11M) | 54.45 | 64.12 |
| 32 | 2.44% (14.2M) | 54.27 | 63.83 |
| 64 | 4.77% (28.4M) | 54.41 | 64.30 |

Performance is **rank-robust**: across an 8× range in trainable parameters, Legal
varies by only 0.56 points and MTEB by 0.47, with no clear optimum (r=8 is highest
on Legal, r=64 on MTEB). Even r=8 at 0.62% of parameters matches r=64. This
reinforces that the grounded synthetic data — not adapter capacity — drives
performance; r=32 is a reasonable default, but the method is insensitive to this choice.

Pinned known-good stack (see `requirements.txt`). The pins matter: bleeding-edge
torch/transformers caused NaN training collapse; this stack trained the full matrix
with zero NaN.
