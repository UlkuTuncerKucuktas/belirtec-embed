from __future__ import annotations

from pathlib import Path

from belirtec.config import Config

# canonical final file per bucket: legal is scored, retrieval mined, sts/cls raw
CANONICAL = {
    "legal": "data/scored/legal.jsonl",
    "retrieval": "data/mined/retrieval.jsonl",
    "sts": "data/raw/sts.jsonl",
    "classification": "data/raw/classification.jsonl",
}


def push_all(cfg: Config) -> None:
    repo = cfg.hub.dataset_repo
    if "<CONFIRM>" in repo or not repo.strip():
        raise ValueError("configs/generation.yaml hub.dataset_repo is unset")
    from datasets import load_dataset

    for bucket, path in CANONICAL.items():
        if not Path(path).exists():
            print(f"[push] skip {bucket}: {path} missing")
            continue
        ds = load_dataset("json", data_files=path, split="train")
        ds.push_to_hub(repo, config_name=bucket)
        print(f"[push] {bucket} ({len(ds)} rows) -> {repo}:{bucket}")
