from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from tqdm import tqdm

from belirtec import llm
from belirtec.bucket import generate_bucket
from belirtec.config import load_config
from belirtec.io import JsonlWriter

RAW_DIR = Path("data/raw")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--buckets", nargs="+", required=True)
    args = ap.parse_args()

    cfg = load_config()
    model = cfg.models[args.index]
    n_shards = len(cfg.models)
    sampling = replace(cfg.sampling, seed=cfg.sampling.seed + args.index)

    gen = llm.load(model, cfg.vllm, sampling)

    for bucket in args.buckets:
        target = -(-cfg.counts[bucket] // n_shards)
        shard = RAW_DIR / f"{bucket}.worker{args.index}.jsonl"
        with JsonlWriter(shard) as w:
            for row in tqdm(generate_bucket(bucket, gen, cfg, sampling, args.index, n_shards),
                            total=target, desc=f"gpu{args.index} {bucket}", unit="row"):
                row["_model"] = model.id
                w.write(row)
        print(f"[gpu{args.index}] {model.id} {bucket}: {w._n} rows -> {shard}", flush=True)


if __name__ == "__main__":
    main()
