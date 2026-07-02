from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from belirtec.config import Config, load_config
from belirtec.io import pool_dedup

RAW_DIR = Path("data/raw")


def _launch(cfg: Config, buckets: list[str]) -> list[subprocess.Popen]:
    procs = []
    for idx, model in enumerate(cfg.models):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(idx))
        cmd = [sys.executable, "-m", "belirtec.worker", "--index", str(idx), "--buckets", *buckets]
        log = open(RAW_DIR / f"worker{idx}.log", "w")
        procs.append(subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT))
        print(f">>> gpu{idx}: {model.id} ({buckets})", flush=True)
    return procs


def _manifest(cfg: Config, buckets: list[str], produced: dict[str, int]) -> dict:
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": _git_commit(),
        "models": cfg.model_ids(),
        "buckets": buckets,
        "target_counts": {b: cfg.counts[b] for b in buckets},
        "produced_counts": produced,
    }


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def run(buckets: list[str] | None = None, do_push: bool = False) -> None:
    cfg = load_config()
    buckets = buckets or list(cfg.counts)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    procs = _launch(cfg, buckets)
    codes = [p.wait() for p in procs]
    failed = [i for i, c in enumerate(codes) if c != 0]
    if failed:
        print(f"[warn] workers failed: {failed} (pooling survivors)", flush=True)

    produced = {}
    for b in buckets:
        shards = [RAW_DIR / f"{b}.worker{i}.jsonl" for i in range(len(cfg.models))]
        produced[b] = pool_dedup(shards, RAW_DIR / f"{b}.jsonl")
        for s in shards:
            s.unlink(missing_ok=True)
        print(f"  {b}: {produced[b]} rows -> {RAW_DIR / f'{b}.jsonl'}", flush=True)

    manifest = _manifest(cfg, buckets, produced)
    (RAW_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"[done] manifest -> {RAW_DIR / 'manifest.json'}", flush=True)

    if do_push:
        from belirtec.push import push_all
        push_all(cfg)
