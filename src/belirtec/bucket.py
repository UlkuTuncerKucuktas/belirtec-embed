from __future__ import annotations

import math
from collections.abc import Iterator

from belirtec import corpora, grounded, synthetic
from belirtec.config import Config, Sampling

GROUNDED = {"legal": "hukuki", "retrieval": "genel"}


def _cap(it: Iterator[dict], n: int) -> Iterator[dict]:
    for i, row in enumerate(it):
        if i >= n:
            return
        yield row


def generate_bucket(
    name: str, gen, cfg: Config, sampling: Sampling, shard_idx: int, n_shards: int
) -> Iterator[dict]:
    target = math.ceil(cfg.counts[name] / n_shards)
    if name in GROUNDED:
        domain = GROUNDED[name]
        stream = corpora.stream_legal if name == "legal" else corpora.stream_retrieval
        passages = stream(cfg, shard_idx, n_shards)
        yield from _cap(grounded.generate(gen, passages, cfg.axes[name], domain, sampling), target)
    elif name == "sts":
        yield from _cap(synthetic.generate_sts(gen, cfg, target), target)
    elif name == "classification":
        yield from _cap(synthetic.generate_classification(gen, cfg, target), target)
    else:
        raise ValueError(f"unknown bucket: {name}")
