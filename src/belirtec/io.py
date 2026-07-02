from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Passage:
    id: str
    text: str


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


class JsonlWriter:
    # append + periodic flush so a crash costs only in-flight rows, not the whole run
    def __init__(self, path: str | Path, flush_every: int = 200):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w", encoding="utf-8")
        self._flush_every = flush_every
        self._n = 0

    def write(self, row: dict) -> None:
        self._f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._n += 1
        if self._n % self._flush_every == 0:
            self._f.flush()

    def close(self) -> int:
        self._f.flush()
        self._f.close()
        return self._n

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def pool_dedup(shard_paths, out_path: str | Path, key: str = "anchor") -> int:
    seen: set = set()
    total = 0
    with JsonlWriter(out_path) as out:
        for shard in shard_paths:
            if not Path(shard).exists():
                continue
            for row in read_jsonl(shard):
                k = row.get(key)
                if k in seen:
                    continue
                seen.add(k)
                out.write(row)
                total += 1
    return total
