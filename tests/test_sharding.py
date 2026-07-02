import json

from belirtec.config import load_config
from belirtec.corpora import stream_legal


def test_shards_disjoint_and_complete(tmp_path, monkeypatch):
    f = tmp_path / "legal.jsonl"
    with open(f, "w", encoding="utf-8") as out:
        for i in range(100):
            out.write(json.dumps({"positive": "x" * 250, "id": i}) + "\n")

    cfg = load_config()
    object.__setattr__(cfg.corpora.legal, "clean_file", str(f))

    n = 4
    seen = [set() for _ in range(n)]
    for shard in range(n):
        for p in stream_legal(cfg, shard, n):
            seen[shard].add(p.id)

    all_ids = set().union(*seen)
    assert len(all_ids) == 100                       # complete
    assert sum(len(s) for s in seen) == 100          # disjoint (no overlap)
