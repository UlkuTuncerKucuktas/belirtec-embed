from __future__ import annotations

import random
import re
from collections.abc import Iterator

from belirtec.config import Config, RetrievalSource
from belirtec.io import Passage, read_jsonl
from belirtec.json_utils import clean

_TR_CHARS = set("abcçdefgğhıijklmnoöprsştuüvyzABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ ")
_ELLIPSIS = re.compile(r"\.\s*\.\s*\.|…")
_WORD = re.compile(r"\b\w+\b", re.UNICODE)
_LOWER_SENT = re.compile(r"\.\s+[a-zçğıöşü]")


def _is_low_quality(text: str) -> bool:
    # web text (FineWeb2/CulturaX) carries MT-slop with broken spacing, ellipsis runs,
    # low lexical variety. mechanically detectable; drop before grounding.
    n = len(text)
    if n < 50:
        return True
    if len(_ELLIPSIS.findall(text)) >= 3:
        return True
    if sum(c in _TR_CHARS for c in text) / n < 0.85:
        return True
    words = _WORD.findall(text.lower())
    if len(words) < 10:
        return True
    if len(set(words)) / len(words) < 0.45:
        return True
    if sum(len(w) for w in words) / len(words) < 3.0:
        return True
    if len(_LOWER_SENT.findall(text)) > 4:                  # OCR-mangled PDF: sentences starting mid-word
        return True
    if len(words) >= 12:                                    # templated SEO: same phrase repeated
        tri = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
        if tri and max(tri.count(t) for t in set(tri)) >= 3:
            return True
    return False


def _clip(text: str, min_chars: int, max_chars: int, quality: bool) -> str | None:
    text = clean(text)
    if len(text) < min_chars:
        return None
    if quality and _is_low_quality(text):
        return None
    return text[:max_chars]


def stream_legal(cfg: Config, shard_idx: int = 0, n_shards: int = 1) -> Iterator[Passage]:
    lc = cfg.corpora.legal
    for i, row in enumerate(read_jsonl(lc.clean_file)):
        if i % n_shards != shard_idx:
            continue
        text = _clip(row.get(lc.text_field, ""), lc.min_chars, lc.max_chars, quality=False)
        if text is None:
            continue
        yield Passage(id=str(row.get("id", i)), text=text)


def _source_stream(
    src: RetrievalSource, cfg: Config, shard_idx: int, n_shards: int
) -> Iterator[Passage]:
    from datasets import load_dataset

    rc = cfg.corpora.retrieval
    args = (src.hf_dataset, src.hf_config) if src.hf_config else (src.hf_dataset,)
    ds = load_dataset(*args, split=rc.split, streaming=rc.streaming)
    tag = (src.hf_config or "default")
    for i, row in enumerate(ds):
        if i % n_shards != shard_idx:
            continue
        text = _clip(row.get(src.text_field, ""), rc.min_chars, rc.max_chars, src.quality_filter)
        if text is None:
            continue
        yield Passage(id=f"{src.hf_dataset.split('/')[-1]}-{tag}-{i}", text=text)


def stream_retrieval(cfg: Config, shard_idx: int = 0, n_shards: int = 1) -> Iterator[Passage]:
    # weighted round-robin over N sources: each turn, pick a source by weight and yield its
    # next passage. seeded per shard so the mixture is reproducible and shards differ.
    sources = cfg.corpora.retrieval.sources
    streams = [_source_stream(s, cfg, shard_idx, n_shards) for s in sources]
    weights = [s.weight for s in sources]
    rng = random.Random(1000 + shard_idx)
    alive = list(range(len(streams)))

    while alive:
        idx = rng.choices(alive, weights=[weights[i] for i in alive], k=1)[0]
        try:
            yield next(streams[idx])
        except StopIteration:
            alive.remove(idx)  # exhausted source drops out; others keep the mixture going
