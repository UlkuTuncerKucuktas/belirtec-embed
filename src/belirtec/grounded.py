from __future__ import annotations

from collections.abc import Iterable, Iterator

from belirtec.config import Axes, Sampling
from belirtec.io import Passage
from belirtec.json_utils import clean, parse_json
from belirtec.prompts import AxisSampler, grounded_query_prompt


def generate(
    gen,
    passages: Iterable[Passage],
    axes: Axes,
    domain: str,
    sampling: Sampling,
    min_query_chars: int = 8,
    batch_size: int = 256,
) -> Iterator[dict]:
    # gen: (prompts, temp, max_tokens) -> texts. Passed in from llm.py so this engine
    # has no vLLM dependency and is testable with a stub. Legal and retrieval both call
    # this; they differ only in passages, axes, and domain.
    sampler = AxisSampler(axes, seed=sampling.seed)

    for batch in _chunk(passages, batch_size):
        conditions = [sampler.sample() for _ in batch]
        prompts = [grounded_query_prompt(p.text, c, domain) for p, c in zip(batch, conditions)]
        responses = gen(prompts, sampling.temp_samples, sampling.max_tokens_query)

        for passage, cond, resp in zip(batch, conditions, responses):
            parsed = parse_json(resp)
            if not isinstance(parsed, dict):
                continue
            query = clean(parsed.get("query", ""))
            if len(query) < min_query_chars:
                continue
            yield {
                "anchor": query,
                "positive": passage.text,
                "source": passage.id,
                "persona": cond["persona"],
                "intent": cond["intent"],
                "difficulty": cond["difficulty"],
            }


def _chunk(it: Iterable, n: int) -> Iterator[list]:
    batch: list = []
    for x in it:
        batch.append(x)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch
