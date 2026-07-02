from __future__ import annotations

import math
from collections.abc import Iterator

from belirtec.config import Config
from belirtec.json_utils import clean, parse_json
from belirtec import prompts

NUM_NEGATIVES = 4


def _triple(ex, a_key, b_key, instruction, min_a, min_b):
    if not isinstance(ex, dict):
        return None
    a, b = clean(ex.get(a_key, "")), clean(ex.get(b_key, ""))
    if len(a) < min_a or len(b) < min_b or a == b:
        return None
    hn = ex.get("hard_negatives", [])
    if isinstance(hn, str):
        hn = [hn]
    negs = [clean(x) for x in hn if clean(x)]
    negs = [n for n in dict.fromkeys(negs) if n and n not in (a, b)]
    if not negs:
        return None
    row = {"anchor": a, "positive": b, "instruction": instruction}
    for j, ng in enumerate(negs[:NUM_NEGATIVES], 1):
        row[f"negative_{j}"] = ng
    return row


def _brainstorm(gen, prompt_list, temp, want):
    out, seen = [], set()
    for r in gen(prompt_list, temp, 2048):
        arr = parse_json(r)
        if not isinstance(arr, list):
            continue
        for it in arr:
            s = clean(it if isinstance(it, str) else it.get("task", "") if isinstance(it, dict) else "")
            if len(s) > 6 and s.lower() not in seen:
                seen.add(s.lower())
                out.append(s)
    return out[:want]


def generate_sts(gen, cfg: Config, target: int, per_task: int = 6) -> Iterator[dict]:
    s = cfg.sampling
    tasks_needed = math.ceil(target / per_task)
    per_angle = math.ceil(tasks_needed / len(prompts.STS_ANGLES))
    topics = _brainstorm(
        gen, [prompts.sts_topics_prompt(per_angle, a) for a in prompts.STS_ANGLES], s.temp_tasks, tasks_needed
    )
    seen = set()
    for topic, r in zip(topics, gen([prompts.sts_samples_prompt(t, per_task) for t in topics], s.temp_samples, s.max_tokens_samples)):
        arr = parse_json(r)
        if not isinstance(arr, list):
            continue
        for ex in arr:
            row = _triple(ex, "text_a", "text_b", prompts.STS_INSTRUCTION, 6, 6)
            if row and row["anchor"] not in seen:
                seen.add(row["anchor"])
                yield row


def generate_classification(gen, cfg: Config, target: int, per_label: int = 25) -> Iterator[dict]:
    s = cfg.sampling
    tasks_needed = math.ceil(target / (3 * per_label))
    per_dom = math.ceil(tasks_needed / len(prompts.CLS_DOMAINS))

    tasks = []
    for r in gen([prompts.cls_tasks_prompt(per_dom, d) for d in prompts.CLS_DOMAINS], s.temp_tasks, s.max_tokens_samples):
        arr = parse_json(r)
        if not isinstance(arr, list):
            continue
        for it in arr:
            if not isinstance(it, dict):
                continue
            t = clean(it.get("task", ""))
            labels = list(dict.fromkeys(clean(x) for x in it.get("labels", []) if clean(x)))
            if t and 2 <= len(labels) <= 6 and not any(prompts.is_foreign_label(l) for l in labels):
                tasks.append({"task": t, "labels": labels})
    tasks = tasks[:tasks_needed]

    prompt_list, meta = [], []
    for t in tasks:
        for lab in t["labels"]:
            prompt_list.append(prompts.cls_samples_prompt(t["task"], t["labels"], lab, per_label))
            meta.append((t, lab))

    seen = set()
    for (t, target_lab), r in zip(meta, gen(prompt_list, s.temp_samples, s.max_tokens_samples)):
        arr = parse_json(r)
        if not isinstance(arr, list):
            continue
        others = [l for l in t["labels"] if l != target_lab]
        for ex in arr:
            if not isinstance(ex, dict):
                continue
            txt = clean(ex.get("text", ""))
            if len(txt) < 8 or txt in seen:
                continue
            if target_lab.lower() in txt.lower():
                continue
            negs = [clean(m) for m in ex.get("misleading_labels", []) if clean(m) and clean(m) != target_lab]
            negs = list(dict.fromkeys(negs))
            for ol in others:  # pad from the real other labels so every row has negatives
                if len(negs) >= NUM_NEGATIVES:
                    break
                if ol not in negs:
                    negs.append(ol)
            if not negs:
                continue
            seen.add(txt)
            row = {"anchor": txt, "positive": target_lab, "instruction": t["task"]}
            for j, ng in enumerate(negs[:NUM_NEGATIVES], 1):
                row[f"negative_{j}"] = ng
            yield row
