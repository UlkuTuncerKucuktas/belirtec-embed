from __future__ import annotations

import json
import re


def clean(s) -> str:
    return " ".join(str(s).split()).strip()


def parse_json(text: str):
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text).strip()
    # array first (sample prompts return lists), then object (grounded returns {"query": ...})
    for opn, cls in (("[", "]"), ("{", "}")):
        start = text.find(opn)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == opn:
                depth += 1
            elif text[i] == cls:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except Exception:
                        break
    return None
