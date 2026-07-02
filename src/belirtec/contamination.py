from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from tqdm import tqdm

from belirtec.config import Config

_ESAS = re.compile(r"(\d{4}/\d+)\s*E\.")
_KARAR = re.compile(r"(\d{4}/\d+)\s*K\.")


def _norm(s) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(s)).replace("\xa0", " ").split()).strip()


def _eval_case_numbers(eval_sets: list[str]):
    from datasets import load_dataset

    pairs, esas_all, karar_all, hashes = set(), set(), set(), set()
    for name in eval_sets:
        corpus = load_dataset(name, "corpus")
        docs = corpus[list(corpus.keys())[0]]
        for row in tqdm(docs, desc=f"eval {name.split('/')[-1]}", unit="doc"):
            txt = _norm(row.get("text", ""))
            if not txt:
                continue
            hashes.add(hash(txt.lower()[:500]))
            if "caselaw" in name:
                es, ks = _ESAS.findall(txt), _KARAR.findall(txt)
                for e, k in zip(es, ks):
                    pairs.add((e, k))
                esas_all |= set(es)
                karar_all |= set(ks)
    return pairs, esas_all, karar_all, hashes


def filter_legal(cfg: Config) -> dict:
    lc = cfg.corpora.legal
    from datasets import load_dataset

    pairs, esas_all, karar_all, hashes = _eval_case_numbers(lc.eval_sets)
    ds = load_dataset(lc.raw_source, split="train", streaming=True)

    out_path = Path(lc.clean_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {"total": 0, "kept": 0, "drop_case": 0, "drop_hash": 0, "drop_short": 0}

    with open(out_path, "w", encoding="utf-8") as fout:
        for r in tqdm(ds, desc="filter legal", unit="doc"):
            counts["total"] += 1
            txt = _norm(r.get("text", ""))
            if len(txt) < lc.min_chars:
                counts["drop_short"] += 1
                continue
            esas, karar = str(r.get("esasNo", "")), str(r.get("kararNo", ""))
            if esas and karar and ((esas, karar) in pairs or (esas in esas_all and karar in karar_all)):
                counts["drop_case"] += 1
                continue
            if hash(txt.lower()[:500]) in hashes:
                counts["drop_hash"] += 1
                continue
            row = {
                "anchor": None, "positive": txt[: lc.max_chars], "source": r.get("source", ""),
                "esasNo": esas, "kararNo": karar, "id": r.get("id", ""),
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts["kept"] += 1

    Path("data/contamination_report.json").write_text(json.dumps(counts, ensure_ascii=False, indent=2))
    print(f"[filter] {counts['total']} -> {counts['kept']} kept "
          f"(case={counts['drop_case']} hash={counts['drop_hash']} short={counts['drop_short']})")
    return counts
