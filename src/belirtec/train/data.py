from __future__ import annotations

import json
import random
from pathlib import Path

from belirtec.train.train_config import BucketSpec, DataCfg, TrainingConfig

REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------- row helpers (ported from champion, behavior-preserving) ----------------
def _read_jsonl(path: str) -> list[dict]:
    p = REPO_ROOT / path if not Path(path).is_absolute() else Path(path)
    rows = []
    with open(p, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                try:
                    rows.append(json.loads(ln))
                except Exception:
                    pass
    return rows


def _neg_list(r: dict) -> list[str]:
    ks = sorted([k for k in r if k.startswith("negative_")], key=lambda k: int(k.split("_")[1]))
    return [r[k] for k in ks if str(r[k]).strip()]


def _pad_negs(negs: list[str], k: int) -> list[str] | None:
    negs = [n for n in negs if str(n).strip()]
    if not negs:
        return None
    out = list(negs[:k])
    i = 0
    while len(out) < k:
        out.append(negs[i % len(negs)])
        i += 1
    return out


def _fmt_anchor(anchor: str, instruction: str | None, use_instr: bool) -> str:
    if use_instr and instruction:
        return f"Instruct: {instruction}\nQuery: {anchor}"
    return anchor


def _contrastive_row(anchor, positive, negs, k, instruction, use_instr) -> dict | None:
    anchor = " ".join(str(anchor).split())
    positive = " ".join(str(positive).split())
    if len(anchor) < 2 or len(positive) < 1 or anchor == positive:
        return None
    p = _pad_negs(negs, k)
    if p is None:
        return None
    row = {"anchor": _fmt_anchor(anchor, instruction, use_instr), "positive": positive}
    for j, n in enumerate(p, 1):
        row[f"negative_{j}"] = " ".join(str(n).split())
    return row


# ---------------- selection: forgetting + limit ----------------
def _apply_forgetting(rows: list[dict], spec: BucketSpec) -> list[dict]:
    mode = spec.forgetting.mode
    if not mode:
        return rows
    thr = spec.forgetting.threshold
    scored = [r for r in rows if "_forget_cos" in r]
    if not scored:
        return rows  # bucket has no scores (only legal does); no-op elsewhere
    if mode == "filter":            # keep SAFE (high cos)
        return [r for r in rows if r.get("_forget_cos", thr + 1) > thr]
    if mode == "keep_risky":        # keep RISKY tail (low/negative cos)
        return [r for r in rows if r.get("_forget_cos", thr + 1) <= thr]
    if mode == "weight":            # keep all; weighting handled downstream (not in baseline)
        return rows
    raise ValueError(f"unknown forgetting mode: {mode}")


def _apply_limit(rows: list[dict], spec: BucketSpec, global_seed: int) -> list[dict]:
    if spec.limit is None or spec.limit >= len(rows):
        return rows
    if spec.shuffle:
        rng = random.Random(spec.seed if spec.seed is not None else global_seed)
        rows = list(rows)
        rng.shuffle(rows)
    return rows[: spec.limit]


def _select(rows: list[dict], spec: BucketSpec, global_seed: int) -> list[dict]:
    rows = _apply_forgetting(rows, spec)
    rows = _apply_limit(rows, spec, global_seed)
    return rows


# ---------------- bucket loading (local or hf) ----------------
def _load_bucket_rows(name: str, data: DataCfg) -> list[dict]:
    if data.synthetic_source == "local":
        return _read_jsonl(data.local_paths[name])
    if data.synthetic_source == "hf":
        from datasets import load_dataset

        ds = load_dataset(data.synthetic_repo, name=name, split="train")
        return [dict(r) for r in ds]
    raise ValueError(f"unknown synthetic_source: {data.synthetic_source}")


def _rows_to_contrastive(rows, data: DataCfg) -> list[dict]:
    out = []
    for r in rows:
        row = _contrastive_row(
            r.get("anchor"), r.get("positive"), _neg_list(r),
            data.num_negatives, r.get("instruction"), data.use_instructions,
        )
        if row:
            out.append(row)
    return out


# ---------------- real NLI + real stsb (ported) ----------------
def _build_real_nli(data: DataCfg, seed: int) -> list[dict]:
    if not data.real_nli.enabled:
        return []
    from datasets import load_dataset

    try:
        nli = load_dataset(data.real_nli_hf, split="train")
    except Exception as e:
        print(f"[warn] real NLI load failed ({e}); continuing without it")
        return []
    cols = nli.column_names
    trip = None
    for cand in (["sent0", "sent1", "hard_neg"], ["anchor", "positive", "negative"],
                 ["premise", "entailment", "contradiction"]):
        if all(c in cols for c in cand):
            trip = cand
            break
    if trip is None and len(cols) >= 3:
        trip = [cols[0], cols[1], cols[2]]
    a_c, p_c, n_c = trip
    idx = list(range(len(nli)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n = data.real_nli.limit if data.real_nli.limit is not None else data.nli_samples
    if n and n > 0:
        idx = idx[:n]
    out = []
    for i in idx:
        ex = nli[i]
        row = _contrastive_row(ex[a_c], ex[p_c], [ex[n_c]], data.num_negatives, None, data.use_instructions)
        if row:
            out.append(row)
    return out


def _build_msmarco(data: DataCfg, seed: int) -> list[dict]:
    """Load newmindai/ms-marco-turkish-triplets (query_text/pos_text/neg_text) as
    contrastive rows. Controlled by data.msmarco = {enabled, limit, repo}."""
    ms = data.msmarco
    if not ms or not ms.get("enabled"):
        return []
    from datasets import load_dataset

    repo = ms.get("repo", "newmindai/ms-marco-turkish-triplets")
    try:
        ds = load_dataset(repo, split="train")
    except Exception as e:
        print(f"[warn] MS MARCO load failed ({e}); continuing without it")
        return []
    idx = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    lim = ms.get("limit")
    if lim:
        idx = idx[:lim]
    out = []
    for i in idx:
        ex = ds[i]
        row = _contrastive_row(
            ex.get("query_text"), ex.get("pos_text"), [ex.get("neg_text")],
            data.num_negatives, None, data.use_instructions,
        )
        if row:
            out.append(row)
    return out


def build_stsb(cfg: TrainingConfig):
    from datasets import load_dataset

    hf = cfg.data.real_stsb_hf

    def find_cols(cols):
        score_c = None
        for cand in ("score", "label", "similarity_score", "labels"):
            if cand in cols:
                score_c = cand
                break
        text_cs = [c for c in cols if c != score_c][:2]
        return text_cs[0], text_cs[1], score_c

    ds = load_dataset(hf, split="train")
    s1c, s2c, sc = find_cols(ds.column_names)
    raw = [float(x[sc]) for x in ds]
    div = 5.0 if (raw and max(raw) > 1.5) else 1.0
    train = []
    for ex in ds:
        try:
            s = float(ex[sc]) / div
        except Exception:
            continue
        a, b = " ".join(str(ex[s1c]).split()), " ".join(str(ex[s2c]).split())
        if a and b:
            train.append({"sentence1": a, "sentence2": b, "score": s})

    ev = None
    for split in ("validation", "test"):
        try:
            d = load_dataset(hf, split=split)
            s1, s2, scc = find_cols(d.column_names)
            S1, S2, SC = [], [], []
            er = [float(x[scc]) for x in d]
            ediv = 5.0 if (er and max(er) > 1.5) else 1.0
            for ex in d:
                try:
                    s = float(ex[scc]) / ediv
                except Exception:
                    continue
                a = " ".join(str(ex[s1]).split())
                b = " ".join(str(ex[s2]).split())
                if not a or not b:      # empty/whitespace -> NaN embedding -> eval crash
                    continue
                S1.append(a)
                S2.append(b)
                SC.append(s)
            if S1:
                ev = (S1, S2, SC)
                break
        except Exception:
            continue
    return train, ev


# ---------------- top-level builder ----------------
def build_datasets(cfg: TrainingConfig) -> tuple[dict, dict, tuple | None]:
    """Returns (train_datasets_dict, counts, stsb_eval).

    Route A preserved: classification is its OWN dataset (label-name positives need
    NO_DUPLICATES so same-label rows never collide in a batch). Everything else
    (real NLI + synthetic retrieval/sts/legal) goes in the 'contrastive' dataset.
    """
    data = cfg.data
    seed = cfg.train.seed
    counts: dict[str, int] = {}

    # --- contrastive dataset: real NLI + retrieval + sts + legal ---
    contrastive: list[dict] = []

    nli_rows = _build_real_nli(data, seed)
    counts["nli"] = len(nli_rows)
    contrastive += nli_rows

    ms_rows = _build_msmarco(data, seed)
    counts["msmarco"] = len(ms_rows)
    contrastive += ms_rows

    for name in ("retrieval", "sts", "legal"):
        spec = data.buckets.get(name)
        if not spec or not spec.enabled:
            counts[name] = 0
            continue
        rows = _load_bucket_rows(name, data)
        rows = _select(rows, spec, seed)
        conv = _rows_to_contrastive(rows, data)
        counts[name] = len(conv)
        contrastive += conv

    rng = random.Random(seed)
    rng.shuffle(contrastive)

    # --- classification dataset (separate, Route A) ---
    class_rows: list[dict] = []
    cspec = data.buckets.get("classification")
    if cspec and cspec.enabled:
        rows = _load_bucket_rows("classification", data)
        rows = _select(rows, cspec, seed)
        class_rows = _rows_to_contrastive(rows, data)
        random.Random(seed + 1).shuffle(class_rows)
    counts["classification"] = len(class_rows)

    # --- stsb (graded, CoSENT) ---
    stsb_rows, stsb_eval = build_stsb(cfg)
    counts["stsb"] = len(stsb_rows)

    from datasets import Dataset

    train_datasets = {"contrastive": Dataset.from_list(contrastive)}
    if class_rows:
        train_datasets["classification"] = Dataset.from_list(class_rows)
    if stsb_rows:
        train_datasets["stsb"] = Dataset.from_list(stsb_rows)

    return train_datasets, counts, stsb_eval
