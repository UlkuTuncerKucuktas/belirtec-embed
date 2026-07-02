from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from belirtec.io import JsonlWriter, read_jsonl

TARGET_SUBSTR = ("query", "key", "value", "dense")


def _negs(r, k):
    ks = sorted([x for x in r if x.startswith("negative_")], key=lambda x: int(x.split("_")[1]))
    return [str(r[x]) for x in ks if str(r[x]).strip()][:k]


def _target_params(model):
    ps = []
    for n, p in model.named_parameters():
        keep = any(s in n for s in TARGET_SUBSTR)
        p.requires_grad_(keep)
        if keep:
            ps.append(p)
    return ps


def _embed(model, texts, device):
    import torch

    feats = model.tokenize(texts)
    feats = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in feats.items()}
    return model(feats)["sentence_embedding"]


def _triple_loss(model, a, p, negs, device, scale=20.0):
    import torch

    emb = _embed(model, [a, p] + negs, device)
    logits = scale * (emb[0:1] @ emb[1:].t())
    return torch.nn.functional.cross_entropy(logits, torch.zeros(1, dtype=torch.long, device=device))


def _flat_grad(params):
    import torch

    return torch.cat([p.grad.detach().reshape(-1).float() for p in params if p.grad is not None])


def run(candidate: str, out_file: str, preserve_files: list[str], model_name: str = "BAAI/bge-m3",
        num_negatives: int = 4, preserve_n: int = 512) -> int:
    import random
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name).to(device)
    model.eval()
    params = _target_params(model)

    pres = []
    for pf in preserve_files:
        if not Path(pf).exists():
            continue
        rows = [r for r in read_jsonl(pf) if r.get("anchor") and r.get("positive")]
        random.shuffle(rows)
        for r in rows[:preserve_n]:
            ns = _negs(r, num_negatives)
            if ns:
                pres.append((r["anchor"], r["positive"], ns))

    model.zero_grad(set_to_none=True)
    for a, p, ns in tqdm(pres, desc="preserve grad", unit="row"):
        _triple_loss(model, str(a), str(p), ns, device).backward()
    g_pre = _flat_grad(params)
    g_pre = g_pre / (g_pre.norm() + 1e-8)

    cands = [r for r in read_jsonl(candidate) if r.get("anchor") and r.get("positive") and _negs(r, num_negatives)]
    scored = []
    for r in tqdm(cands, desc="score forgetting", unit="row"):
        a, p, ns = r["anchor"], r["positive"], _negs(r, num_negatives)
        model.zero_grad(set_to_none=True)
        _triple_loss(model, str(a), str(p), ns, device).backward()
        g = _flat_grad(params)
        cos = float(torch.dot(g, g_pre) / (g.norm() + 1e-8))
        with torch.no_grad():
            emb = _embed(model, [str(a), str(p)] + ns, device)
            margin = float(emb[0] @ emb[1]) - max(float(emb[0] @ emb[i]) for i in range(2, emb.shape[0]))
        rr = dict(r)
        rr["_forget_cos"] = round(cos, 5)
        rr["_base_margin"] = round(margin, 4)
        scored.append(rr)

    scored.sort(key=lambda x: -x["_forget_cos"])
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    with JsonlWriter(out_file) as w:
        for r in scored:
            w.write(r)
    print(f"[score] {candidate}: {len(scored)} rows -> {out_file}")
    return len(scored)
