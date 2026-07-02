from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from belirtec.io import JsonlWriter, read_jsonl

NUM_NEGATIVES = 4


def run(in_file: str, out_file: str, model_name: str = "BAAI/bge-m3",
        top_k: int = 30, fn_ratio: float = 0.95, batch_size: int = 128) -> int:
    # positive-aware filter: any retrieved passage scoring >= fn_ratio * the true positive
    # is treated as a false negative and dropped (NV-Retriever / ARHN style).
    import numpy as np
    from sentence_transformers import SentenceTransformer

    rows = [r for r in read_jsonl(in_file) if r.get("anchor") and r.get("positive")]
    if not rows:
        print(f"[mine] {in_file}: no rows, skipping")
        return 0
    anchors = [r["anchor"] for r in rows]
    positives = [r["positive"] for r in rows]

    model = SentenceTransformer(model_name)
    a_emb = model.encode(anchors, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
    p_emb = model.encode(positives, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)

    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with JsonlWriter(out_file) as w:
        for i, row in enumerate(tqdm(rows, desc="mine negatives", unit="row")):
            sims = a_emb[i] @ p_emb.T
            self_score = sims[i]
            negs = []
            for j in np.argsort(-sims):
                if j == i or sims[j] >= fn_ratio * self_score:
                    continue
                negs.append(positives[j])
                if len(negs) >= NUM_NEGATIVES:
                    break
            if not negs:
                continue
            out = {k: v for k, v in row.items() if not k.startswith("negative_")}
            for n, neg in enumerate(negs, 1):
                out[f"negative_{n}"] = neg
            w.write(out)
            written += 1
    print(f"[mine] {in_file}: {written} rows -> {out_file}")
    return written
