#!/usr/bin/env python3
"""Push a trained SentenceTransformer to the HF Hub with a model card."""
import argparse


CARD = """---
license: mit
language:
- tr
library_name: sentence-transformers
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- feature-extraction
- sentence-similarity
- turkish
- legal
- bge-m3
base_model: BAAI/bge-m3
---

# {repo_name}

Turkish embedding model fine-tuned from [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
on the [VistalabBelirtecSyntheticDataset](https://huggingface.co/datasets/{ns}/VistalabBelirtecSyntheticDataset),
a grounded synthetic Turkish corpus (legal + retrieval + STS + classification), using
LoRA (r=32) with CachedGISTEmbedLoss and Matryoshka representation learning (dims 1024, 256).

## Results (MTEB-Turkish, calibrated vs the Mizan leaderboard)

| Metric | Score |
|---|---|
| Legal (Contracts/Regulation/Caselaw avg) | **54.51** |
| MTEB Turkish (Mean TaskType) | **64.50** |

The model targets Turkish legal retrieval; the legal score reflects performance on
Turkish contract, regulation, and case-law retrieval tasks.

## Usage

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("{repo_id}")
emb = model.encode(["Türkçe bir cümle", "başka bir cümle"])
```

## Training

- Base: BAAI/bge-m3 (XLM-RoBERTa, CLS pooling, max_seq 512)
- Adaptation: LoRA r=32, alpha=64, targets [query, key, value, dense]
- Loss: CachedGISTEmbedLoss (guide bge-m3) + Matryoshka [1024, 256]; classification MNRL; STS CoSENT
- Data: grounded synthetic (legal 19.8K, retrieval 20K, STS 7.6K, classification 14.2K) +
  real NLI/STS ballast + MS MARCO-TR
- Precision bf16, 2 epochs, lr 2e-4
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="path to the merged model (final/)")
    ap.add_argument("--repo", required=True, help="HF repo id, e.g. user/bge-m3-vistalab")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    ns = args.repo.split("/")[0]
    repo_name = args.repo.split("/")[-1]

    from sentence_transformers import SentenceTransformer

    print(f"[load] {args.model_dir}")
    model = SentenceTransformer(args.model_dir)
    v = model.encode(["hukuki test cümlesi"])
    dim = len(v[0]) if hasattr(v[0], "__len__") else v.shape[-1]
    print(f"[ok] encodes, dim={dim}")

    print(f"[push] -> {args.repo} (private={args.private})")
    model.push_to_hub(args.repo, private=args.private)

    from huggingface_hub import HfApi
    card = CARD.format(repo_name=repo_name, repo_id=args.repo, ns=ns)
    HfApi().upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="model",
    )
    print(f"[done] https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
