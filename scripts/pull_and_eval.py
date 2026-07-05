#!/usr/bin/env python3
"""Pull a model from HF and run the MTEB-Turkish + legal eval on it.

Verifies the published artifact evaluates to the expected scores end-to-end.
"""
import argparse
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="HF repo id, e.g. user/bge-m3-vistalab")
    ap.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES")
    ap.add_argument("--output-dir", default="results_published")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"[gpu] CUDA_VISIBLE_DEVICES={args.gpu}")

    from huggingface_hub import snapshot_download

    print(f"[pull] {args.repo}")
    local = snapshot_download(repo_id=args.repo, repo_type="model")
    print(f"[pull] -> {local}")

    from belirtec.train.evaluate import run as eval_run

    print(f"[eval] {args.repo}")
    df = eval_run(local, output_dir=args.output_dir, batch_size=args.batch_size,
                  device=None, overwrite=True)
    if df is not None:
        print("\n=== published model eval ===")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
