#!/usr/bin/env python3
import argparse
import os


def main():
    p = argparse.ArgumentParser(description="Evaluate a checkpoint on MTEB-Turkish + legal tasks.")
    p.add_argument("--model", "-m", required=True, help="checkpoint path or HF model id")
    p.add_argument("--output-dir", "-o", default="results")
    p.add_argument("--batch-size", "-b", type=int, default=32)
    p.add_argument("--gpu", default=None, help="GPU id for CUDA_VISIBLE_DEVICES (e.g. '3')")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"[gpu] CUDA_VISIBLE_DEVICES={args.gpu}")

    from belirtec.train.evaluate import run

    run(args.model, output_dir=args.output_dir, batch_size=args.batch_size,
        device=None, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
