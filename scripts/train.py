#!/usr/bin/env python3
import argparse
import os


def main():
    p = argparse.ArgumentParser(description="Train bge-m3 Turkish embedding (config-driven).")
    p.add_argument("--config", default=None, help="base training.yaml (default: configs/training.yaml)")
    p.add_argument("--experiment", default=None,
                   help="name under configs/experiments/ to overlay (e.g. gist, staged)")
    p.add_argument("--gpu", default=None,
                   help="GPU id(s) for CUDA_VISIBLE_DEVICES, e.g. '0' or '0,1'. "
                        "Set BEFORE torch import so the process sees only these GPUs.")
    args = p.parse_args()

    # MUST set before any torch/sentence-transformers import, or it's ignored.
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"[gpu] CUDA_VISIBLE_DEVICES={args.gpu}")

    from belirtec.train.loop import run

    final = run(args.config, args.experiment)
    print(f"[train] final model -> {final}")


if __name__ == "__main__":
    main()
