#!/usr/bin/env python3
import argparse
import os


def main():
    p = argparse.ArgumentParser(description="Train an experiment then run MTEB eval on the result.")
    p.add_argument("--config", default=None)
    p.add_argument("--experiment", default=None)
    p.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES (set before torch import)")
    p.add_argument("--eval-output-dir", default=None, help="default: <output_dir>/eval")
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--skip-eval", action="store_true", help="train only")
    args = p.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"[gpu] CUDA_VISIBLE_DEVICES={args.gpu}")

    from belirtec.train.loop import run

    final = run(args.config, args.experiment)
    print(f"[train] final model -> {final}")

    if args.skip_eval:
        return

    from belirtec.train.evaluate import run as eval_run

    out = args.eval_output_dir or os.path.join(os.path.dirname(final), "eval")
    print(f"[eval] evaluating {final} -> {out}")
    eval_run(final, output_dir=out, batch_size=args.eval_batch_size, device=None, overwrite=True)


if __name__ == "__main__":
    main()
