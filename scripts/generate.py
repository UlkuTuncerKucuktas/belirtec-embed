import argparse

from belirtec.config import load_config
from belirtec.orchestrate import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--buckets", nargs="+", default=None, choices=list(load_config().counts))
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()
    run(args.buckets, do_push=args.push)
