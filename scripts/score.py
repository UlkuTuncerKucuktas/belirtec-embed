import argparse

from belirtec import score

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--out_file", required=True)
    ap.add_argument("--preserve", nargs="+", required=True)
    args = ap.parse_args()
    score.run(args.candidate, args.out_file, args.preserve)
