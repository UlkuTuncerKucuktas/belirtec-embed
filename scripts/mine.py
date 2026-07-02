import argparse

from belirtec import mine

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_file", required=True)
    ap.add_argument("--out_file", required=True)
    args = ap.parse_args()
    mine.run(args.in_file, args.out_file)
