#!/usr/bin/env python3
import argparse
from pathstar.trajectory import build_and_save

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="outputs/age_analysis")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--min-samples", type=int, default=200)
    ap.add_argument("--normalize", default="zscore", choices=["zscore","none"])
    args = ap.parse_args()

    norm = None if args.normalize == "none" else "zscore"
    build_and_save(args.csv, args.out, args.window, args.alpha, args.min_samples, norm)
