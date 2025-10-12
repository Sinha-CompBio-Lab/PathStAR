#!/usr/bin/env python3
import argparse
import pandas as pd
from pathstar.individual import analyze_multiple_tissues

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="outputs/individual_scores")
    ap.add_argument("--min-tissue-samples", type=int, default=200)
    ap.add_argument("--min-younger", type=int, default=5)
    ap.add_argument("--younger-years", type=int, default=10)
    ap.add_argument("--zscore-by-sex", action="store_true", help="default: combine sexes")
    ap.add_argument("--sex-matched-younger", action="store_true")
    ap.add_argument("--min-target-age", type=int, default=30)
    ap.add_argument("--tissues", nargs="*", help="If omitted, auto-detect tissues with >= min-tissue-samples")
    args = ap.parse_args()

    if args.tissues:
        tissues = args.tissues
    else:
        df = pd.read_csv(args.csv)
        counts = df["tissue"].value_counts()
        tissues = counts[counts >= args.min_tissue_samples].index.tolist()
        print(f"Tissues to analyze (>= {args.min_tissue_samples} samples): {len(tissues)}")
        for i, t in enumerate(tissues, 1): print(f"{i}. {t}")

    analyze_multiple_tissues(
        csv_file=args.csv,
        tissues=tissues,
        min_younger_samples=args.min_younger,
        out_dir=args.out,
        zscore_by_sex=args.zscore_by_sex,
        sex_matched_younger=args.sex_matched_younger,
        younger_window_years=args.younger_years,
        min_target_age=args.min_target_age,
    )
