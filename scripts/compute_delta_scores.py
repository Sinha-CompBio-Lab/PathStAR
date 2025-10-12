#!/usr/bin/env python3
import argparse
from pathstar.delta import calculate_delta_scores_all_tissues

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--individual-scores", required=True,
                    help="Dir with per-tissue <tissue>_effect_size_deviation.tsv (from individual scores)")
    ap.add_argument("--spline-dir", required=True,
                    help="Dir containing *combined_spline_comparison.csv files")
    ap.add_argument("--gaussian-dir", required=True,
                    help="Dir containing *combined_gaussian_process_comparison.csv files")
    ap.add_argument("--out", default="outputs/delta_scores")
    ap.add_argument("--label", default="combined")
    ap.add_argument("--override", nargs=2, action="append", metavar=("TISSUE", "METHOD"),
                    help="Override method per tissue, e.g. --override Ovary gaussian_process")
    args = ap.parse_args()

    overrides = dict(args.override) if args.override else {}
    calculate_delta_scores_all_tissues(
        individual_score_dir=args.individual_scores,
        spline_dir=args.spline_dir,
        gaussian_dir=args.gaussian_dir,
        output_dir=args.out,
        label=args.label,
        method_overrides=overrides,
    )
