#!/usr/bin/env python3
import argparse
from pathstar.smoothing import smooth_all

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pickle", required=True, help="Path to all_results_combined_sexes_*.pkl")
    ap.add_argument("--out-root", required=True, help="Output root (creates spline_data/ and gaussian_process_data/)")
    ap.add_argument("--label", default="combined")
    ap.add_argument("--no-spline", action="store_true")
    ap.add_argument("--no-gp", action="store_true")
    ap.add_argument("--window", type=int, default=10)
    args = ap.parse_args()

    summary = smooth_all(
        pickle_file=args.pickle,
        out_root=args.out_root,
        label=args.label,
        do_spline=not args.no_spline,
        do_gp=not args.no_gp,
        window_size=args.window,
    )
    print("Done. Tissues smoothed:")
    for k, v in summary.items():
        print(f"  {k}: {len(v)} tissues")
