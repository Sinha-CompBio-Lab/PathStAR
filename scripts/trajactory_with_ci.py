#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trajectory with bootstrap CIs (by sex) — minimal script.

Usage:
  python scripts/trajectory_ci_extra.py \
    --csv /path/to/critical_age_dataset.csv \
    --outdir ./age_analysis_by_sex_ci \
    --windows 10 \
    --n-boot 10000
"""
import os
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval
from scipy import stats
from sklearn.preprocessing import StandardScaler


# ----------------------------- I/O -----------------------------
def load_and_prepare_data(csv_file: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    if "features" not in df.columns:
        raise ValueError("Input CSV must contain a 'features' column.")
    df["features"] = df["features"].apply(literal_eval)
    return df


# ----------------------------- CI helper -----------------------------
def bootstrap_ci_mean_abs(values, n_boot=1000, alpha=0.05, rng=None):
    """Bootstrap CI for mean(|values|)."""
    x = np.asarray(values, float)
    if x.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(rng)
    boot = np.empty(n_boot, float)
    for b in range(n_boot):
        boot[b] = np.mean(np.abs(rng.choice(x, size=x.size, replace=True)))
    lo = np.percentile(boot, 100 * (alpha / 2))
    hi = np.percentile(boot, 100 * (1 - alpha / 2))
    # enforce lo <= mean <= hi
    mean = float(np.mean(np.abs(x)))
    lo, hi = float(min(lo, hi, mean)), float(max(lo, hi, mean))
    return lo, hi


# ----------------------------- Core per-tissue plot -----------------------------
def plot_tissue_by_sex(
    df: pd.DataFrame,
    tissue: str,
    window_size=10,
    alpha=0.05,
    n_boot=1000,
    sign_frac_thresh=0.05,
    line_color="#1f77b4",
    ribbon_color="#f3a6b6",
):
    tdf = df[df["tissue"] == tissue].copy()
    if len(tdf) < window_size * 2:
        return None  # not enough samples

    min_age, max_age = tdf["age"].min(), tdf["age"].max()
    vmin, vmax = min_age + window_size, max_age - window_size
    if vmin > vmax:
        return None

    fig, (ax_m, ax_f) = plt.subplots(1, 2, figsize=(18, 7))
    out = {}

    for sex, ax in (("male", ax_m), ("female", ax_f)):
        sdf = tdf[tdf["sex"] == sex].copy()
        if len(sdf) < window_size * 2:
            ax.text(0.5, 0.5, f"Insufficient {sex} samples", ha="center", va="center", transform=ax.transAxes)
            out[sex] = None
            continue

        # z-score features within (tissue, sex)
        X = np.vstack(sdf["features"].apply(np.asarray)).astype(float)
        Xz = StandardScaler().fit_transform(X)
        sdf = sdf.reset_index(drop=True)
        sdf["_zfeat"] = list(Xz)

        ages = sorted(sdf["age"].unique())
        ages = [a for a in ages if vmin <= a <= vmax]
        if not ages:
            ax.text(0.5, 0.5, f"No valid ages (window={window_size})", ha="center", va="center", transform=ax.transAxes)
            out[sex] = None
            continue

        A, mean_abs, lo, hi, lo_n, hi_n, mean_p, flags = [], [], [], [], [], [], [], []

        for ta in ages:
            m_lo = (sdf["age"] >= ta - window_size) & (sdf["age"] < ta)
            m_hi = (sdf["age"] >= ta) & (sdf["age"] < ta + window_size)
            lo_df, hi_df = sdf[m_lo], sdf[m_hi]
            if len(lo_df) < 2 or len(hi_df) < 2:
                continue

            L = np.vstack(lo_df["_zfeat"]).astype(float)
            U = np.vstack(hi_df["_zfeat"]).astype(float)

            # Welch t per feature
            _, p = stats.ttest_ind(U, L, equal_var=False, axis=0, nan_policy="omit")
            p = np.where(np.isfinite(p), p, 1.0)

            # effect per feature (mean diff)
            d = U.mean(axis=0) - L.mean(axis=0)
            absd = np.abs(d)
            m = float(absd.mean())
            ci_lo, ci_hi = bootstrap_ci_mean_abs(absd, n_boot=n_boot, alpha=alpha)

            A.append(int(ta))
            mean_abs.append(m)
            lo.append(ci_lo); hi.append(ci_hi)
            lo_n.append(int(len(lo_df))); hi_n.append(int(len(hi_df)))
            mean_p.append(float(np.nanmean(p)))
            flags.append(bool((p < alpha).mean() >= sign_frac_thresh))

        if not A:
            ax.text(0.5, 0.5, f"Insufficient {sex} age groups", ha="center", va="center", transform=ax.transAxes)
            out[sex] = None
            continue

        # sort by age
        order = np.argsort(A)
        A = np.asarray(A)[order]
        mean_abs = np.asarray(mean_abs)[order]
        lo = np.asarray(lo)[order]; hi = np.asarray(hi)[order]
        lo_n = np.asarray(lo_n)[order]; hi_n = np.asarray(hi_n)[order]
        flags = [flags[i] for i in order]
        mean_p = np.asarray(mean_p)[order]

        # plot ribbon + line
        ax.fill_between(A, lo, hi, color=ribbon_color, alpha=0.28, linewidth=0)
        ax.plot(A, mean_abs, color=line_color, linewidth=2.0, label="Mean |Δ|")
        for a, y, f in zip(A, mean_abs, flags):
            if f:
                ax.plot(a, y, marker="*", markersize=11, color="red", markeredgecolor="black")
            else:
                ax.plot(a, y, marker="o", markersize=6, color=line_color)

        ax.set_xlabel("Age"); ax.set_ylabel("Mean |Δ| (z-scored features)")
        ax.set_title(sex.capitalize())

        # window counts on twin axis
        ax2 = ax.twinx()
        ax2.plot(A, lo_n, marker="s", linestyle="--", color="#9ecae1", label="Lower")
        ax2.plot(A, hi_n, marker="^", linestyle="--", color="#fdae6b", label="Upper")
        ax2.set_ylabel("Window Samples")

        from matplotlib.lines import Line2D
        sig = Line2D([0],[0], marker="*", color="w", markerfacecolor="red",
                     markeredgecolor="black", markersize=11, label="Significant")
        h1,l1 = ax.get_legend_handles_labels()
        h2,l2 = ax2.get_legend_handles_labels()
        ax.legend([sig]+h1+h2, [sig.get_label()]+l1+l2, loc="best")

        # ci table
        ci_table = pd.DataFrame({
            "Age": A.astype(int),
            "MeanAbsEffectSize(|Δ|)": mean_abs,
            "CI_Lower": lo,
            "CI_Upper": hi,
            "MeanPValue": mean_p,
            f"IsSignificant(>={int(sign_frac_thresh*100)}% p<alpha)": flags,
            "LowerSamples": lo_n,
            "UpperSamples": hi_n,
        })

        out[sex] = {
            "ages": A.astype(int).tolist(),
            "mean_abs": mean_abs.tolist(),
            "ci_lo": lo.tolist(),
            "ci_hi": hi.tolist(),
            "flags": flags,
            "mean_p": mean_p.tolist(),
            "lo_n": lo_n.tolist(),
            "hi_n": hi_n.tolist(),
            "ci_table": ci_table,
        }

    fig.suptitle(tissue, fontsize=20, fontweight="bold")
    fig.tight_layout(rect=[0,0,1,0.93])
    out["fig"] = fig
    return out


# ----------------------------- Batch runner -----------------------------
def analyze_all_tissues_by_sex(
    csv_file: str,
    window_sizes=(10,),
    alpha=0.05,
    outdir="./age_analysis_by_sex_ci",
    n_boot=1000,
    min_tissue_samples=200,
    sign_frac_thresh=0.05,
    unprocessed_dir="./unprocessed_plot_age_custom",
    line_color="#1f77b4",
    ribbon_color="#f3a6b6",
):
    df = load_and_prepare_data(csv_file)

    tissues = [t for t in sorted(df["tissue"].unique())
               if (df["tissue"] == t).sum() >= min_tissue_samples]

    os.makedirs(outdir, exist_ok=True)
    os.makedirs(unprocessed_dir, exist_ok=True)

    all_results = {}

    for w in window_sizes:
        wdir = os.path.join(outdir, f"window_size_{w}")
        os.makedirs(wdir, exist_ok=True)
        simplified = {}
        window_results = {}

        for t in tissues:
            res = plot_tissue_by_sex(
                df, t, window_size=w, alpha=alpha, n_boot=n_boot,
                sign_frac_thresh=sign_frac_thresh,
                line_color=line_color, ribbon_color=ribbon_color
            )
            if res is None:
                continue

            # save figure
            fig_path = os.path.join(wdir, f"{t.replace(' ', '_')}.png")
            res["fig"].savefig(fig_path, dpi=220, bbox_inches="tight")
            plt.close(res["fig"])

            # save per-sex CI tables
            for sex in ("male","female"):
                if res.get(sex) is None: 
                    continue
                out_csv = os.path.join(wdir, f"{t.replace(' ', '_')}_{sex}_CI_table.csv")
                res[sex]["ci_table"].to_csv(out_csv, index=False)

            # simplified arrays for downstream use
            simp = {"male": None, "female": None}
            for sex in ("male","female"):
                if res.get(sex) is not None:
                    simp[sex] = {
                        "target_ages": np.asarray(res[sex]["ages"], np.int32),
                        "mean_abs_effect_size": np.asarray(res[sex]["mean_abs"], np.float32),
                        "significant_flags": np.asarray(res[sex]["flags"], bool),
                        "ci_lowers": np.asarray(res[sex]["ci_lo"], np.float32),
                        "ci_uppers": np.asarray(res[sex]["ci_hi"], np.float32),
                    }
            simplified[t] = simp
            window_results[t] = res

        # write per-window simplified pickle
        pkl_path = os.path.join(unprocessed_dir, f"results_window_{w}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(simplified, f)
        all_results[w] = window_results

    # combined simplified across windows
    combined = {}
    for w, wr in all_results.items():
        for t, res in wr.items():
            combined.setdefault(t, {})
            entry = {}
            for sex in ("male","female"):
                if res.get(sex) is None:
                    entry[sex] = None
                else:
                    entry[sex] = {
                        "target_ages": np.asarray(res[sex]["ages"], np.int32),
                        "mean_abs_effect_size": np.asarray(res[sex]["mean_abs"], np.float32),
                        "significant_flags": np.asarray(res[sex]["flags"], bool),
                        "ci_lowers": np.asarray(res[sex]["ci_lo"], np.float32),
                        "ci_uppers": np.asarray(res[sex]["ci_hi"], np.float32),
                    }
            combined[t][f"window_{w}"] = entry

    combined_pkl = os.path.join(unprocessed_dir, "all_results_combined.pkl")
    with open(combined_pkl, "wb") as f:
        pickle.dump(combined, f)

    return all_results


# ----------------------------- CLI -----------------------------
def _parse_windows(s: str):
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        return [int(x) for x in s.strip("[]").split(",") if x.strip()]
    return [int(x) for x in s.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser(description="Build age trajectories (by sex) with bootstrap CIs.")
    ap.add_argument("--csv", required=True, help="CSV with columns: tissue, sex, age, features (stringified list)")
    ap.add_argument("--outdir", default="./age_analysis_by_sex_ci")
    ap.add_argument("--windows", default="10", help='e.g. "10" or "10,15" or "[10,15]"')
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--min-samples", type=int, default=200)
    ap.add_argument("--sign-frac", type=float, default=0.05)
    ap.add_argument("--unprocessed-dir", default="./unprocessed_plot_age_custom")
    args = ap.parse_args()

    analyze_all_tissues_by_sex(
        csv_file=args.csv,
        window_sizes=_parse_windows(args.windows),
        alpha=args.alpha,
        outdir=args.outdir,
        n_boot=args.n_boot,
        min_tissue_samples=args.min_samples,
        sign_frac_thresh=args.sign_frac,
        unprocessed_dir=args.unprocessed_dir,
    )

if __name__ == "__main__":
    main()
