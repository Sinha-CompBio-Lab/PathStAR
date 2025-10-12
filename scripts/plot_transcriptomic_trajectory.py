#!/usr/bin/env python3
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler

def load_pc_table(path):
    df = pd.read_csv(path) if path.endswith(".csv") else pd.read_parquet(path)
    required = {"Tissue", "AGE", "sex"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required - set(df.columns)}")
    pcs = [c for c in df.columns if c.startswith("PC")]
    if not pcs:
        raise ValueError("No PC columns found (expect PC1, PC2, …)")
    return df, pcs

def bootstrap_ci_mean_abs(data, n_boot=1000, alpha=0.05):
    data = np.asarray(data, float)
    rng = np.random.default_rng(42)
    boot = [np.mean(np.abs(rng.choice(data, size=len(data), replace=True))) for _ in range(n_boot)]
    return np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])

def plot_tissue(df, pcs, tissue, outdir, window=10, alpha=0.05, frac_sig=0.05, n_boot=1000):
    df = df[df["Tissue"] == tissue]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for sex, ax in [("male", axes[0]), ("female", axes[1])]:
        d = df[df["sex"] == sex]
        if len(d) < window * 2:
            ax.text(0.5, 0.5, "Not enough samples", ha="center", va="center")
            continue

        X = StandardScaler().fit_transform(d[pcs])
        d = d.assign(_Z=[x for x in X])
        ages = sorted(d["AGE"].unique())
        res = []
        for a in ages:
            lo = d[(d["AGE"] >= a - window) & (d["AGE"] < a)]
            hi = d[(d["AGE"] >= a) & (d["AGE"] < a + window)]
            if len(lo) < 3 or len(hi) < 3:
                continue
            t, p = stats.ttest_ind(np.vstack(hi["_Z"]), np.vstack(lo["_Z"]), axis=0, equal_var=False)
            diff = np.mean(np.vstack(hi["_Z"]), axis=0) - np.mean(np.vstack(lo["_Z"]), axis=0)
            mean_abs = np.mean(np.abs(diff))
            ci_lo, ci_hi = bootstrap_ci_mean_abs(diff, n_boot=n_boot, alpha=alpha)
            sig = (p < alpha).mean() >= frac_sig
            res.append((a, mean_abs, ci_lo, ci_hi, sig))

        if not res:
            continue
        ages, means, lo, hi, sigs = zip(*res)
        ax.fill_between(ages, lo, hi, color="#c6e2ff", alpha=0.3)
        ax.plot(ages, means, color="#0077b6", lw=2)
        for a, m, s in zip(ages, means, sigs):
            ax.plot(a, m, "*" if s else "o", color="red" if s else "#0077b6")
        ax.set_title(sex)
        ax.set_xlabel("Age")
        ax.set_ylabel("Mean |Δ| (z-scored PCs)")
    fig.suptitle(tissue)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(f"{outdir}/{tissue.replace(' ', '_')}.png", dpi=200)
    plt.close()

def main(pc_table, outdir="./plots_tx", window=10, alpha=0.05, frac_sig=0.05, n_boot=1000):
    df, pcs = load_pc_table(pc_table)
    for tissue in sorted(df["Tissue"].unique()):
        print("→", tissue)
        plot_tissue(df, pcs, tissue, outdir, window, alpha, frac_sig, n_boot)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--pc-table", required=True)
    p.add_argument("--outdir", default="./plots_tx")
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--frac-sig", type=float, default=0.05)
    p.add_argument("--n-boot", type=int, default=1000)
    args = p.parse_args()
    main(**vars(args))
