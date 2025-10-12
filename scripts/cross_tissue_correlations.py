#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-tissue correlations (single script)
- mode=subset14  : analyze predefined 14-tissue panel (+ optional system-grouped heatmaps)
- mode=all       : analyze all tissues
Outputs pivots, correlation & p-value matrices, filtered pivots, heatmaps, and pairwise tables.

Usage (subset of 14 tissues):
  python scripts/cross_tissue_correlations.py \
    --mode subset14 \
    --delta-file ./delta_score_temp/all_tissues_delta_scores.tsv \
    --trischd-file /shares/sinha/anamikay/projects/path_rem_vs/results/patient_tissue_TRISCHD_matrix.csv \
    --outdir cross_organ_14

Usage (all tissues):
  python scripts/cross_tissue_correlations.py \
    --mode all \
    --delta-file ./delta_score_temp/all_tissues_delta_scores.tsv \
    --trischd-file /shares/sinha/anamikay/projects/path_rem_vs/results/patient_tissue_TRISCHD_matrix.csv \
    --outdir cross_organ_all
"""
import warnings
warnings.filterwarnings("ignore")

import argparse
from pathlib import Path
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list
from scipy.spatial.distance import squareform

# -------------------------
# Defaults
# -------------------------
SUBSET14 = [
    "Esophagus - Muscularis",
    "Esophagus - Gastroesophageal Junction",
    "Stomach",
    "Colon - Sigmoid",
    "Small Intestine - Terminal Ileum",
    "Colon - Transverse",
    "Artery - Tibial",
    "Artery - Aorta",
    "Prostate",
    "Testis",
    "Uterus",
    "Vagina",
    "Ovary",
    "Artery - Coronary",
    "Nerve - Tibial",
]

SYSTEMS = {
    "Digestive": [
        "Esophagus - Muscularis",
        "Esophagus - Gastroesophageal Junction",
        "Stomach",
        "Colon - Sigmoid",
        "Colon - Transverse",
        "Small Intestine - Terminal Ileum",
    ],
    "Vascular/Nerve": [
        "Artery - Aorta",
        "Artery - Coronary",
        "Artery - Tibial",
        "Nerve - Tibial",
    ],
    "Female reproductive": ["Ovary", "Uterus", "Vagina"],
    "Male reproductive":   ["Prostate", "Testis"],
}
SYSTEM_ORDER = ["Digestive", "Vascular/Nerve", "Female reproductive", "Male reproductive"]
TISSUE_TO_SYSTEM = {t: sys for sys, lst in SYSTEMS.items() for t in lst}


# -------------------------
# I/O & preprocessing
# -------------------------
def load_and_prepare_long(delta_file: str) -> pd.DataFrame:
    df = pd.read_csv(delta_file, sep="\t")
    # Individual ID = everything before the last hyphen in sample_id
    df["individual_id"] = df["sample_id"].str.rsplit("-", n=1).str[0]
    # Collapse duplicate individual–tissue rows
    df = (
        df.groupby(["individual_id", "tissue"], as_index=False)
          .agg({"delta": "mean", "age": "first", "sex": "first"})
    )
    return df

def merge_trischd(df: pd.DataFrame, trischd_file: str) -> pd.DataFrame:
    tr = pd.read_csv(trischd_file)  # Subject.ID + per-tissue TRISCHD columns
    tr_long = (
        tr.rename(columns={"Subject.ID": "individual_id"})
          .melt(id_vars="individual_id", var_name="tissue", value_name="ischemic_time")
    )
    out = df.merge(tr_long, on=["individual_id", "tissue"], how="left")
    return out

def add_delta_residuals(df: pd.DataFrame,
                        x_col="ischemic_time", y_col="delta",
                        group_col="tissue", min_n=10, min_unique_x=2) -> pd.DataFrame:
    """
    For each tissue, fit y ~ a + b*x on rows with both y and x present; save residuals to 'delta_resid'.
    Rows w/o ischemic_time remain NaN (you may later fill with within-tissue demeaned delta if desired).
    """
    df = df.copy()
    df["delta_resid"] = np.nan
    for tissue, sub in df.groupby(group_col):
        m = sub[[y_col, x_col]].dropna()
        if len(m) >= min_n and m[x_col].nunique() >= min_unique_x:
            x = m[x_col].values.astype(float)
            y = m[y_col].values.astype(float)
            A = np.vstack([x, np.ones_like(x)]).T
            slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
            idx = sub.index[sub[x_col].notna()]
            df.loc[idx, "delta_resid"] = sub.loc[idx, y_col] - (intercept + slope * sub.loc[idx, x_col])
    return df


# -------------------------
# Correlations
# -------------------------
def calculate_correlation_with_pvalue(pivot_df: pd.DataFrame, min_samples=50):
    """
    Pearson correlation across tissues using individuals × tissues pivot of values.
    Keeps tissues with ≥ min_samples non-null values.
    """
    tissue_counts = pivot_df.count()
    valid = tissue_counts[tissue_counts >= min_samples].index
    filt = pivot_df[valid]

    n = len(valid)
    corr = np.zeros((n, n))
    pval = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                corr[i, j] = 1.0
                pval[i, j] = 0.0
            else:
                t1, t2 = valid[i], valid[j]
                paired = filt[[t1, t2]].dropna()
                if len(paired) >= 10:
                    r, p = pearsonr(paired[t1], paired[t2])
                    corr[i, j] = r
                    pval[i, j] = p
                else:
                    corr[i, j] = np.nan
                    pval[i, j] = np.nan

    corr_df = pd.DataFrame(corr, index=valid, columns=valid)
    pval_df = pd.DataFrame(pval, index=valid, columns=valid)
    return corr_df, pval_df, filt


def analyze_significant_correlations(corr_df, pval_df, alpha=0.05) -> pd.DataFrame:
    rows = []
    cols = list(corr_df.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            t1, t2 = cols[i], cols[j]
            r, p = corr_df.at[t1, t2], pval_df.at[t1, t2]
            if not (np.isnan(r) or np.isnan(p)):
                rows.append({
                    "Tissue1": t1, "Tissue2": t2,
                    "Correlation": r, "P-value": p,
                    "Significant": p < alpha, "Abs_Correlation": abs(r),
                })
    return pd.DataFrame(rows)


# -------------------------
# Plotting helpers
# -------------------------
def heatmap_with_stars(corr_df, pval_df, title_suffix="", alpha=0.05,
                       order=None, savepath=None, figsize=(16, 14)):
    if order is not None:
        order = [t for t in order if t in corr_df.index]
        corr_df = corr_df.loc[order, order]
        pval_df = pval_df.loc[order, order]

    annot = np.empty_like(corr_df, dtype=object)
    for i in range(corr_df.shape[0]):
        for j in range(corr_df.shape[1]):
            if i == j:
                annot[i, j] = "1.00"
            elif pd.isna(corr_df.iloc[i, j]):
                annot[i, j] = "N/A"
            else:
                r = corr_df.iloc[i, j]; p = pval_df.iloc[i, j]
                if p < 0.001: s = "***"
                elif p < 0.01: s = "**"
                elif p < 0.05: s = "*"
                else: s = ""
                annot[i, j] = f"{r:.2f}{s}"

    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        corr_df, annot=annot, fmt="",
        cmap="RdBu_r", vmin=-1, vmax=1, center=0,
        square=True, linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Correlation Coefficient"},
    )
    ax.set_title(
        f"Cross-Tissue Delta Score Correlations (ischemia-adjusted){title_suffix}\n"
        f"(Pearson r; * p<0.05, ** p<0.01, *** p<0.001)",
        pad=20,
    )
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
        print(f"Saved heatmap → {savepath}")
    plt.close()


def heatmap_clean(corr_df, pval_df, title_suffix="", alpha=0.05,
                  min_corr=0.10, savepath=None, figsize=(24, 22)):
    show = (pval_df < alpha) & (np.abs(corr_df) >= min_corr)
    corr_clean = corr_df.copy()
    corr_clean[~show] = np.nan

    annot = np.empty_like(corr_clean, dtype=object)
    for i in range(corr_clean.shape[0]):
        for j in range(corr_clean.shape[1]):
            if i == j:
                annot[i, j] = "1.00"
            elif np.isnan(corr_clean.iloc[i, j]):
                annot[i, j] = ""
            else:
                annot[i, j] = f"{corr_clean.iloc[i, j]:.2f}"

    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_clean, annot=annot, fmt="",
        cmap="RdBu_r", center=0, square=True,
        cbar_kws={"label": "Correlation Coefficient"},
        linewidths=0.5, xticklabels=True, yticklabels=True,
        mask=np.isnan(corr_clean),
    )
    plt.title(
        f"Cross-Tissue Delta Score Correlations (ischemia-adjusted){title_suffix}\n"
        f"(Significant ≥|{min_corr}|, p<{alpha})",
        fontsize=16, pad=20,
    )
    plt.xticks(rotation=45, ha="right", fontsize=18)
    plt.yticks(rotation=0, fontsize=18)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
        print(f"Saved heatmap → {savepath}")
    plt.close()


# -------------------------
# System-grouped ordering (subset14 only)
# -------------------------
def _cluster_block(labels, corr_df, method="average"):
    if len(labels) <= 1:
        return labels
    sub = corr_df.loc[labels, labels].copy().fillna(0.0)
    np.fill_diagonal(sub.values, 1.0)
    dist = 1.0 - sub
    d = squareform(dist.values, checks=False)
    try:
        Z = linkage(d, method=method)
        Z = optimal_leaf_ordering(Z, d)
        order_idx = leaves_list(Z)
        return [labels[i] for i in order_idx]
    except Exception:
        return labels

def grouped_cluster_order(corr_df):
    present = [t for t in corr_df.index if t in TISSUE_TO_SYSTEM]
    sys_to_block = {sys: [t for t in SYSTEMS[sys] if t in present] for sys in SYSTEM_ORDER}
    final = []
    for sys in SYSTEM_ORDER:
        block = sys_to_block[sys]
        if not block:
            continue
        final += _cluster_block(block, corr_df)
    return [t for t in final if t in corr_df.index]


# -------------------------
# One runner (ALL / MALE / FEMALE)
# -------------------------
def run_group(df: pd.DataFrame, who: str, outdir: Path,
              min_samples_all=30, min_samples_sex=20,
              mode="subset14", save_system_grouped=False):
    """
    who ∈ {"ALL","MALE","FEMALE"}
    mode ∈ {"subset14", "all"}
    """
    if who == "ALL":
        dfx = df.copy()
        title_suffix = " - ALL SAMPLES"
        min_samp = min_samples_all
    elif who == "MALE":
        dfx = df[df["sex"] == "male"].copy()
        title_suffix = " - MALE SAMPLES"
        min_samp = min_samples_sex
    else:
        dfx = df[df["sex"] == "female"].copy()
        title_suffix = " - FEMALE SAMPLES"
        min_samp = min_samples_sex

    # pivot on delta_resid
    pivot = dfx.pivot(index="individual_id", columns="tissue", values="delta_resid")
    pivot.to_csv(outdir / f"pivot_{who.lower()}_ischemia_adj.tsv", sep="\t", na_rep="NA")

    # correlations
    corr, pval, filt = calculate_correlation_with_pvalue(pivot, min_samples=min_samp)
    corr.to_csv(outdir / f"corr_{who.lower()}_ischemia_adj.tsv", sep="\t", na_rep="NA")
    pval.to_csv(outdir / f"pvals_{who.lower()}_ischemia_adj.tsv", sep="\t", na_rep="NA")
    filt.to_csv(outdir / f"pivot_{who.lower()}_filtered_ischemia_adj.tsv", sep="\t", na_rep="NA")

    # heatmaps
    if mode == "subset14":
        # keep only requested tissues if present
        order = [t for t in SUBSET14 if t in corr.index]
        heatmap_with_stars(
            corr, pval, title_suffix=title_suffix, order=order,
            savepath=outdir / f"heatmap_{who.lower()}_no_dendro_ischemia_adj.png",
            figsize=(16, 14),
        )
        if save_system_grouped and len(corr) > 1:
            order_grouped = grouped_cluster_order(corr)
            heatmap_with_stars(
                corr.loc[order_grouped, order_grouped],
                pval.loc[order_grouped, order_grouped],
                title_suffix=f"{title_suffix} — system-grouped",
                order=None,
                savepath=outdir / f"heatmap_{who.lower()}_system_grouped_plain_ischemia_adj.png",
                figsize=(16, 14),
            )
    else:  # mode == "all"
        heatmap_clean(
            corr, pval, title_suffix=title_suffix,
            savepath=outdir / f"heatmap_{who.lower()}_clean_ischemia_adj.png",
            figsize=(24, 22),
        )

    # pairwise table
    pairwise = analyze_significant_correlations(corr, pval)
    pairwise.to_csv(outdir / f"pairwise_corr_{who.lower()}_ischemia_adj.tsv", sep="\t", index=False)

    return corr, pval, pairwise


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["subset14", "all"], required=True)
    ap.add_argument("--delta-file", required=True,
                    help="TSV of all_tissues_delta_scores (columns include sample_id, tissue, sex, delta, ...)")
    ap.add_argument("--trischd-file", required=True,
                    help="CSV with Subject.ID and tissue-specific TRISCHD columns")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--min-samples-all", type=int, default=30,
                    help="Min non-null per tissue for ALL cohort correlations")
    ap.add_argument("--min-samples-sex", type=int, default=20,
                    help="Min non-null per tissue for sex-specific correlations")
    ap.add_argument("--save-system-grouped", action="store_true",
                    help="(subset14 mode) also save system-grouped heatmaps")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load & prep
    df = load_and_prepare_long(args.delta_file)
    print(f"Loaded: {len(df)} individual–tissue rows across {df['tissue'].nunique()} tissues.")
    print("Sex distribution (pre-merge):")
    print(df["sex"].value_counts(dropna=False))

    # Merge ischemic time (TRISCHD) and residualize
    df = merge_trischd(df, args.trischd_file)
    have_tr = df["ischemic_time"].notna().sum()
    print(f"Merged TRISCHD: {have_tr} / {len(df)} rows with ischemic_time.")

    df = add_delta_residuals(df)
    avail = df["delta_resid"].notna().sum()
    print(f"Residuals available: {avail} / {len(df)} rows.")

    # Mode-specific filtering
    if args.mode == "subset14":
        before = len(df)
        df = df[df["tissue"].isin(SUBSET14)].copy()
        print(f"Filtered to subset14: {len(df)} rows (from {before}) "
              f"across {df['tissue'].nunique()} tissues.")

    # Run groups
    print("\n=== ANALYSIS: ALL ===")
    corr_all, pval_all, pw_all = run_group(
        df, who="ALL", outdir=outdir,
        min_samples_all=args.min_samples_all,
        min_samples_sex=args.min_samples_sex,
        mode=args.mode, save_system_grouped=args.save_system_grouped,
    )

    print("\n=== ANALYSIS: MALE ===")
    corr_male, pval_male, pw_male = run_group(
        df, who="MALE", outdir=outdir,
        min_samples_all=args.min_samples_all,
        min_samples_sex=args.min_samples_sex,
        mode=args.mode, save_system_grouped=args.save_system_grouped,
    )

    print("\n=== ANALYSIS: FEMALE ===")
    corr_fem, pval_fem, pw_fem = run_group(
        df, who="FEMALE", outdir=outdir,
        min_samples_all=args.min_samples_all,
        min_samples_sex=args.min_samples_sex,
        mode=args.mode, save_system_grouped=args.save_system_grouped,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
