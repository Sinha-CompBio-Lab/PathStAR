# pathstar/delta.py
from __future__ import annotations
import os
from glob import glob
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# -----------------------------
# Helpers (filenames & I/O)
# -----------------------------
def to_safe_name(tissue: str) -> str:
    return tissue.replace(" ", "_").replace("/", "_")

def from_safe_name(safe: str) -> str:
    return safe.replace("_", " ")

def discover_tissues_from_spline_dir(spline_dir: str, label: str = "combined") -> list[str]:
    pattern = os.path.join(spline_dir, f"*_{label}_spline_comparison.csv")
    paths = glob(pattern)
    tissues = []
    for p in paths:
        base = os.path.basename(p)
        safe_root = base.replace(f"_{label}_spline_comparison.csv", "")
        tissues.append(from_safe_name(safe_root))
    return sorted(set(tissues))

def load_curve_data(
    tissue: str,
    method: str,                          # "spline" | "gaussian_process"
    label: str = "combined",
    spline_dir: str = "",
    gaussian_dir: str = "",
):
    safe = to_safe_name(tissue)
    if method == "spline":
        path = os.path.join(spline_dir, f"{safe}_{label}_spline_comparison.csv")
    elif method == "gaussian_process":
        path = os.path.join(gaussian_dir, f"{safe}_{label}_gaussian_process_comparison.csv")
    else:
        raise ValueError(f"Unknown method: {method}")

    if not os.path.exists(path):
        print(f"  [WARN] Curve file not found for {tissue} ({method}): {path}")
        return None, None

    try:
        df = pd.read_csv(path, comment="#")
        df.columns = df.columns.str.strip()
        if not {"age", "smoothed_effect_size"}.issubset(df.columns):
            print(f"  [WARN] Missing required columns in {os.path.basename(path)}")
            return None, None
        return df, method
    except Exception as e:
        print(f"  [ERROR] Reading {path}: {e}")
        return None, None

def interpolate_effect_size_at_age(curve_df: pd.DataFrame, target_age: float) -> float:
    if curve_df is None or len(curve_df) == 0:
        return np.nan
    ages = curve_df["age"].astype(float).values
    effects = curve_df["smoothed_effect_size"].astype(float).values
    # clamp to ends
    if target_age <= ages.min():
        return float(effects[0])
    if target_age >= ages.max():
        return float(effects[-1])
    try:
        f = interp1d(ages, effects, kind="linear", fill_value="extrapolate")
        return float(f(target_age))
    except Exception as e:
        print(f"  [WARN] Interp error at age {target_age}: {e}")
        return np.nan

def load_patient_data(individual_score_dir: str, tissue: str):
    safe = to_safe_name(tissue)
    path = os.path.join(individual_score_dir, safe, f"{safe}_effect_size_deviation.tsv")
    if not os.path.exists(path):
        print(f"  [WARN] Patient data not found: {path}")
        return None
    try:
        return pd.read_csv(path, sep="\t")
    except Exception as e:
        print(f"  [ERROR] Loading patient data for {tissue}: {e}")
        return None

# -----------------------------
# Per-tissue delta computation
# -----------------------------
def calculate_delta_for_tissue(
    tissue: str,
    method: str,  # "spline" | "gaussian_process"
    individual_score_dir: str,
    spline_dir: str,
    gaussian_dir: str,
    label: str = "combined",
):
    """
    delta = patient_custom_effect_size - smoothed_curve_effect_size_at_patient_age
    """
    print(f"Processing tissue: {tissue}  [method={method}]")

    patient_df = load_patient_data(individual_score_dir, tissue)
    if patient_df is None:
        print("  Skipping - no patient data.")
        return None

    curve_df, method_used = load_curve_data(
        tissue, method=method, label=label,
        spline_dir=spline_dir, gaussian_dir=gaussian_dir
    )
    if curve_df is None:
        print("  Skipping - no curve data.")
        return None

    print(f"  Curve points: {len(curve_df)} | Age range: "
          f"{curve_df['age'].min():.1f}–{curve_df['age'].max():.1f}")

    deltas = []
    for _, row in patient_df.iterrows():
        age = row.get("age")
        eff = row.get("custom_effect_size")
        if pd.isna(age) or pd.isna(eff):
            deltas.append(np.nan)
            continue
        ref = interpolate_effect_size_at_age(curve_df, float(age))
        deltas.append(np.nan if pd.isna(ref) else float(eff) - ref)

    out = patient_df.copy()
    out["delta"] = deltas
    out["curve_method_used"] = method_used
    out["curve_label_used"] = label

    valid = out["delta"].dropna()
    if len(valid):
        print(f"  Delta computed for {len(valid)}/{len(out)} samples "
              f"(mean={valid.mean():.4f}, sd={valid.std():.4f})")
    else:
        print("  No valid deltas computed.")
    return out

# -----------------------------
# Driver for all tissues
# -----------------------------
def calculate_delta_scores_all_tissues(
    individual_score_dir: str,
    spline_dir: str,
    gaussian_dir: str,
    output_dir: str,
    label: str = "combined",
    method_overrides: dict[str, str] | None = None,   # e.g., {"Ovary": "gaussian_process"}
):
    os.makedirs(output_dir, exist_ok=True)
    tissues = discover_tissues_from_spline_dir(spline_dir, label=label)
    print(f"Discovered {len(tissues)} tissues from spline dir.")

    method_overrides = method_overrides or {}
    all_results = []
    summary_rows = []

    for i, tissue in enumerate(tissues, 1):
        method = method_overrides.get(tissue, "spline")
        print("\n" + "=" * 60)
        print(f" {i}/{len(tissues)}  {tissue}")
        print("=" * 60)

        try:
            res = calculate_delta_for_tissue(
                tissue=tissue,
                method=method,
                individual_score_dir=individual_score_dir,
                spline_dir=spline_dir,
                gaussian_dir=gaussian_dir,
                label=label,
            )
            if res is None:
                summary_rows.append({
                    "tissue": tissue, "method": method,
                    "total_samples": 0, "valid_delta": 0,
                    "mean_delta": np.nan, "sd_delta": np.nan,
                    "min_delta": np.nan, "max_delta": np.nan, "median_delta": np.nan
                })
                continue

            safe = to_safe_name(tissue)
            out_path = os.path.join(output_dir, f"{safe}_delta_scores.tsv")
            res.to_csv(out_path, sep="\t", index=False)
            print(f"  Saved: {out_path}")

            all_results.append(res)
            valid = res["delta"].dropna()

            summary_rows.append({
                "tissue": tissue, "method": method,
                "total_samples": len(res), "valid_delta": len(valid),
                "mean_delta": valid.mean() if len(valid) else np.nan,
                "sd_delta": valid.std() if len(valid) else np.nan,
                "min_delta": valid.min() if len(valid) else np.nan,
                "max_delta": valid.max() if len(valid) else np.nan,
                "median_delta": valid.median() if len(valid) else np.nan
            })
        except Exception as e:
            print(f"  [ERROR] {tissue}: {e}")
            summary_rows.append({
                "tissue": tissue, "method": method,
                "total_samples": 0, "valid_delta": 0,
                "mean_delta": np.nan, "sd_delta": np.nan,
                "min_delta": np.nan, "max_delta": np.nan, "median_delta": np.nan
            })

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined_path = os.path.join(output_dir, "all_tissues_delta_scores.tsv")
        combined.to_csv(combined_path, sep="\t", index=False)
        print(f"\nCombined results → {combined_path} ({len(combined)} rows)")
        valid = combined["delta"].dropna()
        if len(valid):
            print(f"Overall: n_valid={len(valid)} mean={valid.mean():.4f} sd={valid.std():.4f} "
                  f"range=({valid.min():.4f}, {valid.max():.4f})")

    summary_df = pd.DataFrame(summary_rows).sort_values(["tissue"])
    summary_path = os.path.join(output_dir, "delta_calculation_summary.tsv")
    summary_df.to_csv(summary_path, sep="\t", index=False)
    print(f"\nSummary saved → {summary_path}")
    return summary_df
