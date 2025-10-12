# pathstar/individual.py
from __future__ import annotations
import os, re
import numpy as np
import pandas as pd
from ast import literal_eval
from sklearn.preprocessing import StandardScaler

# --------- Utilities ----------
def create_safe_filename(name: str) -> str:
    safe = re.sub(r"[^\w\-_]", "_", str(name))
    return re.sub(r"_+", "_", safe).strip("_")

def clean_features_column(features_str):
    """
    Robustly parse 'features' which may be a stringified list.
    Returns list[float] or None on failure.
    """
    try:
        if isinstance(features_str, list):
            return features_str
        s = str(features_str)
        i = s.find("[")
        if i != -1:
            s = s[i:]
        if not (s.startswith("[") and s.endswith("]")):
            m = re.search(r"\[.*\]", s)
            if m:
                s = m.group(0)
        return literal_eval(s)
    except Exception:
        return None

def load_and_filter_data(csv_file: str, target_tissue: str, debug_first: bool = False) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    if debug_first:
        print("Columns:", df.columns.tolist())
        print("Shape:", df.shape)
        for i in range(min(3, len(df))):
            s = str(df.loc[i, "features"])
            print(f"Row {i} features (head): {repr(s[:200])}")

    feats = [clean_features_column(x) for x in df["features"]]
    df = df.assign(features=feats).dropna(subset=["features"]).reset_index(drop=True)

    tdf = df[df["tissue"] == target_tissue].copy()
    if len(tdf):
        try:
            print(f"[{target_tissue}] n={len(tdf)} | age {tdf['age'].min()}–{tdf['age'].max()} | "
                  f"sex {tdf['sex'].value_counts().to_dict() if 'sex' in tdf.columns else 'NA'}")
            print(f"[{target_tissue}] feature length example: {len(tdf['features'].iloc[0])}")
        except Exception:
            pass
    else:
        print(f"[{target_tissue}] no rows after cleaning")
    return tdf

def zscore_features_inplace(tissue_df: pd.DataFrame, by_sex: bool = False) -> pd.DataFrame:
    """
    Z-score per feature across samples for this tissue.
    by_sex=False (default) → combine sexes (your combined-sex mode).
    """
    if len(tissue_df) == 0:
        return tissue_df

    if by_sex and "sex" in tissue_df.columns:
        parts = []
        for _, g in tissue_df.groupby("sex", group_keys=False):
            X = np.vstack(g["features"].apply(np.asarray).to_list()).astype(float)
            z = StandardScaler().fit_transform(X)
            gg = g.copy()
            gg["features"] = list(z)
            parts.append(gg)
        return pd.concat(parts, axis=0).sort_index()
    else:
        X = np.vstack(tissue_df["features"].apply(np.asarray).to_list()).astype(float)
        z = StandardScaler().fit_transform(X)
        out = tissue_df.copy()
        out["features"] = list(z)
        return out

def _effect_for_sample(target_row: pd.Series, younger_df: pd.DataFrame) -> tuple[float, float]:
    """
    On z-scored features:
      Δ = target - mean(younger)
      d = Δ / sd(younger)
    Returns mean(|d|), mean(|Δ|).
    """
    x = np.asarray(target_row["features"], dtype=float)
    Y = np.asarray(younger_df["features"].to_list(), dtype=float)
    mu = Y.mean(axis=0)
    sd = Y.std(axis=0, ddof=1)

    delta = x - mu
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.divide(delta, sd, out=np.zeros_like(delta), where=np.isfinite(sd) & (sd != 0))

    return float(np.mean(np.abs(d))), float(np.mean(np.abs(delta)))

# --------- Core per-tissue analysis ----------
def analyze_tissue_individual_scores(
    csv_file: str,
    target_tissue: str,
    min_younger_samples: int = 5,
    out_dir: str = "./outputs/individual_scores",
    zscore_by_sex: bool = False,          # keep False for combined-sex z-scoring
    sex_matched_younger: bool = False,    # ignore sex for younger group
    younger_window_years: int = 10,
    min_target_age: int = 30,
) -> pd.DataFrame:
    """
    Computes per-sample |d| (Cohen's d on z-scored features) and |Δ|.
    Saves a TSV per tissue and returns the DataFrame.
    """
    tdf = load_and_filter_data(csv_file, target_tissue, debug_first=(target_tissue == "Ovary"))
    if len(tdf) == 0:
        print(f"[{target_tissue}] skip")
        return pd.DataFrame()

    # z-score across ALL samples (sexes combined unless zscore_by_sex=True)
    tdf = zscore_features_inplace(tdf, by_sex=zscore_by_sex)

    targets = tdf[tdf["age"] >= min_target_age].copy()
    print(f"[{target_tissue}] targets (age≥{min_target_age}): {len(targets)}")

    results = []
    has_sample_col = "sample_id" in tdf.columns

    for idx, row in targets.iterrows():
        age = int(row["age"])
        sex = row["sex"] if "sex" in tdf.columns else None
        sid = row["sample_id"] if has_sample_col else idx

        lo, hi = age - younger_window_years, age - 1
        mask = (tdf["age"] >= lo) & (tdf["age"] <= hi)
        if sex_matched_younger and sex is not None:
            mask = mask & (tdf["sex"] == sex)
        younger = tdf[mask]

        if len(younger) < min_younger_samples:
            results.append({
                "sample_id": sid, "age": age, "sex": sex, "tissue": target_tissue,
                "cohens_d_effect_size": np.nan, "custom_effect_size": np.nan,
                "younger_group_size": int(len(younger)),
                "younger_group_age_range": f"{lo}-{hi}",
            })
            continue

        try:
            mean_abs_d, mean_abs_delta = _effect_for_sample(row, younger)
            results.append({
                "sample_id": sid, "age": age, "sex": sex, "tissue": target_tissue,
                "cohens_d_effect_size": mean_abs_d,     # |d|
                "custom_effect_size": mean_abs_delta,   # |Δ|
                "younger_group_size": int(len(younger)),
                "younger_group_age_range": f"{lo}-{hi}",
            })
        except Exception as e:
            print(f"[{target_tissue}] error @ sample {sid} age {age}: {e}")
            results.append({
                "sample_id": sid, "age": age, "sex": sex, "tissue": target_tissue,
                "cohens_d_effect_size": np.nan, "custom_effect_size": np.nan,
                "younger_group_size": int(len(younger)),
                "younger_group_age_range": f"{lo}-{hi}",
            })

    res_df = pd.DataFrame(results)

    # Save per-tissue TSV
    safe = create_safe_filename(target_tissue)
    tdir = os.path.join(out_dir, safe)
    os.makedirs(tdir, exist_ok=True)
    out_path = os.path.join(tdir, f"{safe}_effect_size_deviation.tsv")
    res_df.to_csv(out_path, sep="\t", index=False)
    print(f"[{target_tissue}] saved → {out_path}")

    return res_df

def analyze_multiple_tissues(
    csv_file: str,
    tissues: list[str],
    min_younger_samples: int = 5,
    out_dir: str = "./outputs/individual_scores",
    zscore_by_sex: bool = False,
    sex_matched_younger: bool = False,
    younger_window_years: int = 10,
    min_target_age: int = 30,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    os.makedirs(out_dir, exist_ok=True)

    all_results, summary_rows = {}, []
    for i, tissue in enumerate(tissues, 1):
        print("\n" + "=" * 60)
        print(f"[{i}/{len(tissues)}] {tissue}")
        print("=" * 60)

        try:
            tdf = analyze_tissue_individual_scores(
                csv_file=csv_file,
                target_tissue=tissue,
                min_younger_samples=min_younger_samples,
                out_dir=out_dir,
                zscore_by_sex=zscore_by_sex,
                sex_matched_younger=sex_matched_younger,
                younger_window_years=younger_window_years,
                min_target_age=min_target_age,
            )
            all_results[tissue] = tdf

            if len(tdf):
                valid = tdf.dropna(subset=["cohens_d_effect_size"])
                summary_rows.append({
                    "tissue": tissue,
                    "total_samples": len(tdf),
                    "successful_calculations": len(valid),
                    "mean_abs_cohens_d": valid["cohens_d_effect_size"].mean() if len(valid) else np.nan,
                    "std_abs_cohens_d": valid["cohens_d_effect_size"].std() if len(valid) else np.nan,
                    "mean_abs_delta": valid["custom_effect_size"].mean() if len(valid) else np.nan,
                    "std_abs_delta": valid["custom_effect_size"].std() if len(valid) else np.nan,
                })
            else:
                summary_rows.append({
                    "tissue": tissue,
                    "total_samples": 0,
                    "successful_calculations": 0,
                    "mean_abs_cohens_d": np.nan,
                    "std_abs_cohens_d": np.nan,
                    "mean_abs_delta": np.nan,
                    "std_abs_delta": np.nan,
                })
        except Exception as e:
            print(f"[{tissue}] ERROR: {e}")
            summary_rows.append({
                "tissue": tissue,
                "total_samples": 0,
                "successful_calculations": 0,
                "mean_abs_cohens_d": np.nan,
                "std_abs_cohens_d": np.nan,
                "mean_abs_delta": np.nan,
                "std_abs_delta": np.nan,
            })

    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, "tissue_analysis_summary.tsv")
    summary.to_csv(summary_path, sep="\t", index=False)
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Summary saved → {summary_path}")
    return all_results, summary
