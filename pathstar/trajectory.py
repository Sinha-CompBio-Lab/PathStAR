# pathstar/trajectory.py
import os, pickle, numpy as np, pandas as pd
from ast import literal_eval
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

REQUIRED = {"tissue", "age", "features"}

def load_dataset(csv_file: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    if not REQUIRED.issubset(df.columns):
        missing = REQUIRED - set(df.columns)
        raise ValueError(f"Missing columns: {missing}")
    df["features"] = df["features"].apply(literal_eval)
    return df

def analyze_tissue(df_t, window_size: int, alpha: float, normalize: str | None):
    if len(df_t) < window_size * 2:
        return None

    X = np.vstack(df_t["features"].apply(np.asarray).tolist()).astype(float)
    if normalize == "zscore":
        X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

    ages = df_t["age"].to_numpy()
    min_age, max_age = ages.min(), ages.max()
    lo_age, hi_age = min_age + window_size, max_age - window_size
    target_ages = [a for a in sorted(np.unique(ages)) if lo_age <= a <= hi_age]
    if not target_ages:
        return None

    d_eff, sig, mean_p, lo_n, up_n = [], [], [], [], []
    feat_dim = X.shape[1]

    for a in target_ages:
        lm = (ages >= a - window_size) & (ages < a)
        um = (ages >= a) & (ages < a + window_size)
        L, U = X[lm], X[um]
        lo_n.append(L.shape[0]); up_n.append(U.shape[0])
        if L.shape[0] >= 2 and U.shape[0] >= 2:
            pvals = []
            diffs = []
            for k in range(feat_dim):
                t, p = stats.ttest_ind(L[:, k], U[:, k], equal_var=False, nan_policy="omit")
                pvals.append(1.0 if np.isnan(p) else p)
                diffs.append(np.nanmean(U[:, k]) - np.nanmean(L[:, k]))
            pvals = np.asarray(pvals)
            diffs = np.asarray(diffs)
            d_eff.append(float(np.nanmean(np.abs(diffs))))
            sig.append(bool((pvals < alpha).sum() >= 0.05 * feat_dim))
            mean_p.append(float(np.nanmean(pvals)))
        else:
            d_eff.append(np.nan); sig.append(False); mean_p.append(np.nan)

    mask = ~np.isnan(d_eff)
    if not np.any(mask):
        return None

    return {
        "target_ages": np.array([v for v, m in zip(target_ages, mask) if m], dtype=int),
        "effect_sizes": np.array([v for v, m in zip(d_eff, mask) if m], dtype=float),
        "significant_flags": np.array([v for v, m in zip(sig, mask) if m], dtype=bool),
        "mean_p_values": np.array([v for v, m in zip(mean_p, mask) if m], dtype=float),
        "lower_counts": np.array([v for v, m in zip(lo_n, mask) if m], dtype=int),
        "upper_counts": np.array([v for v, m in zip(up_n, mask) if m], dtype=int),
    }

def plot_transition(res, tissue, window_size, alpha, normalize):
    ta, eff, sig = res["target_ages"], res["effect_sizes"], res["significant_flags"]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlabel("Age"); ax.set_ylabel("Mean |effect|")
    ax.plot(ta, eff, color="blue")
    for a, e, s in zip(ta, eff, sig):
        ax.plot(a, e, marker="*" if s else "o", markersize=12 if s else 6,
                color="red" if s else "blue", markeredgecolor="black" if s else None)
    ax.set_title(f"{tissue} — window={window_size}, α={alpha}, norm={normalize or 'raw'}")
    fig.tight_layout()
    return fig

def build_and_save(csv_file: str,
                   out_dir: str = "outputs/age_analysis",
                   window_size: int = 10,
                   alpha: float = 0.05,
                   min_samples: int = 200,
                   normalize: str | None = "zscore"):
    os.makedirs(out_dir, exist_ok=True)
    df = load_dataset(csv_file)
    results = {}

    for tissue in sorted(df["tissue"].unique()):
        df_t = df[df["tissue"] == tissue]
        if len(df_t) < min_samples:
            continue
        res = analyze_tissue(df_t, window_size, alpha, normalize)
        if res is None:
            continue
        results[tissue] = res

        # save plot
        fig = plot_transition(res, tissue, window_size, alpha, normalize)
        png = os.path.join(out_dir, f"{tissue.replace(' ', '_')}.png")
        fig.savefig(png, dpi=200, bbox_inches="tight")
        plt.close(fig)

    # save pickle of all results
    pkl = os.path.join(out_dir, f"results_window_{window_size}_{normalize or 'raw'}.pkl")
    with open(pkl, "wb") as f:
        pickle.dump(results, f)
    print("Saved:", pkl)
    return results

