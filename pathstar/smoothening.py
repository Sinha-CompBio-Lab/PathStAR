# pathstar/smoothing.py
from __future__ import annotations
import os, pickle
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# ---------- FS helpers ----------
def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def _safe_tissue(t: str) -> str:
    return t.replace(" ", "_").replace("/", "_")

# ---------- Load pickle ----------
def load_results_pickle(pickle_file: str):
    with open(pickle_file, "rb") as f:
        return pickle.load(f)

# ---------- Select points (uses your significance fallback) ----------
def extract_significant_data(data, min_sig_points=5, window_keys=None, fallback_use_all_if_none=True):
    out = {}
    for tissue, obj in data.items():
        if not isinstance(obj, dict):
            continue
        keys = list(obj.keys())
        win_keys = [k for k in (window_keys or keys) if isinstance(k, str) and k.startswith("window_") and k in obj]
        if not win_keys:
            continue
        node = obj.get(win_keys[0], {})
        if not isinstance(node, dict) or "combined" not in node:
            continue

        comb = node["combined"]
        if comb is None or not {"target_ages", "effect_sizes", "significant_flags"}.issubset(comb):
            continue

        ages = np.asarray(comb["target_ages"])
        effects = np.asarray(comb["effect_sizes"])
        sig = np.asarray(comb["significant_flags"]).astype(bool)

        mask = sig & np.isfinite(ages) & np.isfinite(effects)
        a, e = ages[mask], effects[mask]
        if len(a) >= min_sig_points:
            use_a, use_e = a, e
        elif fallback_use_all_if_none:
            m2 = np.isfinite(ages) & np.isfinite(effects)
            use_a, use_e = ages[m2], effects[m2]
        else:
            continue

        idx = np.argsort(use_a)
        use_a, use_e = use_a[idx], use_e[idx]
        if len(use_a) >= 4:
            out[tissue] = {"combined": {"target_ages": use_a, "effect_sizes": use_e}}
    return out

def preprocess_curves(data):
    processed = {}
    for tissue, obj in data.items():
        if "combined" not in obj:
            continue
        a = np.asarray(obj["combined"]["target_ages"])
        e = np.asarray(obj["combined"]["effect_sizes"])
        if not np.all(np.diff(a) >= 0):
            idx = np.argsort(a)
            a, e = a[idx], e[idx]
        processed[tissue] = {"combined": {"target_ages": a, "effect_sizes": e}}
    return processed

# ---------- Smoothers ----------
def fit_spline(ages, effects, window_size=10):
    n = len(ages)
    s = (window_size / max(n, 1) * 0.5) * n
    return UnivariateSpline(ages, effects, s=s)

def fit_gaussian_process(ages, effects, window_size=10):
    ages = np.asarray(ages, float); effects = np.asarray(effects, float)
    lo, hi = float(ages.min()), float(ages.max())
    if hi == lo:
        mu = float(effects.mean())
        return lambda x: np.full_like(np.asarray(x, float), mu, dtype=float)

    x = (ages - lo) / (hi - lo)
    length_scale = min(0.5, (window_size / max(len(ages), 1)) * 2)
    noise_level  = 0.1 * (window_size / 5)
    kernel = 1.0 * RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_level)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gp.fit(x.reshape(-1, 1), effects)

    def predict(xnew):
        xnew = np.asarray(xnew, float)
        z = (xnew - lo) / (hi - lo)
        return gp.predict(z.reshape(-1, 1))
    return predict

# ---------- Writers (exact filenames you already use) ----------
def save_smoothed_files(tissue: str, label: str, method: str,
                        ages: np.ndarray, effects: np.ndarray,
                        dense_age: np.ndarray, dense_effect: np.ndarray,
                        fitted_func, out_dir: str):
    out_dir = _ensure_dir(out_dir)
    stem = f"{_safe_tissue(tissue)}_{label}_{method}"
    cmp_path   = os.path.join(out_dir, f"{stem}_comparison.csv")
    dense_path = os.path.join(out_dir, f"{stem}_dense.csv")

    sm_at_orig = fitted_func(ages)
    cmp = pd.DataFrame({
        "age": ages,
        "original_effect_size": effects,
        "smoothed_effect_size": sm_at_orig,
        "difference": sm_at_orig - effects,
        "abs_difference": np.abs(sm_at_orig - effects),
    })
    dense = pd.DataFrame({"age": dense_age, "smoothed_effect_size": dense_effect})

    with open(cmp_path, "w") as f:
        f.write(f"# Tissue: {tissue}\n# Gender: {label}\n# Method: {method}\n")
        f.write("# Original vs smoothed at original age points\n#\n")
        cmp.to_csv(f, index=False)

    with open(dense_path, "w") as f:
        f.write(f"# Tissue: {tissue}\n# Gender: {label}\n# Method: {method}\n")
        f.write("# Dense smoothed curve (1000 points)\n#\n")
        dense.to_csv(f, index=False)

    return cmp_path, dense_path

# ---------- High-level runner ----------
def smooth_all(pickle_file: str, out_root: str, label: str = "combined",
               do_spline: bool = True, do_gp: bool = True, window_size: int = 10) -> dict:
    out_root   = _ensure_dir(out_root)
    spline_dir = _ensure_dir(os.path.join(out_root, "spline_data"))
    gp_dir     = _ensure_dir(os.path.join(out_root, "gaussian_process_data"))

    data = load_results_pickle(pickle_file)
    sig  = extract_significant_data(data)
    proc = preprocess_curves(sig)

    summary = {"spline": [], "gaussian_process": []}
    for tissue, obj in proc.items():
        ages = np.asarray(obj["combined"]["target_ages"], float)
        effs = np.asarray(obj["combined"]["effect_sizes"], float)
        if len(ages) < 4:
            continue

        dense_age = np.linspace(ages.min(), ages.max(), 1000)

        if do_spline:
            sp = fit_spline(ages, effs, window_size=window_size)
            dense_sp = sp(dense_age)
            save_smoothed_files(tissue, label, "spline", ages, effs, dense_age, dense_sp, sp, spline_dir)
            summary["spline"].append(tissue)

        if do_gp:
            gp = fit_gaussian_process(ages, effs, window_size=window_size)
            dense_gp = gp(dense_age)
            save_smoothed_files(tissue, label, "gaussian_process", ages, effs, dense_age, dense_gp, gp, gp_dir)
            summary["gaussian_process"].append(tissue)

    return summary
