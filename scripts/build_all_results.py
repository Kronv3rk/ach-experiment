"""
build_all_results.py – Rebuild results/all_results.json from per-run JSONs.

Reads run_*.json files from each series directory, computes aggregate
statistics, paired Wilcoxon signed-rank tests (ACH vs each baseline),
and merges in C5 sensitivity data.

Usage
-----
    python scripts/build_all_results.py [--results-dir results]
"""

import os
import sys
import json
import glob
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy import stats


ALGO_ORDER = ["Static-W", "Static-Wt", "Dynamic-R", "Bounded-Loads", "ACH"]
METRICS    = ["D_mean", "D_max", "M_cum", "pi_chg"]
SERIES     = ["C1", "C2", "C3", "C4a", "C4b", "C4c"]  # C5 = sensitivity only


def ci95(arr: np.ndarray) -> float:
    valid = arr[~np.isnan(arr)]
    n = len(valid)
    if n < 2:
        return float("nan")
    se = float(np.nanstd(arr, ddof=1)) / np.sqrt(n)
    t  = float(stats.t.ppf(0.975, df=n - 1))
    return t * se


def load_runs(series_dir: str) -> list:
    files = sorted(glob.glob(os.path.join(series_dir, "run_*.json")))
    runs = []
    for path in files:
        with open(path) as f:
            runs.append(json.load(f))
    return runs


def build_series(runs: list) -> dict:
    """Build per-series result dict from a list of run JSONs."""
    algos_present = [
        a for a in ALGO_ORDER
        if any(a in r.get("algorithms", {}) for r in runs)
    ]
    n = len(runs)

    result = {
        "n_runs": n,
        "algorithms": {},
        "wilcoxon_vs_ach": {},
        "total_violations": 0,
    }

    ach_vals: dict = {}

    for algo in algos_present:
        agg = {}
        for m in METRICS:
            vals = np.array(
                [r.get("algorithms", {}).get(algo, {}).get("agg", {}).get(m, float("nan"))
                 for r in runs],
                dtype=np.float64,
            )
            agg[f"{m}_mean"] = float(np.nanmean(vals))
            agg[f"{m}_std"]  = float(np.nanstd(vals, ddof=1)) if n > 1 else float("nan")
            agg[f"{m}_ci95"] = ci95(vals)
        agg["n_runs"] = n
        result["algorithms"][algo] = agg

        if algo == "ACH":
            for m in METRICS:
                ach_vals[m] = np.array(
                    [r.get("algorithms", {}).get("ACH", {}).get("agg", {}).get(m, float("nan"))
                     for r in runs],
                    dtype=np.float64,
                )

    # Violations
    for r in runs:
        for a in algos_present:
            result["total_violations"] += r.get("algorithms", {}).get(a, {}).get("n_violations", 0)

    # Wilcoxon: ACH vs each baseline, for D_mean and M_cum
    if "ACH" in algos_present:
        for algo in algos_present:
            if algo == "ACH":
                continue
            row = {}
            for m in ["D_mean", "M_cum"]:
                a_vals = ach_vals.get(m, np.array([]))
                b_vals = np.array(
                    [r.get("algorithms", {}).get(algo, {}).get("agg", {}).get(m, float("nan"))
                     for r in runs],
                    dtype=np.float64,
                )
                valid = ~(np.isnan(a_vals) | np.isnan(b_vals))
                if valid.sum() >= 4:
                    try:
                        _, p = stats.wilcoxon(a_vals[valid], b_vals[valid])
                        row[f"p_{m}"] = float(p)
                    except Exception:
                        row[f"p_{m}"] = None
                else:
                    row[f"p_{m}"] = None

            # Effect size: how much better/worse is ACH on D_mean
            a_d = ach_vals.get("D_mean", np.array([]))
            b_d = np.array(
                [r.get("algorithms", {}).get(algo, {}).get("agg", {}).get("D_mean", float("nan"))
                 for r in runs],
                dtype=np.float64,
            )
            valid = ~(np.isnan(a_d) | np.isnan(b_d))
            if valid.sum() > 0:
                delta = float(np.nanmean(b_d) - np.nanmean(a_d))   # positive = ACH better
                baseline_mean = float(np.nanmean(b_d))
                pct = delta / max(abs(baseline_mean), 1e-12) * 100
                row["ach_D_delta"] = delta
                row["ach_D_pct"]   = pct
            result["wilcoxon_vs_ach"][algo] = row

    return result


def main():
    parser = argparse.ArgumentParser(description="Build all_results.json from per-run JSONs.")
    parser.add_argument("--results-dir", default="results",
                        help="Root results directory (default: results)")
    args = parser.parse_args()

    root = args.results_dir
    all_results = {}

    for s in SERIES:
        series_dir = os.path.join(root, s)
        if not os.path.isdir(series_dir):
            print(f"[SKIP] {s}: directory not found ({series_dir})")
            continue
        runs = load_runs(series_dir)
        if not runs:
            print(f"[SKIP] {s}: no run_*.json files")
            continue
        result = build_series(runs)
        all_results[s] = result

        ach_d = result["algorithms"].get("ACH", {}).get("D_mean_mean", float("nan"))
        ach_m = result["algorithms"].get("ACH", {}).get("M_cum_mean", float("nan"))
        viols = result["total_violations"]
        print(f"{s}: {result['n_runs']} runs  "
              f"ACH D_mean={ach_d:.4f}  M_cum={ach_m:.4f}  violations={viols}")

    # C5: sensitivity sweep only
    sens_path = os.path.join(root, "C5", "sensitivity.json")
    if os.path.isfile(sens_path):
        with open(sens_path) as f:
            all_results["C5_sensitivity"] = json.load(f)
        print(f"C5_sensitivity: loaded from {sens_path}")
    else:
        print(f"[SKIP] C5_sensitivity: {sens_path} not found")

    out_path = os.path.join(root, "all_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {out_path}  (keys: {list(all_results.keys())})")


if __name__ == "__main__":
    main()
