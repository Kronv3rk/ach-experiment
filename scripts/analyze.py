"""
analyze.py – Statistical analysis of experiment results.

Outputs a table of mean +/- 95% CI and Wilcoxon signed-rank p-values
comparing ACH against each baseline algorithm.

Usage
-----
    python scripts/analyze.py --series C1 --results-dir results/C1
"""

import sys
import os
import argparse
import json
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy import stats


ALGO_ORDER = ["Static-W", "Static-Wt", "Dynamic-R", "Bounded-Loads", "ACH"]
METRICS    = ["D_mean", "D_max", "M_cum", "pi_chg"]

METRIC_LABELS = {
    "D_mean":  "D_mean",
    "D_max":   "D_max",
    "M_cum":   "M_cum",
    "pi_chg":  "pi_chg",
}


def _load_results(results_dir: str) -> list:
    """Load all run_NNN.json files from results_dir."""
    pattern = os.path.join(results_dir, "run_*.json")
    files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No run_*.json files found in {results_dir}"
        )
    data = []
    for path in files:
        with open(path, "r") as f:
            data.append(json.load(f))
    return data


def _extract_metric(runs: list, algo: str, metric: str) -> np.ndarray:
    """Extract one metric across all runs for a given algorithm."""
    values = []
    for run in runs:
        agg = run.get("algorithms", {}).get(algo, {}).get("agg", {})
        v   = agg.get(metric, float("nan"))
        values.append(v)
    return np.array(values, dtype=np.float64)


def _ci95(arr: np.ndarray) -> float:
    """95% confidence interval half-width using t-distribution."""
    n = len(arr[~np.isnan(arr)])
    if n < 2:
        return float("nan")
    se = float(np.nanstd(arr, ddof=1)) / np.sqrt(n)
    t  = float(stats.t.ppf(0.975, df=n - 1))
    return t * se


def analyze(results_dir: str, series: str):
    """Load results and print analysis table.

    Parameters
    ----------
    results_dir : str
        Directory containing run_*.json files.
    series : str
        Series name (for display only).
    """
    runs = _load_results(results_dir)
    n    = len(runs)
    print(f"\nSeries {series}  -  {n} runs from {results_dir}")
    print()

    # Determine which algorithms are present
    algos_present = []
    for a in ALGO_ORDER:
        if any(a in run.get("algorithms", {}) for run in runs):
            algos_present.append(a)

    # --- Table header ---
    header_algo = f"{'Algorithm':<18}"
    for m in METRICS:
        header_algo += f"  {METRIC_LABELS[m]:>22}"
    print(header_algo)
    print("-" * (18 + len(METRICS) * 24))

    # Store ACH values for Wilcoxon tests
    ach_vals: dict = {}

    # --- Per-algorithm rows ---
    for algo in algos_present:
        row = f"{algo:<18}"
        for m in METRICS:
            vals = _extract_metric(runs, algo, m)
            mu   = float(np.nanmean(vals))
            ci   = _ci95(vals)
            row += f"  {mu:8.4f} +/- {ci:7.4f}     "
        print(row)

        if algo == "ACH":
            for m in METRICS:
                ach_vals[m] = _extract_metric(runs, algo, m)

    print()

    # --- Wilcoxon p-values: ACH vs each baseline ---
    if "ACH" in algos_present and ach_vals:
        print("Wilcoxon signed-rank p-values (ACH vs baseline, two-sided):")
        print(f"{'Baseline':<18}", end="")
        for m in METRICS:
            print(f"  {METRIC_LABELS[m]:>18}", end="")
        print()
        print("-" * (18 + len(METRICS) * 20))

        for algo in algos_present:
            if algo == "ACH":
                continue
            row = f"{algo:<18}"
            for m in METRICS:
                a_vals = ach_vals.get(m, np.array([]))
                b_vals = _extract_metric(runs, algo, m)
                # Paired test on matched runs
                valid = ~(np.isnan(a_vals) | np.isnan(b_vals))
                if valid.sum() < 4:
                    row += f"  {'N/A':>18}"
                    continue
                try:
                    _, p = stats.wilcoxon(a_vals[valid], b_vals[valid])
                    row += f"  {p:>18.4f}"
                except Exception:
                    row += f"  {'error':>18}"
            print(row)

    print()

    # --- Violation summary ---
    total_viols = sum(
        sum(run.get("algorithms", {}).get(a, {}).get("n_violations", 0)
            for a in algos_present)
        for run in runs
    )
    print(f"Total invariant violations across all runs: {total_viols}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze ACH experiment results."
    )
    parser.add_argument("--series",      default="",
                        help="Series name (for display). Ignored when --input-dir is used.")
    parser.add_argument("--results-dir", default=None,
                        help="Directory with run_*.json files for a single series.")
    parser.add_argument("--input-dir",   default=None,
                        help="Root results directory; auto-discovers C1..C5 subdirectories.")
    args = parser.parse_args()

    if args.input_dir:
        # Auto-discover all C* series subdirectories
        import glob as _glob
        root = args.input_dir
        series_dirs = sorted(_glob.glob(os.path.join(root, "C*")))
        for series_dir in series_dirs:
            if os.path.isdir(series_dir):
                series_name = os.path.basename(series_dir)
                try:
                    analyze(results_dir=series_dir, series=series_name)
                except FileNotFoundError as e:
                    print(f"[SKIP] {series_name}: {e}")
    elif args.results_dir:
        analyze(results_dir=args.results_dir, series=args.series)
    else:
        parser.error("Provide either --results-dir or --input-dir.")


if __name__ == "__main__":
    main()
