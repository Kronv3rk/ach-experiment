"""
effect_size.py – Cliff's delta and distribution analysis for ACH vs baselines.

Computes Cliff's delta |delta| as effect size for paired D_mean comparisons
between ACH and each baseline across all runs of each series. Optionally
generates boxplot and ECDF figures of D_mean distributions per series.

Usage
-----
    python scripts/effect_size.py [--results-dir results] [--plots]

Cliff's delta interpretation (Romano et al., 2006):
    |delta| < 0.147 : negligible
    |delta| < 0.33  : small
    |delta| < 0.474 : medium
    |delta| >= 0.474: large
"""

import sys
import os
import argparse
import json
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

ALGO_ORDER = ["Static-W", "Static-Wt", "Dynamic-R", "Bounded-Loads", "ACH"]
SERIES     = ["C1", "C2", "C3", "C4a", "C4b", "C4c"]


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cliff's delta effect size for two independent samples.

    delta = (#(x_i > y_j) - #(x_i < y_j)) / (n_x * n_y)

    Range: [-1, 1]. Sign indicates direction; magnitude indicates strength.

    Parameters
    ----------
    x, y : np.ndarray
        Sample arrays (1-D, no NaN).

    Returns
    -------
    float
        Cliff's delta in [-1, 1].
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return float("nan")

    # Pairwise comparison via broadcasting
    diff = x[:, None] - y[None, :]   # (nx, ny)
    greater = int(np.sum(diff > 0))
    less    = int(np.sum(diff < 0))
    return (greater - less) / (nx * ny)


def interpret_delta(d: float) -> str:
    """Romano et al. (2006) qualitative thresholds for |Cliff's delta|."""
    ad = abs(d)
    if ad < 0.147:
        return "negligible"
    if ad < 0.330:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"


def load_series_runs(results_dir: str, series: str) -> list:
    pattern = os.path.join(results_dir, series, "run_*.json")
    files = sorted(glob.glob(pattern))
    runs = []
    for path in files:
        with open(path) as f:
            runs.append(json.load(f))
    return runs


def extract_D_mean(runs: list, algo: str) -> np.ndarray:
    """Get D_mean across all runs for one algorithm."""
    vals = []
    for r in runs:
        v = r.get("algorithms", {}).get(algo, {}).get("agg", {}).get("D_mean")
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            vals.append(v)
    return np.array(vals, dtype=np.float64)


def analyze_series(results_dir: str, series: str) -> dict:
    """Compute Cliff's delta for ACH vs each baseline in one series."""
    runs = load_series_runs(results_dir, series)
    if not runs:
        return {}

    ach_vals = extract_D_mean(runs, "ACH")
    if len(ach_vals) == 0:
        return {}

    out = {"n_runs": len(runs), "deltas": {}, "distributions": {}}
    out["distributions"]["ACH"] = ach_vals.tolist()

    for algo in ALGO_ORDER:
        if algo == "ACH":
            continue
        b_vals = extract_D_mean(runs, algo)
        if len(b_vals) == 0:
            continue
        d = cliffs_delta(ach_vals, b_vals)
        out["deltas"][algo] = {
            "cliffs_delta": float(d),
            "abs_cliffs_delta": float(abs(d)),
            "interpretation": interpret_delta(d),
            "ach_smaller_count": int(np.sum(ach_vals[:, None] < b_vals[None, :])),
            "ach_larger_count": int(np.sum(ach_vals[:, None] > b_vals[None, :])),
            "n_pairs": int(len(ach_vals) * len(b_vals)),
        }
        out["distributions"][algo] = b_vals.tolist()

    return out


def plot_distributions(all_results: dict, plot_dir: str):
    """Generate boxplot and ECDF figures per series."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots")
        return

    os.makedirs(plot_dir, exist_ok=True)

    for series, data in all_results.items():
        dists = data.get("distributions", {})
        if not dists:
            continue

        labels = [a for a in ALGO_ORDER if a in dists]
        samples = [np.array(dists[a]) for a in labels]

        # --- Boxplot ---
        fig, ax = plt.subplots(figsize=(8, 4.5))
        bp = ax.boxplot(samples, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'],
                                ['#1f77b4', '#ff7f0e', '#2ca02c',
                                 '#d62728', '#9467bd'][:len(labels)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        ax.set_ylabel(r"$\bar{D}$ over 30 runs")
        ax.set_title(f"Series {series}: distribution of mean imbalance")
        ax.grid(True, axis="y", linestyle=":", alpha=0.5)
        fig.tight_layout()
        out_box = os.path.join(plot_dir, f"boxplot_{series}.svg")
        fig.savefig(out_box, format="svg", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_box}")

        # --- ECDF ---
        fig, ax = plt.subplots(figsize=(8, 4.5))
        colors = {'Static-W': '#1f77b4', 'Static-Wt': '#ff7f0e',
                  'Dynamic-R': '#2ca02c', 'Bounded-Loads': '#d62728',
                  'ACH': '#9467bd'}
        for algo, sample in zip(labels, samples):
            xs = np.sort(sample)
            ys = np.arange(1, len(xs) + 1) / len(xs)
            ax.step(xs, ys, where="post",
                    label=algo, color=colors.get(algo, "black"),
                    linewidth=2 if algo == "ACH" else 1.4)
        ax.set_xlabel(r"$\bar{D}$")
        ax.set_ylabel(r"$F(\bar{D})$")
        ax.set_title(f"Series {series}: ECDF of mean imbalance")
        ax.legend(loc="lower right")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_ylim(0, 1.02)
        fig.tight_layout()
        out_ecdf = os.path.join(plot_dir, f"ecdf_{series}.svg")
        fig.savefig(out_ecdf, format="svg", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_ecdf}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute Cliff's delta and analyze distributions."
    )
    parser.add_argument("--results-dir", default="results",
                        help="Root results directory (default: results)")
    parser.add_argument("--plots", action="store_true",
                        help="Generate boxplot and ECDF SVGs")
    parser.add_argument("--plot-dir", default=None,
                        help="Directory for plots (default: <results-dir>/plots)")
    args = parser.parse_args()

    all_results = {}
    for s in SERIES:
        result = analyze_series(args.results_dir, s)
        if result:
            all_results[s] = result

    # Print summary table
    print(f"\n{'='*72}")
    print(f"  Cliff's delta: ACH vs baseline (D_mean, paired across runs)")
    print(f"{'='*72}")
    print(f"  {'Series':<6} {'Baseline':<16} {'delta':>10} {'|delta|':>9} {'class':>12}")
    print(f"  {'-'*6} {'-'*16} {'-'*10} {'-'*9} {'-'*12}")
    for series, data in all_results.items():
        for algo, info in data.get("deltas", {}).items():
            d = info["cliffs_delta"]
            print(f"  {series:<6} {algo:<16} {d:>10.4f} "
                  f"{abs(d):>9.4f} {info['interpretation']:>12}")

    # Save JSON
    out_path = os.path.join(args.results_dir, "effect_size.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")

    if args.plots:
        plot_dir = args.plot_dir or os.path.join(args.results_dir, "plots")
        plot_distributions(all_results, plot_dir)


if __name__ == "__main__":
    main()
