"""
plot_traces.py – Generate matplotlib trace plots for each experiment series.

Produces D(t) traces with median and inter-quartile bands for each algorithm.
Saves plots as SVG files.

Usage
-----
    python scripts/plot_traces.py --series C1 --results-dir results/C1 \
        --output-dir plots/C1
"""

import sys
import os
import argparse
import json
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

ALGO_STYLES = {
    "Static-W":      {"color": "#1f77b4", "linestyle": "-",  "linewidth": 1.5},
    "Static-Wt":     {"color": "#ff7f0e", "linestyle": "--", "linewidth": 1.5},
    "Dynamic-R":     {"color": "#2ca02c", "linestyle": "-.", "linewidth": 1.5},
    "Bounded-Loads": {"color": "#d62728", "linestyle": ":",  "linewidth": 1.5},
    "ACH":           {"color": "#9467bd", "linestyle": "-",  "linewidth": 2.2},
}

ALGO_ORDER = ["Static-W", "Static-Wt", "Dynamic-R", "Bounded-Loads", "ACH"]

FONT_FAMILY = "serif"
FONT_SIZE   = 12


def _set_style():
    """Apply Times New Roman-style matplotlib settings."""
    plt.rcParams.update({
        "font.family":       FONT_FAMILY,
        "font.serif":        ["Times New Roman", "DejaVu Serif"],
        "font.size":         FONT_SIZE,
        "axes.labelsize":    FONT_SIZE,
        "axes.titlesize":    FONT_SIZE + 1,
        "legend.fontsize":   FONT_SIZE - 1,
        "xtick.labelsize":   FONT_SIZE - 1,
        "ytick.labelsize":   FONT_SIZE - 1,
        "figure.dpi":        150,
    })


def _load_results(results_dir: str) -> list:
    pattern = os.path.join(results_dir, "run_*.json")
    files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No run_*.json files in {results_dir}")
    data = []
    for path in files:
        with open(path, "r") as f:
            data.append(json.load(f))
    return data


def _gather_traces(runs: list, algo: str, key: str) -> np.ndarray:
    """Stack traces across runs into a 2D array (n_runs, T)."""
    traces = []
    for run in runs:
        trace = run.get("algorithms", {}).get(algo, {}).get(key, [])
        if trace:
            traces.append(np.array(trace, dtype=np.float64))
    if not traces:
        return np.empty((0, 0))
    min_len = min(len(t) for t in traces)
    return np.vstack([t[:min_len] for t in traces])


def plot_D_traces(runs: list, series: str, output_dir: str):
    """Plot D(t) median + IQR bands for all algorithms.

    Parameters
    ----------
    runs : list
        Loaded run data.
    series : str
        Series name (used in title and filename).
    output_dir : str
        Directory to save the SVG.
    """
    _set_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))

    algos_present = [a for a in ALGO_ORDER
                     if any(a in run.get("algorithms", {}) for run in runs)]

    for algo in algos_present:
        mat = _gather_traces(runs, algo, "D_trace")
        if mat.shape[0] == 0:
            continue

        T      = mat.shape[1]
        t_axis = np.arange(T)

        med  = np.median(mat, axis=0)
        q25  = np.percentile(mat, 25, axis=0)
        q75  = np.percentile(mat, 75, axis=0)

        style = ALGO_STYLES.get(algo, {"color": "black", "linestyle": "-", "linewidth": 1.5})
        ax.plot(t_axis, med, label=algo, **style)
        ax.fill_between(t_axis, q25, q75, color=style["color"], alpha=0.15)

    ax.set_xlabel("Time step $t$")
    ax.set_ylabel("$D(t) = \\max(\\ell) - \\bar{\\ell}$")
    ax.set_title(f"Series {series}: Load Imbalance $D(t)$")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_xlim(0, None)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"D_traces_{series}.svg")
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_M_traces(runs: list, series: str, output_dir: str):
    """Plot M_keys(t) median + IQR bands for adaptive algorithms.

    Parameters
    ----------
    runs : list
    series : str
    output_dir : str
    """
    _set_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))

    adaptive = ["Dynamic-R", "Bounded-Loads", "ACH"]
    algos_present = [a for a in adaptive
                     if any(a in run.get("algorithms", {}) for run in runs)]

    for algo in algos_present:
        mat = _gather_traces(runs, algo, "M_trace")
        if mat.shape[0] == 0:
            continue

        T      = mat.shape[1]
        t_axis = np.arange(T)

        med = np.median(mat, axis=0)
        q75 = np.percentile(mat, 75, axis=0)

        style = ALGO_STYLES.get(algo, {"color": "black", "linestyle": "-", "linewidth": 1.5})
        ax.plot(t_axis, med, label=algo, **style)
        ax.fill_between(t_axis, 0, q75, color=style["color"], alpha=0.12)

    ax.set_xlabel("Time step $t$")
    ax.set_ylabel("$M_{\\rm keys}(t)$ (fraction of key space moved)")
    ax.set_title(f"Series {series}: Key Churn $M_{{\\rm keys}}(t)$")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_xlim(0, None)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"M_traces_{series}.svg")
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate trace plots for ACH experiment series."
    )
    parser.add_argument("--series",      nargs="+", default=[],
                        help="Series names (e.g. C1 C2 C3 C4). "
                             "If omitted with --results-dir, a single unnamed series is plotted.")
    parser.add_argument("--results-dir", default=None,
                        help="Directory containing run_*.json files (single series).")
    parser.add_argument("--input-dir",   default=None,
                        help="Root results directory; auto-discovers series subdirectories.")
    parser.add_argument("--output-dir",  default=None,
                        help="Directory to save SVG plots. Default: <results-dir>/plots/")
    args = parser.parse_args()

    # Determine (results_dir, series_name) pairs to process
    jobs = []

    if args.input_dir:
        series_list = args.series if args.series else ["C1", "C2", "C3", "C4", "C5"]
        root = args.input_dir
        for s in series_list:
            d = os.path.join(root, s)
            if os.path.isdir(d):
                out = args.output_dir or os.path.join(root, "plots")
                jobs.append((d, s, out))
            else:
                print(f"[SKIP] {s}: directory not found at {d}")
    elif args.results_dir:
        s = args.series[0] if args.series else ""
        out = args.output_dir or os.path.join(args.results_dir, "plots")
        jobs.append((args.results_dir, s, out))
    else:
        parser.error("Provide either --results-dir or --input-dir.")

    for results_dir, series_name, output_dir in jobs:
        try:
            runs = _load_results(results_dir)
            print(f"Loaded {len(runs)} runs from {results_dir}")
            plot_D_traces(runs, series=series_name, output_dir=output_dir)
            plot_M_traces(runs, series=series_name, output_dir=output_dir)
        except FileNotFoundError as e:
            print(f"[SKIP] {series_name}: {e}")


if __name__ == "__main__":
    main()
