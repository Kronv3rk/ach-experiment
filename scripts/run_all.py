"""
run_all.py – Run all five experiment series (C1–C5) sequentially.

Usage
-----
    python scripts/run_all.py [--n-runs 30] [--base-seed 42]
                               [--output-dir results] [--config-dir configs]

Each series uses the default n_runs from its YAML config unless overridden.
"""

import sys
import os
import argparse
import subprocess
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


ALL_SERIES = ["C1", "C2", "C3", "C4", "C5"]

DEFAULT_N_RUNS = {
    "C1": 30,
    "C2": 30,
    "C3": 30,
    "C4": 30,
    "C5": 10,
}


def main():
    parser = argparse.ArgumentParser(
        description="Run all ACH experiment series."
    )
    parser.add_argument("--n-runs",     type=int, default=None,
                        help="Override n_runs for all series.")
    parser.add_argument("--base-seed",  type=int, default=42)
    parser.add_argument("--output-dir", default=None,
                        help="Root output directory (default: <repo>/results).")
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--series",     nargs="+", default=ALL_SERIES,
                        help="Which series to run (default: all).")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir   = os.path.join(script_dir, "..")
    config_dir = args.config_dir or os.path.join(repo_dir, "configs")
    output_dir = args.output_dir or os.path.join(repo_dir, "results")

    run_series_script = os.path.join(script_dir, "run_series.py")

    overall_t0 = time.time()

    for series in args.series:
        n_runs = args.n_runs if args.n_runs is not None else DEFAULT_N_RUNS.get(series, 30)
        series_output = os.path.join(output_dir, series)

        print(f"{'='*60}")
        print(f"Starting series {series}  (n_runs={n_runs})")
        print(f"{'='*60}")

        cmd = [
            sys.executable, run_series_script,
            "--series",     series,
            "--n-runs",     str(n_runs),
            "--output-dir", series_output,
            "--base-seed",  str(args.base_seed),
            "--config-dir", config_dir,
        ]

        t0 = time.time()
        result = subprocess.run(cmd, check=False)
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"[ERROR] Series {series} failed with return code {result.returncode}")
        else:
            print(f"Series {series} completed in {elapsed:.1f}s")
        print()

    total = time.time() - overall_t0
    print(f"All series completed in {total:.1f}s")


if __name__ == "__main__":
    main()
