"""
run_series.py – Run one experiment series across multiple independent runs.

Usage
-----
    python scripts/run_series.py --series C1 --n-runs 30 \
        --output-dir results/C1 --base-seed 42

For each run r in [0, n_runs):
  seed = base_seed + r * 1000

All 5 algorithms are evaluated on the same load sequence per run.
Invariants are checked every step; violations are logged.
Results are saved as JSON: results/<series>/run_NNN.json
"""

import sys
import os
import argparse
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import yaml

from src.experiment import run_experiment


def _load_config(series: str, config_dir: str) -> dict:
    """Load and merge series config with base config."""
    base_path   = os.path.join(config_dir, "base.yaml")
    series_path = os.path.join(config_dir, f"series_{series.lower()}.yaml")

    with open(base_path, "r") as f:
        base_cfg = yaml.safe_load(f)

    with open(series_path, "r") as f:
        series_cfg = yaml.safe_load(f)

    # Deep merge: series values override base
    merged = dict(base_cfg)
    for k, v in series_cfg.items():
        if k != "extends":
            if isinstance(v, dict) and isinstance(merged.get(k), dict):
                merged[k] = {**merged[k], **v}
            else:
                merged[k] = v
    return merged


def _convert_for_json(obj):
    """Recursively convert numpy types to Python native types for JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _convert_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_for_json(v) for v in obj]
    return obj


def run_series(series: str, n_runs: int, output_dir: str,
               base_seed: int, config_dir: str):
    """Run a complete experiment series.

    Parameters
    ----------
    series : str
        Series name, e.g. 'C1'.
    n_runs : int
        Number of independent runs.
    output_dir : str
        Directory to save results JSON files.
    base_seed : int
        Base random seed; run r uses seed = base_seed + r * 1000.
    config_dir : str
        Directory containing YAML config files.
    """
    cfg = _load_config(series, config_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Series {series}: {cfg.get('description', '')}")
    print(f"  T={cfg.get('T', 500)}  n_runs={n_runs}  base_seed={base_seed}")
    print(f"  Output: {output_dir}")
    print()

    series_summary = []

    for run_id in range(n_runs):
        seed = base_seed + run_id * 1000
        t0   = time.time()

        results = run_experiment(cfg, seed=seed)

        elapsed = time.time() - t0

        # Build per-run summary
        run_data = {
            "run_id":  run_id,
            "seed":    seed,
            "series":  series,
            "elapsed": elapsed,
            "algorithms": {}
        }

        n_violations = 0
        for algo_name, data in results.items():
            viols = data.get("violations", [])
            n_violations += len(viols)
            run_data["algorithms"][algo_name] = {
                "D_trace":    data["D_trace"],
                "M_trace":    data["M_trace"],
                "agg":        data["agg"],
                "violations": viols[:20],  # cap to first 20
                "n_violations": len(viols),
            }

        # Warn if invariants violated
        if n_violations > 0:
            print(f"  [WARN] run {run_id:03d}: {n_violations} invariant violations")

        out_path = os.path.join(output_dir, f"run_{run_id:03d}.json")
        with open(out_path, "w") as f:
            json.dump(_convert_for_json(run_data), f, indent=2)

        series_summary.append({
            "run_id": run_id,
            "seed":   seed,
            "agg": {name: data["agg"] for name, data in results.items()},
            "n_violations": n_violations,
        })

        # Progress
        D_ach = results.get("ACH", {}).get("agg", {}).get("D_mean", float("nan"))
        D_sw  = results.get("Static-W", {}).get("agg", {}).get("D_mean", float("nan"))
        print(f"  run {run_id:03d}  seed={seed}  elapsed={elapsed:.1f}s  "
              f"D_ACH={D_ach:.3f}  D_SW={D_sw:.3f}")

    # Save series-level summary
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(_convert_for_json(series_summary), f, indent=2)

    print(f"\nSeries {series} complete. Results saved to {output_dir}/")
    print(f"Summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run one ACH experiment series."
    )
    parser.add_argument("--series",     required=True,
                        help="Series name, e.g. C1")
    parser.add_argument("--n-runs",     type=int, default=30)
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: results/<series>)")
    parser.add_argument("--base-seed",  type=int, default=42)
    parser.add_argument("--config-dir", default=None,
                        help="Config directory (default: <repo>/configs)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir   = os.path.join(script_dir, "..")

    config_dir = args.config_dir or os.path.join(repo_dir, "configs")
    output_dir = args.output_dir or os.path.join(repo_dir, "results",
                                                  args.series)

    run_series(
        series=args.series,
        n_runs=args.n_runs,
        output_dir=output_dir,
        base_seed=args.base_seed,
        config_dir=config_dir,
    )


if __name__ == "__main__":
    main()
