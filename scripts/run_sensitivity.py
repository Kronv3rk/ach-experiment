"""
run_sensitivity.py – C5 parameter sensitivity sweep for ACH.

Sweeps eps_on, kappa, tau, M_max independently (one at a time, others at
their base.yaml defaults). Runs 10 independent seeds per grid point and
reports mean D_mean, M_cum, pi_chg.

Usage
-----
    python scripts/run_sensitivity.py [--n-runs 10] [--output-dir results/C5]
                                       [--config-dir configs]
"""

import sys
import os
import argparse
import json
import copy
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import yaml

from src.experiment import run_experiment

# ---------------------------------------------------------------------------
# Parameter grids
# ---------------------------------------------------------------------------

SWEEPS = [
    {
        "param":  "eps_on",
        "values": [0.05, 0.08, 0.10, 0.15, 0.20],
        "extra":  lambda v: {"eps_off": round(v * 0.4, 4)},
    },
    {
        "param":  "kappa",
        "values": [0.05, 0.10, 0.15, 0.20],
        "extra":  None,
    },
    {
        "param":  "tau",
        "values": [0.70, 0.80, 0.85, 0.90, 0.95],
        "extra":  None,
    },
    {
        "param":  "M_max",
        "values": [0.005, 0.01, 0.02, 0.05, 0.10],
        "extra":  None,
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_base(config_dir: str) -> dict:
    with open(os.path.join(config_dir, "base.yaml")) as f:
        cfg = yaml.safe_load(f)
    # Force constant load at lambda=1437.5
    cfg["load"] = {"type": "constant", "lambda": 1437.5}
    return cfg


def _run_point(base_cfg: dict, param: str, value,
               extra: dict, n_runs: int, base_seed: int) -> dict:
    """Run n_runs with one ACH parameter overridden. Return mean +/- CI stats."""
    cfg = copy.deepcopy(base_cfg)
    cfg["ach"][param] = value
    if extra:
        cfg["ach"].update(extra)

    D_vals, M_vals, pi_vals = [], [], []
    for r in range(n_runs):
        res = run_experiment(cfg, seed=base_seed + r * 1000)
        agg = res["ACH"]["agg"]
        D_vals.append(agg["D_mean"])
        M_vals.append(agg["M_cum"])
        pi_vals.append(agg["pi_chg"])

    n = len(D_vals)
    ci = lambda a: float(1.96 * np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return {
        "D_mean": round(float(np.mean(D_vals)), 5),
        "D_ci":   round(ci(D_vals), 5),
        "M_cum":  round(float(np.mean(M_vals)), 5),
        "M_ci":   round(ci(M_vals), 5),
        "pi_chg": round(float(np.mean(pi_vals)), 5),
        "pi_ci":  round(ci(pi_vals), 5),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_sensitivity(n_runs: int, output_dir: str,
                    config_dir: str, base_seed: int = 42):

    base_cfg = _load_base(config_dir)
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    for sweep in SWEEPS:
        param  = sweep["param"]
        values = sweep["values"]
        extra_fn = sweep.get("extra")

        print(f"\n{'='*52}")
        print(f"Sweep: {param}   (n_runs={n_runs} per point)")
        print(f"  {'Value':>8}  {'D':>9}  {'M_cum':>9}  {'pi_chg':>9}")
        print(f"  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*9}")

        sweep_results = {}
        for v in values:
            extra = extra_fn(v) if extra_fn else None
            t0 = time.time()
            row = _run_point(base_cfg, param, v, extra, n_runs, base_seed)
            elapsed = time.time() - t0

            label = f"{v:.3f}" if isinstance(v, float) else str(v)
            print(f"  {label:>8}  {row['D_mean']:9.4f}  "
                  f"{row['M_cum']:9.4f}  {row['pi_chg']:9.4f}  ({elapsed:.1f}s)")

            sweep_results[str(v)] = {"value": v, **row}
            if extra:
                sweep_results[str(v)]["extra"] = extra

        all_results[param] = sweep_results

    # Save JSON
    out_path = os.path.join(output_dir, "sensitivity.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Final table with CIs
    print(f"\n{'='*70}")
    print("SUMMARY  (mean +/- 95% CI)")
    print(f"{'='*70}")
    for sweep in SWEEPS:
        param = sweep["param"]
        print(f"\n  {param}")
        print(f"  {'Value':>8}  {'D +/- CI':>18}  {'M_cum +/- CI':>18}  {'pi_chg +/- CI':>18}")
        print(f"  {'-'*8}  {'-'*18}  {'-'*18}  {'-'*18}")
        for v_str, row in all_results[param].items():
            v_lbl = f"{float(v_str):.3f}"
            print(f"  {v_lbl:>8}  "
                  f"{row['D_mean']:7.4f}+/-{row['D_ci']:.4f}  "
                  f"  {row['M_cum']:7.4f}+/-{row['M_ci']:.4f}  "
                  f"  {row['pi_chg']:7.4f}+/-{row['pi_ci']:.4f}")

    print(f"\nSaved -> {out_path}")
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="ACH parameter sensitivity sweep (series C5)."
    )
    parser.add_argument("--n-runs",     type=int, default=10)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--base-seed",  type=int, default=42)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir   = os.path.join(script_dir, "..")
    config_dir = args.config_dir or os.path.join(repo_dir, "configs")
    output_dir = args.output_dir or os.path.join(repo_dir, "results", "C5")

    run_sensitivity(
        n_runs=args.n_runs,
        output_dir=output_dir,
        config_dir=config_dir,
        base_seed=args.base_seed,
    )


if __name__ == "__main__":
    main()
