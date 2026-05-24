"""
calibrate.py – Binary search for Lambda such that the load distribution
matches the target imbalance profile for the ACH experiment.

Target:
  - Mean ell for weak nodes (cap=0.25)   > ell_star + eps_on = 0.75
  - Mean ell for strong nodes (cap=1.0)  < ell_star - eps_on = 0.55
  - D(t) in steady state ~ 0.20-0.30

Uses the Static-W baseline (equal token distribution) with 100 warmup steps.

Usage
-----
    python scripts/calibrate.py [--config configs/base.yaml]
"""

import sys
import os
import argparse
import numpy as np

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
from src.ring import HashRing
from src.cluster import Cluster
from src.load_generator import LoadGenerator
from src.telemetry import Telemetry
from src.metrics import compute_D


def _load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _run_warmup(lam: float, cfg: dict, warmup: int = 100, seed: int = 42) -> dict:
    """Run `warmup` steps with Static-W and return steady-state stats."""
    rng = np.random.RandomState(seed)
    V       = int(cfg.get("V", 1000))
    n_nodes = int(cfg.get("n0", 10))
    NK      = int(cfg.get("NK", 50000))
    caps    = list(cfg.get("capacities", [1.0] * n_nodes))
    mu_base = float(cfg.get("mu_base", 350.0))
    tel_p   = cfg.get("telemetry", {})

    ring    = HashRing(V=V, n_nodes=n_nodes, seed=seed)
    cluster = Cluster(capacities=caps, mu_base=mu_base)
    gen     = LoadGenerator(NK=NK, zipf_s=0.0, seed=seed)
    tel     = Telemetry(n_nodes, tel_p)

    ells = []
    for _ in range(warmup):
        keys, wf = gen.generate(lam, rng)
        raw, _   = cluster.simulate_load(ring, keys, wf)
        tel.update(raw)
        ells.append(tel.compute_ell())

    caps_arr = np.array(caps)
    strong_ids = np.where(caps_arr >= 0.99)[0]
    weak_ids   = np.where(caps_arr <= 0.26)[0]

    # Use last 50 steps for steady-state estimate
    recent = np.array(ells[-50:])
    ell_strong = float(np.mean(recent[:, strong_ids]))
    ell_weak   = float(np.mean(recent[:, weak_ids]))
    D_vals     = [compute_D(e) for e in recent]
    D_mean     = float(np.mean(D_vals))

    return {
        "ell_strong": ell_strong,
        "ell_weak":   ell_weak,
        "D_mean":     D_mean,
    }


def calibrate(cfg: dict,
              lam_lo: float = 500.0,
              lam_hi: float = 8000.0,
              tol: float = 10.0,
              max_iter: int = 30,
              warmup: int = 100,
              seed: int = 42) -> float:
    """Binary search for Lambda satisfying the imbalance targets.

    Parameters
    ----------
    cfg : dict
        Base configuration.
    lam_lo, lam_hi : float
        Initial search bracket.
    tol : float
        Convergence tolerance on Lambda.
    max_iter : int
        Maximum binary search iterations.
    warmup : int
        Warm-up steps for each evaluation.
    seed : int
        Random seed.

    Returns
    -------
    float
        Calibrated Lambda.
    """
    ach_cfg  = cfg.get("ach", {})
    ell_star = float(ach_cfg.get("ell_star", 0.65))
    eps_on   = float(ach_cfg.get("eps_on",   0.10))

    target_weak_min   = ell_star + eps_on       # = 0.75
    target_strong_max = ell_star - eps_on       # = 0.55
    target_D_lo, target_D_hi = 0.18, 0.32

    print(f"Targets: ell_weak > {target_weak_min:.2f}, "
          f"ell_strong < {target_strong_max:.2f}, "
          f"D in [{target_D_lo:.2f}, {target_D_hi:.2f}]")
    print(f"Search bracket: [{lam_lo:.0f}, {lam_hi:.0f}]")
    print()

    best_lam = None
    best_score = float("inf")

    for it in range(max_iter):
        lam_mid = 0.5 * (lam_lo + lam_hi)
        stats   = _run_warmup(lam_mid, cfg, warmup=warmup, seed=seed)

        ew = stats["ell_weak"]
        es = stats["ell_strong"]
        D  = stats["D_mean"]

        # Score: sum of constraint violations (we want all <= 0)
        score = max(0.0, target_weak_min - ew) + max(0.0, es - target_strong_max)

        print(f"  iter {it+1:2d}: lambda={lam_mid:8.1f}  "
              f"ell_weak={ew:.3f}  ell_strong={es:.3f}  D={D:.3f}  score={score:.4f}")

        if score < best_score:
            best_score = score
            best_lam   = lam_mid

        if score == 0.0 and target_D_lo <= D <= target_D_hi:
            print(f"\nConverged at lambda={lam_mid:.1f}")
            return lam_mid

        # Steer search: if weak nodes not overloaded, increase lambda
        if ew < target_weak_min:
            lam_lo = lam_mid
        else:
            lam_hi = lam_mid

        if lam_hi - lam_lo < tol:
            break

    print(f"\nBest lambda found: {best_lam:.1f}  (score={best_score:.4f})")
    stats = _run_warmup(best_lam, cfg, warmup=warmup, seed=seed)
    print(f"Final stats: ell_weak={stats['ell_weak']:.3f}  "
          f"ell_strong={stats['ell_strong']:.3f}  D={stats['D_mean']:.3f}")
    return best_lam


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate Lambda for ACH experiment."
    )
    parser.add_argument(
        "--config", default="configs/base.yaml",
        help="Path to base YAML config file."
    )
    parser.add_argument(
        "--lam-lo", type=float, default=500.0,
        help="Lower bound of Lambda search bracket."
    )
    parser.add_argument(
        "--lam-hi", type=float, default=8000.0,
        help="Upper bound of Lambda search bracket."
    )
    parser.add_argument(
        "--warmup", type=int, default=100,
        help="Number of warmup steps per evaluation."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed."
    )
    args = parser.parse_args()

    # Resolve config path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path   = os.path.join(script_dir, "..", args.config) \
                 if not os.path.isabs(args.config) else args.config

    cfg = _load_config(cfg_path)
    calibrate(
        cfg,
        lam_lo=args.lam_lo,
        lam_hi=args.lam_hi,
        warmup=args.warmup,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
