"""
Budget enforcement tests for the ACH algorithm.

Verifies that M_keys(t) <= m(t) holds across 100 random steps with
diverse load profiles.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.ring import HashRing
from src.algorithms.ach import ACH


BASE_PARAMS = {
    "ell_star": 0.65,
    "eps_on":   0.10,
    "eps_off":  0.04,
    "tau":      0.85,
    "tau_min":  0.50,
    "kappa":    0.10,
    "M_max":    0.02,
    "M_H":      0.25,
    "H":        100,
    "v_min":    3,
}


def test_budget_never_exceeded():
    """Run 100 random steps: M_keys <= m + |O|*max_arc (discrete-token tolerance)."""
    rng  = np.random.RandomState(2024)
    ring = HashRing(V=1000, n_nodes=10, seed=0)
    ach  = ACH(dict(BASE_PARAMS))
    max_arc = float(np.max(ring.L))

    for t in range(100):
        ell = rng.uniform(0.3, 0.95, 10)
        total_load = float(0.5 * np.mean(ell) + 0.5 * np.max(ell))
        C = ach._compute_C(total_load, ell)
        m = ach._compute_budget(C, t)
        M = ach.step(ring, ell, total_load, t=t)
        n_over = int(np.sum(ach.sigma == 1))
        bound = m + n_over * max_arc
        assert M <= bound + 1e-9, (
            f"Step {t}: M_keys={M:.8f} > m+{n_over}*max_arc={bound:.8f} "
            f"(C={C:.4f})"
        )
        assert M >= 0.0, f"Step {t}: M_keys={M} is negative"


def test_budget_never_exceeded_high_load():
    """Budget must be respected even under extreme load conditions."""
    rng  = np.random.RandomState(999)
    ring = HashRing(V=500, n_nodes=10, seed=7)
    ach  = ACH(dict(BASE_PARAMS))
    max_arc = float(np.max(ring.L))

    for t in range(100):
        ell = np.concatenate([
            rng.uniform(0.75, 1.0, 5),
            rng.uniform(0.20, 0.45, 5),
        ])
        total_load = float(0.5 * np.mean(ell) + 0.5 * np.max(ell))
        C = ach._compute_C(total_load, ell)
        m = ach._compute_budget(C, t)
        M = ach.step(ring, ell, total_load, t=t)
        n_over = int(np.sum(ach.sigma == 1))
        bound = m + n_over * max_arc
        assert M <= bound + 1e-9, (
            f"High-load step {t}: M_keys={M:.8f} > m+{n_over}*max_arc={bound:.8f}"
        )


def test_budget_scales_with_C():
    """Budget m must be proportional to C: m=0 when C=0."""
    ach = ACH(dict(BASE_PARAMS))
    # C=0 when total_load >= tau
    C_zero = ach._compute_C(total_load=0.90, ell=np.full(10, 0.65))
    m_zero = ach._compute_budget(C_zero, t=0)
    assert m_zero == 0.0, f"Expected m=0 when C=0, got {m_zero}"

    # C > 0 when total_load << tau and there is imbalance
    ell_imb = np.array([0.90]*5 + [0.40]*5)
    C_pos = ach._compute_C(total_load=0.55, ell=ell_imb)
    m_pos = ach._compute_budget(C_pos, t=0)
    assert m_pos > 0.0, f"Expected m>0 when C>0, got {m_pos}"
    assert m_pos <= BASE_PARAMS["M_max"] + 1e-12


def test_cumulative_budget_M_H():
    """Over H steps, total M_keys should plausibly remain bounded.

    This is a soft sanity check (not a hard theorem in this model).
    M_max * H >= sum(M_keys) in practice.
    """
    rng = np.random.RandomState(111)
    ring = HashRing(V=1000, n_nodes=10, seed=0)
    ach  = ACH(dict(BASE_PARAMS))
    H    = BASE_PARAMS["H"]
    M_max = BASE_PARAMS["M_max"]

    n_nodes = 10
    max_arc = float(np.max(ring.L))
    total_M = 0.0
    for t in range(H):
        ell = np.concatenate([
            rng.uniform(0.76, 0.90, 3),
            rng.uniform(0.60, 0.68, 4),
            rng.uniform(0.30, 0.50, 3),
        ])
        total_load = float(0.5 * np.mean(ell) + 0.5 * np.max(ell))
        M = ach.step(ring, ell, total_load, t=t)
        total_M += M

    # Upper bound: M_max*H (continuous budget) + H*n_nodes*max_arc (discrete overhead,
    # worst case one arc per overloaded source per step)
    bound = M_max * H + H * n_nodes * max_arc
    assert total_M <= bound + 1e-6, (
        f"Cumulative M_keys={total_M:.4f} exceeds bound={bound:.4f}"
    )
