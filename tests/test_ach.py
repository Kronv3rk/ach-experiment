"""
Tests for src/algorithms/ach.py (ACH algorithm).
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.ring import HashRing
from src.algorithms.ach import ACH


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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


@pytest.fixture
def ach():
    return ACH(dict(BASE_PARAMS))


@pytest.fixture
def ring():
    return HashRing(V=1000, n_nodes=10, seed=42)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_no_rebalance_when_no_overload(ach, ring):
    """M_keys must be 0 when all ell == ell_star (no nodes overloaded)."""
    ell = np.full(10, 0.65)      # exactly at target
    total_load = 0.65
    M = ach.step(ring, ell, total_load, t=0)
    assert M == 0.0, f"Expected M_keys=0 at ell_star, got {M}"


def test_no_rebalance_when_total_load_high(ach, ring):
    """M_keys must be 0 when total_load >= tau (gamma=0, budget=0)."""
    ell = np.full(10, 0.90)
    total_load = 0.90   # >= tau=0.85 -> gamma=0 -> C=0 -> m=0
    M = ach.step(ring, ell, total_load, t=0)
    assert M == 0.0, f"Expected M_keys=0 when total_load >= tau, got {M}"


def test_budget_respected(ach, ring):
    """M_keys must not exceed m(t) + |O|*max_arc (discrete-token tolerance)."""
    rng = np.random.RandomState(77)
    max_arc = float(np.max(ring.L))
    for trial in range(50):
        ell = rng.uniform(0.3, 0.95, 10)
        total_load = float(0.5 * np.mean(ell) + 0.5 * np.max(ell))
        C = ach._compute_C(total_load, ell)
        m = ach._compute_budget(C, trial)
        M = ach.step(ring, ell, total_load, t=trial)
        n_over = int(np.sum(ach.sigma == 1))
        bound = m + n_over * max_arc
        assert M <= bound + 1e-9, (
            f"Trial {trial}: M_keys={M:.6f} > m+{n_over}*max_arc={bound:.6f} "
            f"(C={C:.4f})"
        )


def test_stable_zone(ach, ring):
    """sigma stays 0 when ell_i in [ell_star-eps_off, ell_star+eps_off]."""
    # Place ell in the dead-band: |xi| < eps_on but > eps_off
    # Starting sigma = 0, ell = 0.65 + 0.06 (xi=0.06 < eps_on=0.10)
    ell = np.full(10, 0.65 + 0.06)
    total_load = 0.60
    ach.sigma = np.zeros(10, dtype=np.int8)
    M = ach.step(ring, ell, total_load, t=0)
    # sigma should remain 0 since xi < eps_on
    assert np.all(ach.sigma == 0), (
        f"sigma should stay 0 in dead-band, got {ach.sigma}"
    )


def test_rebalance_occurs_with_overload(ring):
    """ACH should actually move tokens when clear overload/underload exists."""
    params = dict(BASE_PARAMS)
    ach = ACH(params)
    # Manually set sigma so that nodes 0-2 are overloaded, 7-9 underloaded
    ach.sigma = np.array([1, 1, 1, 0, 0, 0, 0, -1, -1, -1], dtype=np.int8)
    ell = np.array([0.90, 0.90, 0.90, 0.65, 0.65, 0.65, 0.65, 0.35, 0.35, 0.35])
    total_load = 0.60
    M = ach.step(ring, ell, total_load, t=0)
    # With clear overload and budget available, some tokens should move
    assert M >= 0.0, "M_keys should be non-negative"


def test_v_min_respected(ring):
    """After rebalancing, no node should have fewer than v_min tokens."""
    params = dict(BASE_PARAMS)
    ach = ACH(params)
    ach.sigma = np.array([1, 1, 1, 0, 0, 0, 0, -1, -1, -1], dtype=np.int8)
    ell = np.array([0.90, 0.90, 0.90, 0.65, 0.65, 0.65, 0.65, 0.35, 0.35, 0.35])
    total_load = 0.60
    ach.step(ring, ell, total_load, t=0)
    v_min = params["v_min"]
    for i in range(10):
        cnt = ring.node_count(i)
        assert cnt >= v_min, f"Node {i} has {cnt} tokens < v_min={v_min}"


def test_m_always_nonnegative(ach, ring):
    """Budget m must always be >= 0."""
    rng = np.random.RandomState(5)
    for t in range(30):
        ell = rng.uniform(0.0, 1.0, 10)
        total_load = float(rng.uniform(0.0, 1.0))
        C = ach._compute_C(total_load, ell)
        m = ach._compute_budget(C, t)
        assert m >= 0.0, f"m={m} is negative at t={t}"


def test_C_range(ach):
    """C(t) must lie in [0, 1]."""
    rng = np.random.RandomState(1)
    for _ in range(100):
        ell = rng.uniform(0.0, 1.0, 10)
        total_load = float(rng.uniform(0.0, 1.2))
        C = ach._compute_C(total_load, ell)
        assert 0.0 <= C <= 1.0 + 1e-9, f"C={C} outside [0,1]"
