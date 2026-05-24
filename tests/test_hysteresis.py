"""
Tests for ACH hysteresis logic and budget gating.
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

def test_overload_trigger(ach, ring):
    """sigma must transition 0 -> +1 when xi = ell_i - ell_star >= eps_on."""
    ach.sigma = np.zeros(10, dtype=np.int8)
    # All nodes overloaded: ell_star + eps_on = 0.75
    ell = np.full(10, 0.80)   # xi = 0.15 >= eps_on=0.10
    ach._update_hysteresis(ell)
    assert np.all(ach.sigma == 1), f"Expected all sigma=+1, got {ach.sigma}"


def test_overload_clear(ach, ring):
    """sigma must transition +1 -> 0 when xi <= eps_off."""
    ach.sigma = np.ones(10, dtype=np.int8)
    # xi = 0.03 <= eps_off=0.04 -> clear overload
    ell = np.full(10, 0.65 + 0.03)
    ach._update_hysteresis(ell)
    assert np.all(ach.sigma == 0), f"Expected all sigma=0, got {ach.sigma}"


def test_underload_trigger(ach, ring):
    """sigma must transition 0 -> -1 when xi <= -eps_on."""
    ach.sigma = np.zeros(10, dtype=np.int8)
    # ell = 0.65 - 0.12 = 0.53; xi = -0.12 <= -eps_on=-0.10
    ell = np.full(10, 0.53)
    ach._update_hysteresis(ell)
    assert np.all(ach.sigma == -1), f"Expected all sigma=-1, got {ach.sigma}"


def test_underload_clear(ach, ring):
    """sigma must transition -1 -> 0 when xi >= -eps_off."""
    ach.sigma = np.full(10, -1, dtype=np.int8)
    # xi = -0.02 >= -eps_off=-0.04 -> clear underload
    ell = np.full(10, 0.65 - 0.02)
    ach._update_hysteresis(ell)
    assert np.all(ach.sigma == 0), f"Expected all sigma=0, got {ach.sigma}"


def test_hysteresis_zone_stable(ach, ring):
    """sigma must stay 0 when |xi| < eps_on for nodes starting at sigma=0."""
    ach.sigma = np.zeros(10, dtype=np.int8)
    # xi = 0.05 < eps_on=0.10, and > -eps_on
    ell = np.full(10, 0.65 + 0.05)
    ach._update_hysteresis(ell)
    assert np.all(ach.sigma == 0), (
        f"sigma should stay 0 inside hysteresis zone, got {ach.sigma}"
    )


def test_no_rebalance_when_C_zero(ach, ring):
    """M_keys must be 0 when total_load >= tau (C=0 because gamma=0)."""
    # total_load = tau = 0.85 -> gamma = 0 -> C = 0 -> m = 0
    ell = np.full(10, 0.85)
    total_load = 0.90  # above tau
    M = ach.step(ring, ell, total_load, t=0)
    assert M == 0.0, f"Expected M_keys=0 when C=0, got {M}"


def test_sigma_stays_neutral_at_ell_star(ach):
    """sigma stays 0 when ell == ell_star exactly."""
    ach.sigma = np.zeros(10, dtype=np.int8)
    ell = np.full(10, 0.65)   # exactly ell_star
    ach._update_hysteresis(ell)
    assert np.all(ach.sigma == 0)


def test_hysteresis_no_oscillation():
    """Once in overload state, sigma must not flip back immediately if xi > eps_off."""
    ach = ACH(dict(BASE_PARAMS))
    ach.sigma = np.array([1], dtype=np.int8)
    # xi = 0.06 > eps_off=0.04 -> stay in +1
    ell = np.array([0.65 + 0.06])
    ach._update_hysteresis(ell)
    assert ach.sigma[0] == 1, (
        "sigma should remain +1 when xi > eps_off"
    )


def test_mixed_states():
    """Test simultaneous overload, underload, and neutral nodes."""
    ach = ACH(dict(BASE_PARAMS))
    ach.sigma = np.zeros(3, dtype=np.int8)
    ell = np.array([
        0.65 + 0.15,   # xi=+0.15 >= eps_on -> should become +1
        0.65 - 0.15,   # xi=-0.15 <= -eps_on -> should become -1
        0.65 + 0.05,   # |xi|=0.05 < eps_on -> stays 0
    ])
    ach._update_hysteresis(ell)
    assert ach.sigma[0] == +1
    assert ach.sigma[1] == -1
    assert ach.sigma[2] == 0
