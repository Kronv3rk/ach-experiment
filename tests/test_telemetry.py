"""
Tests for src/telemetry.py (Telemetry).
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.telemetry import Telemetry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DEFAULT_PARAMS = {
    "beta":    0.3,
    "weights": [0.40, 0.25, 0.35],
    "p":       4,
    "alpha":   0.5,
}


@pytest.fixture
def tel():
    return Telemetry(n_nodes=10, params=DEFAULT_PARAMS)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_zero_load_ell_zero(tel):
    """ell must be 0 for all nodes when all raw observations are 0."""
    raw = np.zeros((10, 3), dtype=np.float64)
    for _ in range(20):   # warm up EWMA
        tel.update(raw)
    ell = tel.compute_ell()
    np.testing.assert_allclose(ell, 0.0, atol=1e-12)


def test_full_load_ell_near_one(tel):
    """ell must be close to 1 for all nodes when all raw observations are 1."""
    raw = np.ones((10, 3), dtype=np.float64)
    for _ in range(50):   # let EWMA converge
        tel.update(raw)
    ell = tel.compute_ell()
    # With all inputs = 1 and weights summing to 1, ell = 1.0 exactly
    np.testing.assert_allclose(ell, 1.0, atol=1e-6)


def test_smoothing_reduces_variance():
    """EWMA smoothed output should have lower variance than raw input."""
    rng = np.random.RandomState(42)
    tel = Telemetry(n_nodes=5, params=DEFAULT_PARAMS)
    raw_signals = []
    smoothed_signals = []
    for _ in range(200):
        raw = rng.uniform(0.0, 1.0, (5, 3))
        raw_signals.append(raw.copy())
        smoothed = tel.update(raw)
        smoothed_signals.append(smoothed.copy())

    raw_var = float(np.var([r[:, 0] for r in raw_signals]))
    smo_var = float(np.var([s[:, 0] for s in smoothed_signals]))
    assert smo_var < raw_var, (
        f"Smoothed variance {smo_var:.4f} not less than raw variance {raw_var:.4f}"
    )


def test_total_load_range(tel):
    """total_load must lie in [0, 1] for valid ell inputs."""
    rng = np.random.RandomState(0)
    for _ in range(100):
        raw = rng.uniform(0.0, 1.0, (10, 3))
        tel.update(raw)
        ell = tel.compute_ell()
        tl = tel.compute_total_load(ell)
        assert 0.0 <= tl <= 1.0 + 1e-9, f"total_load={tl} outside [0,1]"


def test_ell_shape(tel):
    """compute_ell must return array of shape (n_nodes,)."""
    raw = np.random.uniform(0, 1, (10, 3))
    tel.update(raw)
    ell = tel.compute_ell()
    assert ell.shape == (10,)


def test_update_shape(tel):
    """update() must return smoothed array of shape (n_nodes, 3)."""
    raw = np.random.uniform(0, 1, (10, 3))
    smoothed = tel.update(raw)
    assert smoothed.shape == (10, 3)


def test_ewma_convergence():
    """Smoothed values must converge to a constant input after enough steps."""
    tel = Telemetry(n_nodes=3, params=DEFAULT_PARAMS)
    raw = np.full((3, 3), 0.7)
    for _ in range(100):
        tel.update(raw)
    np.testing.assert_allclose(tel.smoothed, 0.7, atol=1e-4)


def test_invalid_raw_shape(tel):
    """update() must raise ValueError for wrong-shaped input."""
    with pytest.raises(ValueError):
        tel.update(np.zeros((5, 3)))   # wrong n_nodes


def test_reset_clears_state(tel):
    """reset() must zero the smoothed state."""
    raw = np.ones((10, 3))
    tel.update(raw)
    tel.reset()
    np.testing.assert_array_equal(tel.smoothed, np.zeros((10, 3)))


def test_total_load_monotone_in_ell():
    """total_load should increase when ell increases."""
    tel = Telemetry(n_nodes=5, params=DEFAULT_PARAMS)
    ell_low  = np.full(5, 0.3)
    ell_high = np.full(5, 0.8)
    tl_low  = tel.compute_total_load(ell_low)
    tl_high = tel.compute_total_load(ell_high)
    assert tl_high > tl_low
