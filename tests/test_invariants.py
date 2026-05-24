"""
Tests for src/invariants.py (check_all).
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.ring import HashRing
from src.invariants import check_all


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_all_invariants_valid_ring():
    """A freshly created ring must pass all invariants."""
    ring = HashRing(V=300, n_nodes=10, seed=42)
    violations = check_all(ring, n_nodes=10, v_min=3)
    assert violations == [], f"Unexpected violations: {violations}"


def test_all_invariants_small_ring():
    """Small ring (V=30, n=3) must also pass all invariants."""
    ring = HashRing(V=30, n_nodes=3, seed=7)
    violations = check_all(ring, n_nodes=3, v_min=1)
    assert violations == [], f"Unexpected violations on small ring: {violations}"


def test_detect_invalid_assignment():
    """Inv3 must fire when a token has a node id >= n_nodes."""
    ring = HashRing(V=100, n_nodes=5, seed=0)
    ring.a[10] = 99   # out-of-range
    violations = check_all(ring, n_nodes=5, v_min=1)
    assert len(violations) > 0, "Expected Inv3 violation"
    assert any("Inv3" in v for v in violations)


def test_detect_negative_assignment():
    """Inv3 must fire when a token has a negative node id."""
    ring = HashRing(V=100, n_nodes=5, seed=0)
    ring.a[5] = -1    # invalid
    violations = check_all(ring, n_nodes=5, v_min=1)
    assert len(violations) > 0
    assert any("Inv3" in v for v in violations)


def test_detect_arc_sum_error():
    """Inv1 must fire when arc lengths don't sum to 1.

    We simulate this by directly corrupting L.
    """
    ring = HashRing(V=100, n_nodes=5, seed=0)
    ring.L[0] += 0.5   # inflate one arc -> sum != 1
    violations = check_all(ring, n_nodes=5, v_min=1)
    assert len(violations) > 0, "Expected Inv1 violation"
    assert any("Inv1" in v for v in violations)


def test_detect_too_few_tokens():
    """Inv2 must fire when a node has fewer tokens than v_min."""
    ring = HashRing(V=30, n_nodes=3, seed=0)
    # Steal all tokens from node 1
    stolen = np.where(ring.a == 1)[0]
    ring.reassign(stolen, np.zeros(len(stolen), dtype=np.int32))
    violations = check_all(ring, n_nodes=3, v_min=3)
    assert len(violations) > 0
    assert any("Inv2" in v for v in violations)


def test_detect_unsorted_positions():
    """Inv4 must fire when positions are not sorted."""
    ring = HashRing(V=50, n_nodes=5, seed=0)
    # Swap two positions to break sorted order
    ring.pos[0], ring.pos[49] = ring.pos[49], ring.pos[0]
    violations = check_all(ring, n_nodes=5, v_min=1)
    assert len(violations) > 0
    assert any("Inv4" in v for v in violations)


def test_detect_zero_arc():
    """Inv5 must fire when an arc length is zero or negative."""
    ring = HashRing(V=50, n_nodes=5, seed=0)
    ring.L[3] = 0.0
    violations = check_all(ring, n_nodes=5, v_min=1)
    assert len(violations) > 0
    assert any("Inv5" in v for v in violations)


def test_multiple_violations_reported():
    """Multiple invariant violations should all be reported."""
    ring = HashRing(V=30, n_nodes=3, seed=0)
    ring.a[0] = 99    # Inv3
    ring.L[0] += 0.5  # Inv1
    violations = check_all(ring, n_nodes=3, v_min=1)
    assert len(violations) >= 2
