"""
Comprehensive tests for src/ring.py (HashRing).
"""

import numpy as np
import pytest
import sys
import os

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.ring import HashRing


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ring():
    """Default 1000-token, 10-node ring."""
    return HashRing(V=1000, n_nodes=10, seed=42)


@pytest.fixture
def small_ring():
    """Small 30-token, 3-node ring for manual checks."""
    return HashRing(V=30, n_nodes=3, seed=0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_positions_sorted(ring):
    """Token positions must be non-decreasingly sorted on [0,1)."""
    assert np.all(np.diff(ring.pos) >= 0), "positions are not sorted"


def test_positions_in_unit_interval(ring):
    """All positions must lie in [0, 1)."""
    assert np.all(ring.pos >= 0.0)
    assert np.all(ring.pos < 1.0)


def test_arc_lengths_sum(ring):
    """Sum of all arc lengths must equal 1.0 within floating-point tolerance."""
    total = float(np.sum(ring.L))
    assert abs(total - 1.0) < 1e-9, f"sum(L) = {total:.12f} != 1.0"


def test_arc_lengths_positive(ring):
    """All arc lengths must be strictly positive."""
    assert np.all(ring.L > 0), "some arc lengths are non-positive"


def test_arc_lengths_sum_small(small_ring):
    """Arc sum invariant holds for the small ring too."""
    assert abs(np.sum(small_ring.L) - 1.0) < 1e-9


def test_lookup_correctness(ring):
    """lookup() must return a valid node id for arbitrary hash values."""
    rng = np.random.RandomState(7)
    hashes = rng.uniform(0.0, 1.0, 500)
    for h in hashes:
        node = ring.lookup(h)
        assert 0 <= node < ring.n_nodes, f"node {node} out of range"


def test_lookup_batch_matches_single(ring):
    """lookup_batch() must agree with repeated lookup() calls."""
    rng = np.random.RandomState(99)
    hashes = rng.uniform(0.0, 1.0, 200)
    batch_result = ring.lookup_batch(hashes)
    single_result = np.array([ring.lookup(h) for h in hashes])
    np.testing.assert_array_equal(batch_result, single_result)


def test_M_keys_zero_no_change(ring):
    """M_keys must be 0 when no tokens are reassigned."""
    prev = ring.snapshot()
    M = ring.get_M_keys(prev)
    assert M == 0.0, f"Expected M_keys=0, got {M}"


def test_M_keys_full_reassign(small_ring):
    """M_keys must equal 1.0 when every token is moved to a different node.

    We reassign all tokens that are on node 0 to node 1 and vice versa,
    and those on node 2 to node 0 - i.e., every token changes owner.
    For a 30-token, 3-node ring with round-robin init this always covers all.
    """
    ring = small_ring
    prev = ring.snapshot()
    # Assign all tokens to a single different rotation
    new_assign = (ring.a + 1) % ring.n_nodes
    ring.reassign(np.arange(ring.V, dtype=np.int32),
                  new_assign.astype(np.int32))
    M = ring.get_M_keys(prev)
    # All tokens changed, so M_keys = sum(L) = 1.0
    assert abs(M - 1.0) < 1e-9, f"Expected M_keys=1.0, got {M}"


def test_M_keys_partial_reassign(ring):
    """M_keys should be < 1 and > 0 when only some tokens move."""
    prev = ring.snapshot()
    # Move first 100 tokens to a different node
    moved_tokens = np.arange(100, dtype=np.int32)
    new_nodes = np.full(100, (ring.a[0] + 1) % ring.n_nodes, dtype=np.int32)
    ring.reassign(moved_tokens, new_nodes)
    M = ring.get_M_keys(prev)
    assert 0.0 < M < 1.0, f"Expected partial M_keys, got {M}"


def test_invariants_pass(ring):
    """verify_invariants must return an empty list for a fresh ring."""
    violations = ring.verify_invariants(n_nodes=10, v_min=3)
    assert violations == [], f"Unexpected violations: {violations}"


def test_invariant_too_few_tokens():
    """Inv2 violation detected when a node has fewer than v_min tokens."""
    ring = HashRing(V=30, n_nodes=3, seed=0)
    # Force all tokens onto node 0, leaving node 1 and 2 empty
    ring.reassign(np.arange(ring.V, dtype=np.int32),
                  np.zeros(ring.V, dtype=np.int32))
    violations = ring.verify_invariants(n_nodes=3, v_min=3)
    assert len(violations) > 0, "Expected Inv2 violation, none found"
    assert any("Inv2" in v for v in violations)


def test_invariant_invalid_assignment():
    """Inv4 violation detected when a token has an out-of-range node id."""
    ring = HashRing(V=30, n_nodes=3, seed=0)
    # Corrupt one token's assignment
    ring.a[0] = 99
    violations = ring.verify_invariants(n_nodes=3, v_min=1)
    assert len(violations) > 0, "Expected Inv4 violation, none found"
    assert any("Inv4" in v for v in violations)


def test_node_share_sums_to_one(ring):
    """Sum of all node shares must equal 1.0."""
    total = sum(ring.node_share(i) for i in range(ring.n_nodes))
    assert abs(total - 1.0) < 1e-9, f"sum of shares = {total:.12f} != 1.0"


def test_node_count_sums_to_V(ring):
    """Total token count across all nodes must equal V."""
    total = sum(ring.node_count(i) for i in range(ring.n_nodes))
    assert total == ring.V, f"total tokens = {total} != V={ring.V}"


def test_snapshot_is_copy(ring):
    """snapshot() must return an independent copy."""
    snap = ring.snapshot()
    snap[0] = 999
    assert ring.a[0] != 999, "snapshot() returned a view, not a copy"


def test_reassign_updates_assignment(ring):
    """reassign() must update the assignment array correctly."""
    ring.reassign(np.array([0, 1, 2], dtype=np.int32),
                  np.array([5, 5, 5], dtype=np.int32))
    assert ring.a[0] == 5
    assert ring.a[1] == 5
    assert ring.a[2] == 5
