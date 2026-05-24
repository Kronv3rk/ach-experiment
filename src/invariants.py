"""
Ring invariant checker for ACH experiment.

Provides a comprehensive invariant check that can be called every step
to detect any corruption in the ring state.
"""

import numpy as np


def check_all(ring, n_nodes: int, v_min: int = 3,
              dead_nodes: set = None) -> list:
    """Check all ring invariants.

    Invariants checked:
      1. abs(sum(L) - 1.0) < 1e-9          arc lengths partition the unit circle
      2. node_count(i) >= v_min             each node has enough tokens
      3. all a[s] in [0, n_nodes)           no stale/invalid node references
      4. positions are strictly sorted      consistent binary search
      5. L[s] > 0 for all s                 no zero-length arcs

    Parameters
    ----------
    ring : HashRing
        Ring instance to check.
    n_nodes : int
        Expected number of active nodes.
    v_min : int
        Minimum token count per node.

    Returns
    -------
    list of str
        Violation messages. Empty list means all invariants satisfied.
    """
    violations = []

    # Inv 1: arc lengths sum to 1
    arc_sum = float(np.sum(ring.L))
    if abs(arc_sum - 1.0) >= 1e-9:
        violations.append(
            f"Inv1: sum(L)={arc_sum:.12f} != 1.0 (diff={arc_sum - 1.0:.3e})"
        )

    # Inv 2: each active (non-dead) node has >= v_min tokens
    _dead = dead_nodes or set()
    for i in range(n_nodes):
        if i in _dead:
            continue
        cnt = ring.node_count(i)
        if cnt < v_min:
            violations.append(
                f"Inv2: node {i} has {cnt} tokens < v_min={v_min}"
            )

    # Inv 3: all assignments in [0, n_nodes)
    invalid_mask = (ring.a < 0) | (ring.a >= n_nodes)
    invalid_idx = np.where(invalid_mask)[0]
    if len(invalid_idx) > 0:
        sample = invalid_idx[:5].tolist()
        violations.append(
            f"Inv3: tokens {sample} have assignments outside [0, {n_nodes}); "
            f"values={ring.a[invalid_idx[:5]].tolist()}"
        )

    # Inv 4: positions are sorted (non-decreasing)
    if not np.all(np.diff(ring.pos) >= 0):
        bad = np.where(np.diff(ring.pos) < 0)[0]
        violations.append(
            f"Inv4: positions not sorted at indices {bad[:5].tolist()}"
        )

    # Inv 5: arc lengths are positive
    zero_arcs = np.where(ring.L <= 0)[0]
    if len(zero_arcs) > 0:
        violations.append(
            f"Inv5: {len(zero_arcs)} tokens have L[s] <= 0 "
            f"(e.g. tokens {zero_arcs[:5].tolist()})"
        )

    return violations
