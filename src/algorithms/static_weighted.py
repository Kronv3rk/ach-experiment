"""
A1: Static-Wt – capacity-proportional token distribution, no adaptive rebalancing.

Tokens are distributed proportional to node capacity at initialisation and
are never moved thereafter. This is the weighted static baseline.
"""

import numpy as np


class StaticWeighted:
    """Capacity-proportional token distribution with no adaptive rebalancing.

    Token counts are assigned proportional to each node's capacity at
    construction time. No tokens are ever moved during the experiment.

    Parameters
    ----------
    capacities : np.ndarray
        Capacity of each node (float array, e.g. [1.0, 0.55, 0.25, ...]).
    V : int
        Total number of virtual tokens in the ring.

    Attributes
    ----------
    name : str
        Human-readable algorithm identifier.
    """

    name = "Static-Wt"

    def __init__(self, capacities: np.ndarray, V: int):
        self.capacities = np.asarray(capacities, dtype=np.float64)
        self.V = V
        self._initialized = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_ring(self, ring):
        """Perform one-time capacity-weighted token redistribution."""
        n_nodes = len(self.capacities)
        total_cap = float(np.sum(self.capacities))

        # Compute target token counts proportional to capacity
        fracs = self.capacities / total_cap
        targets = np.round(fracs * self.V).astype(int)

        # Adjust rounding errors to ensure sum == V
        diff = self.V - int(np.sum(targets))
        if diff != 0:
            # Add/remove tokens to the node with the largest fractional part
            residuals = fracs * self.V - targets
            order = np.argsort(-residuals if diff > 0 else residuals)
            for k in range(abs(diff)):
                targets[order[k]] += (1 if diff > 0 else -1)

        # Build new assignment: assign token s to node based on targets
        new_assignment = np.empty(self.V, dtype=np.int32)
        idx = 0
        for node_id in range(n_nodes):
            end = idx + targets[node_id]
            new_assignment[idx:end] = node_id
            idx = end

        # Reassign tokens by their sorted position index
        token_indices = np.arange(self.V, dtype=np.int32)
        ring.reassign(token_indices, new_assignment)
        self._initialized = True

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step(self, ring, ell: np.ndarray, total_load: float, t: int) -> float:
        """Perform one algorithm step.

        On the first call, performs the one-time capacity-proportional
        redistribution. On subsequent calls, does nothing.

        Parameters
        ----------
        ring : HashRing
            Current ring state; modified only on first call.
        ell : np.ndarray
            Per-node load indicator, shape (n_nodes,).
        total_load : float
            Scalar total-load indicator.
        t : int
            Current time step index.

        Returns
        -------
        float
            M_keys moved (non-zero only on first call, then 0.0).
        """
        if not self._initialized:
            prev = ring.snapshot()
            self._init_ring(ring)
            return ring.get_M_keys(prev)
        return 0.0
