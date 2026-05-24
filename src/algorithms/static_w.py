"""
A0: Static-W – equal token distribution, no rebalancing.

This is the baseline algorithm. Tokens are distributed uniformly across
nodes (s % n_nodes) and are never moved. It serves as the lower bound
for adaptive algorithms to beat.
"""

import numpy as np


class StaticW:
    """Equal token distribution with no adaptive rebalancing.

    Tokens are assigned round-robin at initialisation and never moved.
    M_keys is always 0.

    Attributes
    ----------
    name : str
        Human-readable algorithm identifier.
    """

    name = "Static-W"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step(self, ring, ell: np.ndarray, total_load: float, t: int) -> float:
        """Perform one algorithm step (no-op for Static-W).

        Parameters
        ----------
        ring : HashRing
            Current ring state (not modified).
        ell : np.ndarray
            Per-node load indicator, shape (n_nodes,).
        total_load : float
            Scalar total-load indicator.
        t : int
            Current time step index.

        Returns
        -------
        float
            M_keys = 0.0 (no tokens moved).
        """
        return 0.0
