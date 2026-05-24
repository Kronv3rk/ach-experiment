"""
A3: Bounded-Loads – bounded loads heuristic for consistent hashing.

For each overloaded node, moves the top `max_tokens` tokens (by arc length)
to the least loaded eligible receiver. Provides a hard upper bound on
per-node load relative to the mean.
"""

import numpy as np


class BoundedLoads:
    """Bounded loads heuristic.

    For each overloaded node, selects up to `max_tokens` tokens and
    reassigns them to underloaded nodes. Overloaded = ell_i > ell_star + upper.
    Underloaded = ell_i < ell_star - lower.

    Parameters
    ----------
    ell_star : float
        Target load level.
    upper : float
        Overload threshold above ell_star.
    lower : float
        Underload threshold below ell_star.
    max_tokens : int
        Maximum number of tokens to move from each overloaded node per step.

    Attributes
    ----------
    name : str
        Human-readable algorithm identifier.
    """

    name = "Bounded-Loads"

    def __init__(self, ell_star: float = 0.65, upper: float = 0.12,
                 lower: float = 0.05, max_tokens: int = 2, v_min: int = 3):
        self.ell_star = float(ell_star)
        self.upper = float(upper)
        self.lower = float(lower)
        self.max_tokens = int(max_tokens)
        self.v_min = int(v_min)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step(self, ring, ell: np.ndarray, total_load: float, t: int) -> float:
        """Perform one bounded-loads rebalancing step.

        Parameters
        ----------
        ring : HashRing
            Current ring state; may be modified.
        ell : np.ndarray
            Per-node load indicator, shape (n_nodes,).
        total_load : float
            Scalar total-load indicator (not used by this algorithm).
        t : int
            Current time step index.

        Returns
        -------
        float
            M_keys: fraction of key space whose assignment changed.
        """
        ell = np.asarray(ell, dtype=np.float64)
        n_nodes = len(ell)

        O = np.where(ell > self.ell_star + self.upper)[0]
        U = np.where(ell < self.ell_star - self.lower)[0]

        if len(O) == 0 or len(U) == 0:
            return 0.0

        prev = ring.snapshot()

        for src in O:
            src_tokens = np.where(ring.a == src)[0]
            if len(src_tokens) <= self.v_min:
                continue

            # Select top `max_tokens` tokens by descending arc length
            arcs = ring.L[src_tokens]
            order = np.argsort(-arcs)
            n_move = min(self.max_tokens, len(src_tokens) - self.v_min)
            selected = src_tokens[order[:n_move]]

            # Find least loaded receiver
            if len(U) == 0:
                break
            dst = U[np.argmin(ell[U])]

            ring.reassign(selected, np.full(len(selected), dst, dtype=np.int32))

        return ring.get_M_keys(prev)
