"""
Consistent hash ring with V virtual tokens on [0, 1).

Each token occupies a position on the unit circle. Arc length L[s] is the
fraction of the circle assigned to token s (and thus to node a[s]).
"""

import numpy as np


class HashRing:
    """Consistent hash ring with V virtual tokens.

    Tokens are placed at sorted positions on [0, 1). Each token is assigned to
    a node. The arc length L[s] of token s equals the fraction of the ring
    covered by that token.

    Parameters
    ----------
    V : int
        Number of virtual tokens.
    n_nodes : int
        Number of nodes.
    seed : int
        Random seed for reproducible token placement.
    """

    def __init__(self, V: int, n_nodes: int, seed: int = 42):
        self.V = V
        self.n_nodes = n_nodes
        rng = np.random.RandomState(seed)

        # Generate V sorted positions on [0, 1)
        self.pos = np.sort(rng.uniform(0.0, 1.0, V))

        # Initial assignment: token s -> node s % n_nodes
        self.a = np.array([s % n_nodes for s in range(V)], dtype=np.int32)

        # Pre-compute arc lengths
        self._update_arc_lengths()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_arc_lengths(self):
        """Recompute arc lengths from the current token positions."""
        # L[s] = (pos[s] - pos[s-1]) % 1  (circular distance)
        self.L = np.empty(self.V, dtype=np.float64)
        self.L[0] = (self.pos[0] - self.pos[-1]) % 1.0
        self.L[1:] = np.diff(self.pos)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def lookup(self, h_k: float) -> int:
        """Return the node responsible for key hash h_k.

        Uses O(log V) binary search: finds min{s : pos[s] >= h_k} mod V.

        Parameters
        ----------
        h_k : float
            Hash value in [0, 1).

        Returns
        -------
        int
            Node id.
        """
        idx = np.searchsorted(self.pos, h_k, side='left')
        token = idx % self.V
        return int(self.a[token])

    def lookup_batch(self, h_keys: np.ndarray) -> np.ndarray:
        """Vectorised lookup for an array of key hashes.

        Parameters
        ----------
        h_keys : np.ndarray
            Array of hash values in [0, 1).

        Returns
        -------
        np.ndarray
            Array of node ids, shape (len(h_keys),).
        """
        indices = np.searchsorted(self.pos, h_keys, side='left') % self.V
        return self.a[indices]

    def reassign(self, token_indices: np.ndarray, new_nodes: np.ndarray):
        """Reassign a set of tokens to new nodes.

        Parameters
        ----------
        token_indices : np.ndarray
            Token indices to reassign (integers in [0, V)).
        new_nodes : np.ndarray
            Corresponding new node ids.
        """
        self.a[token_indices] = new_nodes

    def get_M_keys(self, prev_assignment: np.ndarray) -> float:
        """Compute the fraction of key space whose assignment changed.

        M_keys = sum of L[s] for tokens where a[s] != prev_assignment[s].

        Parameters
        ----------
        prev_assignment : np.ndarray
            Previous token-to-node assignment array, shape (V,).

        Returns
        -------
        float
            Fraction of ring affected (in [0, 1]).
        """
        moved = self.a != prev_assignment
        return float(np.sum(self.L[moved]))

    def node_share(self, node_id: int) -> float:
        """Return the total arc length assigned to node_id.

        Parameters
        ----------
        node_id : int

        Returns
        -------
        float
            Arc fraction in [0, 1].
        """
        return float(np.sum(self.L[self.a == node_id]))

    def node_count(self, node_id: int) -> int:
        """Return the number of tokens assigned to node_id.

        Parameters
        ----------
        node_id : int

        Returns
        -------
        int
        """
        return int(np.sum(self.a == node_id))

    def snapshot(self) -> np.ndarray:
        """Return a copy of the current assignment array.

        Returns
        -------
        np.ndarray
            Shape (V,), dtype int32.
        """
        return self.a.copy()

    def verify_invariants(self, n_nodes: int, v_min: int = 3,
                          v_max: int = None) -> list:
        """Check ring invariants.

        Parameters
        ----------
        n_nodes : int
            Expected number of nodes.
        v_min : int
            Minimum tokens per node.
        v_max : int or None
            Maximum tokens per node (if None, not checked).

        Returns
        -------
        list of str
            Violation messages. Empty list means all invariants satisfied.
        """
        violations = []

        # Inv 1: arc lengths sum to 1
        total = np.sum(self.L)
        if abs(total - 1.0) >= 1e-9:
            violations.append(
                f"Inv1 FAIL: sum(L) = {total:.12f} != 1.0 (diff={total - 1.0:.3e})"
            )

        # Inv 2: each node has >= v_min tokens
        for i in range(n_nodes):
            cnt = self.node_count(i)
            if cnt < v_min:
                violations.append(
                    f"Inv2 FAIL: node {i} has {cnt} tokens < v_min={v_min}"
                )

        # Inv 3: each node has <= v_max tokens (if set)
        if v_max is not None:
            for i in range(n_nodes):
                cnt = self.node_count(i)
                if cnt > v_max:
                    violations.append(
                        f"Inv3 FAIL: node {i} has {cnt} tokens > v_max={v_max}"
                    )

        # Inv 4: all token assignments in [0, n_nodes)
        invalid = np.where((self.a < 0) | (self.a >= n_nodes))[0]
        if len(invalid) > 0:
            violations.append(
                f"Inv4 FAIL: tokens {invalid[:5]} have assignments outside [0, {n_nodes})"
            )

        return violations
