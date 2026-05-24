"""
A2: Dynamic-R – dynamic rebalancing without budget constraint.

Overloaded nodes donate a fraction of their token arc to underloaded
nodes each step. No budget constraint is applied.
"""

import numpy as np


class DynamicR:
    """Dynamic rebalancing without budget constraint.

    At each step, overloaded nodes (ell_i > ell_star + threshold) donate
    tokens to underloaded nodes (ell_i < ell_star - threshold). The amount
    of arc transferred per step is capped at `rate` fraction of the
    overloaded node's current arc share.

    Parameters
    ----------
    ell_star : float
        Target load level (e.g. 0.65).
    threshold : float
        Dead-band half-width around ell_star (e.g. 0.03).
    rate : float
        Maximum fraction of arc to transfer per step (e.g. 0.05).

    Attributes
    ----------
    name : str
        Human-readable algorithm identifier.
    """

    name = "Dynamic-R"

    def __init__(self, ell_star: float = 0.65, threshold: float = 0.03,
                 rate: float = 0.05, v_min: int = 3):
        self.ell_star = float(ell_star)
        self.threshold = float(threshold)
        self.rate = float(rate)
        self.v_min = int(v_min)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step(self, ring, ell: np.ndarray, total_load: float, t: int) -> float:
        """Perform one rebalancing step.

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
        n_nodes = len(ell)
        ell = np.asarray(ell, dtype=np.float64)

        O = np.where(ell > self.ell_star + self.threshold)[0]
        U = np.where(ell < self.ell_star - self.threshold)[0]

        if len(O) == 0 or len(U) == 0:
            return 0.0

        prev = ring.snapshot()

        # For each overloaded node, move `rate` of its arc to underloaded nodes
        # Compute receive weights for underloaded nodes
        r = np.maximum(self.ell_star - ell[U], 0.0)
        r_total = float(np.sum(r))
        if r_total < 1e-12:
            return 0.0
        r_norm = r / r_total

        for src in O:
            src_share = ring.node_share(src)
            arc_to_give = self.rate * src_share

            # Collect tokens from src, sorted by ascending arc length
            src_tokens = np.where(ring.a == src)[0]
            if len(src_tokens) == 0:
                continue

            arcs = ring.L[src_tokens]
            order = np.argsort(arcs)
            src_tokens_sorted = src_tokens[order]
            arcs_sorted = arcs[order]

            # Greedily pick tokens until we meet arc_to_give budget
            cumulative = 0.0
            selected = []
            for tok, arc in zip(src_tokens_sorted, arcs_sorted):
                if cumulative >= arc_to_give:
                    break
                # Enforce v_min: same constraint as ACH for fair comparison
                if len(src_tokens) - len(selected) <= self.v_min:
                    break
                selected.append(tok)
                cumulative += arc

            if not selected:
                continue

            # Distribute selected tokens to underloaded nodes proportionally
            selected = np.array(selected, dtype=np.int32)
            n_sel = len(selected)

            # Assign each token to a receiver proportional to r_norm
            counts = np.round(r_norm * n_sel).astype(int)
            counts_diff = n_sel - int(np.sum(counts))
            if counts_diff != 0:
                residuals = r_norm * n_sel - counts
                order_res = np.argsort(-residuals if counts_diff > 0 else residuals)
                for k in range(abs(counts_diff)):
                    counts[order_res[k]] += (1 if counts_diff > 0 else -1)

            idx = 0
            for j, dst in enumerate(U):
                c = counts[j]
                if c > 0 and idx < n_sel:
                    ring.reassign(selected[idx:idx + c],
                                  np.full(c, dst, dtype=np.int32))
                    idx += c

        return ring.get_M_keys(prev)
