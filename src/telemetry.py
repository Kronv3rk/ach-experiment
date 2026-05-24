"""
Telemetry module: EWMA smoothing and composite load metric.

Computes the per-node load indicator ell_i using an Lp-norm weighted
combination of CPU, memory and latency metrics, with EWMA smoothing.
"""

import numpy as np


class Telemetry:
    """EWMA-smoothed telemetry and composite load metric.

    Parameters
    ----------
    n_nodes : int
        Number of nodes to track.
    params : dict
        Must contain:
          beta   - EWMA smoothing factor (e.g. 0.3)
          weights - list of 3 weights [w_cpu, w_mem, w_lat], must sum to 1
          p      - Lp norm exponent (e.g. 4)
          alpha  - blend factor for total_load = alpha*mean + (1-alpha)*max
    """

    def __init__(self, n_nodes: int, params: dict):
        self.n_nodes = n_nodes
        self.beta = float(params.get("beta", 0.3))
        self.w = np.array(params.get("weights", [0.40, 0.25, 0.35]),
                          dtype=np.float64)
        self.p = float(params.get("p", 4))
        self.alpha = float(params.get("alpha", 0.5))

        # Validate weights
        if len(self.w) != 3:
            raise ValueError(f"Expected 3 weights, got {len(self.w)}")

        # Initialise smoothed state to zeros
        self.smoothed = np.zeros((n_nodes, 3), dtype=np.float64)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, raw: np.ndarray) -> np.ndarray:
        """Apply EWMA smoothing to raw telemetry observations.

        smoothed_{t} = (1 - beta) * smoothed_{t-1} + beta * raw_t

        Parameters
        ----------
        raw : np.ndarray
            Shape (n_nodes, 3): [cpu, mem, latency] each in [0, 1].

        Returns
        -------
        np.ndarray
            Smoothed values, shape (n_nodes, 3).
        """
        raw = np.asarray(raw, dtype=np.float64)
        if raw.shape != (self.n_nodes, 3):
            raise ValueError(
                f"Expected raw shape ({self.n_nodes}, 3), got {raw.shape}"
            )
        self.smoothed = (1.0 - self.beta) * self.smoothed + self.beta * raw
        return self.smoothed.copy()

    def compute_ell(self) -> np.ndarray:
        """Compute per-node composite load indicator.

        ell_i = (sum_j w_j * smoothed_{ij}^p)^{1/p}

        Returns
        -------
        np.ndarray
            Shape (n_nodes,), values in [0, 1].
        """
        # smoothed: (n_nodes, 3), w: (3,)
        weighted = self.w * (self.smoothed ** self.p)   # (n_nodes, 3)
        ell = np.sum(weighted, axis=1) ** (1.0 / self.p)
        return ell

    def compute_total_load(self, ell: np.ndarray) -> float:
        """Compute scalar total-load indicator.

        total_load = alpha * mean(ell) + (1 - alpha) * max(ell)

        Parameters
        ----------
        ell : np.ndarray
            Per-node load indicator, shape (n_nodes,).

        Returns
        -------
        float
            Scalar in [0, 1].
        """
        ell = np.asarray(ell, dtype=np.float64)
        return float(self.alpha * np.mean(ell) + (1.0 - self.alpha) * np.max(ell))

    def resize(self, new_n_nodes: int):
        """Resize smoothed state when nodes are added or removed.

        New nodes get zero initial smoothed state; removed nodes are dropped.

        Parameters
        ----------
        new_n_nodes : int
            Target number of nodes.
        """
        if new_n_nodes == self.n_nodes:
            return
        if new_n_nodes > self.n_nodes:
            extra = np.zeros((new_n_nodes - self.n_nodes, 3), dtype=np.float64)
            self.smoothed = np.vstack([self.smoothed, extra])
        else:
            self.smoothed = self.smoothed[:new_n_nodes]
        self.n_nodes = new_n_nodes

    def reset(self):
        """Reset smoothed state to zeros (useful when nodes change)."""
        self.smoothed = np.zeros((self.n_nodes, 3), dtype=np.float64)
