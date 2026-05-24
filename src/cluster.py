"""
Cluster simulation module.

Models a heterogeneous set of nodes with varying service capacities.
Given a ring assignment and a batch of requests, computes per-node
utilisation and returns telemetry (CPU, memory, latency) in [0, 1].
"""

import numpy as np

# Default capacity classes for 10 nodes (Table 3.1)
DEFAULT_CAPACITIES = [
    1.00, 1.00, 1.00,                    # strong
    0.55, 0.55, 0.55, 0.55,              # medium
    0.25, 0.25, 0.25,                    # weak
]


class Cluster:
    """Heterogeneous cluster simulation.

    Parameters
    ----------
    capacities : list or np.ndarray
        Relative capacity of each node (float, e.g. 1.0 / 0.55 / 0.25).
    mu_base : float
        Service rate (requests/step) for a node with capacity 1.0.
    """

    def __init__(self, capacities=None, mu_base: float = 350.0):
        if capacities is None:
            capacities = DEFAULT_CAPACITIES
        self.capacities = np.asarray(capacities, dtype=np.float64)
        self.n_nodes = len(self.capacities)
        self.mu_base = float(mu_base)
        # Per-node service rate
        self.mu = self.capacities * self.mu_base

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def simulate_load(self, ring, requests: np.ndarray,
                      write_flags: np.ndarray) -> tuple:
        """Simulate one time step and return raw telemetry.

        Each request is routed to a node via the ring. Per-node CPU
        utilisation (rho) is computed as an M/M/1 queue:
            rho_i = lambda_i / mu_i
        where lambda_i is the request rate hitting node i.

        Telemetry mapping:
            cpu     = rho (clipped to 1.0)
            memory  = 0.6 * rho + 0.1 (memory pressure proxy)
            latency = M/M/1 normalised: rho/(1-rho), capped and normalised

        Parameters
        ----------
        ring : HashRing
            Current ring state used for routing.
        requests : np.ndarray
            Hash values of incoming keys, shape (n_requests,).
        write_flags : np.ndarray
            Boolean array, True = write, shape (n_requests,).

        Returns
        -------
        tuple (raw_telemetry, rho)
            raw_telemetry : np.ndarray, shape (n_nodes, 3), values in [0, 1]
            rho           : np.ndarray, shape (n_nodes,), utilisation
        """
        n_nodes = ring.n_nodes if hasattr(ring, 'n_nodes') else self.n_nodes

        # Count requests per node
        if len(requests) == 0:
            counts = np.zeros(n_nodes, dtype=np.float64)
        else:
            node_ids = ring.lookup_batch(requests)
            counts = np.bincount(node_ids, minlength=n_nodes).astype(np.float64)

        # Writes cost ~1.5x CPU (read-write asymmetry)
        if len(requests) > 0 and np.any(write_flags):
            write_counts = np.bincount(
                node_ids[write_flags], minlength=n_nodes
            ).astype(np.float64)
            effective_counts = counts + 0.5 * write_counts
        else:
            effective_counts = counts

        # Utilisation (cap at 1.0 to keep M/M/1 stable)
        rho = effective_counts / np.maximum(self.mu[:n_nodes], 1e-9)
        rho = np.clip(rho, 0.0, 1.0)

        # CPU metric
        cpu = rho.copy()

        # Memory proxy: linear with rho, slight offset
        mem = np.clip(0.6 * rho + 0.1 * (rho > 0.5).astype(float), 0.0, 1.0)

        # Latency via M/M/1 waiting time: rho/(1-rho), capped at rho=0.85
        rho_for_lat = np.minimum(rho, 0.849)
        lat_raw = rho_for_lat / np.maximum(1.0 - rho_for_lat, 1e-9)
        # Normalise: at rho=0.85, lat_raw=5.67; map to 1.0
        lat_max = 0.849 / (1.0 - 0.849)
        latency = np.clip(lat_raw / lat_max, 0.0, 1.0)
        # For saturated nodes set latency=1
        latency = np.where(rho >= 0.85, 1.0, latency)

        raw_telemetry = np.stack([cpu, mem, latency], axis=1)  # (n_nodes, 3)
        return raw_telemetry, rho

    def degrade_node(self, node_id: int, factor: float):
        """Reduce node capacity by a multiplicative factor.

        Parameters
        ----------
        node_id : int
        factor : float
            New capacity = old capacity * factor.
        """
        self.capacities[node_id] *= factor
        self.mu[node_id] = self.capacities[node_id] * self.mu_base

    def add_node(self, cap_class: float):
        """Add a new node with given capacity.

        Parameters
        ----------
        cap_class : float
            Capacity of the new node (e.g. 1.0, 0.55, 0.25).

        Returns
        -------
        int
            New node id.
        """
        self.capacities = np.append(self.capacities, cap_class)
        self.mu = np.append(self.mu, cap_class * self.mu_base)
        self.n_nodes += 1
        return self.n_nodes - 1

    def remove_node(self, node_id: int):
        """Mark a node as failed (capacity -> 0).

        Parameters
        ----------
        node_id : int
        """
        self.capacities[node_id] = 0.0
        self.mu[node_id] = 0.0
