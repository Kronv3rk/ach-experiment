"""
Metrics computation for ACH experiment.

Provides scalar and trace-level metrics used for comparing algorithm
performance across experiment series.
"""

import numpy as np


def compute_D(ell: np.ndarray) -> float:
    """Compute imbalance dispersion D(t).

    D(t) = max(ell) - mean(ell)

    Parameters
    ----------
    ell : np.ndarray
        Per-node load indicator, shape (n_nodes,).

    Returns
    -------
    float
        Non-negative scalar.
    """
    ell = np.asarray(ell, dtype=np.float64)
    return float(np.max(ell) - np.mean(ell))


def compute_imbalance(ring, n_nodes: int) -> float:
    """Compute ring imbalance based on token arc lengths.

    Imbalance = std(node_share) / mean(node_share), where node_share_i
    is the total arc length assigned to node i.

    Parameters
    ----------
    ring : HashRing
        Current ring state.
    n_nodes : int
        Number of nodes.

    Returns
    -------
    float
        Coefficient of variation of arc shares (>0 = imbalanced).
    """
    shares = np.array([ring.node_share(i) for i in range(n_nodes)],
                      dtype=np.float64)
    mean_share = float(np.mean(shares))
    if mean_share < 1e-12:
        return 0.0
    return float(np.std(shares) / mean_share)


def aggregate_metrics(D_trace: np.ndarray, M_trace: np.ndarray,
                      warmup: int = 40) -> dict:
    """Compute aggregate summary metrics after warmup period.

    Parameters
    ----------
    D_trace : np.ndarray
        Per-step D(t) values, shape (T,).
    M_trace : np.ndarray
        Per-step M_keys(t) values, shape (T,).
    warmup : int
        Number of initial steps to discard.

    Returns
    -------
    dict with keys:
        D_mean   : float  mean imbalance after warmup
        D_max    : float  worst-case imbalance after warmup
        M_cum    : float  cumulative key churn after warmup
        pi_chg   : float  fraction of steps with M_keys > 0
        ell_max  : float  mean of max(ell) per step (peak-load proxy)
    """
    D_trace = np.asarray(D_trace, dtype=np.float64)
    M_trace = np.asarray(M_trace, dtype=np.float64)

    D_post = D_trace[warmup:]
    M_post = M_trace[warmup:]

    D_mean = float(np.mean(D_post)) if len(D_post) > 0 else 0.0
    D_max  = float(np.max(D_post))  if len(D_post) > 0 else 0.0
    M_cum  = float(np.sum(M_post))
    pi_chg = float(np.mean(M_post > 0.0)) if len(M_post) > 0 else 0.0
    # ell_max was previously duplicating D_mean; corrected to D_max (true worst-case)
    ell_max = D_max

    return {
        "D_mean":  D_mean,
        "D_max":   D_max,
        "M_cum":   M_cum,
        "pi_chg":  pi_chg,
        "ell_max": ell_max,
    }
