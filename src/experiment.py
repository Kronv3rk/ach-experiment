"""
Experiment runner for the ACH dissertation experiment.

Runs all five algorithms (A0–A4) on an identical load sequence and
returns per-algorithm traces and aggregate metrics.
"""

import copy
import numpy as np
from typing import Dict, Any

from .ring import HashRing
from .cluster import Cluster
from .load_generator import LoadGenerator
from .telemetry import Telemetry
from .noise_model import NoiseModel
from .metrics import compute_D, aggregate_metrics
from .invariants import check_all
from .algorithms.static_w import StaticW
from .algorithms.static_weighted import StaticWeighted
from .algorithms.dynamic_r import DynamicR
from .algorithms.bounded_loads import BoundedLoads
from .algorithms.ach import ACH


# Capacity class lookup for node churn events
CAP_CLASSES = [1.0, 0.55, 0.25]


def _build_algorithms(cfg: dict, ring_snapshot: HashRing) -> list:
    """Construct one instance of each algorithm.

    Parameters
    ----------
    cfg : dict
        Full experiment configuration.
    ring_snapshot : HashRing
        The initial ring (used to extract V and capacities for StaticWeighted).

    Returns
    -------
    list of algorithm objects
    """
    caps = np.array(cfg.get("capacities", [1.0]*10), dtype=np.float64)
    V    = int(cfg.get("V", 1000))
    ach_params = cfg.get("ach", {})

    algos = [
        StaticW(),
        StaticWeighted(caps, V),
        DynamicR(
            ell_star=float(ach_params.get("ell_star", 0.65)),
            threshold=0.03,
            rate=0.05,
            v_min=int(ach_params.get("v_min", 3)),
        ),
        BoundedLoads(
            ell_star=float(ach_params.get("ell_star", 0.65)),
            upper=0.12,
            lower=0.05,
            max_tokens=2,
            v_min=int(ach_params.get("v_min", 3)),
        ),
        ACH(ach_params),
    ]
    return algos


def _make_load_sequence(cfg: dict, rng: np.random.RandomState,
                        gen: LoadGenerator) -> list:
    """Pre-generate the full load sequence so all algorithms see the same data.

    Parameters
    ----------
    cfg : dict
        Experiment config, must contain 'load' and optionally 'T'.
    rng : np.random.RandomState
    gen : LoadGenerator

    Returns
    -------
    list of (key_hashes, write_flags) tuples, length T
    """
    T = int(cfg.get("T", 500))
    load_cfg = cfg.get("load", {})
    load_type = load_cfg.get("type", "constant")

    sequence = []
    for t in range(T):
        lam = _get_lambda(load_cfg, t, T)
        kh, wf = gen.generate(lam, rng)
        sequence.append((kh, wf))
    return sequence


def _get_lambda(load_cfg: dict, t: int, T: int) -> float:
    """Compute arrival rate lambda at time step t.

    Supports load types:
      constant       : fixed lambda
      step_periodic  : step-up at step_t, periodic modulation after periodic_t
    """
    load_type = load_cfg.get("type", "constant")

    if load_type == "constant":
        return float(load_cfg.get("lambda", 1800.0))

    elif load_type == "step_periodic":
        lam_low  = float(load_cfg.get("lambda_low",  1800.0))
        lam_high = float(load_cfg.get("lambda_high", 3200.0))
        step_t   = int(load_cfg.get("step_t", 200))
        per_t    = int(load_cfg.get("periodic_t", 350))
        period   = int(load_cfg.get("period", 50))
        amp      = float(load_cfg.get("amplitude", 0.25))

        lam = lam_low if t < step_t else lam_high
        if t >= per_t:
            lam = lam * (1.0 + amp * np.sin(2.0 * np.pi * (t - per_t) / period))
        return float(lam)

    else:
        raise ValueError(f"Unknown load type: {load_type!r}")


def run_experiment(cfg: dict, seed: int = 42) -> dict:
    """Run all 5 algorithms on the same load sequence.

    Parameters
    ----------
    cfg : dict
        Full experiment configuration (loaded from YAML).
    seed : int
        Random seed for this run.

    Returns
    -------
    dict with keys per algorithm name:
        {
          'D_trace'   : list of float (length T)
          'M_trace'   : list of float (length T)
          'ell_trace' : list of np.ndarray (T x n_nodes)
          'violations': list of str (invariant violations)
          'agg'       : dict of aggregate metrics
        }
    """
    rng = np.random.RandomState(seed)

    # --- Setup ---
    V       = int(cfg.get("V", 1000))
    n_nodes = int(cfg.get("n0", 10))
    T       = int(cfg.get("T", 500))
    NK      = int(cfg.get("NK", 50000))
    warmup  = int(cfg.get("warmup", 40))

    caps     = list(cfg.get("capacities", [1.0]*n_nodes))
    mu_base  = float(cfg.get("mu_base", 350.0))
    zipf_s   = float(cfg.get("zipf_s", 0.0))

    tel_params = cfg.get("telemetry", {})
    ach_params = cfg.get("ach", {})
    noise_cfg  = cfg.get("noise", {})
    degrade_cfg = cfg.get("degradation", {})
    churn_cfg  = cfg.get("churn", [])

    # --- Build shared infrastructure ---
    cluster = Cluster(capacities=caps, mu_base=mu_base)
    gen = LoadGenerator(NK=NK, zipf_s=zipf_s, seed=seed)

    _sigma_noise = float(noise_cfg.get("sigma", 0.0))
    _p_miss      = float(noise_cfg.get("p_miss", 0.0))
    _lag         = int(noise_cfg.get("lag", 0))

    # One NoiseModel per algorithm so lag buffers are independent,
    # but Gaussian delta and missing mask are generated ONCE per step
    # (see K2 fix) and shared across all algorithms via apply() arguments.
    noises = [
        NoiseModel(sigma_noise=_sigma_noise, lag=_lag,
                   p_miss=_p_miss, seed=seed + 1 + i)
        for i in range(5)
    ]

    # Pre-generate load sequence (all algorithms use the same)
    load_seq = _make_load_sequence(cfg, rng, gen)

    # --- Build one ring per algorithm (all identical at t=0) ---
    rings = [HashRing(V=V, n_nodes=n_nodes, seed=seed) for _ in range(5)]
    telemetries = [Telemetry(n_nodes, tel_params) for _ in range(5)]
    algos = _build_algorithms(cfg, rings[0])

    # Track nodes removed via churn (capacity zeroed, not compacted)
    dead_nodes: set = set()

    # Results containers
    results: Dict[str, Any] = {}
    for algo in algos:
        results[algo.name] = {
            "D_trace":    [],
            "M_trace":    [],
            "ell_trace":  [],
            "violations": [],
            "agg":        {},
        }

    # --- Main simulation loop ---
    for t in range(T):
        key_hashes, write_flags = load_seq[t]

        # Handle churn events
        for event_group in churn_cfg:
            if event_group.get("t") == t:
                for ev in event_group.get("events", []):
                    if ev["type"] == "remove":
                        node_id = ev["node"]
                        cluster.remove_node(node_id)
                        dead_nodes.add(node_id)
                        for ring in rings:
                            # Reassign orphaned tokens to node 0
                            orphaned = np.where(ring.a == node_id)[0]
                            if len(orphaned) > 0:
                                ring.reassign(orphaned,
                                              np.zeros(len(orphaned), dtype=np.int32))
                    elif ev["type"] == "add":
                        cap_class_idx = ev.get("cap_class", 0)
                        cap = CAP_CLASSES[cap_class_idx]
                        new_id = cluster.add_node(cap)
                        v_min_cfg = int(ach_params.get("v_min", 3))
                        for ring in rings:
                            ring.n_nodes = cluster.n_nodes
                            ring.a = np.where(ring.a >= ring.n_nodes,
                                              ring.a % ring.n_nodes, ring.a)
                            # Bootstrap: steal v_min tokens from the node
                            # with the most tokens so the new node is valid
                            donor = int(np.argmax(
                                [ring.node_count(i)
                                 for i in range(ring.n_nodes - 1)]
                            ))
                            donor_toks = np.where(ring.a == donor)[0]
                            if len(donor_toks) > v_min_cfg:
                                give = donor_toks[:v_min_cfg]
                                ring.reassign(
                                    give,
                                    np.full(len(give), new_id, dtype=np.int32),
                                )
                        for tel in telemetries:
                            tel.resize(cluster.n_nodes)

        # Handle degradation events
        if degrade_cfg:
            t_start = int(degrade_cfg.get("t_start", T))
            if t == t_start:
                for node_str, factor in degrade_cfg.get("nodes", {}).items():
                    cluster.degrade_node(int(node_str), float(factor))

        # --- K2 fix: generate noise perturbation ONCE per step ---
        # All algorithms receive the same Gaussian delta and missing-data mask
        # applied to their own ring's raw telemetry. This makes the noise
        # realization identical across algorithms, keeping the paired
        # Wilcoxon test valid.
        _cur_nodes = cluster.n_nodes
        _noise_shape = (_cur_nodes, 3)
        gauss_delta = (rng.normal(0.0, _sigma_noise, _noise_shape)
                       if _sigma_noise > 0.0 else None)
        miss_mask   = (rng.random_sample(_noise_shape) < _p_miss
                       if _p_miss > 0.0 else None)

        for algo_idx, (algo, ring, tel, noise_mdl) in enumerate(
                zip(algos, rings, telemetries, noises)):

            raw_i, _ = cluster.simulate_load(ring, key_hashes, write_flags)
            noisy_i = noise_mdl.apply(raw_i,
                                      gauss_delta=gauss_delta,
                                      miss_mask=miss_mask)

            tel.update(noisy_i)
            ell = tel.compute_ell()
            total_load = tel.compute_total_load(ell)

            prev = ring.snapshot()
            M_keys = algo.step(ring, ell, total_load, t)

            D = compute_D(ell)
            results[algo.name]["D_trace"].append(D)
            results[algo.name]["M_trace"].append(M_keys)
            results[algo.name]["ell_trace"].append(ell.copy())

            # Check invariants every step (skip dead/zeroed-out nodes)
            viols = check_all(ring, ring.n_nodes, v_min=int(ach_params.get("v_min", 3)),
                              dead_nodes=dead_nodes)
            for v in viols:
                results[algo.name]["violations"].append(f"t={t}: {v}")

    # --- Compute aggregates ---
    for algo in algos:
        name = algo.name
        D_arr = np.array(results[name]["D_trace"])
        M_arr = np.array(results[name]["M_trace"])
        results[name]["agg"] = aggregate_metrics(D_arr, M_arr, warmup=warmup)
        # Convert ell_trace to numpy array when shapes are homogeneous
        # (may be inhomogeneous during node churn)
        try:
            results[name]["ell_trace"] = np.array(results[name]["ell_trace"])
        except ValueError:
            pass  # keep as list of arrays when n_nodes changes mid-run

    return results
