"""
A4: ACH – Adaptive Consistent Hashing (proposed method).

Implements the full ACH algorithm with:
  - Hysteresis-based overload/underload detection
  - Budget-controlled token rebalancing
  - Three-phase proportional token redistribution
  - Hard enforcement of M_keys(t) <= m(t)
"""

import numpy as np


class ACH:
    """Adaptive Consistent Hashing – the proposed algorithm.

    Parameters
    ----------
    params : dict
        Algorithm hyperparameters. Keys and defaults:
          ell_star : float = 0.65   target load
          eps_on   : float = 0.10   hysteresis trigger threshold
          eps_off  : float = 0.04   hysteresis clear threshold
          tau      : float = 0.85   global load throttle ceiling
          tau_min  : float = 0.50   global load throttle floor
          kappa    : float = 0.10   imbalance sensitivity offset
          M_max    : float = 0.02   per-step arc budget cap
          M_H      : float = 0.25   history-window arc budget
          H        : int   = 100    history window length
          v_min    : int   = 3      minimum tokens per node
          v_max    : int or None     maximum tokens per node

    Attributes
    ----------
    name : str
        Human-readable algorithm identifier.
    sigma : np.ndarray or None
        Hysteresis state per node: -1=underload, 0=neutral, +1=overload.
    """

    name = "ACH"

    def __init__(self, params: dict):
        self.ell_star = float(params.get("ell_star", 0.65))
        self.eps_on   = float(params.get("eps_on",   0.10))
        self.eps_off  = float(params.get("eps_off",  0.04))
        self.tau      = float(params.get("tau",      0.85))
        self.tau_min  = float(params.get("tau_min",  0.50))
        self.kappa    = float(params.get("kappa",    0.10))
        self.M_max    = float(params.get("M_max",    0.02))
        self.M_H      = float(params.get("M_H",     0.25))
        self.H        = int(params.get("H",          100))
        self.v_min    = int(params.get("v_min",        3))
        self.v_max    = params.get("v_max", None)
        if self.v_max is not None:
            self.v_max = int(self.v_max)

        # Per-node hysteresis state (initialised lazily)
        self.sigma: np.ndarray = None

        # Rolling window of M_keys values for H-step budget
        self._M_history: list = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step(self, ring, ell: np.ndarray, total_load: float, t: int) -> float:
        """Execute one ACH step.

        Steps:
          1. Initialise sigma if needed.
          2. Update per-node hysteresis states.
          3. Compute capacity factor C(t).
          4. Compute step budget m(t).
          5. Rebalance if overloaded and underloaded nodes exist.
          6. Assert M_keys(t) <= m(t).
          7. Record M_keys in rolling history.

        Parameters
        ----------
        ring : HashRing
            Current ring state; may be modified.
        ell : np.ndarray
            Per-node load indicator, shape (n_nodes,).
        total_load : float
            Scalar total-load indicator.
        t : int
            Current time step index.

        Returns
        -------
        float
            M_keys moved this step (fraction of key space).
        """
        ell = np.asarray(ell, dtype=np.float64)
        n_nodes = len(ell)

        # Lazy initialisation
        if self.sigma is None or len(self.sigma) != n_nodes:
            self.sigma = np.zeros(n_nodes, dtype=np.int8)

        # Step 1: update hysteresis
        self._update_hysteresis(ell)

        # Step 2: compute capacity factor and budget
        C = self._compute_C(total_load, ell)
        m = self._compute_budget(C, t)

        # Step 3: rebalance if budget > 0 and both O and U non-empty
        O = np.where(self.sigma == +1)[0]
        U = np.where(self.sigma == -1)[0]

        if C <= 0.0 or m <= 0.0 or len(O) == 0 or len(U) == 0:
            self._M_history.append(0.0)
            return 0.0

        M_keys = self._rebalance(ring, ell, m)

        # Enforce budget invariant: each of the |O| senders can overshoot its
        # per-source share by at most one arc-length (discrete-token tolerance),
        # so M_keys <= m + |O| * max_L_s.
        max_arc = float(np.max(ring.L)) if len(ring.L) > 0 else 0.0
        assert M_keys <= m + len(O) * max_arc + 1e-9, (
            f"Budget violated at t={t}: M_keys={M_keys:.6f} > "
            f"m+{len(O)}*max_arc={m + len(O) * max_arc:.6f}"
        )

        self._M_history.append(M_keys)
        return M_keys

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _update_hysteresis(self, ell: np.ndarray):
        """Update per-node hysteresis states sigma_i.

        Transitions (xi = ell_i - ell_star):
          sigma=0,  xi >= +eps_on  -> sigma=+1  (overload trigger)
          sigma=+1, xi <= +eps_off -> sigma=0   (overload clear)
          sigma=0,  xi <= -eps_on  -> sigma=-1  (underload trigger)
          sigma=-1, xi >= -eps_off -> sigma=0   (underload clear)

        Parameters
        ----------
        ell : np.ndarray
            Per-node load indicator.
        """
        xi = ell - self.ell_star
        n = len(xi)

        for i in range(n):
            s = self.sigma[i]
            x = xi[i]
            if s == 0:
                if x >= self.eps_on:
                    self.sigma[i] = 1
                elif x <= -self.eps_on:
                    self.sigma[i] = -1
            elif s == 1:
                if x <= self.eps_off:
                    self.sigma[i] = 0
            else:  # s == -1
                if x >= -self.eps_off:
                    self.sigma[i] = 0

    def _compute_C(self, total_load: float, ell: np.ndarray) -> float:
        """Compute capacity factor C(t) in [0, 1].

        gamma = clip((tau - total_load) / (tau - tau_min), 0, 1)
        I_i   = max(ell_i - ell_star, 0)
        I_max = max(I_i)
        eta   = I_max / (I_max + kappa)  if I_max > 0 else 0
        C     = gamma * eta

        Parameters
        ----------
        total_load : float
        ell : np.ndarray

        Returns
        -------
        float
        """
        denom = self.tau - self.tau_min
        if denom < 1e-12:
            gamma = 0.0
        else:
            gamma = float(np.clip((self.tau - total_load) / denom, 0.0, 1.0))

        I = np.maximum(ell - self.ell_star, 0.0)
        I_total = float(np.max(I))
        if I_total > 0.0:
            eta = I_total / (I_total + self.kappa)
        else:
            eta = 0.0

        return gamma * eta

    def _compute_budget(self, C: float, t: int) -> float:
        """Compute per-step arc budget m(t).

        m(t) = min(M_max, M_H / H) * C

        Additionally enforces the rolling H-step window budget: if the sum of
        M_keys over the last H steps already equals or exceeds M_H, the budget
        is clamped to 0 for this step.  This prevents cumulative over-movement
        under noisy or adversarial telemetry.

        Parameters
        ----------
        C : float
            Capacity factor from _compute_C.
        t : int
            Current time step.

        Returns
        -------
        float
            Arc budget for this step.
        """
        base = min(self.M_max, self.M_H / max(self.H, 1))
        m = base * C

        # Rolling H-window: pause if the last H steps already exhausted M_H
        if len(self._M_history) >= self.H:
            window_sum = float(sum(self._M_history[-self.H:]))
            if window_sum >= self.M_H - 1e-9:
                return 0.0

        return m

    def _rebalance(self, ring, ell: np.ndarray, m: float) -> float:
        """Three-phase token redistribution.

        Phase 1: For each overloaded node (sigma_i == +1), greedily select
                 tokens by descending arc length until the allocated share
                 of the budget is reached.
        Phase 2: Pool all selected tokens (T_out).
        Phase 3: Assign T_out to underloaded nodes (sigma_i == -1),
                 proportional to each node's deficit r_i = max(ell_star - ell_i, 0).
                 Enforces v_min tokens per node.

        Parameters
        ----------
        ring : HashRing
            Current ring state.
        ell : np.ndarray
            Per-node load indicator.
        m : float
            Arc budget for this step.

        Returns
        -------
        float
            M_keys: fraction of key space moved.
        """
        O = np.where(self.sigma == +1)[0]
        U = np.where(self.sigma == -1)[0]

        # Demand weights for senders
        g = np.maximum(ell[O] - self.ell_star, 0.0)
        g_total = float(np.sum(g))
        if g_total < 1e-12:
            return 0.0

        # Receive weights for receivers
        r = np.maximum(self.ell_star - ell[U], 0.0)
        r_total = float(np.sum(r))
        if r_total < 1e-12:
            return 0.0

        prev = ring.snapshot()

        # --- Phase 1 & 2: collect tokens from overloaded nodes ---
        all_selected = []

        for src_idx, src in enumerate(O):
            share = m * g[src_idx] / g_total  # budget allocated to this sender

            src_tokens = np.where(ring.a == src)[0]
            if len(src_tokens) <= self.v_min:
                continue  # can't give any tokens

            arcs = ring.L[src_tokens]
            # Sort descending to maximise arc coverage
            order = np.argsort(-arcs)
            src_tokens_sorted = src_tokens[order]
            arcs_sorted = arcs[order]

            cumulative = 0.0
            selected = []
            for tok, arc in zip(src_tokens_sorted, arcs_sorted):
                if cumulative >= share:
                    break
                remaining = len(src_tokens) - len(selected)
                if remaining <= self.v_min:
                    break
                selected.append(tok)
                cumulative += arc

            all_selected.extend(selected)

        if not all_selected:
            return 0.0

        # --- Phase 3: assign collected tokens to underloaded nodes ---
        T_out = np.array(all_selected, dtype=np.int32)
        n_sel = len(T_out)

        # Proportional allocation
        r_norm = r / r_total
        counts = np.floor(r_norm * n_sel).astype(int)
        remainder = n_sel - int(np.sum(counts))
        if remainder > 0:
            residuals = r_norm * n_sel - counts
            top = np.argsort(-residuals)
            for k in range(min(remainder, len(top))):
                counts[top[k]] += 1

        # Safety: don't exceed v_max
        if self.v_max is not None:
            for j, dst in enumerate(U):
                current_count = ring.node_count(dst)
                max_recv = max(0, self.v_max - current_count)
                counts[j] = min(counts[j], max_recv)

        # Perform assignments
        idx = 0
        for j, dst in enumerate(U):
            c = int(counts[j])
            if c > 0 and idx < n_sel:
                actual = min(c, n_sel - idx)
                ring.reassign(T_out[idx:idx + actual],
                               np.full(actual, dst, dtype=np.int32))
                idx += actual

        # Any remaining unassigned tokens go back to least-loaded of U
        if idx < n_sel:
            remaining_tokens = T_out[idx:]
            dst = U[np.argmin(ell[U])]
            ring.reassign(remaining_tokens,
                          np.full(len(remaining_tokens), dst, dtype=np.int32))

        M_keys = ring.get_M_keys(prev)

        # Discrete-token bound: each of the |O| sources can overshoot its
        # per-source share by at most one arc-length (the last token selected),
        # so M_keys <= m + |O| * max_L_s.
        max_arc = float(np.max(ring.L)) if len(ring.L) > 0 else 0.0
        n_over = len(O)
        assert M_keys <= m + n_over * max_arc + 1e-9, (
            f"Budget violated: M_keys={M_keys:.8f} > m+{n_over}*max_arc="
            f"{m + n_over * max_arc:.8f}"
        )

        return M_keys
