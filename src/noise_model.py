"""
Noise model for telemetry corruption experiments.

Applies three independent perturbations to raw telemetry:
  1. Gaussian additive noise (sigma_noise)
  2. Temporal lag (observation delay in steps)
  3. Random missing data (replaced with last-good value)
"""

import numpy as np
from collections import deque


class NoiseModel:
    """Telemetry noise and corruption model.

    Parameters
    ----------
    sigma_noise : float
        Standard deviation of Gaussian noise added to observations.
        0.0 = no noise.
    lag : int
        Number of steps of observation delay. 0 = no lag.
    p_miss : float
        Probability that any single observation is missing (NaN replaced
        by last good value). 0.0 = no missing data.
    seed : int
        Random seed for the internal RNG (used in apply() if no rng given).
    """

    def __init__(self, sigma_noise: float = 0.0, lag: int = 0,
                 p_miss: float = 0.0, seed: int = 0):
        self.sigma_noise = float(sigma_noise)
        self.lag = int(lag)
        self.p_miss = float(p_miss)
        self._rng = np.random.RandomState(seed)

        # Circular buffer for lag: stores past observations
        self._buffer: deque = deque(maxlen=max(lag + 1, 1))
        self._last_good = None   # last complete observation (for imputation)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def apply(self, raw: np.ndarray,
              rng: np.random.RandomState = None,
              gauss_delta: np.ndarray = None,
              miss_mask: np.ndarray = None) -> np.ndarray:
        """Apply noise model to a raw telemetry snapshot.

        Processing order:
          1. Store in lag buffer, retrieve delayed observation.
          2. Apply Gaussian noise (pre-computed delta or fresh sample).
          3. Randomly mask entries (pre-computed mask or fresh sample).
          4. Clip to [0, 1].

        Parameters
        ----------
        raw : np.ndarray
            Shape (n_nodes, 3), values in [0, 1].
        rng : np.random.RandomState, optional
            External RNG; uses internal one if None. Only consulted when
            gauss_delta / miss_mask are not pre-supplied.
        gauss_delta : np.ndarray, optional
            Pre-computed Gaussian noise array (same shape as raw).
            When supplied, used directly instead of sampling from rng.
            Pass zeros or None to skip Gaussian noise.
        miss_mask : np.ndarray of bool, optional
            Pre-computed missing-data mask (same shape as raw, dtype bool).
            When supplied, used directly instead of sampling from rng.

        Returns
        -------
        np.ndarray
            Corrupted telemetry, same shape as raw, clipped to [0, 1].

        Notes
        -----
        Callers that run multiple algorithms on the same time-step should
        pre-compute gauss_delta and miss_mask once and pass them to every
        call so that all algorithms receive identical noise perturbations.
        """
        if rng is None:
            rng = self._rng

        raw = np.asarray(raw, dtype=np.float64)

        # --- Step 1: lag ---
        self._buffer.append(raw.copy())
        if len(self._buffer) <= self.lag:
            # Not enough history yet; return zeros (conservative)
            obs = np.zeros_like(raw)
        else:
            # buffer[0] is the oldest; retrieve delayed observation
            obs = self._buffer[0].copy()

        # --- Step 2: additive Gaussian noise ---
        if self.sigma_noise > 0.0:
            if gauss_delta is not None:
                obs = obs + gauss_delta
            else:
                obs = obs + rng.normal(0.0, self.sigma_noise, size=obs.shape)

        # --- Step 3: missing data imputation ---
        if self.p_miss > 0.0:
            if miss_mask is not None:
                mask = miss_mask
            else:
                mask = rng.random_sample(obs.shape) < self.p_miss
            if self._last_good is not None:
                obs[mask] = self._last_good[mask]
            else:
                obs[mask] = 0.0

        # Update last good
        self._last_good = obs.copy()

        # --- Step 4: clip ---
        obs = np.clip(obs, 0.0, 1.0)
        return obs

    def reset(self):
        """Reset internal buffer and last-good state."""
        self._buffer.clear()
        self._last_good = None
