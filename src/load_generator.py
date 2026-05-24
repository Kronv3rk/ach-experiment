"""
Load generator module.

Generates batches of key hashes drawn from a (optionally Zipf-distributed)
key universe. Arrival rate is Poisson(lambda). 30% of requests are writes.
"""

import numpy as np


class LoadGenerator:
    """Request load generator.

    Parameters
    ----------
    NK : int
        Size of the key universe.
    zipf_s : float
        Zipf exponent. 0.0 = uniform; >0 = Zipf skew.
    seed : int
        Random seed for reproducible key sequences.
    """

    def __init__(self, NK: int = 50_000, zipf_s: float = 0.0, seed: int = 42):
        self.NK = int(NK)
        self.zipf_s = float(zipf_s)
        self._seed = seed

        # Pre-compute key hashes: each key k has a fixed hash h(k) in [0,1)
        rng = np.random.RandomState(seed)
        self._key_hashes = rng.uniform(0.0, 1.0, self.NK)

        # Pre-compute Zipf probabilities if needed
        if zipf_s > 0.0:
            ranks = np.arange(1, self.NK + 1, dtype=np.float64)
            probs = 1.0 / (ranks ** zipf_s)
            self._key_probs = probs / probs.sum()
        else:
            self._key_probs = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(self, lam: float,
                 rng: np.random.RandomState) -> tuple:
        """Generate a batch of requests for one time step.

        Parameters
        ----------
        lam : float
            Mean arrival rate (Poisson parameter).
        rng : np.random.RandomState
            External RNG for reproducible experiments.

        Returns
        -------
        tuple (key_hashes, write_flags)
            key_hashes  : np.ndarray, shape (n_requests,), values in [0,1)
            write_flags : np.ndarray, shape (n_requests,), dtype bool
        """
        n_requests = int(rng.poisson(lam))

        if n_requests == 0:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=bool)

        # Sample key indices
        if self._key_probs is not None:
            key_indices = rng.choice(self.NK, size=n_requests,
                                     replace=True, p=self._key_probs)
        else:
            key_indices = rng.randint(0, self.NK, size=n_requests)

        key_hashes = self._key_hashes[key_indices]

        # 30% writes
        write_flags = rng.random_sample(n_requests) < 0.30

        return key_hashes, write_flags
