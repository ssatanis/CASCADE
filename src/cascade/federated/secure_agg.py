"""Secure aggregation via pairwise additive masking (Bonawitz et al. 2017).

Each pair of clients (i, j) deterministically derives a shared random vector
r_ij from a pair seed. Client i adds +r_ij, client j adds −r_ij. Every mask
appears once with each sign, so the masks cancel in the SUM — the server learns
Σ updates exactly while each individual masked update looks uniformly random.

This is the mandatory floor of FSCP. It is NOT sufficient alone (post-aggregation
inference is still possible), which is why `dp` layers differential privacy on
top. Here we provide an honest, exact simulation used to prove the cancellation
property in tests; a production deployment derives pair seeds via authenticated
Diffie-Hellman key agreement rather than a shared integer.
"""

from __future__ import annotations

import numpy as np


def pairwise_masks(n_clients: int, dim: int, seed: int = 0) -> list[np.ndarray]:
    """Return per-client mask vectors whose sum is the zero vector."""
    if n_clients < 1:
        raise ValueError("need >= 1 client")
    masks = [np.zeros(dim, dtype=float) for _ in range(n_clients)]
    for i in range(n_clients):
        for j in range(i + 1, n_clients):
            rng = np.random.default_rng([seed, i, j])  # shared pair seed
            r = rng.standard_normal(dim)
            masks[i] += r
            masks[j] -= r
    return masks


class SecureAggregator:
    """Simulates the secure-aggregation round over a set of client updates."""

    def __init__(self, seed: int = 0):
        self.seed = seed

    def mask(self, updates: list[np.ndarray]) -> list[np.ndarray]:
        """Add cancelling pairwise masks to each client update."""
        updates = [np.asarray(u, dtype=float) for u in updates]
        n = len(updates)
        if n == 0:
            return []
        dim = updates[0].shape[0]
        if any(u.shape[0] != dim for u in updates):
            raise ValueError("all updates must have the same dimension")
        masks = pairwise_masks(n, dim, self.seed)
        return [u + m for u, m in zip(updates, masks)]

    def aggregate(self, masked_updates: list[np.ndarray]) -> np.ndarray:
        """Server side: sum the masked updates (masks cancel → exact Σ)."""
        if not masked_updates:
            raise ValueError("no updates to aggregate")
        return np.sum(np.vstack([np.asarray(m, dtype=float) for m in masked_updates]), axis=0)

    def run(self, updates: list[np.ndarray]) -> np.ndarray:
        """Convenience: mask then aggregate. Returns Σ updates."""
        return self.aggregate(self.mask(updates))
