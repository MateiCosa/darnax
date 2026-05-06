"""Spatially-aware FC adapters for use alongside conv layers."""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from darnax.modules.fully_connected import FullyConnected, FrozenRescaledFullyConnected


class ChannelWBack(FrozenRescaledFullyConnected):
    """(N, num_classes) → FC → broadcast over H×W → (N, H, W, C).

    Label feedback with channel-only spatial specificity: every spatial position
    receives the same channel-level signal with no spatial bias.
    """
    out_spatial: tuple = eqx.field(static=True)

    def __init__(self, num_classes: int, H: int, W: int, Cv: int,
                 strength: float, key: jax.Array):
        super().__init__(in_features=num_classes, out_features=Cv,
                         strength=strength, threshold=0.0, key=key)
        self.out_spatial = (H, W, Cv)

    def __call__(self, y: jax.Array, rng=None) -> jax.Array:
        H, W, Cv = self.out_spatial
        ch = super().__call__(y, rng)          # (N, C)
        return jnp.broadcast_to(ch[:, None, None, :], (ch.shape[0], H, W, Cv))


class PooledFlattenFC(FullyConnected):
    """POOL×POOL avg-pool + flatten + FC: (N, H, W, C) → (N, n_classes)."""
    pool: int = eqx.field(static=True)

    def __init__(self, pool: int, H: int, W: int, C_in: int, n_classes: int,
                 strength: float, threshold: float, key: jax.Array,
                 lr: float = 1.0, weight_decay: float = 0.0):
        super().__init__(
            in_features=(H // pool) * (W // pool) * C_in,
            out_features=n_classes,
            strength=strength, threshold=threshold,
            key=key, lr=lr, weight_decay=weight_decay,
        )
        self.pool = pool

    def _pool(self, x: jax.Array) -> jax.Array:
        N, H, W, Cv = x.shape
        p = self.pool
        return x.reshape(N, H // p, p, W // p, p, Cv).mean(axis=(2, 4)).reshape(N, -1)

    def __call__(self, x: jax.Array, rng=None) -> jax.Array:
        return self._pool(x) @ self.W

    def backward(self, x, y, y_hat, gate=None):
        return super().backward(self._pool(x), y, y_hat, gate)
