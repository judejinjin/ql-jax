"""Stochastic process array — joint multi-dimensional process.

Container for an array of correlated 1D stochastic processes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax.numpy as jnp


@dataclass(frozen=True)
class StochasticProcessArray:
    """Array of correlated stochastic processes.

    Parameters
    ----------
    processes : sequence of process objects (each must have drift/diffusion)
    correlation : (n x n) correlation matrix, or None for independent
    """
    processes: tuple
    correlation: object = None

    @staticmethod
    def create(processes: Sequence, correlation=None):
        return StochasticProcessArray(
            processes=tuple(processes),
            correlation=correlation,
        )

    @property
    def size(self):
        return len(self.processes)

    def drift_array(self, t, x):
        """Compute drift for each process."""
        return jnp.array([p.drift(t, x[i]) for i, p in enumerate(self.processes)])

    def diffusion_array(self, t, x):
        """Compute diffusion for each process."""
        return jnp.array([p.diffusion(t, x[i]) for i, p in enumerate(self.processes)])

    def evolve(self, t0, x0, dt, dw):
        """Evolve all processes jointly.

        Parameters
        ----------
        x0 : array of shape (n,) — current state for each process
        dw : array of shape (n,) — independent standard normals * sqrt(dt)

        Returns
        -------
        array of shape (n,) — new states
        """
        n = self.size

        # Apply correlation if present
        if self.correlation is not None:
            # Cholesky decomposition of correlation matrix
            L = jnp.linalg.cholesky(self.correlation)
            z = dw / jnp.sqrt(dt)
            corr_z = L @ z
            dw_corr = corr_z * jnp.sqrt(dt)
        else:
            dw_corr = dw

        # Evolve each process
        new_states = []
        for i, p in enumerate(self.processes):
            x_new = p.evolve(t0, x0[i], dt, dw_corr[i])
            new_states.append(x_new)

        return jnp.array(new_states)
