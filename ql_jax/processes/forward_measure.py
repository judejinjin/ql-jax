"""Forward measure process.

Process with drift adjusted for the T-forward measure.
"""

from __future__ import annotations

from dataclasses import dataclass
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class ForwardMeasureProcess:
    """Process under the T-forward measure.

    Under the T-forward measure, the drift of the underlying is:
    mu^T = mu - sigma * sigma_P(t, T)
    where sigma_P(t,T) is the bond price volatility.

    Parameters
    ----------
    x0 : initial value
    sigma : process volatility (float or callable(t))
    T_forward : forward measure maturity
    bond_vol_fn : callable(t, T) -> sigma_P(t,T),
                  or None for zero bond vol (risk-neutral = forward measure)
    drift_fn : callable(t, x) -> drift under risk-neutral, or float
    """
    x0: float
    sigma: float = 0.2
    T_forward: float = 1.0
    _bond_vol: float = 0.0  # simplified: constant bond vol
    _drift_rn: float = 0.0  # risk-neutral drift

    def drift(self, t, x):
        """Forward-measure drift = RN drift - sigma * sigma_bond."""
        return self._drift_rn - self.sigma * self._bond_vol

    def diffusion(self, t, x):
        return self.sigma

    def simulate(self, T, n_steps, n_paths, key):
        dt = T / n_steps
        sqrt_dt = jnp.sqrt(dt)
        x = jnp.ones(n_paths) * self.x0
        paths = [x]
        for step in range(n_steps):
            key, subkey = jax.random.split(key)
            t = step * dt
            dW = jax.random.normal(subkey, (n_paths,)) * sqrt_dt
            x = x + self.drift(t, x) * dt + self.diffusion(t, x) * dW
            paths.append(x)
        return jnp.stack(paths)
