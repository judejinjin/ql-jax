"""Ultimate Forward Rate (UFR) term structure.

Smith-Wilson extrapolation method used by insurance regulators (Solvency II)
to extrapolate yield curves beyond the liquid point to a specified UFR.
"""

import jax.numpy as jnp
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class UltimateForwardTermStructure:
    """Smith-Wilson extrapolation to an ultimate forward rate.

    Parameters
    ----------
    base_curve_fn : callable(t) -> zero rate for t <= last_liquid_point
    last_liquid_point : float – time beyond which extrapolation starts
    ufr : float – ultimate forward rate (e.g. 0.042)
    alpha : float – convergence speed parameter
    """
    base_curve_fn: Callable
    last_liquid_point: float
    ufr: float
    alpha: float = 0.1

    def _wilson_function(self, t, u):
        """Wilson function W(t, u) for Smith-Wilson method."""
        return jnp.exp(-self.ufr * (t + u)) * (
            self.alpha * jnp.minimum(t, u) -
            0.5 * jnp.exp(-self.alpha * jnp.abs(t - u)) +
            0.5 * jnp.exp(-self.alpha * (t + u))
        )

    def zero_rate(self, t):
        """Zero rate: base curve within liquid range, Smith-Wilson beyond."""
        in_liquid = t <= self.last_liquid_point
        base_rate = self.base_curve_fn(jnp.minimum(t, self.last_liquid_point))

        # Smith-Wilson extrapolation
        llp = self.last_liquid_point
        convergence = 1.0 - jnp.exp(-self.alpha * (t - llp))
        extrapolated = base_rate + convergence * (self.ufr - base_rate)

        return jnp.where(in_liquid, base_rate, extrapolated)

    def discount(self, t):
        return jnp.exp(-self.zero_rate(t) * t)

    def forward_rate(self, t1, t2):
        return -jnp.log(self.discount(t2) / self.discount(t1)) / (t2 - t1)
