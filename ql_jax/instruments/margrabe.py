"""Margrabe option — exchange option (option to exchange one asset for another)."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class MargrabeOption:
    """Option to exchange asset 2 for asset 1.

    Call payoff: max(S1 - S2, 0)
    This is equivalent to a European option with S1 as the underlying
    and S2 as the strike that is itself stochastic.

    Parameters
    ----------
    maturity : time to maturity
    quantity1 : quantity of asset 1
    quantity2 : quantity of asset 2
    """
    maturity: float
    quantity1: float = 1.0
    quantity2: float = 1.0

    def payoff(self, s1, s2):
        return jnp.maximum(self.quantity1 * s1 - self.quantity2 * s2, 0.0)
