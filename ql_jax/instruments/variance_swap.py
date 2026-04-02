"""Variance swap instrument."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class VarianceSwap:
    """Variance swap.

    The buyer receives realized variance minus strike variance at maturity.
    NPV = notional * (realized_var - strike_var) * df

    Parameters
    ----------
    strike_variance : the strike (delivery) variance
    notional : notional amount (often expressed per variance point)
    maturity : time to maturity
    is_long : True if long variance (receives realized)
    """
    strike_variance: float
    notional: float
    maturity: float
    is_long: bool = True

    @property
    def strike_volatility(self):
        return jnp.sqrt(self.strike_variance)

    def payoff(self, realized_variance):
        """Payoff at maturity."""
        diff = realized_variance - self.strike_variance
        if not self.is_long:
            diff = -diff
        return self.notional * diff

    def npv(self, realized_variance, discount_factor):
        """Net present value."""
        return self.payoff(realized_variance) * discount_factor


def realized_variance_from_prices(prices, dt, annualize=True):
    """Compute realized variance from a price series.

    Parameters
    ----------
    prices : array of prices
    dt : time step between observations
    annualize : if True, annualize the result

    Returns
    -------
    Realized variance
    """
    log_returns = jnp.diff(jnp.log(prices))
    rv = jnp.mean(log_returns**2)
    if annualize:
        rv = rv / dt
    return rv
