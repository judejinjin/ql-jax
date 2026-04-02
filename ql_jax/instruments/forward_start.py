"""Forward-starting vanilla option."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class ForwardVanillaOption:
    """Forward-starting vanilla option.

    The strike is set at some future date (reset_date) as a fraction
    (moneyness) of the spot price at that time.

    Parameters
    ----------
    moneyness : strike = moneyness * S(reset_date)
    reset_date : time at which the strike is determined
    maturity : final expiry
    is_call : True for call
    """
    moneyness: float
    reset_date: float
    maturity: float
    is_call: bool = True

    def payoff(self, spot_at_reset, spot_at_expiry):
        strike = self.moneyness * spot_at_reset
        if self.is_call:
            return jnp.maximum(spot_at_expiry - strike, 0.0)
        return jnp.maximum(strike - spot_at_expiry, 0.0)
