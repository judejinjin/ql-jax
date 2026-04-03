"""CMS spread coupons and lognormal CMS spread pricer.

A CMS spread coupon pays the spread between two CMS rates,
optionally capped and/or floored.
"""

from __future__ import annotations

import jax.numpy as jnp
from dataclasses import dataclass
from ql_jax.math.distributions.bivariate import bivariate_normal_cdf
from jax.scipy.stats import norm as jax_norm


@dataclass(frozen=True)
class CmsSpreadCoupon:
    """Coupon paying CMS spread = CMS_1 - CMS_2 + spread.

    Parameters
    ----------
    notional : float
    accrual_fraction : float – year fraction for coupon accrual
    cms_rate_1 : float – first CMS fixing
    cms_rate_2 : float – second CMS fixing
    gearing : float – multiplier on spread (default 1.0)
    spread : float – additive spread (default 0.0)
    cap : float or None
    floor : float or None
    """
    notional: float
    accrual_fraction: float
    cms_rate_1: float
    cms_rate_2: float
    gearing: float = 1.0
    spread: float = 0.0
    cap: float = None
    floor: float = None

    @property
    def raw_rate(self):
        """Unmodified spread rate."""
        return self.cms_rate_1 - self.cms_rate_2

    @property
    def adjusted_rate(self):
        """Rate after gearing and spread."""
        rate = self.gearing * self.raw_rate + self.spread
        if self.floor is not None:
            rate = jnp.maximum(rate, self.floor)
        if self.cap is not None:
            rate = jnp.minimum(rate, self.cap)
        return rate

    @property
    def amount(self):
        """Cash flow amount."""
        return self.notional * self.accrual_fraction * self.adjusted_rate


@dataclass(frozen=True)
class DigitalCmsSpreadCoupon:
    """Digital coupon on CMS spread: pays fixed rate if spread > strike.

    Parameters
    ----------
    notional : float
    accrual_fraction : float
    cms_rate_1, cms_rate_2 : float
    strike : float – digital barrier
    digital_rate : float – rate paid if condition met
    is_call : bool – True = pays if spread > strike, False = pays if spread < strike
    """
    notional: float
    accrual_fraction: float
    cms_rate_1: float
    cms_rate_2: float
    strike: float
    digital_rate: float
    is_call: bool = True

    @property
    def amount(self):
        spread = self.cms_rate_1 - self.cms_rate_2
        if self.is_call:
            pays = jnp.where(spread > self.strike, self.digital_rate, 0.0)
        else:
            pays = jnp.where(spread < self.strike, self.digital_rate, 0.0)
        return self.notional * self.accrual_fraction * pays


def lognormal_cms_spread_price(
    forward1: float, forward2: float,
    vol1: float, vol2: float,
    rho: float, expiry: float,
    strike: float = 0.0,
    is_call: bool = True,
    discount: float = 1.0,
    notional: float = 1.0,
) -> float:
    """Price a CMS spread caplet/floorlet using log-normal model.

    Assumes both CMS rates follow correlated log-normal processes:
        S1 = F1 * exp(-0.5*v1^2*T + v1*sqrt(T)*Z1)
        S2 = F2 * exp(-0.5*v2^2*T + v2*sqrt(T)*Z2)
        corr(Z1, Z2) = rho

    The spread payoff max(S1 - S2 - K, 0) is priced via Margrabe-type formula
    extended to non-zero strike using Kirk's approximation.

    Parameters
    ----------
    forward1, forward2 : CMS forward rates
    vol1, vol2 : Black vols for CMS rates
    rho : correlation between the two CMS rates
    expiry : time to fixing
    strike : spread strike
    is_call : call on spread (True) or floor (False)
    discount : discount factor
    notional : notional amount
    """
    # Kirk's approximation: treat S2+K as a single asset
    if is_call:
        # Call: E[max(S1 - S2 - K, 0)]
        f_adj = forward2 + strike
        if f_adj <= 0:
            return float(discount * notional * jnp.maximum(forward1 - forward2 - strike, 0.0))
        sigma_s = jnp.sqrt(
            vol1 ** 2
            - 2.0 * rho * vol1 * vol2 * forward2 / f_adj
            + (vol2 * forward2 / f_adj) ** 2
        )
        sigma_s = jnp.maximum(sigma_s, 1e-10) * jnp.sqrt(expiry)
        d1 = (jnp.log(forward1 / f_adj) + 0.5 * sigma_s ** 2) / sigma_s
        d2 = d1 - sigma_s
        price = forward1 * jax_norm.cdf(d1) - f_adj * jax_norm.cdf(d2)
    else:
        # Floor: E[max(K + S2 - S1, 0)] = E[max(S1 - S2 - K, 0)] - (F1 - F2 - K)
        call_price = lognormal_cms_spread_price(
            forward1, forward2, vol1, vol2, rho, expiry,
            strike, True, 1.0, 1.0,
        )
        price = call_price - (forward1 - forward2 - strike)

    return float(discount * notional * jnp.maximum(price, 0.0))
