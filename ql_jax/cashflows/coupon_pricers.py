"""Additional coupon pricers and cashflow utilities."""

import jax.numpy as jnp
from jax.scipy.stats import norm
from dataclasses import dataclass


@dataclass
class StrippedCapFlooredCoupon:
    """Strips the embedded cap/floor from a capped/floored coupon.

    Given a capped/floored coupon, decomposes into:
    - pure floating coupon (no cap/floor)
    - caplet (if capped)
    - floorlet (if floored)

    Parameters
    ----------
    rate : floating rate
    notional : coupon notional
    accrual : accrual fraction
    cap : cap rate (None if uncapped)
    floor : floor rate (None if unfloored)
    """
    rate: float
    notional: float
    accrual: float
    cap: float = None
    floor: float = None

    def pure_coupon_amount(self):
        """Amount of the pure coupon (no cap/floor)."""
        return self.rate * self.notional * self.accrual

    def caplet_amount(self, vol, df, T):
        """Value of the embedded caplet."""
        if self.cap is None:
            return 0.0
        return _bachelier_caplet(self.rate, self.cap, vol, T, df,
                                  self.notional * self.accrual)

    def floorlet_amount(self, vol, df, T):
        """Value of the embedded floorlet."""
        if self.floor is None:
            return 0.0
        return _bachelier_floorlet(self.rate, self.floor, vol, T, df,
                                    self.notional * self.accrual)


def quanto_coupon_amount(domestic_rate, fx_rate, fx_vol, rate_vol,
                          correlation, notional, accrual, df):
    """Quanto coupon pricer — FX-adjusted floating coupon.

    Adjusts a foreign rate to domestic currency using quanto adjustment.

    Parameters
    ----------
    domestic_rate : domestic forward rate
    fx_rate : FX rate (domestic per foreign)
    fx_vol : FX volatility
    rate_vol : rate volatility
    correlation : rate-FX correlation
    notional : coupon notional
    accrual : accrual fraction
    df : discount factor

    Returns
    -------
    adjusted_amount : quanto-adjusted coupon amount
    """
    # Quanto adjustment: F_adj = F * exp(-rho * sigma_r * sigma_fx * T)
    # For simplicity, use the convexity adjustment
    adjusted_rate = domestic_rate - correlation * rate_vol * fx_vol
    return adjusted_rate * notional * accrual * df


def linear_tsr_cms_rate(swap_rate, annuity, swap_rate_vol, T, shift=0.0):
    """Linear Terminal Swap Rate (TSR) model for CMS coupon.

    The TSR model gives a convexity-adjusted CMS rate:
    E[S(T)] ≈ forward_swap_rate + convexity_adjustment

    Parameters
    ----------
    swap_rate : forward swap rate
    annuity : PV01 (annuity)
    swap_rate_vol : swaption vol
    T : fixing time
    shift : lognormal shift (for shifted lognormal)

    Returns
    -------
    cms_rate : convexity-adjusted CMS rate
    """
    # Convexity adjustment in linear TSR is:
    # E[S(T)] ≈ S0 + S0^2 * sigma^2 * T / annuity * dA/dS
    # For a simple approximation:
    ca = (swap_rate + shift)**2 * swap_rate_vol**2 * T / (1.0 + swap_rate)
    return swap_rate + ca


def black_overnight_coupon(overnight_rate, strike, vol, T, df, notional,
                            accrual, option_type=1):
    """Black model for overnight indexed coupon.

    Parameters
    ----------
    overnight_rate : forward overnight rate
    strike : strike rate
    vol : volatility
    T : fixing time
    df : discount factor
    notional : notional
    accrual : accrual fraction
    option_type : 1=cap, -1=floor

    Returns
    -------
    value : option value
    """
    phi = option_type
    vol_sqrt_T = vol * jnp.sqrt(T)
    d1 = (jnp.log(overnight_rate / strike) + 0.5 * vol**2 * T) / (vol_sqrt_T + 1e-15)
    d2 = d1 - vol_sqrt_T
    value = phi * df * notional * accrual * (
        overnight_rate * norm.cdf(phi * d1) - strike * norm.cdf(phi * d2))
    return float(jnp.maximum(value, 0.0))


@dataclass
class DividendSchedule:
    """Container for discrete dividend schedule."""
    ex_div_times: jnp.ndarray  # (n,) times
    amounts: jnp.ndarray  # (n,) dividend amounts

    def pv(self, discount_factors):
        """Present value of all future dividends."""
        return float((self.amounts * jnp.asarray(discount_factors)).sum())

    def dividends_between(self, t1, t2):
        """Get dividends between two times."""
        mask = (self.ex_div_times > t1) & (self.ex_div_times <= t2)
        return self.amounts[mask]


def _bachelier_caplet(F, K, vol, T, df, notional):
    """Bachelier (normal) caplet price."""
    d = (F - K) / (vol * jnp.sqrt(T) + 1e-15)
    price = df * notional * ((F - K) * norm.cdf(d) +
                              vol * jnp.sqrt(T) * norm.pdf(d))
    return float(jnp.maximum(price, 0.0))


def _bachelier_floorlet(F, K, vol, T, df, notional):
    """Bachelier (normal) floorlet price."""
    d = (F - K) / (vol * jnp.sqrt(T) + 1e-15)
    price = df * notional * ((K - F) * norm.cdf(-d) +
                              vol * jnp.sqrt(T) * norm.pdf(d))
    return float(jnp.maximum(price, 0.0))
