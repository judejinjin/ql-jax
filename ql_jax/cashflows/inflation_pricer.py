"""Inflation coupon pricer – pricing zero-coupon and YoY inflation coupons."""

import jax.numpy as jnp
from dataclasses import dataclass
from jax.scipy.stats import norm


@dataclass(frozen=True)
class InflationCouponPricer:
    """Pricer for inflation-linked coupons.

    Parameters
    ----------
    vol : float – inflation vol
    correlation : float – correlation between rates and inflation
    """
    vol: float = 0.0
    correlation: float = 0.0


def zero_inflation_coupon_npv(notional, base_cpi, fixing_cpi, accrual,
                                payment_date, discount_fn, floor=None, cap=None,
                                vol=0.0, fixing_time=0.0):
    """Price a zero-coupon inflation coupon with optional cap/floor.

    Parameters
    ----------
    notional : float
    base_cpi : float – CPI at base date
    fixing_cpi : float – projected CPI at fixing date
    accrual : float – day count fraction
    payment_date : float
    discount_fn : callable(t) -> DF
    floor, cap : float or None – floor/cap on inflation rate
    vol : float – inflation rate vol
    fixing_time : float – time to fixing

    Returns
    -------
    float – NPV
    """
    inflation_rate = fixing_cpi / base_cpi - 1.0

    npv = notional * accrual * inflation_rate * discount_fn(payment_date)

    # Embedded floor
    if floor is not None and vol > 0 and fixing_time > 0:
        floor_val = _inflation_capfloor(
            inflation_rate, floor, vol, fixing_time, is_cap=False
        )
        npv += notional * accrual * floor_val * discount_fn(payment_date)

    # Embedded cap
    if cap is not None and vol > 0 and fixing_time > 0:
        cap_val = _inflation_capfloor(
            inflation_rate, cap, vol, fixing_time, is_cap=True
        )
        npv -= notional * accrual * cap_val * discount_fn(payment_date)

    return npv


def yoy_inflation_coupon_npv(notional, yoy_rate, accrual, gearing, spread,
                               payment_date, discount_fn, floor=None, cap=None,
                               vol=0.0, fixing_time=0.0):
    """Price a year-on-year inflation coupon with optional cap/floor.

    Parameters
    ----------
    notional : float
    yoy_rate : float – forward YoY inflation rate
    accrual : float
    gearing, spread : float
    payment_date : float
    discount_fn : callable(t) -> DF
    floor, cap : float or None
    vol : float – YoY inflation vol
    fixing_time : float

    Returns
    -------
    float – NPV
    """
    effective = gearing * yoy_rate + spread
    npv = notional * accrual * effective * discount_fn(payment_date)

    if floor is not None and vol > 0 and fixing_time > 0:
        adj_floor = (floor - spread) / gearing if gearing != 0 else floor
        floor_val = _inflation_capfloor(yoy_rate, adj_floor, vol, fixing_time, False)
        npv += notional * accrual * gearing * floor_val * discount_fn(payment_date)

    if cap is not None and vol > 0 and fixing_time > 0:
        adj_cap = (cap - spread) / gearing if gearing != 0 else cap
        cap_val = _inflation_capfloor(yoy_rate, adj_cap, vol, fixing_time, True)
        npv -= notional * accrual * gearing * cap_val * discount_fn(payment_date)

    return npv


def _inflation_capfloor(forward, strike, vol, T, is_cap):
    """Black formula for inflation cap/floor."""
    total_vol = vol * jnp.sqrt(T)
    d1 = (jnp.log(forward / strike) + 0.5 * total_vol**2) / total_vol
    d2 = d1 - total_vol

    if is_cap:
        return forward * norm.cdf(d1) - strike * norm.cdf(d2)
    else:
        return strike * norm.cdf(-d2) - forward * norm.cdf(-d1)
