"""Coupon pricer framework – abstract coupon pricing dispatch."""

import jax.numpy as jnp
from dataclasses import dataclass
from jax.scipy.stats import norm


@dataclass(frozen=True)
class BlackIborCouponPricer:
    """Black-formula pricer for IBOR coupons.

    Parameters
    ----------
    vol : float – Black volatility
    timing_adjustment : str – 'Black76' or 'BivariateLognormal'
    """
    vol: float
    timing_adjustment: str = 'Black76'


@dataclass(frozen=True)
class AnalyticHaganPricer:
    """Analytic Hagan pricer for CMS coupons.

    Parameters
    ----------
    vol_surface : callable(K, T) -> vol
    mean_reversion : float
    """
    vol_surface: object  # callable(K, T) -> vol
    mean_reversion: float = 0.0


@dataclass(frozen=True)
class LinearTSRPricer:
    """Linear terminal swap rate pricer for CMS.

    Parameters
    ----------
    vol_surface : callable(K, T) -> vol
    mean_reversion : float
    """
    vol_surface: object
    mean_reversion: float = 0.0


def black_ibor_coupon_npv(forward, vol, fixing_time, accrual, notional,
                           gearing, spread, discount, floor=None, cap=None):
    """Price an IBOR coupon with optional embedded cap/floor.

    Parameters
    ----------
    forward : float – forward IBOR rate
    vol : float – Black vol
    fixing_time : float – time to fixing
    accrual : float – day count fraction
    notional : float
    gearing, spread : float
    discount : float – discount factor to payment
    floor, cap : float or None

    Returns
    -------
    float – NPV
    """
    effective_rate = gearing * forward + spread
    npv = notional * accrual * effective_rate * discount

    # Embedded floor
    if floor is not None:
        adjusted_floor = (floor - spread) / gearing if gearing != 0 else floor
        floor_val = _black_formula(forward, adjusted_floor, vol, fixing_time,
                                    is_call=False)
        npv += notional * accrual * gearing * floor_val * discount

    # Embedded cap
    if cap is not None:
        adjusted_cap = (cap - spread) / gearing if gearing != 0 else cap
        cap_val = _black_formula(forward, adjusted_cap, vol, fixing_time,
                                  is_call=True)
        npv -= notional * accrual * gearing * cap_val * discount

    return npv


def _black_formula(forward, strike, vol, T, is_call=True):
    """Standard Black formula for caplet/floorlet."""
    total_vol = vol * jnp.sqrt(jnp.maximum(T, 1e-12))
    d1 = (jnp.log(forward / strike) + 0.5 * total_vol**2) / total_vol
    d2 = d1 - total_vol

    if is_call:
        return forward * norm.cdf(d1) - strike * norm.cdf(d2)
    else:
        return strike * norm.cdf(-d2) - forward * norm.cdf(-d1)


def cms_coupon_npv(swap_rate, vol, fixing_time, accrual, notional,
                    gearing, spread, discount, annuity_mapping_fn,
                    floor=None, cap=None):
    """Price a CMS coupon using a generic annuity mapping function.

    Parameters
    ----------
    swap_rate : float – forward swap rate
    vol : float – swaption vol
    fixing_time : float
    accrual, notional, gearing, spread : float
    discount : float – DF to payment
    annuity_mapping_fn : callable(S) -> G(S) – annuity mapping
    floor, cap : float or None

    Returns
    -------
    float – NPV
    """
    # Convexity-adjusted forward
    G = annuity_mapping_fn(swap_rate)
    G_prime = (annuity_mapping_fn(swap_rate + 1e-4) -
               annuity_mapping_fn(swap_rate - 1e-4)) / 2e-4

    convexity = swap_rate + G_prime / G * swap_rate**2 * vol**2 * fixing_time

    effective = gearing * convexity + spread
    npv = notional * accrual * effective * discount

    return npv
