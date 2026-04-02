"""Bachelier (normal model) and additional cap/floor engines."""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.stats import norm as jnorm


# ---------------------------------------------------------------------------
# Bachelier (normal vol) cap/floor pricing
# ---------------------------------------------------------------------------

def bachelier_capfloor_price(
    cap_floor, discount_curve_fn, forward_rates, normal_vols,
):
    """Price a cap/floor using Bachelier (normal) model.

    Parameters
    ----------
    cap_floor : CapFloor instrument
    discount_curve_fn : t -> discount factor
    forward_rates : array of forward rates per period
    normal_vols : array of normal volatilities per period

    Returns
    -------
    float : total cap/floor price
    """
    total = 0.0
    for i, cf in enumerate(cap_floor.legs):
        if not hasattr(cf, 'accrual_start'):
            continue
        T = cf.payment_date.serial_float if hasattr(cf.payment_date, 'serial_float') else float(i + 1) * 0.25
        F = forward_rates[i] if i < len(forward_rates) else cf.rate
        K = cap_floor.strike
        sigma_n = normal_vols[i] if i < len(normal_vols) else normal_vols[-1]
        tau = cf.accrual_fraction if hasattr(cf, 'accrual_fraction') else 0.25
        df = discount_curve_fn(T)
        omega = cap_floor.cap_floor_type  # 1 = cap, -1 = floor
        notional = cf.nominal

        price = _bachelier_caplet_normal(F, K, T, sigma_n, tau, df, omega, notional)
        total = total + price
    return total


def _bachelier_caplet_normal(F, K, T, sigma_n, tau, df, omega, notional):
    """Single caplet/floorlet under Bachelier (normal) model."""
    sigma_n = jnp.float64(sigma_n)
    T = jnp.maximum(jnp.float64(T), 1e-10)
    sqrtT = jnp.sqrt(T)

    d = (F - K) / (sigma_n * sqrtT + 1e-15)
    price = omega * (F - K) * jnorm.cdf(omega * d) + sigma_n * sqrtT * jnorm.pdf(d)
    return notional * tau * df * price


# ---------------------------------------------------------------------------
# Flat normal vol cap/floor
# ---------------------------------------------------------------------------

def bachelier_capfloor_price_flat(
    cap_floor, discount_curve_fn, forward_rates, flat_normal_vol,
):
    """Price a cap/floor with a flat normal volatility."""
    n = len(forward_rates)
    vols = [flat_normal_vol] * n
    return bachelier_capfloor_price(cap_floor, discount_curve_fn, forward_rates, vols)


# ---------------------------------------------------------------------------
# Tree-based cap/floor pricing using Hull-White trinomial tree
# ---------------------------------------------------------------------------

def tree_capfloor_price(
    cap_floor, a, sigma, discount_curve_fn, n_steps=100,
):
    """Price a cap/floor using a Hull-White trinomial tree.

    Parameters
    ----------
    cap_floor : CapFloor instrument
    a : HW mean-reversion
    sigma : HW volatility
    discount_curve_fn : t -> discount factor
    n_steps : number of tree steps
    """
    from ql_jax.engines.lattice.short_rate_tree import hw_trinomial_tree

    # Use capfloor maturity
    T = 0.0
    for cf in cap_floor.legs:
        if hasattr(cf, 'payment_date'):
            pd = cf.payment_date.serial_float if hasattr(cf.payment_date, 'serial_float') else 1.0
            T = max(T, pd)
    if T <= 0:
        T = 5.0  # default

    tree = hw_trinomial_tree(a, sigma, T, n_steps, discount_curve_fn)

    # Roll back each caplet/floorlet through tree and sum
    total = 0.0
    omega = cap_floor.cap_floor_type
    K = cap_floor.strike

    for cf in cap_floor.legs:
        if not hasattr(cf, 'accrual_start'):
            continue
        notional = cf.nominal if hasattr(cf, 'nominal') else 1.0
        tau = cf.accrual_fraction if hasattr(cf, 'accrual_fraction') else 0.25
        # Approximate payoff at tree nodes
        pay = notional * tau * jnp.maximum(omega * (tree['rates'] - K), 0.0)
        # Discount to today using tree
        df_tree = tree.get('discount_factors', jnp.exp(-tree['rates'] * tree['dt']))
        total = total + jnp.mean(pay * df_tree)

    return total
