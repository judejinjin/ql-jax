"""Cashflow utilities: leg builders and rate averaging."""

from __future__ import annotations

import jax.numpy as jnp


def cashflow_vectors_fixed_leg(notionals, accrual_times, rates, day_count_fractions=None):
    """Build fixed leg cashflow vector.

    Parameters
    ----------
    notionals : array of notional amounts per period
    accrual_times : array [n, 2] of (start, end) year fractions
    rates : array of fixed coupon rates
    day_count_fractions : optional precomputed

    Returns
    -------
    dict with 'amounts', 'payment_times'
    """
    notionals = jnp.asarray(notionals, dtype=jnp.float64)
    rates = jnp.asarray(rates, dtype=jnp.float64)
    accrual_times = jnp.asarray(accrual_times, dtype=jnp.float64)

    if day_count_fractions is not None:
        dcf = jnp.asarray(day_count_fractions, dtype=jnp.float64)
    elif accrual_times.ndim == 2:
        dcf = accrual_times[:, 1] - accrual_times[:, 0]
    else:
        # 1D: treat as payment times starting from 0
        dcf = jnp.diff(jnp.concatenate([jnp.array([0.0]), accrual_times]))

    amounts = notionals * rates * dcf
    payment_times = accrual_times if accrual_times.ndim == 1 else accrual_times[:, 1]
    return {'amounts': amounts, 'payment_times': payment_times}


def cashflow_vectors_floating_leg(
    notionals, accrual_times, forward_rates,
    gearings=None, spreads=None, day_count_fractions=None,
):
    """Build floating leg cashflow vector.

    Parameters
    ----------
    notionals : array
    accrual_times : array [n, 2]
    forward_rates : array
    gearings : optional array (default 1.0)
    spreads : optional array (default 0.0)

    Returns
    -------
    dict with 'amounts', 'payment_times'
    """
    notionals = jnp.asarray(notionals, dtype=jnp.float64)
    forward_rates = jnp.asarray(forward_rates, dtype=jnp.float64)
    accrual_times = jnp.asarray(accrual_times, dtype=jnp.float64)
    n = len(notionals)

    g = jnp.ones(n, dtype=jnp.float64) if gearings is None else jnp.asarray(gearings, dtype=jnp.float64)
    s = jnp.zeros(n, dtype=jnp.float64) if spreads is None else jnp.asarray(spreads, dtype=jnp.float64)
    if day_count_fractions is not None:
        dcf = jnp.asarray(day_count_fractions, dtype=jnp.float64)
    elif accrual_times.ndim == 2:
        dcf = accrual_times[:, 1] - accrual_times[:, 0]
    else:
        dcf = jnp.diff(jnp.concatenate([jnp.array([0.0]), accrual_times]))

    amounts = notionals * (g * forward_rates + s) * dcf
    payment_times = accrual_times if accrual_times.ndim == 1 else accrual_times[:, 1]
    return {'amounts': amounts, 'payment_times': payment_times}


class RateAveraging:
    """Rate averaging methods for overnight coupons."""

    @staticmethod
    def compound(rates, accrual_fractions):
        """Compound averaging: prod(1 + r_i * tau_i) - 1."""
        rates = jnp.asarray(rates, dtype=jnp.float64)
        taus = jnp.asarray(accrual_fractions, dtype=jnp.float64)
        total_tau = jnp.sum(taus)
        compounded = jnp.prod(1.0 + rates * taus) - 1.0
        return compounded / jnp.maximum(total_tau, 1e-10)

    @staticmethod
    def simple(rates, accrual_fractions):
        """Simple weighted averaging."""
        rates = jnp.asarray(rates, dtype=jnp.float64)
        taus = jnp.asarray(accrual_fractions, dtype=jnp.float64)
        return jnp.sum(rates * taus) / jnp.maximum(jnp.sum(taus), 1e-10)
