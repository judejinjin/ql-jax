"""Risky bond pricing engine — credit-risky bond discounting."""

from __future__ import annotations

import jax.numpy as jnp


def risky_bond_npv(
    bond,
    discount_curve_fn,
    survival_curve_fn,
    recovery_rate: float = 0.4,
    settlement_date=None,
) -> float:
    """Price a bond accounting for issuer default risk.

    NPV = sum_i [ cf_i * DF(t_i) * Q(t_i) ]
        + R * sum_i [ notional_i * DF(t_i) * (Q(t_{i-1}) - Q(t_i)) ]

    Parameters
    ----------
    bond : Bond instrument
    discount_curve_fn : t -> discount factor
    survival_curve_fn : t -> survival probability
    recovery_rate : expected recovery rate
    settlement_date : valuation date

    Returns
    -------
    float : NPV
    """
    from ql_jax.time.daycounter import year_fraction

    today = settlement_date or bond.settlement_date() or bond.issue_date
    cfs = bond.cashflows

    npv = 0.0
    t_prev = 0.0

    for cf in cfs:
        if hasattr(cf, 'payment_date'):
            pay_date = cf.payment_date
        else:
            pay_date = cf.date

        t = year_fraction(today, pay_date, "Actual/365 (Fixed)")
        if t <= 0:
            continue

        amount = cf.amount if hasattr(cf, 'amount') else 0.0
        df = discount_curve_fn(t)
        q = survival_curve_fn(t)
        q_prev = survival_curve_fn(t_prev) if t_prev > 0 else 1.0

        # Cash flow conditional on survival
        npv += amount * df * q

        # Recovery on default between t_prev and t
        npv += recovery_rate * bond.notional * df * (q_prev - q)

        t_prev = t

    return npv


def risky_bond_spread(
    bond,
    market_price: float,
    discount_curve_fn,
    recovery_rate: float = 0.4,
    initial_guess: float = 0.01,
) -> float:
    """Implied credit spread from market price.

    Parameters
    ----------
    bond : Bond instrument
    market_price : observed clean price
    discount_curve_fn : risk-free discount curve
    recovery_rate : recovery rate assumption
    initial_guess : starting spread

    Returns
    -------
    float : implied hazard rate (flat)
    """
    from ql_jax.math.solvers.brent import brent_solve

    def objective(h):
        survival_fn = lambda t: jnp.exp(-h * t)
        npv = risky_bond_npv(bond, discount_curve_fn, survival_fn, recovery_rate)
        return npv - market_price * bond.notional / 100.0

    return brent_solve(objective, 1e-6, 1.0, tol=1e-8)
