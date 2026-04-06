"""Mid-point CDS pricing engine.

Uses piecewise-constant hazard rate and mid-point integration for
the protection and premium legs.
"""

from __future__ import annotations

import jax.numpy as jnp


def midpoint_cds_npv(cds, discount_curve_fn, survival_curve_fn):
    """Compute CDS NPV using mid-point integration.

    Parameters
    ----------
    cds : CreditDefaultSwap
    discount_curve_fn : callable(t) -> discount factor P(0,t)
    survival_curve_fn : callable(t) -> survival probability Q(0,t)

    Returns
    -------
    npv : NPV from protection buyer's perspective (positive = buyer receives)
    """
    protection_leg = _protection_leg(cds, discount_curve_fn, survival_curve_fn)
    premium_leg = _premium_leg(cds, discount_curve_fn, survival_curve_fn)

    sign = 1.0 if cds.protection_side == 'buyer' else -1.0
    return sign * (protection_leg - premium_leg)


def cds_fair_spread(cds, discount_curve_fn, survival_curve_fn):
    """Compute the fair CDS spread (par spread).

    The spread at which the CDS NPV = 0.

    Parameters
    ----------
    cds : CreditDefaultSwap
    discount_curve_fn : callable(t) -> P(0,t)
    survival_curve_fn : callable(t) -> Q(0,t)

    Returns
    -------
    fair_spread
    """
    protection = _protection_leg(cds, discount_curve_fn, survival_curve_fn)
    risky_annuity = _risky_annuity(cds, discount_curve_fn, survival_curve_fn)

    return protection / (cds.notional * risky_annuity)


def _protection_leg(cds, discount_fn, survival_fn):
    """Value of the protection leg (what buyer receives on default).

    Protection = (1-R) * N * integral_0^T P(0,t) * dQ(t)
    Approximated at midpoints of premium periods.
    """
    N = cds.notional
    R = cds.recovery_rate
    lgd = (1.0 - R) * N

    dates = cds.payment_dates
    n = cds.n_periods
    dt = cds.payment_frequency

    total = 0.0
    t_prev = 0.0
    for i in range(n):
        t_curr = dates[i]
        t_mid = 0.5 * (t_prev + t_curr)

        P_mid = discount_fn(t_mid)
        Q_prev = survival_fn(t_prev)
        Q_curr = survival_fn(t_curr)
        default_prob = Q_prev - Q_curr  # probability of default in [t_prev, t_curr]

        total = total + lgd * P_mid * default_prob
        t_prev = t_curr

    return total


def _premium_leg(cds, discount_fn, survival_fn):
    """Value of the premium leg (what buyer pays = spread * risky annuity)."""
    return cds.notional * cds.spread * _risky_annuity(cds, discount_fn, survival_fn)


def _risky_annuity(cds, discount_fn, survival_fn):
    """Risky annuity (risky PV01): sum of tau_i * P(0,t_i) * Q(0,t_i).

    Also includes accrual-on-default correction.
    """
    dates = cds.payment_dates
    n = cds.n_periods
    dt = cds.payment_frequency

    annuity = 0.0
    t_prev = 0.0
    for i in range(n):
        t_curr = dates[i]
        tau = dt

        # Survival at payment date
        P_curr = discount_fn(t_curr)
        Q_curr = survival_fn(t_curr)

        # Regular premium payment
        annuity = annuity + tau * P_curr * Q_curr

        # Accrual-on-default: half-period premium if default occurs in period
        Q_prev = survival_fn(t_prev)
        default_prob = Q_prev - Q_curr
        t_mid = 0.5 * (t_prev + t_curr)
        P_mid = discount_fn(t_mid)
        annuity = annuity + 0.5 * tau * P_mid * default_prob

        t_prev = t_curr

    return annuity


def hazard_rate_from_spread(spread, recovery_rate=0.4):
    """Approximate flat hazard rate from CDS spread.

    h ≈ spread / (1 - R)
    """
    return spread / (1.0 - recovery_rate)


def survival_probability(hazard_rate, t):
    """Survival probability under constant hazard rate: Q(t) = exp(-h*t)."""
    return jnp.exp(-hazard_rate * t)
