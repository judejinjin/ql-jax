"""ISDA standard CDS pricing engine."""

from __future__ import annotations

import jax.numpy as jnp


def isda_cds_npv(
    spread: float, recovery: float,
    payment_dates: jnp.ndarray,
    day_fractions: jnp.ndarray,
    discount_fn,
    survival_fn,
    notional: float = 1.0,
    is_buyer: bool = True,
    accrual_on_default: bool = True,
) -> float:
    """ISDA standard model CDS NPV.

    Follows ISDA CDS Standard Model conventions:
    - Accrual on default uses midpoint approximation
    - Payment dates assumed to be adjusted for business days

    Parameters
    ----------
    spread : CDS spread (annualized)
    recovery : recovery rate
    payment_dates : array of payment dates (year fracs from valuation)
    day_fractions : day count fractions for each period
    discount_fn : t -> discount factor
    survival_fn : t -> survival probability
    notional : notional amount
    is_buyer : True = protection buyer (pays spread)
    accrual_on_default : include accrued premium on default
    """
    n = payment_dates.shape[0]
    t_prev = jnp.concatenate([jnp.zeros(1), payment_dates[:-1]])

    # Premium leg: spread * sum(tau_i * df_i * Q_i)
    df = jnp.array([discount_fn(t) for t in payment_dates])
    Q = jnp.array([survival_fn(t) for t in payment_dates])
    Q_prev = jnp.array([survival_fn(t) for t in t_prev])

    premium = spread * jnp.sum(day_fractions * df * Q)

    # Accrual on default: sum(0.5 * tau_i * df_mid * (Q_{i-1} - Q_i))
    if accrual_on_default:
        t_mid = 0.5 * (t_prev + payment_dates)
        df_mid = jnp.array([discount_fn(t) for t in t_mid])
        accrual = spread * jnp.sum(0.5 * day_fractions * df_mid * (Q_prev - Q))
        premium = premium + accrual

    # Protection leg: (1 - R) * sum(df_mid * (Q_{i-1} - Q_i))
    t_mid = 0.5 * (t_prev + payment_dates)
    df_mid = jnp.array([discount_fn(t) for t in t_mid])
    protection = (1.0 - recovery) * jnp.sum(df_mid * (Q_prev - Q))

    # Buyer pays premium, receives protection
    npv = protection - premium
    if not is_buyer:
        npv = -npv

    return notional * npv


def isda_cds_fair_spread(
    recovery: float,
    payment_dates: jnp.ndarray,
    day_fractions: jnp.ndarray,
    discount_fn,
    survival_fn,
    accrual_on_default: bool = True,
) -> float:
    """ISDA standard model CDS fair spread (par spread).

    Parameters
    ----------
    recovery, payment_dates, day_fractions, discount_fn, survival_fn:
        Same as isda_cds_npv.
    """
    n = payment_dates.shape[0]
    t_prev = jnp.concatenate([jnp.zeros(1), payment_dates[:-1]])

    df = jnp.array([discount_fn(t) for t in payment_dates])
    Q = jnp.array([survival_fn(t) for t in payment_dates])
    Q_prev = jnp.array([survival_fn(t) for t in t_prev])

    # Risky duration
    risky_pv01 = jnp.sum(day_fractions * df * Q)
    if accrual_on_default:
        t_mid = 0.5 * (t_prev + payment_dates)
        df_mid = jnp.array([discount_fn(t) for t in t_mid])
        risky_pv01 = risky_pv01 + jnp.sum(0.5 * day_fractions * df_mid * (Q_prev - Q))

    # Protection leg PV per unit notional
    t_mid = 0.5 * (t_prev + payment_dates)
    df_mid = jnp.array([discount_fn(t) for t in t_mid])
    protection = (1.0 - recovery) * jnp.sum(df_mid * (Q_prev - Q))

    return protection / risky_pv01
